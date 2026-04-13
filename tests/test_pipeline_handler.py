from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
import warnings

from src import GateBlock as TopLevelGateBlock
from src.mlpipelineholder import ExecutionError, GateBlock, PipelineHandler, RegistrationError, ResolutionError
from src.mlpipelineholder.models import ArtifactRecord, RuntimeValueReference, TorchStateArtifactRecord


@dataclass
class DemoConfig:
    base: int
    factor: int = 2


def produce_seed(base: int) -> int:
    return base + 1


def multiply(seed: int, factor: int) -> int:
    return seed * factor


def branch_left(seed: int) -> int:
    return seed + 10


def branch_right(seed: int) -> int:
    return seed + 20


def combine(left: int, right: int) -> int:
    return left + right


def save_text(seed: int) -> str:
    return f"value={seed}"


def memory_text(seed: int) -> str:
    return f"memory={seed}"


def save_json(seed: int) -> dict[str, int]:
    return {"seed": seed, "double": seed * 2}


def save_array(seed: int):
    from importlib import import_module

    return import_module("numpy").array([seed, seed + 1, seed + 2])


def read_text(saved_blob: str) -> str:
    return saved_blob.upper()


def late_seed(seed: int) -> int:
    return seed + 100


def pair(seed: int) -> tuple[int, int]:
    return seed, seed + 1


def needs_missing(missing: int) -> int:
    return missing


def logger_step(seed: int, logger) -> int:
    logger.info(f"seed={seed}")
    logger.result(f"final-seed={seed}")
    return seed


def print_step(seed: int) -> int:
    print(f"printed-seed={seed}")
    return seed


def debug_and_info_step(seed: int, logger) -> int:
    logger.debug(f"debug-seed={seed}")
    logger.info(f"info-seed={seed}")
    return seed


def verbose_step(seed: int, verbose: bool = True) -> int:
    return seed


def always_skip() -> bool:
    return False


def child_value(seed: int, base: int) -> int:
    return seed + base


def unique_child_output(seed: int) -> int:
    return seed * 10


def always_true() -> bool:
    return True


def always_false() -> bool:
    return False


def needs_seed_gate(seed: int) -> bool:
    return seed > 0


def local_variadic_sum(base: int, *extra_values: int, factor: int = 1, **extra_items: int) -> int:
    return (base + sum(extra_values) + sum(extra_items.values())) * factor


def build_torch_model():
    from importlib import import_module

    torch = import_module("torch")
    return torch.nn.Linear(2, 1)


def build_torch_optimizer():
    from importlib import import_module

    torch = import_module("torch")
    model = torch.nn.Linear(2, 1)
    return torch.optim.SGD(model.parameters(), lr=0.1)


def build_torch_model_optimizer_pairs():
    from importlib import import_module

    torch = import_module("torch")
    me_model = torch.nn.Linear(2, 1)
    me_optimizer = torch.optim.SGD(me_model.parameters(), lr=0.1)
    predictor_model = torch.nn.Linear(3, 1)
    predictor_optimizer = torch.optim.Adam(predictor_model.parameters(), lr=0.01)
    return me_model, me_optimizer, predictor_model, predictor_optimizer


def build_unserializable_object():
    from threading import Lock

    return Lock()


def use_stock_project_root(stock_project_root: str) -> str:
    return stock_project_root


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class PipelineHandlerTests(unittest.TestCase):
    def test_pipeline_runs_full_and_partial_flow(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("demo", DemoConfig(base=3, factor=4), tmp_path)
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])

            branch = pipeline.add_block("branch", 2)
            branch.register_function(branch_left, ["left"])
            branch.register_function(branch_right, ["right"])

            final = pipeline.add_block("final", 3)
            final.register_function(combine, ["total"])
            final.register_function(multiply, ["scaled_total"])

            run = pipeline.run_all()

            self.assertEqual(run.status, "success")
            self.assertEqual(pipeline.para_value_dict["seed"], 4)
            self.assertEqual(pipeline.para_value_dict["left"], 14)
            self.assertEqual(pipeline.para_value_dict["right"], 24)
            self.assertEqual(pipeline.para_value_dict["total"], 38)
            self.assertEqual(pipeline.para_value_dict["scaled_total"], 16)

            pipeline.run_block("setup", overrides={"base": 10})
            self.assertEqual(pipeline.para_value_dict["seed"], 11)
            self.assertNotIn("left", pipeline.para_value_dict)
            self.assertNotIn("total", pipeline.para_value_dict)

            pipeline.run_from("branch")
            self.assertEqual(pipeline.para_value_dict["left"], 21)
            self.assertEqual(pipeline.para_value_dict["right"], 31)
            self.assertEqual(pipeline.para_value_dict["total"], 52)

    def test_disk_artifact_is_saved_and_loaded_for_downstream_use(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("disk", DemoConfig(base=2), tmp_path)
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])
            disk_block = pipeline.add_block("disk_write", 2)
            disk_block.register_function(save_text, ["saved_blob"], save_to_disk=["saved_blob"])
            consumer = pipeline.add_block("consumer", 3)
            consumer.register_function(read_text, ["upper_blob"])

            pipeline.run_all()

            artifact = pipeline.para_value_dict["saved_blob"]
            self.assertIsInstance(artifact, ArtifactRecord)
            self.assertTrue(Path(artifact.file_path).exists())
            self.assertEqual(pipeline.para_value_dict["upper_blob"], "VALUE=3")

    def test_project_can_be_saved_and_loaded(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            project_dir = tmp_path / "project"
            pipeline = PipelineHandler("persisted", DemoConfig(base=5), project_dir)
            block = pipeline.add_block("setup", 1)
            block.register_function(produce_seed, ["seed"])
            pipeline.run_all()

            save_dir = tmp_path / "save_bundle"
            pipeline.save_project(save_dir)
            loaded = PipelineHandler.load_project(save_dir)

            self.assertEqual(loaded.registration_name, "persisted")
            self.assertEqual(loaded.para_value_dict["seed"], 6)
            self.assertEqual(list(loaded.blocks_by_name), ["setup"])

    def test_pipeline_logger_starts_with_blank_file_on_create(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            metadata_root = tmp_path / "metadata"
            metadata_root.mkdir(parents=True, exist_ok=True)
            log_path = metadata_root / "pipeline.log"
            log_path.write_text("old log\n", encoding="utf-8")

            pipeline = PipelineHandler("blank-log", DemoConfig(base=1), tmp_path)

            self.assertEqual(pipeline.logger.log_file_path.read_text(encoding="utf-8"), "")

    def test_load_pipeline_recreates_blank_runtime_log(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            project_dir = tmp_path / "project"
            pipeline = PipelineHandler("persisted", DemoConfig(base=5), project_dir)
            block = pipeline.add_block("setup", 1)
            block.register_function(produce_seed, ["seed"])
            pipeline.run_all()
            save_dir = tmp_path / "bundle"
            pipeline.save_pipeline(save_dir)

            log_path = save_dir / "metadata" / "pipeline.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("stale log\n", encoding="utf-8")

            loaded = PipelineHandler.load_pipeline(save_dir)

            self.assertEqual(loaded.logger.log_file_path.read_text(encoding="utf-8"), "")

    def test_save_pipeline_does_not_export_log_by_default(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            project_dir = tmp_path / "project"
            pipeline = PipelineHandler("persisted", DemoConfig(base=5), project_dir)
            block = pipeline.add_block("setup", 1)
            block.register_function(produce_seed, ["seed"])
            pipeline.run_all()

            save_dir = tmp_path / "bundle"
            pipeline.save_pipeline(save_dir)

            self.assertFalse((save_dir / "exported.log").exists())

    def test_save_pipeline_can_export_log_when_requested(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            project_dir = tmp_path / "project"
            pipeline = PipelineHandler("persisted", DemoConfig(base=5), project_dir)
            block = pipeline.add_block("setup", 1)
            block.register_function(produce_seed, ["seed"])
            pipeline.run_all()

            save_dir = tmp_path / "bundle"
            export_log = tmp_path / "exported.log"
            pipeline.save_pipeline(save_dir, save_log_to_file=export_log)

            self.assertTrue(export_log.exists())
            self.assertIn(" INFO ", export_log.read_text(encoding="utf-8"))

    def test_save_pipeline_persists_live_torch_model_as_artifact(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("torch-save", {}, tmp_path / "project")
            block = pipeline.add_block("model", 1)
            block.register_function(build_torch_model, ["model_obj"])
            pipeline.run_all()

            save_dir = tmp_path / "bundle"
            pipeline.save_pipeline(save_dir)
            loaded = PipelineHandler.load_pipeline(save_dir)
            loaded_value = loaded.get_value("model_obj")

            self.assertEqual(type(loaded.para_value_dict["model_obj"]).__name__, "ArtifactRecord")
            self.assertEqual(loaded_value.__class__.__name__, "Linear")

    def test_save_pipeline_warns_and_uses_reference_for_unserializable_value(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("ref-save", {}, tmp_path / "project")
            block = pipeline.add_block("weird", 1)
            block.register_function(build_unserializable_object, ["weird_obj"])
            pipeline.run_all()

            save_dir = tmp_path / "bundle"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipeline.save_pipeline(save_dir)
            loaded = PipelineHandler.load_pipeline(save_dir)

            self.assertTrue(
                any("reference placeholder" in str(item.message) for item in caught)
            )
            self.assertIsInstance(loaded.get_value("weird_obj"), RuntimeValueReference)

    def test_save_pipeline_persists_live_torch_optimizer_as_artifact(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("torch-save", {}, tmp_path / "project")
            block = pipeline.add_block("optimizer", 1)
            block.register_function(build_torch_optimizer, ["optimizer_obj"])
            pipeline.run_all()

            save_dir = tmp_path / "bundle"
            pipeline.save_pipeline(save_dir)
            loaded = PipelineHandler.load_pipeline(save_dir)
            loaded_value = loaded.get_value("optimizer_obj")

            self.assertIsInstance(loaded.para_value_dict["optimizer_obj"], TorchStateArtifactRecord)
            self.assertIsInstance(loaded_value, TorchStateArtifactRecord)

    def test_save_pipeline_keeps_optimizer_model_pairs_separate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("torch-pairs", {}, tmp_path / "project")
            block = pipeline.add_block("pairs", 1)
            block.register_function(
                build_torch_model_optimizer_pairs,
                ["me_model", "me_optimizer", "predictor_model", "predictor_optimizer"],
            )
            pipeline.run_all()

            save_dir = tmp_path / "bundle"
            pipeline.save_pipeline(save_dir)
            loaded = PipelineHandler.load_pipeline(save_dir)

            me_optimizer = loaded.get_value("me_optimizer")
            predictor_optimizer = loaded.get_value("predictor_optimizer")

            self.assertIsInstance(me_optimizer, TorchStateArtifactRecord)
            self.assertIsInstance(predictor_optimizer, TorchStateArtifactRecord)
            self.assertEqual(me_optimizer.metadata.get("linked_model_variable"), "me_model")
            self.assertEqual(
                predictor_optimizer.metadata.get("linked_model_variable"),
                "predictor_model",
            )

    def test_save_pipeline_warns_when_optimizer_has_no_linked_model(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("torch-save", {}, tmp_path / "project")
            block = pipeline.add_block("optimizer", 1)
            block.register_function(build_torch_optimizer, ["me_optimizer"])
            pipeline.run_all()

            save_dir = tmp_path / "bundle"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipeline.save_pipeline(save_dir)

            self.assertTrue(
                any("without a linked model artifact" in str(item.message) for item in caught)
            )

    def test_logger_uses_persistent_file_handle(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("logging", DemoConfig(base=1), tmp_path)

            first_handle = pipeline.logger._file_handle
            pipeline.logger.info("first")
            second_handle = pipeline.logger._file_handle
            pipeline.logger.info("second")

            self.assertIsNotNone(first_handle)
            self.assertIs(first_handle, second_handle)

    def test_logger_disables_file_logging_after_oserror(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("logging", DemoConfig(base=1), tmp_path)
            file_handle = pipeline.logger._file_handle
            self.assertIsNotNone(file_handle)
            if file_handle is None:
                self.fail("logger file handle should exist")
            file_handle.close()
            pipeline.logger._file_handle = MagicMock()
            pipeline.logger._file_handle.write.side_effect = OSError(24, "Too many open files")
            pipeline.logger._file_handle.flush.side_effect = OSError(24, "Too many open files")

            pipeline.logger.info("still logs to console")

            self.assertFalse(pipeline.logger._file_logging_enabled)

    def test_logger_flush_keeps_log_export_working(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            project_dir = tmp_path / "project"
            pipeline = PipelineHandler("persisted", DemoConfig(base=5), project_dir)
            pipeline.logger.info("before export")

            export_log = tmp_path / "exported.log"
            pipeline.save_pipeline(save_log_to_file=export_log)

            self.assertIn("before export", export_log.read_text(encoding="utf-8"))

    def test_duplicate_outputs_override_later_and_are_reported(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("dup", DemoConfig(base=1), tmp_path)
            first = pipeline.add_block("first", 1)
            second = pipeline.add_block("second", 2)

            first.register_function(produce_seed, ["seed"])
            second.register_function(late_seed, ["seed"])
            pipeline.run_all()

            self.assertEqual(pipeline.get_value("seed"), 102)
            conflicts = pipeline.get_output_conflicts()
            self.assertEqual(conflicts["seed"]["created_by"], "dup/first")
            self.assertEqual(conflicts["seed"]["overridden_by"], ["dup/second"])

    def test_float_priority_branch_group_executes_first_matching_node_only(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("branching", {"pick_first": True, "base": 1, "factor": 2}, tmp_path)
            first = pipeline.add_block("first", 5.1)
            first.register_function(produce_seed, ["seed"])

            second_child = PipelineHandler("second_child", {"pick_first": False, "base": 100}, tmp_path / "child")
            second_child.set_gate_block("pick_first")
            child_block = second_child.add_block("child_block", 1.0)
            child_block.register_function(child_value, ["seed"])
            pipeline.add_child_pipeline(second_child, 5.3)

            final = pipeline.add_block("final", 6.0)
            final.register_function(multiply, ["scaled_total"])

            pipeline.run_all()

            self.assertEqual(pipeline.get_value("seed"), 2)
            self.assertEqual(pipeline.get_value("scaled_total"), 4)

    def test_get_priority_group_returns_group_names_and_active_node(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("branching", {"run_child": True, "base": 1}, tmp_path)
            first_child = PipelineHandler("child_a", {"base": 2}, tmp_path / "a")
            first_child.set_gate_block("run_child")
            first_child.add_block("child_block", 1.0)
            pipeline.add_child_pipeline(first_child, 5.1)

            second = pipeline.add_block("second", 5.2)
            second.register_function(produce_seed, ["seed"])

            names, active = pipeline.get_priority_group(5)

            self.assertEqual(names, ["child_a", "second"])
            self.assertEqual(active, "child_a")

    def test_get_priority_group_assumes_true_when_callable_gate_inputs_are_not_ready(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("branching", {"base": 1}, tmp_path)
            child = PipelineHandler("child_a", {"base": 2}, tmp_path / "a")
            child.set_gate_block(needs_seed_gate)
            child_block = child.add_block("child_block", 1.0)
            child_block.register_function(child_value, ["child_result"])
            pipeline.add_child_pipeline(child, 5.1)

            second = pipeline.add_block("second", 5.2)
            second.register_function(produce_seed, ["seed"])

            names, active = pipeline.get_priority_group(5)

            self.assertEqual(names, ["child_a", "second"])
            self.assertEqual(active, "child_a")

    def test_same_integer_priority_uses_next_node_when_first_child_gate_is_false(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("branching", {"run_child": False, "base": 1, "factor": 3}, tmp_path)
            first_child = PipelineHandler("child_a", {"base": 2}, tmp_path / "a")
            first_child.set_gate_block("run_child")
            child_block = first_child.add_block("child_block", 1.0)
            child_block.register_function(child_value, ["seed"])
            pipeline.add_child_pipeline(first_child, 5.1)

            second = pipeline.add_block("second", 5.2)
            second.register_function(produce_seed, ["seed"])
            final = pipeline.add_block("final", 6.0)
            final.register_function(multiply, ["scaled_total"])

            pipeline.run_all()

            self.assertEqual(pipeline.get_value("seed"), 2)
            self.assertEqual(pipeline.get_value("scaled_total"), 6)

    def test_chart_greys_child_pipeline_with_false_config_gate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            parent = PipelineHandler("parent", {"run_child": False, "base": 1}, tmp_path / "parent")
            child = PipelineHandler("child", {"base": 2}, tmp_path / "child")
            child.set_gate_block("run_child")
            child_block = child.add_block("child_block", 1.0)
            child_block.register_function(child_value, ["child_result"])
            parent.add_child_pipeline(child, 5.1)

            chart = parent.describe_pipeline()

            self.assertRegex(chart, r"\x1b\[(3[1-6])m")

    def test_chart_greys_child_pipeline_when_config_misses_expected_value(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            parent = PipelineHandler(
                "parent", {"model_cls": "cls_b", "base": 1}, tmp_path / "parent"
            )
            child = PipelineHandler("child", {"base": 2}, tmp_path / "child")
            child.set_gate_block("model_cls", "cls_a")
            child_block = child.add_block("child_block", 1.0)
            child_block.register_function(child_value, ["child_result"])
            parent.add_child_pipeline(child, 5.1)

            chart = parent.describe_pipeline()

            self.assertRegex(chart, r"\x1b\[(3[1-6])m")
            self.assertRegex(chart, r"\x1b\[(37|97)m[├└│─ ]+")

    def test_overridden_disk_artifact_is_cleaned_when_later_value_is_in_memory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("cleanup", DemoConfig(base=2), tmp_path)
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])
            first = pipeline.add_block("first", 2)
            first.register_function(save_text, ["shared"], save_to_disk=["shared"])
            second = pipeline.add_block("second", 3)
            second.register_function(memory_text, ["shared"])

            pipeline.run_all()

            artifact_dir = tmp_path / "artifacts"
            artifact_files = list(artifact_dir.rglob("*")) if artifact_dir.exists() else []
            self.assertEqual(pipeline.get_value("shared"), "memory=3")
            self.assertFalse(any(path.is_file() for path in artifact_files))

    def test_earlier_block_individual_run_does_not_see_later_override(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("stale", DemoConfig(base=1, factor=1), tmp_path)
            first = pipeline.add_block("first", 1)
            second = pipeline.add_block("second", 2)
            first.register_function(produce_seed, ["seed"])
            second.register_function(late_seed, ["seed"])
            pipeline.run_all()

            pipeline.run_block("first", overrides={"base": 10})

            self.assertEqual(pipeline.get_value("seed"), 11)

    def test_multiple_outputs_require_matching_return_arity(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("pair", DemoConfig(base=2), tmp_path)
            first = pipeline.add_block("first", 1)
            first.register_function(produce_seed, ["seed"])
            second = pipeline.add_block("second", 2)
            second.register_function(pair, ["first_value", "second_value"])

            pipeline.run_all()
            self.assertEqual(pipeline.para_value_dict["first_value"], 3)
            self.assertEqual(pipeline.para_value_dict["second_value"], 4)

    def test_multi_output_error_reports_returned_type_and_declared_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("shape-error", DemoConfig(base=2), tmp_path)
            first = pipeline.add_block("first", 1)
            first.register_function(produce_seed, ["seed"])
            second = pipeline.add_block("second", 2)
            second.register_function(branch_left, ["left", "right"])

            with self.assertRaises(ExecutionError) as exc_info:
                pipeline.run_all()

            message = str(exc_info.exception)
            self.assertIn("branch_left", message)
            self.assertIn("returned int", message)
            self.assertIn("['left', 'right']", message)

    def test_missing_argument_raises_resolution_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("missing", DemoConfig(base=1), tmp_path)
            block = pipeline.add_block("broken", 1)
            block.register_function(needs_missing, ["x"])

            with self.assertRaises(ResolutionError):
                pipeline.run_all()

    def test_duplicate_block_priority_is_rejected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("priority", DemoConfig(base=1), tmp_path)
            first = pipeline.add_block("first", 1)
            with self.assertRaises(RegistrationError):
                pipeline.add_block("second", 1)

            self.assertIsNotNone(first)
            self.assertEqual(list(pipeline.nodes_by_name), ["first"])

    def test_duplicate_block_can_be_replaced_with_force(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("priority", DemoConfig(base=1), tmp_path)
            first = pipeline.add_block("first", 1)
            first.register_function(produce_seed, ["seed"])

            replacement = pipeline.add_block("first", 1, forced=True)
            replacement.register_function(branch_left, ["left"])

            self.assertEqual(list(pipeline.nodes_by_name), ["first"])
            self.assertIs(pipeline.nodes_by_name["first"], replacement)

    def test_different_block_name_same_priority_raises_even_with_force(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("priority", DemoConfig(base=1), tmp_path)
            pipeline.add_block("first", 1)

            with self.assertRaises(RegistrationError):
                pipeline.add_block("second", 1, forced=True)

    def test_gate_block_can_be_replaced_with_force(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("gate", DemoConfig(base=1), tmp_path)
            first = pipeline.add_gate_block(always_true)
            second = pipeline.add_gate_block(always_skip)
            replacement = pipeline.add_gate_block(always_skip, forced=True)

            self.assertIsNotNone(first)
            self.assertIsNone(second)
            self.assertIsNotNone(replacement)
            gate_block = pipeline.gate_block
            self.assertIsNotNone(gate_block)
            if gate_block is None:
                self.fail("gate block should exist")
            self.assertEqual(gate_block.registration.function_name, "always_skip")

    def test_callable_gate_expected_value_round_trips(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("gate", DemoConfig(base=1), tmp_path / "project")
            pipeline.set_gate_block(always_false, expected_value=False)
            block = pipeline.add_block("setup", 1)
            block.register_function(produce_seed, ["seed"])

            save_dir = tmp_path / "bundle"
            pipeline.save_pipeline(save_dir)
            loaded = PipelineHandler.load_pipeline(save_dir)
            run = loaded.run_all()

            self.assertEqual(run.status, "success")
            self.assertEqual(loaded.get_value("seed"), 2)

    def test_boolean_config_field_can_define_gate_block(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("gate", {"run_enabled": False}, tmp_path)
            pipeline.add_gate_block("run_enabled")
            block = pipeline.add_block("setup", 1)
            block.register_function(produce_seed, ["seed"])

            run = pipeline.run_all()

            self.assertEqual(run.status, "skipped")
            self.assertIsNone(pipeline.get_value("seed"))

    def test_config_gate_can_use_custom_expected_value(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("gate", {"model_cls": "cls_b", "base": 1}, tmp_path)
            pipeline.add_gate_block("model_cls", "cls_a")
            block = pipeline.add_block("setup", 1)
            block.register_function(produce_seed, ["seed"])

            run = pipeline.run_all()

            self.assertEqual(run.status, "skipped")
            self.assertIsNone(pipeline.get_value("seed"))

    def test_config_gate_custom_expected_value_runs_when_matched(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("gate", {"model_cls": "cls_a", "base": 1}, tmp_path)
            pipeline.add_gate_block("model_cls", "cls_a")
            block = pipeline.add_block("setup", 1)
            block.register_function(produce_seed, ["seed"])

            run = pipeline.run_all()

            self.assertEqual(run.status, "success")
            self.assertEqual(pipeline.get_value("seed"), 2)

    def test_boolean_config_gate_round_trips_with_new_api(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("gate", {"run_enabled": False}, tmp_path / "project")
            pipeline.add_gate_block("run_enabled")
            block = pipeline.add_block("setup", 1)
            block.register_function(produce_seed, ["seed"])

            save_dir = tmp_path / "bundle"
            pipeline.save_pipeline(save_dir)
            loaded = PipelineHandler.load_pipeline(save_dir)
            run = loaded.run_all()

            self.assertEqual(run.status, "skipped")
            self.assertIsNone(loaded.get_value("seed"))

    def test_custom_expected_value_gate_round_trips(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler(
                "gate",
                {"model_cls": "cls_b", "base": 1},
                tmp_path / "project",
            )
            pipeline.add_gate_block("model_cls", "cls_a")
            block = pipeline.add_block("setup", 1)
            block.register_function(produce_seed, ["seed"])

            save_dir = tmp_path / "bundle"
            pipeline.save_pipeline(save_dir)
            loaded = PipelineHandler.load_pipeline(save_dir)
            run = loaded.run_all()

            self.assertEqual(run.status, "skipped")
            self.assertIsNone(loaded.get_value("seed"))

    def test_update_config_overrides_known_fields(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("config", DemoConfig(base=1, factor=2), tmp_path)
            pipeline.update_config({"factor": 9})

            self.assertEqual(getattr(pipeline.config, "factor"), 9)

    def test_none_configuration_is_treated_as_empty(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("config", None, tmp_path)

            pipeline.update_config({"new_value": 9})

            self.assertEqual(pipeline.config, {"new_value": 9})

    def test_get_full_config_includes_nested_parent_chain(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            grandparent = PipelineHandler(
                "grandparent", {"shared": "grand", "grand": 1}, tmp_path / "grandparent"
            )
            parent = PipelineHandler("parent", {"shared": "parent", "parent": 2}, tmp_path / "parent")
            child = PipelineHandler("child", {"shared": "child", "child": 3}, tmp_path / "child")

            grandparent.add_child_pipeline(parent, 1)
            parent.add_child_pipeline(child, 1)

            self.assertEqual(
                child.get_full_config(),
                {"shared": "child", "grand": 1, "parent": 2, "child": 3},
            )

    def test_get_config_value_prefers_child_over_parents(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            grandparent = PipelineHandler(
                "grandparent", {"shared": "grand", "grand": 1}, tmp_path / "grandparent"
            )
            parent = PipelineHandler("parent", {"shared": "parent", "parent": 2}, tmp_path / "parent")
            child = PipelineHandler("child", {"shared": "child", "child": 3}, tmp_path / "child")

            grandparent.add_child_pipeline(parent, 1)
            parent.add_child_pipeline(child, 1)

            self.assertEqual(child.get_config_value("shared"), "child")
            self.assertEqual(child.get_config_value("parent"), 2)
            self.assertEqual(child.get_config_value("grand"), 1)

    def test_get_config_value_raises_for_missing_key(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("config", {"present": 1}, tmp_path)

            with self.assertRaises(ResolutionError):
                pipeline.get_config_value("missing")

    def test_save_pipeline_defaults_to_project_root(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("save-default", DemoConfig(base=4), tmp_path)
            block = pipeline.add_block("setup", 1)
            block.register_function(produce_seed, ["seed"])
            pipeline.run_all()

            saved_path = pipeline.save_pipeline()

            self.assertEqual(saved_path, tmp_path)
            self.assertTrue((tmp_path / "pipeline_state.pkl").exists())

    def test_update_config_skips_names_conflicting_with_declared_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("config", DemoConfig(base=1, factor=2), tmp_path)
            block = pipeline.add_block("setup", 1)
            block.register_function(produce_seed, ["seed"])

            pipeline.update_config({"seed": 99})

            self.assertFalse(hasattr(pipeline.config, "seed"))

    def test_update_config_allows_new_non_conflicting_fields(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("config", DemoConfig(base=1, factor=2), tmp_path)

            pipeline.update_config({"missing": 9})

            self.assertEqual(getattr(pipeline.config, "missing"), 9)

    def test_get_value_loads_disk_backed_artifact(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("values", DemoConfig(base=2), tmp_path)
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])
            disk_block = pipeline.add_block("disk_write", 2)
            disk_block.register_function(save_text, ["saved_blob"], save_to_disk=["saved_blob"])
            pipeline.run_all()

            self.assertEqual(pipeline.get_value("saved_blob"), "value=3")

    def test_get_block_returns_registered_block(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("lookup", DemoConfig(base=2), tmp_path)
            setup = pipeline.add_block("setup", 1)

            self.assertIs(pipeline.get_block("setup"), setup)

    def test_get_child_pipeline_returns_registered_child(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            parent = PipelineHandler("parent", DemoConfig(base=2), tmp_path / "parent")
            child = PipelineHandler("child", DemoConfig(base=3), tmp_path / "child")
            parent.add_child_pipeline(child, 1)

            self.assertIs(parent.get_child_pipeline("child"), child)

    def test_get_block_rejects_child_pipeline_name(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            parent = PipelineHandler("parent", DemoConfig(base=2), tmp_path / "parent")
            child = PipelineHandler("child", DemoConfig(base=3), tmp_path / "child")
            parent.add_child_pipeline(child, 1)

            with self.assertRaises(RegistrationError):
                parent.get_block("child")

    def test_get_child_pipeline_rejects_block_name(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("lookup", DemoConfig(base=2), tmp_path)
            pipeline.add_block("setup", 1)

            with self.assertRaises(RegistrationError):
                pipeline.get_child_pipeline("setup")

    def test_reset_gate_block_clears_existing_gate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("gate", {"run_enabled": False, "base": 1}, tmp_path)
            pipeline.add_gate_block("run_enabled")
            block = pipeline.add_block("setup", 1)
            block.register_function(produce_seed, ["seed"])

            pipeline.reset_gate_block()
            run = pipeline.run_all()

            self.assertEqual(run.status, "success")
            self.assertEqual(pipeline.get_value("seed"), 2)

    def test_json_artifact_uses_json_serializer(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("json", DemoConfig(base=2), tmp_path)
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])
            block = pipeline.add_block("json_write", 2)
            block.register_function(save_json, ["json_blob"], save_to_disk=["json_blob"])
            pipeline.run_all()

            artifact = pipeline.para_value_dict["json_blob"]
            self.assertEqual(artifact.serializer, "json")
            self.assertEqual(pipeline.get_value("json_blob"), {"seed": 3, "double": 6})

    def test_numpy_artifact_uses_numpy_serializer(self) -> None:
        from importlib import import_module

        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("numpy", DemoConfig(base=2), tmp_path)
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])
            block = pipeline.add_block("numpy_write", 2)
            block.register_function(save_array, ["array_blob"], save_to_disk=["array_blob"])
            pipeline.run_all()

            artifact = pipeline.para_value_dict["array_blob"]
            self.assertEqual(artifact.serializer, "numpy")
            np = import_module("numpy")
            np.testing.assert_array_equal(pipeline.get_value("array_blob"), np.array([3, 4, 5]))

    def test_describe_pipeline_contains_blocks_functions_and_io(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("describe", DemoConfig(base=2), tmp_path)
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])
            disk_block = pipeline.add_block("disk_write", 2)
            disk_block.register_function(save_text, ["saved_blob"], save_to_disk=["saved_blob"])
            third = pipeline.add_block("third", 3)
            third.register_function(verbose_step, ["kept_seed"])

            chart = strip_ansi(pipeline.describe_pipeline())

            self.assertIn("PipelineHandler(describe)", chart)
            self.assertIn("[1] setup", chart)
            self.assertIn("produce_seed(base) -> seed", chart)
            self.assertIn("[2] disk_write", chart)
            self.assertIn("save_text(seed) -> saved_blob*", chart)
            self.assertIn("verbose_step(seed)", chart)
            self.assertNotIn("verbose_step(seed, verbose)", chart)
            self.assertNotIn("-> bool", chart)

    def test_chart_shows_block_scoped_args_and_kwargs_only_when_referenced(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler(
                "describe-helpers",
                {
                    "base_value": 1,
                    "factor_value": 2,
                    "arg_one": 3,
                    "arg_two": 4,
                    "kw_bonus": 5,
                },
                tmp_path,
            )
            block = pipeline.add_block("block", 1)
            block.register_args("args_a", ("arg_one", "arg_two"))
            block.register_kwargs("kwargs_a", {"bonus": "kw_bonus"})
            block.register_args("unused_args", ("arg_one",))
            block.register_function(
                local_variadic_sum,
                ["result"],
                param_mapping={"base": "base_value", "factor": "factor_value"},
                var_pos_name="args_a",
                var_kw_name="kwargs_a",
            )

            chart = strip_ansi(pipeline.describe_pipeline())

            self.assertIn("local_variadic_sum(base_value, factor_value, args_a, kwargs_a)", chart)
            self.assertNotIn("unused_args", chart)

    def test_str_and_repr_show_pipeline_chart(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("describe", DemoConfig(base=2), tmp_path)
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])

            self.assertEqual(strip_ansi(str(pipeline)), strip_ansi(pipeline.describe_pipeline()))
            self.assertEqual(strip_ansi(repr(pipeline)), strip_ansi(pipeline.describe_pipeline()))

    def test_logger_is_injected_and_result_history_is_recorded(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("logger", DemoConfig(base=4), tmp_path)
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])
            logging_block = pipeline.add_block("logging", 2)
            logging_block.register_function(logger_step, ["logged_seed"])
            pipeline.run_all()

            history = pipeline.get_result_history()
            self.assertEqual(len(history), 1)
            self.assertIn(" RESULT final-seed=5", history[0])
            log_text = (tmp_path / "metadata" / "pipeline.log").read_text(encoding="utf-8")
            self.assertIn(" INFO seed=5", log_text)
            self.assertIn(" RESULT final-seed=5", log_text)

    def test_print_result_history_writes_result_entries(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("logger", DemoConfig(base=4), tmp_path)
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])
            logging_block = pipeline.add_block("logging", 2)
            logging_block.register_function(logger_step, ["logged_seed"])
            pipeline.run_all()

            output = StringIO()
            with patch("sys.stdout", output):
                pipeline.print_result_history()

            self.assertIn(" RESULT final-seed=5", output.getvalue())

    def test_clear_result_history_keeps_disk_log(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("logger", DemoConfig(base=4), tmp_path)
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])
            logging_block = pipeline.add_block("logging", 2)
            logging_block.register_function(logger_step, ["logged_seed"])
            pipeline.run_all()

            log_text_before = (tmp_path / "metadata" / "pipeline.log").read_text(encoding="utf-8")
            pipeline.clear_result_history()

            self.assertEqual(pipeline.get_result_history(), [])
            log_text_after = (tmp_path / "metadata" / "pipeline.log").read_text(encoding="utf-8")
            self.assertEqual(log_text_before, log_text_after)

    def test_print_output_is_tee_logged_by_default(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("print", DemoConfig(base=4), tmp_path)
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])
            printer = pipeline.add_block("printer", 2)
            printer.register_function(print_step, ["printed_seed"])

            output = StringIO()
            with patch("sys.stdout", output):
                pipeline.run_all()

            log_text = (tmp_path / "metadata" / "pipeline.log").read_text(encoding="utf-8")
            self.assertIn("printed-seed=5", output.getvalue())
            self.assertIn(" PRINT printed-seed=5", log_text)

    def test_print_output_can_be_logger_only(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("print", DemoConfig(base=4), tmp_path)
            pipeline.set_print_capture_mode("logger_only")
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])
            printer = pipeline.add_block("printer", 2)
            printer.register_function(print_step, ["printed_seed"])

            output = StringIO()
            with patch("sys.stdout", output):
                pipeline.run_all()

            log_text = (tmp_path / "metadata" / "pipeline.log").read_text(encoding="utf-8")
            self.assertNotIn("printed-seed=5", output.getvalue())
            self.assertIn(" PRINT printed-seed=5", log_text)

    def test_set_log_level_filters_debug_but_keeps_info(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("logging", DemoConfig(base=4), tmp_path)
            pipeline.set_log_level("info")
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])
            logging_block = pipeline.add_block("logging", 2)
            logging_block.register_function(debug_and_info_step, ["logged_seed"])

            pipeline.run_all()

            log_text = (tmp_path / "metadata" / "pipeline.log").read_text(encoding="utf-8")
            self.assertIn(" DEBUG debug-seed=5", log_text)
            self.assertIn(" INFO info-seed=5", log_text)

    def test_set_log_level_rejects_unknown_level(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("logging", DemoConfig(base=4), tmp_path)

            with self.assertRaises(RegistrationError):
                pipeline.set_log_level("nope")

    def test_save_project_defaults_to_project_root(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("save-default", DemoConfig(base=4), tmp_path)
            block = pipeline.add_block("setup", 1)
            block.register_function(produce_seed, ["seed"])
            pipeline.run_all()

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                saved_path = pipeline.save_project()

            self.assertEqual(saved_path, tmp_path)
            self.assertTrue((tmp_path / "pipeline_state.pkl").exists())
            self.assertTrue(
                any("historical function behavior" in str(item.message) for item in caught)
            )

    def test_grandchild_project_root_rebases_under_grandparent(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            grandparent = PipelineHandler("grandparent", DemoConfig(base=1), tmp_path / "grandparent")
            parent = PipelineHandler("parent", DemoConfig(base=2), tmp_path / "parent")
            grandchild = PipelineHandler("grandchild", DemoConfig(base=3), tmp_path / "grandchild")

            child_block = grandchild.add_block("work", 1)
            child_block.register_function(produce_seed, ["seed"])
            parent.add_child_pipeline(grandchild, 1)
            grandparent.add_child_pipeline(parent, 1, forced=True)

            expected_root = (
                grandparent.project_root / "children" / "parent" / "children" / "grandchild"
            )
            self.assertEqual(grandchild.project_root, expected_root)
            self.assertEqual(grandchild.metadata_root, expected_root / "metadata")

    def test_child_attachment_removes_old_root_after_move(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            parent = PipelineHandler("parent", DemoConfig(base=1), tmp_path / "parent")
            child_root = tmp_path / "child"
            child_root.mkdir(parents=True, exist_ok=True)
            (child_root / "marker.txt").write_text("moved", encoding="utf-8")
            child = PipelineHandler("child", DemoConfig(base=2), child_root)

            parent.add_child_pipeline(child, 1)

            self.assertFalse(child_root.exists())
            self.assertTrue((parent.project_root / "children" / "child" / "marker.txt").exists())

    def test_load_project_emits_function_preservation_warning(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            project_dir = tmp_path / "project"
            pipeline = PipelineHandler("persisted", DemoConfig(base=5), project_dir)
            block = pipeline.add_block("setup", 1)
            block.register_function(produce_seed, ["seed"])
            pipeline.run_all()
            save_dir = tmp_path / "save_bundle"
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                pipeline.save_project(save_dir)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                PipelineHandler.load_project(save_dir)

            self.assertTrue(
                any("historical function snapshots" in str(item.message) for item in caught)
            )

    def test_attached_child_reads_historical_result_lines_from_current_log_format(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            parent = PipelineHandler("parent", DemoConfig(base=4), tmp_path / "parent")
            child = PipelineHandler("child", DemoConfig(base=4), tmp_path / "child")
            setup = child.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])
            logging_block = child.add_block("logging", 2)
            logging_block.register_function(logger_step, ["logged_seed"])
            child.run_all()

            parent.add_child_pipeline(child, 1)

            history = child.get_result_history()
            self.assertTrue(any(" RESULT final-seed=5" in line for line in history))

    def test_loaded_child_pipeline_keeps_nested_project_root(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            parent = PipelineHandler("parent", DemoConfig(base=5), tmp_path / "parent")
            child = PipelineHandler("child", DemoConfig(base=5), tmp_path / "child")
            block = child.add_block("setup", 1)
            block.register_function(produce_seed, ["seed"])
            parent.add_child_pipeline(child, 1)

            save_dir = tmp_path / "bundle"
            parent.save_pipeline(save_dir)
            loaded = PipelineHandler.load_pipeline(save_dir)
            loaded_child = loaded.get_child_pipeline("child")

            self.assertEqual(loaded_child.project_root, save_dir / "children" / "child")

    def test_nested_pipeline_with_gate_round_trips_through_save_load(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            parent = PipelineHandler("parent", DemoConfig(base=3, factor=4), tmp_path / "parent")
            setup = parent.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])

            child = PipelineHandler("child", DemoConfig(base=50, factor=1), tmp_path / "child")
            child.set_gate_block(always_skip)
            child_block = child.add_block("child_unique", 1)
            child_block.register_function(unique_child_output, ["child_only"])
            parent.add_child_pipeline(child, 2)
            parent.run_all()

            save_dir = tmp_path / "bundle"
            parent.save_project(save_dir)
            loaded = PipelineHandler.load_project(save_dir)
            loaded.run_all()

            self.assertEqual(loaded.get_value("seed"), 4)
            self.assertIsNone(loaded.get_value("child_only"))

    def test_remove_block_invalidates_removed_and_downstream_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("remove", DemoConfig(base=3, factor=4), tmp_path)
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])
            branch = pipeline.add_block("branch", 2)
            branch.register_function(branch_left, ["left"])
            final = pipeline.add_block("final", 3)
            final.register_function(multiply, ["scaled_total"])
            pipeline.run_all()

            pipeline.remove_block("branch")

            self.assertNotIn("branch", pipeline.blocks_by_name)
            self.assertNotIn("left", pipeline.para_value_dict)
            self.assertNotIn("scaled_total", pipeline.para_value_dict)
            self.assertIn("seed", pipeline.para_value_dict)

    def test_child_pipeline_can_use_parent_outputs_and_own_config(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            parent = PipelineHandler("parent", DemoConfig(base=3, factor=4), tmp_path / "parent")
            parent_setup = parent.add_block("setup", 1)
            parent_setup.register_function(produce_seed, ["seed"])

            child = PipelineHandler("child", DemoConfig(base=100, factor=1), tmp_path / "child")
            child_block = child.add_block("child_block", 1)
            child_block.register_function(child_value, ["child_result"])

            parent.add_child_pipeline(child, 2)
            parent.run_all()

            self.assertEqual(parent.get_value("child_result"), 104)
            self.assertEqual(child.get_value("child_result"), 104)

    def test_grandchild_pipeline_can_use_root_config_value(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root = PipelineHandler(
                "root",
                {"stock_project_root": "/tmp/stocks", "base": 1},
                tmp_path / "root",
            )
            parent = PipelineHandler("parent", {"base": 2}, tmp_path / "parent")
            grandchild = PipelineHandler("grandchild", {"base": 3}, tmp_path / "grandchild")

            grandchild_block = grandchild.add_block("read_root_config", 1.0)
            grandchild_block.register_function(use_stock_project_root, ["resolved_root"])
            parent.add_child_pipeline(grandchild, 1.0)
            root.add_child_pipeline(parent, 1.0, forced=True)

            root.run_all()

            self.assertEqual(root.get_value("resolved_root"), "/tmp/stocks")
            self.assertEqual(parent.get_value("resolved_root"), "/tmp/stocks")
            self.assertEqual(grandchild.get_value("resolved_root"), "/tmp/stocks")

    def test_child_gate_can_resolve_parent_config_field(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            parent = PipelineHandler("parent", {"run_child": False, "base": 3, "factor": 4}, tmp_path / "parent")
            parent_setup = parent.add_block("setup", 1)
            parent_setup.register_function(produce_seed, ["seed"])

            child = PipelineHandler("child", DemoConfig(base=100, factor=1), tmp_path / "child")
            child.set_gate_block("run_child")
            child_block = child.add_block("child_unique", 1)
            child_block.register_function(unique_child_output, ["child_only"])

            parent.add_child_pipeline(child, 2)
            parent.run_all()

            self.assertIsNone(parent.get_value("child_only"))

    def test_gate_block_skip_keeps_existing_parent_value_and_sets_unique_child_output_none(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            parent = PipelineHandler("parent", DemoConfig(base=3, factor=4), tmp_path / "parent")
            parent_setup = parent.add_block("setup", 1)
            parent_setup.register_function(produce_seed, ["seed"])

            child = PipelineHandler("child", DemoConfig(base=50, factor=1), tmp_path / "child")
            child.set_gate_block(always_skip)
            child_block = child.add_block("child_block", 1)
            child_block.register_function(child_value, ["seed"])
            child_unique = child.add_block("child_unique", 2)
            child_unique.register_function(unique_child_output, ["child_only"])

            parent.add_child_pipeline(child, 2)
            parent.run_all()

            self.assertEqual(parent.get_value("seed"), 4)
            self.assertIsNone(parent.get_value("child_only"))

    def test_child_pipeline_priority_conflict_is_rejected_only_at_parent_level(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            parent = PipelineHandler("parent", DemoConfig(base=3, factor=4), tmp_path / "parent")
            parent.add_block("setup", 1)
            child = PipelineHandler("child", DemoConfig(base=10, factor=1), tmp_path / "child")
            child.add_block("internal", 1)

            with self.assertRaises(RegistrationError):
                parent.add_child_pipeline(child, 1)

            self.assertIsNone(child.parent_pipeline)
            self.assertEqual(child.project_root, tmp_path / "child")

    def test_child_pipeline_can_be_replaced_with_force(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            parent = PipelineHandler("parent", DemoConfig(base=3, factor=4), tmp_path / "parent")
            first_child = PipelineHandler("child", DemoConfig(base=10, factor=1), tmp_path / "child-a")
            first_block = first_child.add_block("first", 1)
            first_block.register_function(child_value, ["child_result"])
            parent.add_child_pipeline(first_child, 2)

            second_child = PipelineHandler("child", DemoConfig(base=20, factor=1), tmp_path / "child-b")
            second_block = second_child.add_block("second", 1)
            second_block.register_function(unique_child_output, ["child_only"])
            replacement = parent.add_child_pipeline(second_child, 2, forced=True)

            self.assertIsNotNone(replacement)
            self.assertIs(parent.nodes_by_name["child"], second_child)

    def test_child_standalone_run_updates_parent_visible_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            parent = PipelineHandler("parent", DemoConfig(base=3, factor=4), tmp_path / "parent")
            setup = parent.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])
            final = parent.add_block("final", 3)
            final.register_function(multiply, ["scaled_total"])

            child = PipelineHandler("child", DemoConfig(base=100, factor=1), tmp_path / "child")
            child_block = child.add_block("child_block", 1)
            child_block.register_function(child_value, ["seed"])
            parent.add_child_pipeline(child, 2)

            parent.run_until("setup")
            child.run_all()

            self.assertEqual(parent.get_value("seed"), 104)
            self.assertNotIn("scaled_total", parent.para_value_dict)

    def test_run_until_supports_nested_child_block_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root = PipelineHandler("root", DemoConfig(base=3, factor=4), tmp_path / "root")
            setup = root.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])

            child = PipelineHandler("modeling_pipeline", DemoConfig(base=100, factor=1), tmp_path / "child")
            predictor_components = child.add_block("predictor_components", 10)
            predictor_components.register_function(branch_left, ["left"])
            later = child.add_block("predictor_training", 20)
            later.register_function(branch_right, ["right"])

            root.add_child_pipeline(child, 70)
            root.run_until("modeling_pipeline", "predictor_components")

            self.assertEqual(root.get_value("seed"), 4)
            self.assertEqual(root.get_value("left"), 14)
            self.assertNotIn("right", root.para_value_dict)

    def test_run_block_supports_nested_child_block_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root = PipelineHandler("root", DemoConfig(base=3, factor=4), tmp_path / "root")
            setup = root.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])

            child = PipelineHandler("modeling_pipeline", DemoConfig(base=100, factor=1), tmp_path / "child")
            predictor_components = child.add_block("predictor_components", 10)
            predictor_components.register_function(branch_left, ["left"])
            later = child.add_block("predictor_training", 20)
            later.register_function(branch_right, ["right"])

            root.add_child_pipeline(child, 70)
            root.run_block("modeling_pipeline", "predictor_components")

            self.assertEqual(root.get_value("left"), 14)
            self.assertNotIn("right", root.para_value_dict)

    def test_run_from_supports_nested_child_block_path_and_continues_root_tail(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root = PipelineHandler("root", DemoConfig(base=3, factor=4), tmp_path / "root")
            setup = root.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])

            child = PipelineHandler("modeling_pipeline", DemoConfig(base=100, factor=1), tmp_path / "child")
            predictor_components = child.add_block("predictor_components", 10)
            predictor_components.register_function(branch_left, ["left"])
            later = child.add_block("predictor_training", 20)
            later.register_function(branch_right, ["right"])
            root.add_child_pipeline(child, 70)

            final = root.add_block("final", 80)
            final.register_function(combine, ["total"])

            root.run_until("setup")
            root.run_from("modeling_pipeline", "predictor_components")

            self.assertEqual(root.get_value("left"), 14)
            self.assertEqual(root.get_value("right"), 24)
            self.assertEqual(root.get_value("total"), 38)

    def test_output_name_conflicting_with_config_is_skipped(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("conflict", DemoConfig(base=2), tmp_path)
            block = pipeline.add_block("setup", 1)

            registration = block.register_function(produce_seed, ["base"])

            self.assertIsNone(registration)
            self.assertEqual(len(block.functions), 0)

    def test_existing_state_is_restored_after_execution_failure(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("restore", DemoConfig(base=2, factor=4), tmp_path)
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])
            pipeline.run_all()

            failing = pipeline.add_block("failing", 2)
            failing.register_function(needs_missing, ["x"])

            with self.assertRaises(ResolutionError):
                pipeline.run_all()

            self.assertEqual(pipeline.get_value("seed"), 3)

    def test_successful_earlier_outputs_remain_after_later_failure_in_same_run(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("partial-failure", DemoConfig(base=2, factor=4), tmp_path)
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])
            failing = pipeline.add_block("failing", 2)
            failing.register_function(needs_missing, ["x"])

            with self.assertRaises(ResolutionError):
                pipeline.run_all()

            self.assertEqual(pipeline.get_value("seed"), 3)

    def test_top_level_gate_block_export_is_correct(self) -> None:
        self.assertIs(TopLevelGateBlock, GateBlock)

    def test_unknown_node_raises_registration_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("missing-node", DemoConfig(base=1), tmp_path)

            with self.assertRaises(RegistrationError):
                pipeline.run_block("missing")

    def test_gate_skip_cleans_previous_disk_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("gate-clean", DemoConfig(base=2), tmp_path)
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])
            block = pipeline.add_block("save", 2)
            block.register_function(save_text, ["saved_blob"], save_to_disk=["saved_blob"])
            pipeline.run_all()
            pipeline.set_gate_block(always_skip)

            pipeline.run_all()

            artifact_dir = tmp_path / "artifacts"
            artifact_files = list(artifact_dir.rglob("*")) if artifact_dir.exists() else []
            self.assertFalse(any(path.is_file() for path in artifact_files))
