from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import warnings

from src.mlpipelineholder import PipelineHandler, RegistrationError, ResolutionError
from src.mlpipelineholder.models import ArtifactRecord


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


def save_json(seed: int) -> dict[str, int]:
    return {"seed": seed, "double": seed * 2}


def save_array(seed: int):
    from importlib import import_module

    return import_module("numpy").array([seed, seed + 1, seed + 2])


def read_text(saved_blob: str) -> str:
    return saved_blob.upper()


def pair(seed: int) -> tuple[int, int]:
    return seed, seed + 1


def needs_missing(missing: int) -> int:
    return missing


def logger_step(seed: int, logger) -> int:
    logger.info(f"seed={seed}")
    logger.result(f"final-seed={seed}")
    return seed


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

    def test_duplicate_outputs_are_rejected_across_blocks(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("dup", DemoConfig(base=1), tmp_path)
            first = pipeline.add_block("first", 1)
            second = pipeline.add_block("second", 2)

            first.register_function(produce_seed, ["seed"])
            with self.assertRaises(RegistrationError):
                second.register_function(produce_seed, ["seed"])

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
            pipeline.add_block("first", 1)

            with self.assertRaises(RegistrationError):
                pipeline.add_block("second", 1)

    def test_update_config_overrides_known_fields(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("config", DemoConfig(base=1, factor=2), tmp_path)
            pipeline.update_config({"factor": 9})

            self.assertEqual(pipeline.config.factor, 9)

    def test_update_config_rejects_unknown_fields(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("config", DemoConfig(base=1, factor=2), tmp_path)

            with self.assertRaises(ResolutionError):
                pipeline.update_config({"missing": 9})

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

            chart = strip_ansi(pipeline.describe_pipeline())

            self.assertIn("PipelineHandler(describe)", chart)
            self.assertIn("[1] setup", chart)
            self.assertIn("produce_seed(base) -> seed", chart)
            self.assertIn("[2] disk_write", chart)
            self.assertIn("save_text(seed) -> saved_blob*", chart)

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
            self.assertIn("[RESULT] final-seed=5", history[0])
            log_text = (tmp_path / "metadata" / "pipeline.log").read_text(encoding="utf-8")
            self.assertIn("[INFO] seed=5", log_text)
            self.assertIn("[RESULT] final-seed=5", log_text)

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

            self.assertIn("[RESULT] final-seed=5", output.getvalue())

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
