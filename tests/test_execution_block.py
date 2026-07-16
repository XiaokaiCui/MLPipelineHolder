from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.mlpipelineholder import ExecutionError, PipelineHandler, RegistrationError


@dataclass
class BlockConfig:
    value: int


def source(value: int) -> int:
    return value


def same_block_dependency(intermediate: int) -> int:
    return intermediate + 1


def passthrough(value: int) -> int:
    return value


def variadic_sum(base: int, *extra_values: int, factor: int = 1, **extra_items: int) -> int:
    return (base + sum(extra_values) + sum(extra_items.values())) * factor


def increment(value: int) -> int:
    return value + 1


def source_data(data: int) -> int:
    return data


def increment_data(data: int) -> int:
    return data + 1


def recorder_step(model: int, recorder: list[int] | None = None) -> tuple[int, list[int]]:
    next_recorder = [] if recorder is None else list(recorder)
    next_recorder.append(model)
    return model + 1, next_recorder


def duplicate_mapping_use(training_metric_divider: int, eval_metric_divider: int) -> tuple[int, int]:
    return training_metric_divider, eval_metric_divider


def optional_recorders(train_recorder: list[int] | None, eval_recorder: list[int] | None) -> tuple[list[int], list[int]]:
    next_train = [] if train_recorder is None else list(train_recorder)
    next_eval = [] if eval_recorder is None else list(eval_recorder)
    next_train.append(1)
    next_eval.append(2)
    return next_train, next_eval


def annotated_single_output(value: int) -> int:
    return value


def annotated_two_outputs(value: int) -> tuple[int, int]:
    return value, value + 1


def implicit_input(value: int) -> int:
    return value + 1


def mutate_disk_backed_input(shared: str) -> str:
    return shared + "!"



class ExecutionBlockTests(unittest.TestCase):
    def test_same_block_dependency_is_rejected_at_execution_time(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("same-block", BlockConfig(value=1), tmp_path)
            block = pipeline.add_block("block", 1)
            block.register_function(source, ["intermediate"])
            block.register_function(same_block_dependency, ["result"])

            with self.assertRaises(ExecutionError):
                pipeline.run_all()

    def test_variadic_functions_work_with_renamed_inputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler(
                "variadic",
                {
                    "base_value": 1,
                    "factor_value": 2,
                    "function_args": [3, 4],
                    "function_kwargs": {"bonus": 5},
                },
                tmp_path,
            )
            block = pipeline.add_block("block", 1)
            block.register_function(
                variadic_sum,
                ["result"],
                param_mapping={"base": "base_value", "factor": "factor_value"},
                var_pos_name="function_args",
                var_kw_name="function_kwargs",
            )

            pipeline.run_all()

            self.assertEqual(pipeline.get_value("result"), 26)

    def test_block_scoped_args_and_kwargs_helpers_are_used(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler(
                "helpers",
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
            block.register_function(
                variadic_sum,
                ["result"],
                param_mapping={"base": "base_value", "factor": "factor_value"},
                var_pos_name="args_a",
                var_kw_name="kwargs_a",
            )

            pipeline.run_all()

            self.assertEqual(pipeline.get_value("result"), 26)

    def test_save_to_disk_must_be_subset_of_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("subset", BlockConfig(value=1), tmp_path)
            block = pipeline.add_block("block", 1)

            registration = block.register_function(
                passthrough, ["result"], save_to_disk=["not_result"]
            )

            self.assertIsNone(registration)
            self.assertEqual(len(block.functions), 0)

    def test_no_output_function_registration_is_allowed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("no-output", BlockConfig(value=1), tmp_path)
            block = pipeline.add_block("block", 1)

            registration = block.register_function(passthrough, None)
            pipeline.run_all()

            self.assertIsNotNone(registration)
            self.assertEqual(registration.output_names, [])
            self.assertEqual(pipeline.para_value_dict, {})

    def test_no_output_registration_warns_when_return_annotation_declares_output(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("warn-output", BlockConfig(value=1), tmp_path)
            block = pipeline.add_block("block", 1)

            registration = block.register_function(annotated_single_output, None)

            self.assertIsNotNone(registration)
            log_text = (tmp_path / "metadata" / "pipeline.log").read_text(encoding="utf-8")
            self.assertIn("declares 1 output(s)", log_text)

    def test_registration_fails_when_declared_output_count_mismatches(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("mismatch", BlockConfig(value=1), tmp_path)
            block = pipeline.add_block("block", 1)

            with self.assertRaises(RegistrationError):
                block._register_function_strict(annotated_two_outputs, ["only_one"])

    def test_registration_tolerates_unresolved_type_hints(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("typing", BlockConfig(value=1), tmp_path)
            block = pipeline.add_block("block", 1)

            namespace: dict[str, object] = {}
            exec(
                "def unresolved_annotation_func(value: 'MissingType') -> 'OtherMissingType':\n"
                "    return value\n",
                namespace,
            )
            unresolved_annotation_func = namespace["unresolved_annotation_func"]

            registration = block.register_function(unresolved_annotation_func, ["result"])

            self.assertIsNotNone(registration)

    def test_registration_warns_for_unmapped_resolvable_input(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("warn", {"value": 1}, tmp_path)
            block = pipeline.add_block("block", 1)

            registration = block.register_function(implicit_input, ["result"])

            self.assertIsNotNone(registration)
            log_text = (tmp_path / "metadata" / "pipeline.log").read_text(encoding="utf-8")
            self.assertIn("may be resolved implicitly", log_text)

    def test_registration_warns_for_disk_backed_input_not_declared_as_output(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("warn", BlockConfig(value=1), tmp_path)
            first = pipeline.add_block("first", 1)
            first.register_function(source, ["shared"], save_to_disk=["shared"])
            pipeline.run_all()
            second = pipeline.add_block("second", 2)

            registration = second.register_function(mutate_disk_backed_input, ["result"])

            self.assertIsNotNone(registration)
            log_text = (tmp_path / "metadata" / "pipeline.log").read_text(encoding="utf-8")
            self.assertIn("in-function mutations will not persist", log_text)

    def test_same_name_input_and_output_update_is_allowed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("update", {"source_value": 1}, tmp_path)
            first = pipeline.add_block("first", 1)
            first.register_function(source_data, ["data"], param_mapping={"data": "source_value"})
            second = pipeline.add_block("second", 2)
            second.register_function(increment_data, ["data"])

            pipeline.run_all()

            self.assertEqual(pipeline.get_value("data"), 2)

    def test_run_block_reuses_prior_same_block_outputs_when_inputs_share_output_names(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("rerun", {"source_model": 1}, tmp_path)
            setup = pipeline.add_block("setup", 1)
            setup.register_function(source_data, ["model"], param_mapping={"data": "source_model"})
            block = pipeline.add_block("block", 2)
            block.register_function(
                recorder_step,
                ["model", "recorder"],
            )

            pipeline.run_all()
            self.assertEqual(pipeline.get_value("model"), 2)
            self.assertEqual(pipeline.get_value("recorder"), [1])

            pipeline.run_block("block")

            self.assertEqual(pipeline.get_value("model"), 3)
            self.assertEqual(pipeline.get_value("recorder"), [1, 2])

    def test_multiple_params_can_map_to_same_pipeline_value(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("dup-map", {"me_sample_size": 7}, tmp_path)
            block = pipeline.add_block("block", 1)
            block.register_function(
                duplicate_mapping_use,
                ["train_divider", "eval_divider"],
                param_mapping={
                    "training_metric_divider": "me_sample_size",
                    "eval_metric_divider": "me_sample_size",
                },
            )

            pipeline.run_all()

            self.assertEqual(pipeline.get_value("train_divider"), 7)
            self.assertEqual(pipeline.get_value("eval_divider"), 7)

    def test_first_run_uses_none_for_unassigned_declared_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("optional-recorders", {}, tmp_path)
            block = pipeline.add_block("block", 1)
            block.register_function(
                optional_recorders,
                ["train_recorder", "eval_recorder"],
            )

            pipeline.run_all()

            self.assertEqual(pipeline.get_value("train_recorder"), [1])
            self.assertEqual(pipeline.get_value("eval_recorder"), [2])

    def test_duplicate_function_registration_raises_without_force(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("dup-func", BlockConfig(value=1), tmp_path)
            block = pipeline.add_block("block", 1)
            block.register_function(passthrough, ["result"])

            with self.assertRaises(RegistrationError):
                block.register_function(passthrough, ["result_two"])

    def test_duplicate_function_registration_is_replaced_with_force(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("dup-func", BlockConfig(value=1), tmp_path)
            block = pipeline.add_block("block", 1)
            block.register_function(passthrough, ["result"])

            registration = block.register_function(passthrough, ["new_result"], forced=True)
            pipeline.run_all()

            self.assertIsNotNone(registration)
            self.assertEqual(len(block.functions), 1)
            self.assertIn("new_result", pipeline.para_value_dict)
            self.assertNotIn("result", pipeline.para_value_dict)

    def test_remove_function_invalidates_its_outputs_and_downstream_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("remove-function", BlockConfig(value=2), tmp_path)
            first = pipeline.add_block("first", 1)
            first.register_function(source, ["seed"])
            second = pipeline.add_block("second", 2)
            second.register_function(passthrough, ["middle"])
            third = pipeline.add_block("third", 3)
            third.register_function(passthrough, ["final"])
            pipeline.run_all()

            second.remove_function("passthrough")

            self.assertEqual(len(second.functions), 0)
            self.assertNotIn("middle", pipeline.para_value_dict)
            self.assertNotIn("final", pipeline.para_value_dict)
            self.assertIn("seed", pipeline.para_value_dict)

    def test_remove_function_rejects_missing_name(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("remove-function", BlockConfig(value=2), tmp_path)
            block = pipeline.add_block("block", 1)
            block.register_function(source, ["seed"])

            with self.assertRaises(RegistrationError):
                block.remove_function("missing")
