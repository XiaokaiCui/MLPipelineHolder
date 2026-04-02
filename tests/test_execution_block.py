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
                kw_mapping={"base": "base_value", "factor": "factor_value"},
                var_pos_name="function_args",
                var_kw_name="function_kwargs",
            )

            pipeline.run_all()

            self.assertEqual(pipeline.get_value("result"), 26)

    def test_pos_mapping_is_rejected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("variadic", BlockConfig(value=1), tmp_path)
            block = pipeline.add_block("block", 1)

            registration = block.register_function(passthrough, ["result"], pos_mapping={0: 1})

            self.assertIsNone(registration)
            self.assertEqual(len(block.functions), 0)

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
