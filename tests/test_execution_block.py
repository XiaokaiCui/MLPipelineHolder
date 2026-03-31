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

    def test_variadic_functions_are_not_supported(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("variadic", BlockConfig(value=1), tmp_path)
            block = pipeline.add_block("block", 1)

            def bad(*args: int) -> int:
                return sum(args)

            with self.assertRaises(RegistrationError):
                block.register_function(bad, ["result"])

    def test_save_to_disk_must_be_subset_of_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("subset", BlockConfig(value=1), tmp_path)
            block = pipeline.add_block("block", 1)

            with self.assertRaises(RegistrationError):
                block.register_function(passthrough, ["result"], save_to_disk=["not_result"])

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
