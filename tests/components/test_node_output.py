from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.mlpipelineholder import PipelineHandler
from src.mlpipelineholder.exceptions import RegistrationError, ResolutionError


def produce_one() -> int:
    return 1


def produce_two() -> int:
    return 2


def produce_three() -> int:
    return 3


class BlockRegistrationModeTests(unittest.TestCase):
    def test_forced_expression_is_rejected_after_function_registration(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "root")
            block = pipeline.add_block("mixed", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_function(produce_one, ["function_value"])

            with self.assertRaises(RegistrationError):
                block.register_expression("expression_value = 2", forced=True)

            self.assertEqual(len(block.functions), 1)

    def test_forced_function_is_rejected_after_expression_registration(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "root")
            block = pipeline.add_block("mixed", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_expression("expression_value = 2")

            registration = block.register_function(
                produce_one,
                ["function_value"],
                forced=True,
            )

            self.assertIsNone(registration)
            self.assertEqual(len(block.functions), 1)

    def test_multiple_function_registrations_remain_allowed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "root")
            block = pipeline.add_block("functions", 1)
            if block is None:
                raise AssertionError("add_block should return a block")

            block.register_function(produce_one, ["one"])
            block.register_function(produce_two, ["two"])

            self.assertEqual(len(block.functions), 2)


class GetNodeOutputTests(unittest.TestCase):
    def test_materializes_same_disk_output_from_each_immediate_node(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = PipelineHandler("root", {}, Path(temp_dir) / "root")
            block_a = root.add_block("A", 1)
            block_c = root.add_block("C", 3)
            if block_a is None or block_c is None:
                raise AssertionError("add_block should return a block")
            block_a.register_function(
                produce_one,
                ["same_value"],
                save_to_disk=["same_value"],
            )
            root.create_atom_child_pipeline(
                "B",
                2,
                produce_two,
                output_variable_names=["same_value"],
                save_to_disk_lst=["same_value"],
            )
            block_c.register_function(
                produce_three,
                ["same_value"],
                save_to_disk=["same_value"],
            )
            root.run_all()

            self.assertEqual(root.get_node_output("A", "same_value"), 1)
            self.assertEqual(root.get_node_output("B", "same_value"), 2)
            self.assertEqual(root.get_node_output("C", "same_value"), 3)

    def test_materializes_expression_output_from_immediate_block(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = PipelineHandler("root", {"seed": 2}, Path(temp_dir) / "root")
            block = root.add_block("expression", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_expression("result = seed + 3", save_to_disk=True)
            root.run_all()

            self.assertEqual(root.get_node_output("expression", "result"), 5)

    def test_disk_node_output_survives_save_and_load(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "root"
            root = PipelineHandler("root", {}, path)
            block = root.add_block("block", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_function(
                produce_one,
                ["value"],
                save_to_disk=["value"],
            )
            root.run_all()
            root.save_pipeline()

            loaded = PipelineHandler.load_pipeline(path, forced_deleting=True)

            self.assertEqual(loaded.get_node_output("block", "value"), 1)

    def test_legacy_atom_with_duplicate_internal_outputs_returns_priority_map(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = PipelineHandler("root", {}, base / "root")
            atom = PipelineHandler("atom", {}, base / "atom")
            first = atom.add_block("first", 10)
            second = atom.add_block("second", 20)
            if first is None or second is None:
                raise AssertionError("add_block should return a block")
            first.register_function(produce_one, ["same_value"])
            second.register_function(produce_two, ["same_value"])
            root.add_child_pipeline(atom, 1)
            atom._is_atom = True
            root.run_all()

            self.assertEqual(
                root.get_node_output("atom", "same_value"),
                {10: 1, 20: 2},
            )

    def test_rejects_non_atom_child_pipeline(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = PipelineHandler("root", {}, base / "root")
            child = PipelineHandler("child", {}, base / "child")
            block = child.add_block("inside", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_function(produce_one, ["value"])
            root.add_child_pipeline(child, 1)
            root.run_all()

            with self.assertRaises(ResolutionError):
                root.get_node_output("child", "value")

    def test_requires_node_to_be_immediate_child(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = PipelineHandler("root", {}, base / "root")
            child = PipelineHandler("child", {}, base / "child")
            block = child.add_block("inside", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_function(produce_one, ["value"])
            root.add_child_pipeline(child, 1)
            root.run_all()

            with self.assertRaises(ResolutionError):
                root.get_node_output("inside", "value")

    def test_rejects_unknown_output_for_immediate_block(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = PipelineHandler("root", {}, Path(temp_dir) / "root")
            block = root.add_block("block", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_function(produce_one, ["value"])
            root.run_all()

            with self.assertRaises(ResolutionError):
                root.get_node_output("block", "missing")


if __name__ == "__main__":
    unittest.main()
