from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.mlpipelineholder import ExecutionBlock, PipelineHandler


def produce_value(seed: int) -> int:
    return seed


def produce_interned(seed: int) -> int:
    return 1


def produce_none(seed: int) -> None:
    return None


def add_block(
    pipeline: PipelineHandler,
    registration_name: str,
    execution_priority: int,
) -> ExecutionBlock:
    block = pipeline.add_block(registration_name, execution_priority)
    assert block is not None
    return block


class ValueSyncProvenanceTests(unittest.TestCase):
    def test_parent_own_producer_wins_over_child_after_child_update(self) -> None:
        # Given: the parent's own later block and the child both produce the
        # interned integer 1, and the parent block wins the visible slot.
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler("parent", {"seed": 1}, root / "parent")
            child = PipelineHandler("child", {"seed": 1}, root / "child")
            child_block = add_block(child, "child_block", 1)
            child_block.register_function(produce_interned, ["x"])
            parent_block = add_block(parent, "parent_block", 10)
            parent_block.register_function(produce_interned, ["x"])
            parent.add_child_pipeline(child, 1)
            parent.run_all()
            self.assertEqual(parent.get_value("x"), 1)

            # When: the child's value is updated.
            child.set_value("x", 2)

            # Then: the parent's own later producer still owns the visible slot.
            self.assertEqual(parent.get_value("x"), 1)
            self.assertEqual(child.get_value("x"), 2)

    def test_child_wins_when_child_is_latest_producer_after_update(self) -> None:
        # Given: the child runs later than the parent's own block, so the child
        # owns the visible slot.
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler("parent", {"seed": 1}, root / "parent")
            child = PipelineHandler("child", {"seed": 1}, root / "child")
            parent_block = add_block(parent, "parent_block", 1)
            parent_block.register_function(produce_value, ["x"])
            child_block = add_block(child, "child_block", 1)
            child_block.register_function(produce_value, ["x"])
            parent.add_child_pipeline(child, 10)
            parent.run_all()
            self.assertEqual(parent.get_value("x"), 1)

            # When: the child's value is updated.
            child.set_value("x", 5)

            # Then: the child's new value propagates to the parent's slot.
            self.assertEqual(parent.get_value("x"), 5)
            self.assertEqual(child.get_value("x"), 5)

    def test_none_valued_child_update_propagates_to_parent(self) -> None:
        # Given: the child produces None and owns the visible slot.
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler("parent", {"seed": 1}, root / "parent")
            child = PipelineHandler("child", {"seed": 1}, root / "child")
            child_block = add_block(child, "child_block", 1)
            child_block.register_function(produce_none, ["x"])
            parent.add_child_pipeline(child, 1)
            parent.run_all()
            self.assertIsNone(parent.get_value("x"))

            # When: the child's value is updated.
            child.set_value("x", 7)

            # Then: the parent's visible slot reflects the child's update.
            self.assertEqual(parent.get_value("x"), 7)
            self.assertEqual(child.get_value("x"), 7)

    def test_sibling_update_respects_later_child_producer(self) -> None:
        # Given: both children produce x, and the later child owns the parent slot.
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler("parent", {"seed": 1}, root / "parent")
            first_child = PipelineHandler("first_child", {"seed": 1}, root / "first")
            first_block = add_block(first_child, "first_block", 1)
            first_block.register_function(produce_value, ["x"])
            second_child = PipelineHandler("second_child", {"seed": 1}, root / "second")
            second_block = add_block(second_child, "second_block", 1)
            second_block.register_function(produce_value, ["x"])
            parent.add_child_pipeline(first_child, 1)
            parent.add_child_pipeline(second_child, 2)
            parent.run_all()

            # When: the earlier sibling changes its produced value.
            first_child.set_value("x", 4)

            # Then: the later sibling still owns the parent-visible value.
            self.assertEqual(first_child.get_value("x"), 4)
            self.assertEqual(second_child.get_value("x"), 1)
            self.assertEqual(parent.get_value("x"), 1)

    def test_equal_priority_producers_use_deterministic_node_order(self) -> None:
        # Given: persisted producer metadata contains two same-priority children.
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler("parent", {"seed": 1}, root / "parent")
            first_child = PipelineHandler("a_child", {"seed": 1}, root / "first")
            first_block = add_block(first_child, "first_block", 1)
            first_block.register_function(produce_value, ["x"])
            second_child = PipelineHandler("z_child", {"seed": 1}, root / "second")
            second_block = add_block(second_child, "second_block", 1)
            second_block.register_function(produce_value, ["x"])
            parent.add_child_pipeline(first_child, 1)
            parent.add_child_pipeline(second_child, 2)
            parent.run_all()
            second_child.execution_priority = 1
            parent._refresh_visible_value("x")

            # When: the first child updates its same-priority producer slot.
            first_child.set_value("x", 6)

            # Then: name ordering keeps z_child as the deterministic winner.
            self.assertEqual(parent.get_value("x"), 1)

    def test_grandchild_update_propagates_through_every_ancestor(self) -> None:
        # Given: a grandchild is the only producer of x in the hierarchy.
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler("parent", {"seed": 1}, root / "parent")
            child = PipelineHandler("child", {"seed": 1}, root / "child")
            grandchild = PipelineHandler("grandchild", {"seed": 1}, root / "grandchild")
            block = add_block(grandchild, "producer", 1)
            block.register_function(produce_value, ["x"])
            child.add_child_pipeline(grandchild, 1)
            parent.add_child_pipeline(child, 1)
            parent.run_all()

            # When: the grandchild's produced value changes.
            grandchild.set_value("x", 9)

            # Then: every ancestor mirror receives the new producer value.
            self.assertEqual(grandchild.get_value("x"), 9)
            self.assertEqual(child.get_value("x"), 9)
            self.assertEqual(parent.get_value("x"), 9)


if __name__ == "__main__":
    unittest.main()
