from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.mlpipelineholder import ExecutionBlock, PipelineHandler

observed_b_inputs: list[tuple[int, int | None]] = []
observed_disk_inputs: list[tuple[list[int], list[int] | None]] = []


def produce_a(seed: int) -> int:
    return seed


def run_b(a: int, future_value: int | None = None) -> int:
    observed_b_inputs.append((a, future_value))
    return a + 1


def produce_c(b: int) -> int:
    return b + 1


def run_disk_b(
    a: list[int],
    future_value: list[int] | None = None,
) -> list[int]:
    observed_disk_inputs.append((a, future_value))
    return [*a, 2]


def produce_disk_c(b: list[int]) -> list[int]:
    return [*b, 3]


def double_state(b: list[int]) -> list[int]:
    return b + b


def produce_base(seed: int) -> list[int]:
    return [seed]


def add_block(
    pipeline: PipelineHandler,
    registration_name: str,
    execution_priority: int,
) -> ExecutionBlock:
    block = pipeline.add_block(registration_name, execution_priority)
    assert block is not None
    return block


def build_parent_with_chain(root: Path) -> PipelineHandler:
    parent = PipelineHandler("parent", {"seed": 1}, root / "parent")
    child = PipelineHandler("child", {"seed": 1}, root / "child")
    block_a = add_block(child, "A", 1)
    block_a.register_function(produce_a, ["a"])
    block_b = add_block(child, "B", 2)
    block_b.register_function(run_b, ["b"])
    block_c = add_block(child, "C", 3)
    block_c.register_function(produce_c, ["future_value"])
    parent.add_child_pipeline(child, 1)
    return parent


class NestedRunFromVisibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        observed_b_inputs.clear()
        observed_disk_inputs.clear()

    def test_restart_from_b_hides_future_producer_outputs(self) -> None:
        # Given: A -> B -> C produced a, b, and future_value in a full run.
        with TemporaryDirectory() as temp_dir:
            parent = build_parent_with_chain(Path(temp_dir))
            parent.run_all()
            self.assertEqual(observed_b_inputs, [(1, None)])
            observed_b_inputs.clear()

            # When: the child restarts from B.
            parent.run_from("child", "B")

            # Then: B sees A's earlier output but NOT C's future output.
            self.assertEqual(observed_b_inputs, [(1, None)])
            self.assertEqual(parent.get_value("a"), 1)
            self.assertEqual(parent.get_value("future_value"), 3)

    def test_restart_from_b_preserves_targets_own_previous_output(self) -> None:
        # Given: A produces b, and B consumes b so its restart must see B's
        # own previous value rather than A's upstream copy.
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler("parent", {"seed": 1}, root / "parent")
            child = PipelineHandler("child", {"seed": 1}, root / "child")
            block_a = add_block(child, "A", 1)
            block_a.register_function(produce_base, ["b"])
            block_b = add_block(child, "B", 2)
            block_b.register_function(double_state, ["b"])
            parent.add_child_pipeline(child, 1)
            parent.run_all()
            self.assertEqual(parent.get_value("b"), [1, 1])

            # When: the child restarts from B.
            parent.run_from("child", "B")

            # Then: B saw its own previous output and doubled it again.
            self.assertEqual(parent.get_value("b"), [1, 1, 1, 1])

    def test_multilevel_restart_preserves_a_and_hides_future_value(self) -> None:
        # Given: the target B is inside a grandchild pipeline.
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler("parent", {"seed": 1}, root / "parent")
            child = PipelineHandler("child", {"seed": 1}, root / "child")
            grandchild = PipelineHandler(
                "grandchild",
                {"seed": 1},
                root / "grandchild",
            )
            block_a = add_block(grandchild, "A", 1)
            block_a.register_function(produce_a, ["a"])
            block_b = add_block(grandchild, "B", 2)
            block_b.register_function(run_b, ["b"])
            block_c = add_block(grandchild, "C", 3)
            block_c.register_function(produce_c, ["future_value"])
            child.add_child_pipeline(grandchild, 1)
            parent.add_child_pipeline(child, 1)
            parent.run_all()
            observed_b_inputs.clear()

            # When: the root restarts from the grandchild's B block.
            parent.run_from("child", "grandchild", "B")

            # Then: grandchild A remains available and C is absent inside B.
            self.assertEqual(observed_b_inputs, [(1, None)])
            self.assertEqual(parent.get_value("a"), 1)
            self.assertEqual(parent.get_value("future_value"), 3)

    def test_disk_backed_upstream_survives_nested_restart(self) -> None:
        # Given: A is disk-backed and B consumes it before C runs.
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler("parent", {"seed": 1}, root / "parent")
            child = PipelineHandler("child", {"seed": 1}, root / "child")
            block_a = add_block(child, "A", 1)
            block_a.register_function(
                produce_base,
                ["a"],
                save_to_disk=["a"],
            )
            block_b = add_block(child, "B", 2)
            block_b.register_function(run_disk_b, ["b"])
            block_c = add_block(child, "C", 3)
            block_c.register_function(produce_disk_c, ["future_value"])
            parent.add_child_pipeline(child, 1)
            parent.run_all()
            observed_disk_inputs.clear()

            # When: the nested child restarts from B.
            parent.run_from("child", "B")

            # Then: A's artifact is readable and future C is absent inside B.
            self.assertEqual(observed_disk_inputs, [([1], None)])
            self.assertEqual(parent.get_value("a"), [1])
            self.assertEqual(parent.get_value("future_value"), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
