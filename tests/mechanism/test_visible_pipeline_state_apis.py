from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from src.mlpipelineholder import ExecutionBlock, PipelineHandler, ResolutionError


def produce_seed(seed: int) -> int:
    return seed


def produce_none(seed: int) -> None:
    return None


def make_a(seed: int) -> int:
    return seed


def make_b(a: int, double: bool) -> int:
    return a * 2 if double else a


def read_factor(factor: int) -> int:
    return factor


def make_value(seed: int) -> list[int]:
    return [seed]


def append_two(value: list[int]) -> list[int]:
    value.append(2)
    return value


def add_block(
    pipeline: PipelineHandler,
    registration_name: str,
    execution_priority: int,
) -> ExecutionBlock:
    block = pipeline.add_block(registration_name, execution_priority)
    assert block is not None
    return block


def build_output_hierarchy(
    root: Path,
) -> tuple[PipelineHandler, PipelineHandler, PipelineHandler, PipelineHandler, PipelineHandler]:
    parent = PipelineHandler("parent", {"seed": 1}, root / "parent")
    earlier = PipelineHandler("earlier", {"seed": 1}, root / "earlier")
    current = PipelineHandler("current", {"seed": 1}, root / "current")
    later = PipelineHandler("later", {"seed": 1}, root / "later")
    grand = PipelineHandler("grand", {"seed": 1}, root / "grand")

    earlier_block = add_block(earlier, "earlier_block", 1)
    earlier_block.register_function(produce_seed, ["from_earlier"])

    current_block = add_block(current, "current_block", 1)
    current_block.register_function(produce_seed, ["own_out"])
    none_block = add_block(current, "none_block", 2)
    none_block.register_function(produce_none, ["none_out"])

    grand_block = add_block(grand, "grand_block", 1)
    grand_block.register_function(produce_seed, ["from_grand"])

    later_block = add_block(later, "later_block", 1)
    later_block.register_function(produce_seed, ["from_later"])

    current.add_child_pipeline(grand, 3)
    parent.add_child_pipeline(earlier, 1)
    parent.add_child_pipeline(current, 2)
    parent.add_child_pipeline(later, 3)
    return parent, earlier, current, later, grand


class VisibleOutputApiTests(unittest.TestCase):
    def test_declared_but_unproduced_outputs_are_absent_before_run(self) -> None:
        # Given: a fully attached hierarchy that has not executed yet.
        with TemporaryDirectory() as temp_dir:
            parent, _earlier, current, _later, _grand = build_output_hierarchy(
                Path(temp_dir)
            )

            # Then: declared names produce no visible outputs before running.
            self.assertFalse(parent.has_visible_output("own_out"))
            self.assertNotIn("own_out", parent.list_visible_output())
            self.assertFalse(current.has_visible_output("from_grand"))
            self.assertEqual(parent.list_visible_output(), set())
            self.assertEqual(current.list_visible_output(), set())

    def test_own_none_and_descendant_outputs_visible_after_run(self) -> None:
        # Given: a completed run.
        with TemporaryDirectory() as temp_dir:
            parent, _earlier, current, _later, _grand = build_output_hierarchy(
                Path(temp_dir)
            )
            parent.run_all()

            # Then: own, None-valued, and grandchild-produced outputs count.
            self.assertTrue(parent.has_visible_output("own_out"))
            self.assertTrue(parent.has_visible_output("none_out"))
            self.assertTrue(parent.has_visible_output("from_grand"))
            self.assertIsNone(parent.get_value("none_out"))
            self.assertEqual(parent.get_value("own_out"), 1)
            self.assertEqual(parent.get_value("from_grand"), 1)
            self.assertIn("from_grand", current.list_visible_output())

    def test_sibling_visibility_follows_execution_order(self) -> None:
        # Given: earlier/current/later siblings plus a grandchild all ran.
        with TemporaryDirectory() as temp_dir:
            parent, earlier, current, later, _grand = build_output_hierarchy(
                Path(temp_dir)
            )
            parent.run_all()

            # Then: the earlier sibling sees only its own subtree.
            self.assertIn("from_earlier", earlier.list_visible_output())
            self.assertNotIn("own_out", earlier.list_visible_output())
            self.assertNotIn("from_later", earlier.list_visible_output())

            # Then: the current child sees the earlier sibling and its own
            # descendant, but not the later sibling.
            self.assertIn("from_earlier", current.list_visible_output())
            self.assertNotIn("from_later", current.list_visible_output())
            self.assertIn("from_grand", current.list_visible_output())
            self.assertIn("own_out", current.list_visible_output())

            # Then: the later sibling sees every upstream produced output.
            for name in ("from_earlier", "own_out", "none_out", "from_grand"):
                self.assertIn(name, later.list_visible_output())
            self.assertIn("from_later", later.list_visible_output())

    def test_has_visible_output_equals_list_membership(self) -> None:
        with TemporaryDirectory() as temp_dir:
            parent, _earlier, current, _later, _grand = build_output_hierarchy(
                Path(temp_dir)
            )
            parent.run_all()
            for pipeline in (parent, current):
                visible = pipeline.list_visible_output()
                for name in (
                    "own_out",
                    "none_out",
                    "from_grand",
                    "from_earlier",
                    "from_later",
                    "missing_name",
                ):
                    self.assertEqual(
                        pipeline.has_visible_output(name),
                        name in visible,
                    )

    def test_missing_names_return_false_without_raising(self) -> None:
        with TemporaryDirectory() as temp_dir:
            parent, _earlier, current, _later, _grand = build_output_hierarchy(
                Path(temp_dir)
            )
            parent.run_all()
            self.assertFalse(parent.has_visible_output("does_not_exist"))
            self.assertNotIn("does_not_exist", parent.list_visible_output())
            self.assertFalse(current.has_visible_output("from_later"))

    def test_returned_output_set_is_detached(self) -> None:
        with TemporaryDirectory() as temp_dir:
            parent, _earlier, _current, _later, _grand = build_output_hierarchy(
                Path(temp_dir)
            )
            parent.run_all()
            first = parent.list_visible_output()
            first.add("injected")
            first.clear()
            second = parent.list_visible_output()
            self.assertNotIn("injected", second)
            self.assertTrue(second)

    def test_duplicate_producers_yield_one_entry_without_changing_winner(self) -> None:
        # Given: two sibling children produce the same name.
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler("parent", {"seed": 1}, root / "parent")
            first_child = PipelineHandler("first_child", {"seed": 1}, root / "first")
            second_child = PipelineHandler("second_child", {"seed": 1}, root / "second")
            first_block = add_block(first_child, "first_block", 1)
            first_block.register_function(produce_seed, ["x"])
            second_block = add_block(second_child, "second_block", 1)
            second_block.register_function(produce_seed, ["x"])
            parent.add_child_pipeline(first_child, 1)
            parent.add_child_pipeline(second_child, 2)
            parent.run_all()
            self.assertEqual(parent.get_value("x"), 1)

            # When: the earlier sibling updates its produced value.
            first_child.set_value("x", 4)

            # Then: one set entry, and the later sibling keeps the visible slot.
            self.assertEqual(list(parent.list_visible_output()).count("x"), 1)
            self.assertEqual(first_child.get_value("x"), 4)
            self.assertEqual(parent.get_value("x"), 1)
            self.assertTrue(parent.has_visible_output("x"))

    def test_constants_never_leak_into_visible_outputs(self) -> None:
        # Given: parent and child constants exist alongside produced outputs.
        with TemporaryDirectory() as temp_dir:
            parent, _earlier, current, _later, _grand = build_output_hierarchy(
                Path(temp_dir)
            )
            parent.set_constant_value("parent_const", 10)
            current.set_constant_value("current_const", 20)
            parent.run_all()

            # Then: constants are excluded from output visibility.
            self.assertNotIn("parent_const", parent.list_visible_output())
            self.assertNotIn("current_const", parent.list_visible_output())
            self.assertNotIn("parent_const", current.list_visible_output())
            self.assertNotIn("current_const", current.list_visible_output())
            self.assertFalse(parent.has_visible_output("parent_const"))
            self.assertTrue(parent.has_visible_constant("parent_const"))
            self.assertTrue(current.has_visible_constant("current_const"))

    def test_config_change_rerun_invalidates_downstream_produced_key(self) -> None:
        # Given: A and B produced their outputs from the original config.
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "pipeline",
                {"seed": 1, "double": False},
                Path(temp_dir) / "pipeline",
            )
            a_block = add_block(pipeline, "A", 1)
            a_block.register_function(make_a, ["a"])
            b_block = add_block(pipeline, "B", 2)
            b_block.register_function(make_b, ["b"])
            pipeline.run_all()
            self.assertTrue(pipeline.has_visible_output("a"))
            self.assertTrue(pipeline.has_visible_output("b"))

            # When: the consumed config changes and only A is re-run.
            pipeline.set_config("double", True)
            pipeline.run_block("A")

            # Then: A is reproduced but B's downstream slot was invalidated.
            self.assertTrue(pipeline.has_visible_output("a"))
            self.assertFalse(pipeline.has_visible_output("b"))
            self.assertNotIn("b", pipeline.list_visible_output())
            with self.assertRaises(ResolutionError):
                pipeline.get_value("b")


class VisibleConstantApiTests(unittest.TestCase):
    def test_own_and_ancestor_constants_are_visible(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler("parent", {"seed": 1}, root / "parent")
            child = PipelineHandler("child", {"seed": 1}, root / "child")
            parent.add_child_pipeline(child, 1)
            parent.set_constant_value("parent_const", 10)
            child.set_constant_value("child_const", 20)

            self.assertIn("parent_const", parent.list_visible_constant())
            self.assertNotIn("child_const", parent.list_visible_constant())
            self.assertIn("child_const", child.list_visible_constant())
            self.assertIn("parent_const", child.list_visible_constant())

    def test_none_valued_constant_counts_by_key(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "pipeline",
                {"seed": 1},
                Path(temp_dir) / "pipeline",
            )
            pipeline.set_constant_value("none_const", None)

            self.assertTrue(pipeline.has_visible_constant("none_const"))
            self.assertIn("none_const", pipeline.list_visible_constant())
            self.assertIsNone(pipeline.get_constant_value("none_const"))

    def test_earlier_sibling_constant_visible_later_sibling_hidden(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler("parent", {"seed": 1}, root / "parent")
            earlier = PipelineHandler("earlier", {"seed": 1}, root / "earlier")
            current = PipelineHandler("current", {"seed": 1}, root / "current")
            later = PipelineHandler("later", {"seed": 1}, root / "later")
            parent.add_child_pipeline(earlier, 1)
            parent.add_child_pipeline(current, 2)
            parent.add_child_pipeline(later, 3)
            earlier.set_constant_value("earlier_const", 1)
            current.set_constant_value("current_const", 2)
            later.set_constant_value("later_const", 3)

            self.assertIn("earlier_const", current.list_visible_constant())
            self.assertNotIn("later_const", current.list_visible_constant())
            self.assertIn("current_const", current.list_visible_constant())

            self.assertNotIn("later_const", earlier.list_visible_constant())
            self.assertNotIn("current_const", earlier.list_visible_constant())
            self.assertIn("earlier_const", earlier.list_visible_constant())

            self.assertIn("earlier_const", later.list_visible_constant())
            self.assertIn("current_const", later.list_visible_constant())
            self.assertIn("later_const", later.list_visible_constant())

    def test_shadowing_yields_one_name_and_preserves_getter_precedence(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler("parent", {"seed": 1}, root / "parent")
            earlier = PipelineHandler("earlier", {"seed": 1}, root / "earlier")
            current = PipelineHandler("current", {"seed": 1}, root / "current")
            parent.add_child_pipeline(earlier, 1)
            parent.add_child_pipeline(current, 2)
            parent.set_constant_value("shadow", 9)
            earlier.set_constant_value("shadow", 5)
            current.set_constant_value("shadow", 7)

            self.assertEqual(parent.get_constant_value("shadow"), 9)
            self.assertEqual(earlier.get_constant_value("shadow"), 5)
            self.assertEqual(current.get_constant_value("shadow"), 7)
            for pipeline in (parent, earlier, current):
                self.assertEqual(
                    list(pipeline.list_visible_constant()).count("shadow"),
                    1,
                )
                self.assertTrue(pipeline.has_visible_constant("shadow"))

    def test_descendant_constants_are_excluded(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler("parent", {"seed": 1}, root / "parent")
            child = PipelineHandler("child", {"seed": 1}, root / "child")
            grand = PipelineHandler("grand", {"seed": 1}, root / "grand")
            child.add_child_pipeline(grand, 1)
            parent.add_child_pipeline(child, 1)
            grand.set_constant_value("grand_const", 3)

            self.assertNotIn("grand_const", child.list_visible_constant())
            self.assertNotIn("grand_const", parent.list_visible_constant())
            self.assertIn("grand_const", grand.list_visible_constant())

    def test_atom_inherits_parent_constants(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "pipeline",
                {"seed": 1},
                Path(temp_dir) / "pipeline",
            )
            pipeline.create_atom_child_pipeline(
                "reader",
                26,
                read_factor,
                output_variable_names="result",
            )
            pipeline.set_constant_value("parent_const", 42)
            atom = pipeline.get_child_pipeline("reader")

            self.assertTrue(atom.has_visible_constant("parent_const"))
            self.assertIn("parent_const", atom.list_visible_constant())

    def test_has_visible_constant_equals_membership_and_detached_set(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler("parent", {"seed": 1}, root / "parent")
            child = PipelineHandler("child", {"seed": 1}, root / "child")
            parent.add_child_pipeline(child, 1)
            parent.set_constant_value("c1", 1)
            child.set_constant_value("c2", 2)

            visible = child.list_visible_constant()
            for name in ("c1", "c2", "nope"):
                self.assertEqual(child.has_visible_constant(name), name in visible)
            self.assertFalse(child.has_visible_constant("nope"))

            first = child.list_visible_constant()
            first.clear()
            self.assertIn("c2", child.list_visible_constant())


class VisibleConfigApiTests(unittest.TestCase):
    def test_own_and_ancestor_config_keys_are_visible(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler(
                "parent",
                {"seed": 1, "factor": 2},
                root / "parent",
            )
            child = PipelineHandler(
                "child",
                {"seed": 1, "child_only": 3},
                root / "child",
            )
            parent.add_child_pipeline(child, 1)

            self.assertIn("factor", parent.list_visible_config())
            self.assertNotIn("child_only", parent.list_visible_config())
            self.assertIn("child_only", child.list_visible_config())
            self.assertIn("factor", child.list_visible_config())

    def test_none_valued_config_key_counts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "pipeline",
                {"none_field": None},
                Path(temp_dir) / "pipeline",
            )

            self.assertTrue(pipeline.has_visible_config("none_field"))
            self.assertIn("none_field", pipeline.list_visible_config())

    def test_shadowing_yields_one_name_and_preserves_getter_precedence(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler("parent", {"shared": 1}, root / "parent")
            child = PipelineHandler("child", {"shared": 2}, root / "child")
            parent.add_child_pipeline(child, 1)

            self.assertEqual(parent.get_config("shared"), 1)
            self.assertEqual(child.get_config("shared"), 2)
            self.assertEqual(list(child.list_visible_config()).count("shared"), 1)
            self.assertEqual(list(parent.list_visible_config()).count("shared"), 1)

    def test_sibling_and_descendant_config_keys_are_excluded(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler("parent", {"seed": 1}, root / "parent")
            child1 = PipelineHandler("child1", {"seed": 1, "c1": 1}, root / "child1")
            child2 = PipelineHandler("child2", {"seed": 1, "c2": 2}, root / "child2")
            grand = PipelineHandler("grand", {"seed": 1, "g": 3}, root / "grand")
            child1.add_child_pipeline(grand, 1)
            parent.add_child_pipeline(child1, 1)
            parent.add_child_pipeline(child2, 2)

            self.assertNotIn("c2", child1.list_visible_config())
            self.assertNotIn("g", child1.list_visible_config())
            self.assertNotIn("c1", child2.list_visible_config())
            self.assertIn("g", grand.list_visible_config())
            self.assertIn("c1", grand.list_visible_config())
            self.assertIn("c2", child2.list_visible_config())

    def test_atom_visible_config_equals_parent(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "pipeline",
                {"seed": 1, "factor": 2},
                Path(temp_dir) / "pipeline",
            )
            pipeline.create_atom_child_pipeline(
                "reader",
                26,
                read_factor,
                output_variable_names="result",
            )
            atom = pipeline.get_child_pipeline("reader")

            self.assertEqual(
                atom.list_visible_config(),
                pipeline.list_visible_config(),
            )

            pipeline.set_config("later_field", 5)
            self.assertIn("later_field", atom.list_visible_config())
            self.assertTrue(atom.has_visible_config("later_field"))
            self.assertTrue(atom.has_visible_config("factor"))

    def test_has_visible_config_equals_membership_and_detached_set(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler(
                "parent",
                {"seed": 1, "factor": 2},
                root / "parent",
            )
            child = PipelineHandler(
                "child",
                {"seed": 1, "child_only": 3},
                root / "child",
            )
            parent.add_child_pipeline(child, 1)

            visible = child.list_visible_config()
            for name in ("seed", "factor", "child_only", "nope"):
                self.assertEqual(child.has_visible_config(name), name in visible)
            self.assertFalse(child.has_visible_config("nope"))

            first = child.list_visible_config()
            first.clear()
            self.assertIn("child_only", child.list_visible_config())


class _Lock:
    """Non-picklable value that is persisted as a placeholder."""

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()


def produce_lock() -> _Lock:
    return _Lock()


def produce_pointer() -> object:
    from src.mlpipelineholder.output_pointers import OutputAddress, OutputPointer

    return OutputPointer(OutputAddress("pipeline", "producer", "value"))


class CrossCuttingVisibilityTests(unittest.TestCase):
    def _snapshot(self, pipeline: PipelineHandler) -> dict[str, object]:
        return {
            "producer_outputs": {
                name: dict(outputs)
                for name, outputs in pipeline.producer_outputs.items()
            },
            "manual_values": dict(pipeline.manual_values),
            "para_value_dict": dict(pipeline.para_value_dict),
            "artifact_registry": dict(pipeline.artifact_registry),
            "config": pipeline.config_as_dict(),
        }

    def test_all_six_apis_are_side_effect_free_with_spies(self) -> None:
        # Given: disk-backed, constant, and pointer-producing state exists.
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "pipeline",
                {"seed": 1},
                Path(temp_dir) / "pipeline",
            )
            disk = add_block(pipeline, "disk", 1)
            disk.register_function(make_value, ["disk_out"], save_to_disk=["disk_out"])
            pointer = add_block(pipeline, "pointer", 2)
            pointer.register_function(produce_pointer, ["ptr_out"])
            pipeline.set_constant_value("const_value", [9])
            pipeline.run_all()
            before = self._snapshot(pipeline)

            # When: all six APIs run under fail-fast spies.
            with (
                mock.patch.object(
                    pipeline,
                    "_materialize_stored_value",
                    side_effect=AssertionError("materialized"),
                ),
                mock.patch.object(
                    pipeline,
                    "_rebuild_visible_state",
                    side_effect=AssertionError("rebuilt"),
                ),
                mock.patch.object(
                    pipeline.artifact_store,
                    "load",
                    side_effect=AssertionError("artifact loaded"),
                ),
                mock.patch(
                    "src.mlpipelineholder.pipeline_handler.resolve_pointer_chain",
                    side_effect=AssertionError("pointer resolved"),
                ),
                mock.patch.object(
                    pipeline,
                    "recover_variable_from_backup",
                    side_effect=AssertionError("recovered"),
                ),
                mock.patch(
                    "src.mlpipelineholder.backup_recovery_service.recover_variable_from_backup",
                    side_effect=AssertionError("recovered (service)"),
                ),
            ):
                self.assertTrue(pipeline.has_visible_output("disk_out"))
                self.assertTrue(pipeline.has_visible_output("ptr_out"))
                self.assertTrue(pipeline.has_visible_constant("const_value"))
                self.assertTrue(pipeline.has_visible_config("seed"))
                self.assertIn("disk_out", pipeline.list_visible_output())
                self.assertIn("ptr_out", pipeline.list_visible_output())
                self.assertIn("const_value", pipeline.list_visible_constant())
                self.assertIn("seed", pipeline.list_visible_config())

            # Then: no runtime state changed.
            self.assertEqual(before, self._snapshot(pipeline))

    def test_visibility_survives_save_and_load_without_materialization(self) -> None:
        # Given: a run with a placeholder value, a disk-backed output, a
        # constant, and config fields.
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler(
                "persist",
                {"seed": 1, "factor": 2},
                tmp / "project",
            )
            lock_block = add_block(pipeline, "lock_producer", 1)
            lock_block.register_function(produce_lock, ["out"])
            disk = add_block(pipeline, "disk", 2)
            disk.register_function(make_value, ["disk_out"], save_to_disk=["disk_out"])
            pipeline.set_constant_value("const_key", 7)
            pipeline.run_all()

            before = (
                pipeline.list_visible_output(),
                pipeline.list_visible_constant(),
                pipeline.list_visible_config(),
            )

            # When: the project is saved and loaded.
            pipeline.save_pipeline(tmp / "bundle")
            loaded = PipelineHandler.load_pipeline(tmp / "bundle", forced_deleting=True)

            # Then: the six APIs return identical names from the loaded state,
            # including the not-yet-materialized placeholder output.
            self.assertEqual(
                before,
                (
                    loaded.list_visible_output(),
                    loaded.list_visible_constant(),
                    loaded.list_visible_config(),
                ),
            )
            self.assertIn("out", loaded.list_visible_output())
            self.assertIn("disk_out", loaded.list_visible_output())
            self.assertIn("const_key", loaded.list_visible_constant())
            self.assertIn("factor", loaded.list_visible_config())

            with (
                mock.patch.object(
                    loaded,
                    "_materialize_stored_value",
                    side_effect=AssertionError("materialized"),
                ),
                mock.patch.object(
                    loaded.artifact_store,
                    "load",
                    side_effect=AssertionError("artifact loaded"),
                ),
            ):
                self.assertTrue(loaded.has_visible_output("out"))
                self.assertTrue(loaded.has_visible_output("disk_out"))
                self.assertTrue(loaded.has_visible_constant("const_key"))


if __name__ == "__main__":
    _ = unittest.main()
