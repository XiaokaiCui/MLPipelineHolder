from __future__ import annotations

import __main__
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np

from src.mlpipelineholder import PersistenceError, PipelineHandler, RegistrationError
from src.mlpipelineholder.artifact_store import ArtifactStore
from src.mlpipelineholder.models import ArtifactRecord


def produce() -> int:
    return 1


def produce_y() -> int:
    return 9


def consume(x: int) -> int:
    return x + 1


def produce_with_config(base: int) -> int:
    return base * 2


def scale(value: int, factor: int) -> int:
    return value * factor


def consume_values(*values: int) -> int:
    return sum(values)


def consume_list(values: list[object]) -> int:
    return len(values)


def consume_y(y: int) -> int:
    return y + 1


def transform_x(x: int) -> int:
    return x + 1


@dataclass
class ArraySpec:
    values: Any


def consume_spec(spec: ArraySpec) -> int:
    return int(spec.values.sum())


def call_callback(callback: Callable[[], int]) -> int:
    return callback()


class PickleWithoutDeepcopy:
    def __init__(self, value: int) -> None:
        self.value = value

    def __deepcopy__(self, memo: dict[int, Any]) -> "PickleWithoutDeepcopy":
        raise RuntimeError("no deepcopy")


def make_runtime_callable(value: int = 7) -> Callable[[], int]:
    def inner() -> int:
        return value

    return inner


def make_runtime_transform(delta: int) -> Callable[[int], int]:
    def inner(x: int) -> int:
        return x + delta

    return inner


def bind_runtime_wrapped() -> Callable[[int], int]:
    exec(
        "def partial_runtime_wrapped(value: int) -> int:\n    return value * 10\n",
        __main__.__dict__,
    )
    return getattr(__main__, "partial_runtime_wrapped")


class ForcedReRegistrationTests(unittest.TestCase):
    def test_forced_expression_override_warns_and_identical_reregistration_is_silent(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            block.register_expression("x = 1")
            warnings: list[str] = []

            def capture_warning(message: str) -> None:
                warnings.append(message)

            pipeline.logger.warning = capture_warning

            block.register_expression("x = 2", forced=True)

            self.assertTrue(
                any("was overridden with a different expression" in w for w in warnings)
            )
            warnings.clear()
            block.register_expression("x = 2", forced=True)

            self.assertEqual(warnings, [])

    def test_forbid_invalidate_objects_skips_erasure_and_allow_restores(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            block = pipeline.add_block("b", 1)
            block.register_function(make_runtime_callable(), ["x"])
            pipeline.add_block("c", 2).register_function(consume, ["y"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 7)
            self.assertEqual(pipeline.get_value("y"), 8)

            pipeline.forbid_invalidate_objects()
            block.register_function(make_runtime_callable(11), ["x"], forced=True)

            self.assertNotIn("x", pipeline.para_value_dict)
            self.assertEqual(pipeline.get_value("y"), 8)
            log_text = (tmp / "p" / "metadata" / "pipeline.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("is now FORBIDDEN", log_text)

            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 11)
            self.assertEqual(pipeline.get_value("y"), 12)

            pipeline.allow_invalidate_objects()
            block.register_function(make_runtime_callable(13), ["x"], forced=True)

            self.assertNotIn("x", pipeline.para_value_dict)
            self.assertNotIn("y", pipeline.para_value_dict)
            log_text = (tmp / "p" / "metadata" / "pipeline.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("is now ALLOWED", log_text)

    def test_forbidden_invalidation_expression_replacement_erases_own_keeps_downstream(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            block = pipeline.add_block("b", 1)
            block.register_expression("x = 1")
            pipeline.add_block("c", 2).register_function(consume, ["y"])
            pipeline.run_all()
            pipeline.forbid_invalidate_objects()

            block.register_expression("x = 2", forced=True)

            self.assertNotIn("x", pipeline.para_value_dict)
            self.assertEqual(pipeline.get_value("y"), 2)
            log_text = (tmp / "p" / "metadata" / "pipeline.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("erasing the block's own outputs", log_text)

            pipeline.allow_invalidate_objects()
            block.register_expression("x = 3", forced=True)

            self.assertNotIn("x", pipeline.para_value_dict)
            self.assertNotIn("y", pipeline.para_value_dict)

    def test_forbidden_invalidation_erases_removed_block_but_keeps_downstream(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            block.register_function(produce, ["x"])
            pipeline.add_block("c", 2).register_function(consume, ["y"])
            pipeline.run_all()
            pipeline.forbid_invalidate_objects()

            pipeline.remove_block("b")

            self.assertNotIn("x", pipeline.para_value_dict)
            self.assertEqual(pipeline.get_value("y"), 2)
            self.assertNotIn("b", pipeline.producer_outputs)

    def test_forbidden_invalidation_erases_replaced_block_but_keeps_downstream(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            block.register_function(produce, ["x"])
            pipeline.add_block("c", 2).register_function(consume, ["y"])
            pipeline.run_all()
            pipeline.forbid_invalidate_objects()

            pipeline.add_block("b", 1, forced=True)

            self.assertNotIn("x", pipeline.para_value_dict)
            self.assertEqual(pipeline.get_value("y"), 2)
            self.assertNotIn("b", pipeline.producer_outputs)

    def test_forbidden_invalidation_erases_replaced_pipeline_but_keeps_downstream(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            root = PipelineHandler("root", {}, tmp / "root")
            old_child = PipelineHandler("child", {}, tmp / "old_child")
            old_child.add_block("b", 1).register_function(produce, ["x"])
            root.add_child_pipeline(old_child, 1)
            root.add_block("tail", 2).register_function(consume, ["y"])
            root.run_all()
            root.forbid_invalidate_objects()

            new_child = PipelineHandler("child", {}, tmp / "new_child")
            new_child.add_block("b", 1).register_function(produce_y, ["x"])
            root.add_child_pipeline(new_child, 1, forced=True)

            self.assertNotIn("x", root.para_value_dict)
            self.assertEqual(root.get_value("y"), 2)
            self.assertNotIn("child", root.producer_outputs)
            self.assertEqual(old_child.para_value_dict, {})

    def test_forbidden_invalidation_erases_replaced_atom_but_keeps_downstream(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            pipeline.create_atom_child_pipeline(
                "atom",
                1,
                produce,
                output_variable_names=["x"],
                save_to_disk_lst=["x"],
            )
            pipeline.add_block("tail", 2).register_function(consume, ["y"])
            pipeline.run_all()
            artifact = pipeline.para_value_dict["x"]
            self.assertIsInstance(artifact, ArtifactRecord)
            artifact_path = Path(artifact.file_path)
            self.assertTrue(artifact_path.exists())
            pipeline.forbid_invalidate_objects()

            pipeline.create_atom_child_pipeline(
                "atom", 1, produce_y, output_variable_names=["x"]
            )

            self.assertNotIn("x", pipeline.para_value_dict)
            self.assertEqual(pipeline.get_value("y"), 2)
            self.assertNotIn("atom", pipeline.producer_outputs)
            self.assertFalse(artifact_path.exists())

    def test_forbidden_invalidation_erases_changed_block_when_function_removed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            block.register_function(produce, ["x"])
            pipeline.add_block("c", 2).register_function(consume, ["y"])
            pipeline.run_all()
            pipeline.forbid_invalidate_objects()

            block.remove_function("produce")

            self.assertNotIn("x", pipeline.para_value_dict)
            self.assertEqual(pipeline.get_value("y"), 2)
            self.assertNotIn("b", pipeline.producer_outputs)

    def test_forbidden_invalidation_keeps_outputs_when_gate_is_replaced(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "p", {"gate_enabled": True}, Path(temp_dir) / "p"
            )
            pipeline.set_gate_block("gate_enabled")
            pipeline.add_block("b", 1).register_function(produce, ["x"])
            pipeline.add_block("c", 2).register_function(consume, ["y"])
            pipeline.run_all()
            pipeline.forbid_invalidate_objects()

            pipeline.set_gate_block("gate_enabled", expected_value=False, forced=True)

            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertEqual(pipeline.get_value("y"), 2)

            pipeline.allow_invalidate_objects()
            pipeline.set_gate_block("gate_enabled", expected_value=True, forced=True)
            self.assertNotIn("x", pipeline.para_value_dict)
            self.assertNotIn("y", pipeline.para_value_dict)

    def test_forbidden_invalidation_keeps_outputs_when_gate_is_removed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "p", {"gate_enabled": True}, Path(temp_dir) / "p"
            )
            pipeline.set_gate_block("gate_enabled")
            pipeline.add_block("b", 1).register_function(produce, ["x"])
            pipeline.add_block("c", 2).register_function(consume, ["y"])
            pipeline.run_all()
            pipeline.forbid_invalidate_objects()

            pipeline.reset_gate_block()

            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertEqual(pipeline.get_value("y"), 2)

    def test_invalidation_mode_is_not_persisted(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "p"
            pipeline = PipelineHandler("p", {}, root)
            block = pipeline.add_block("b", 1)
            block.register_expression("x = 1")
            pipeline.add_block("c", 2).register_function(consume, ["y"])
            pipeline.run_all()
            pipeline.forbid_invalidate_objects()
            pipeline.save_pipeline()

            loaded = PipelineHandler.load_pipeline(root)
            loaded.get_block("b").register_expression("x = 2", forced=True)

            self.assertNotIn("x", loaded.para_value_dict)
            self.assertNotIn("y", loaded.para_value_dict)

    def test_invalidate_toggle_warns_only_on_state_change_and_requires_root(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            root = PipelineHandler("root", {}, tmp / "root")
            child = PipelineHandler("child", {}, tmp / "child")
            root.add_child_pipeline(child, 1)
            with self.assertRaisesRegex(RegistrationError, "root pipeline"):
                child.forbid_invalidate_objects()
            with self.assertRaisesRegex(RegistrationError, "root pipeline"):
                child.allow_invalidate_objects()

            root.forbid_invalidate_objects()
            root.forbid_invalidate_objects()
            root.allow_invalidate_objects()
            root.allow_invalidate_objects()

            log_text = (tmp / "root" / "metadata" / "pipeline.log").read_text(
                encoding="utf-8"
            )
            self.assertEqual(log_text.count("is now FORBIDDEN"), 1)
            self.assertEqual(log_text.count("is now ALLOWED"), 1)

    def test_forbid_invalidate_objects_applies_to_descendant_blocks(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            root = PipelineHandler("root", {}, tmp / "root")
            child = PipelineHandler("child", {}, tmp / "child")
            block = child.add_block("b", 1)
            block.register_function(make_runtime_callable(), ["x"])
            child.add_block("c", 2).register_function(consume, ["y"])
            root.add_child_pipeline(child, 1)
            root.run_all()
            self.assertEqual(root.get_value("y"), 8)

            root.forbid_invalidate_objects()
            block.register_function(make_runtime_callable(11), ["x"], forced=True)

            self.assertNotIn("x", child.para_value_dict)
            self.assertEqual(child.get_value("y"), 8)

            root.allow_invalidate_objects()
            block.register_function(make_runtime_callable(13), ["x"], forced=True)

            self.assertNotIn("x", child.para_value_dict)
            self.assertNotIn("y", child.para_value_dict)

    def test_newly_added_pipelines_inherit_invalidation_state_and_sync(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            root = PipelineHandler("root", {}, tmp / "root")
            root.forbid_invalidate_objects()

            child = PipelineHandler("child", {}, tmp / "child")
            block = child.add_block("b", 1)
            block.register_function(make_runtime_callable(), ["x"])
            child.add_block("c", 2).register_function(consume, ["y"])
            root.add_child_pipeline(child, 1)
            root.run_all()
            self.assertEqual(root.get_value("y"), 8)

            block.register_function(make_runtime_callable(11), ["x"], forced=True)

            self.assertNotIn("x", child.para_value_dict)
            self.assertEqual(child.get_value("y"), 8)

            root.allow_invalidate_objects()
            block.register_function(make_runtime_callable(13), ["x"], forced=True)

            self.assertNotIn("x", child.para_value_dict)
            self.assertNotIn("y", child.para_value_dict)

    def test_pipelines_attached_to_descendants_inherit_invalidation_state(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            root = PipelineHandler("root", {}, tmp / "root")
            child = PipelineHandler("child", {}, tmp / "child")
            root.add_child_pipeline(child, 1)
            root.forbid_invalidate_objects()

            grandchild = PipelineHandler("grandchild", {}, tmp / "grandchild")
            block = grandchild.add_block("b", 1)
            block.register_function(make_runtime_callable(), ["x"])
            grandchild.add_block("c", 2).register_function(consume, ["y"])
            child.add_child_pipeline(grandchild, 1)
            root.run_all()
            self.assertEqual(root.get_value("y"), 8)

            block.register_function(make_runtime_callable(11), ["x"], forced=True)

            self.assertNotIn("x", grandchild.para_value_dict)
            self.assertEqual(grandchild.get_value("y"), 8)

    def test_attaching_previous_root_transfers_invalidation_state_to_new_root(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            previous_root = PipelineHandler("old_root", {}, tmp / "old_root")
            block = previous_root.add_block("b", 1)
            block.register_function(make_runtime_callable(), ["x"])
            previous_root.add_block("c", 2).register_function(consume, ["y"])
            previous_root.forbid_invalidate_objects()

            new_root = PipelineHandler("new_root", {}, tmp / "new_root")
            sibling = PipelineHandler("sibling", {}, tmp / "sibling")
            sibling_block = sibling.add_block("sb", 1)
            sibling_block.register_function(make_runtime_callable(), ["sx"])
            sibling.add_block("sc", 2).register_function(
                consume, ["sy"], param_mapping={"x": "sx"}
            )
            new_root.add_child_pipeline(sibling, 1)
            new_root.add_child_pipeline(previous_root, 2)
            new_root.run_all()
            self.assertEqual(new_root.get_value("y"), 8)
            self.assertEqual(new_root.get_value("sy"), 8)
            transfer_log = (tmp / "new_root" / "metadata" / "pipeline.log").read_text(
                encoding="utf-8"
            )
            self.assertEqual(transfer_log.count("transferred FORBIDDEN state"), 1)

            block.register_function(make_runtime_callable(11), ["x"], forced=True)
            sibling_block.register_function(make_runtime_callable(11), ["sx"], forced=True)

            self.assertNotIn("x", previous_root.para_value_dict)
            self.assertEqual(previous_root.get_value("y"), 8)
            self.assertNotIn("sx", sibling.para_value_dict)
            self.assertEqual(sibling.get_value("sy"), 8)

            new_root.allow_invalidate_objects()
            block.register_function(make_runtime_callable(13), ["x"], forced=True)
            sibling_block.register_function(make_runtime_callable(13), ["sx"], forced=True)

            self.assertNotIn("x", previous_root.para_value_dict)
            self.assertNotIn("y", previous_root.para_value_dict)
            self.assertNotIn("sx", sibling.para_value_dict)
            self.assertNotIn("sy", sibling.para_value_dict)

    def test_attaching_forbidden_former_root_to_forbidden_tree_does_not_warn(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            new_root = PipelineHandler("new_root", {}, tmp / "new_root")
            new_root.forbid_invalidate_objects()
            old_root = PipelineHandler("old_root", {}, tmp / "old_root")
            old_root.forbid_invalidate_objects()

            new_root.add_child_pipeline(old_root, 1)

            log_text = (tmp / "new_root" / "metadata" / "pipeline.log").read_text(
                encoding="utf-8"
            )
            self.assertEqual(log_text.count("is now FORBIDDEN"), 1)
            self.assertEqual(log_text.count("transferred FORBIDDEN state"), 0)

    def test_attaching_allowed_former_root_does_not_change_forbidden_tree(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            new_root = PipelineHandler("new_root", {}, tmp / "new_root")
            new_root.forbid_invalidate_objects()
            old_root = PipelineHandler("old_root", {}, tmp / "old_root")
            block = old_root.add_block("b", 1)
            block.register_function(make_runtime_callable(), ["x"])
            old_root.add_block("c", 2).register_function(consume, ["y"])
            new_root.add_child_pipeline(old_root, 1)
            new_root.run_all()
            self.assertEqual(new_root.get_value("y"), 8)

            block.register_function(make_runtime_callable(11), ["x"], forced=True)

            self.assertNotIn("x", old_root.para_value_dict)
            self.assertEqual(old_root.get_value("y"), 8)
            log_text = (tmp / "new_root" / "metadata" / "pipeline.log").read_text(
                encoding="utf-8"
            )
            self.assertEqual(log_text.count("transferred FORBIDDEN state"), 0)

    def test_second_expression_in_block_is_rejected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            block.register_expression("x = 1")
            with self.assertRaisesRegex(RegistrationError, "at most one expression"):
                block.register_expression("y = 2")

    def test_mixed_logging_and_assignment_is_rejected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            block.register_expression("print('hi')")
            with self.assertRaisesRegex(RegistrationError, "at most one expression"):
                block.register_expression("x = 1")
            block2 = pipeline.add_block("b2", 2)
            block2.register_expression("x = 1")
            with self.assertRaisesRegex(RegistrationError, "at most one expression"):
                block2.register_expression("print('hi')")

    def test_same_identity_expression_forced_replaces(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            block.register_expression("x = 1")
            block.register_expression("x = 5", forced=True)
            self.assertEqual(block.functions[0].code, "x = 5")
            self.assertEqual(len(block.functions), 1)

    def test_changing_to_logging_erases_own_and_downstream(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            block.register_expression("x = 1")
            pipeline.add_block("c", 2).register_function(consume, ["y"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertEqual(pipeline.get_value("y"), 2)

            block.register_expression("print('log')", forced=True)

            self.assertNotIn("x", pipeline.para_value_dict)
            self.assertNotIn("y", pipeline.para_value_dict)
            self.assertEqual(block.functions[0].output_names, [])

    def test_changing_to_changing_different_code_erases_own_and_downstream(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            block.register_expression("x = 1")
            pipeline.add_block("c", 2).register_function(consume, ["y"])
            pipeline.run_all()

            block.register_expression("x = 2", forced=True)

            self.assertNotIn("x", pipeline.para_value_dict)
            self.assertNotIn("y", pipeline.para_value_dict)

    def test_logging_to_changing_without_collision_preserves_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            block.register_expression("print('log')")
            pipeline.add_block("c", 2).register_function(produce_y, ["y"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("y"), 9)

            block.register_expression("x = 1", forced=True)

            self.assertEqual(pipeline.get_value("y"), 9)
            self.assertNotIn("x", pipeline.para_value_dict)

    def test_logging_to_changing_collision_warns_and_erases_consumer(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            pipeline.add_block("a", 1).register_function(produce, ["x"])
            block = pipeline.add_block("b", 2)
            block.register_expression("print('log')")
            pipeline.add_block("d", 3).register_function(consume, ["y"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertEqual(pipeline.get_value("y"), 2)

            block.register_expression("x = 5", forced=True)
            log_text = (
                Path(temp_dir) / "p" / "metadata" / "pipeline.log"
            ).read_text(encoding="utf-8")

            self.assertIn(
                "are used as inputs by downstream block(s) 'p.d'", log_text
            )
            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertNotIn("d", pipeline.producer_outputs)
            self.assertNotIn("y", pipeline.para_value_dict)

    def test_logging_to_changing_shielded_by_intermediate_output_skips_erasure(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            block.register_expression("print('log')")
            pipeline.add_block("c", 2).register_function(produce, ["x"])
            pipeline.add_block("d", 3).register_function(consume, ["y"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertEqual(pipeline.get_value("y"), 2)

            block.register_expression("x = 5", forced=True)
            log_text = (
                Path(temp_dir) / "p" / "metadata" / "pipeline.log"
            ).read_text(encoding="utf-8")

            self.assertNotIn(
                "are used as inputs by downstream block(s)", log_text
            )
            self.assertIn("d", pipeline.producer_outputs)
            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertEqual(pipeline.get_value("y"), 2)

    def test_logging_to_changing_collision_in_descendant_cleans_mirror(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            pipeline.add_block("a", 1).register_function(produce, ["x"])
            block = pipeline.add_block("b", 2)
            block.register_expression("print('log')")
            child = PipelineHandler("child", {}, tmp / "child")
            child.add_block("cb", 1).register_function(consume, ["y"])
            pipeline.add_child_pipeline(child, 3)
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertEqual(pipeline.get_value("y"), 2)
            self.assertIn("child", pipeline.producer_outputs)

            block.register_expression("x = 5", forced=True)
            log_text = (tmp / "p" / "metadata" / "pipeline.log").read_text(
                encoding="utf-8"
            )

            self.assertIn(
                "downstream block(s) 'p/child.cb'", log_text
            )
            self.assertNotIn("y", child.para_value_dict)
            self.assertNotIn("child", pipeline.producer_outputs)
            self.assertEqual(pipeline.get_value("x"), 1)

    def test_logging_to_changing_shielded_in_descendant_skips_erasure(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            block = pipeline.add_block("b", 1)
            block.register_expression("print('log')")
            pipeline.add_block("c", 2).register_function(produce, ["x"])
            child = PipelineHandler("child", {}, tmp / "child")
            child.add_block("cb", 1).register_function(consume, ["y"])
            pipeline.add_child_pipeline(child, 3)
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertEqual(pipeline.get_value("y"), 2)

            block.register_expression("x = 5", forced=True)
            log_text = (tmp / "p" / "metadata" / "pipeline.log").read_text(
                encoding="utf-8"
            )

            self.assertNotIn(
                "are used as inputs by downstream block(s)", log_text
            )
            self.assertIn("y", child.para_value_dict)
            self.assertIn("child", pipeline.producer_outputs)
            self.assertEqual(pipeline.get_value("y"), 2)

    def test_logging_to_changing_collision_in_ancestor_erases_consumer(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            pipeline.add_block("a", 0.5).register_function(produce, ["x"])
            child = PipelineHandler("child", {}, tmp / "child")
            block = child.add_block("b", 1)
            block.register_expression("print('log')")
            pipeline.add_child_pipeline(child, 1)
            pipeline.add_block("cons", 2).register_function(consume, ["y"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertEqual(pipeline.get_value("y"), 2)

            block.register_expression("x = 5", forced=True)
            log_text = (tmp / "p" / "metadata" / "pipeline.log").read_text(
                encoding="utf-8"
            )

            self.assertIn(
                "downstream block(s) 'p.cons'", log_text
            )
            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertNotIn("cons", pipeline.producer_outputs)
            self.assertNotIn("y", pipeline.para_value_dict)

    def test_logging_to_changing_shielded_in_ancestor_skips_erasure(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            child = PipelineHandler("child", {}, tmp / "child")
            block = child.add_block("b", 1)
            block.register_expression("print('log')")
            pipeline.add_child_pipeline(child, 1)
            pipeline.add_block("prod", 2).register_function(produce, ["x"])
            pipeline.add_block("cons", 3).register_function(consume, ["y"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertEqual(pipeline.get_value("y"), 2)

            block.register_expression("x = 5", forced=True)
            log_text = (tmp / "p" / "metadata" / "pipeline.log").read_text(
                encoding="utf-8"
            )

            self.assertNotIn(
                "are used as inputs by downstream block(s)", log_text
            )
            self.assertIn("cons", pipeline.producer_outputs)
            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertEqual(pipeline.get_value("y"), 2)

    def test_identical_expression_forced_is_noop(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            block.register_expression("x = 1")
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 1)

            registration = block.register_expression("x = 1", forced=True)

            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertEqual(registration.code, "x = 1")

    def test_changing_to_changing_preserves_unrelated_downstream_block(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            block.register_expression("x = 1")
            pipeline.add_block("c", 2).register_function(produce_y, ["y"])
            pipeline.add_block("d", 3).register_function(consume, ["z"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertEqual(pipeline.get_value("y"), 9)
            self.assertEqual(pipeline.get_value("z"), 2)

            block.register_expression("x = 2", forced=True)

            self.assertNotIn("x", pipeline.para_value_dict)
            self.assertNotIn("d", pipeline.producer_outputs)
            self.assertNotIn("z", pipeline.para_value_dict)
            self.assertEqual(pipeline.get_value("y"), 9)

    def test_function_override_erases_own_and_earliest_consumer_only(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            first = make_runtime_callable()
            block.register_function(first, ["x"])
            pipeline.add_block("c", 2).register_function(produce_y, ["y"])
            pipeline.add_block("d", 3).register_function(consume, ["z"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 7)
            self.assertEqual(pipeline.get_value("y"), 9)
            self.assertEqual(pipeline.get_value("z"), 8)

            block.register_function(make_runtime_callable(), ["x"], forced=True)

            self.assertNotIn("x", pipeline.para_value_dict)
            self.assertNotIn("d", pipeline.producer_outputs)
            self.assertNotIn("z", pipeline.para_value_dict)
            self.assertEqual(pipeline.get_value("y"), 9)

    def test_descendant_invalidation_erases_ancestor_downstream(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            source = pipeline.add_block("source", 1)
            source.register_function(make_runtime_callable(), ["x"])
            child = PipelineHandler("child", {}, tmp / "child")
            child.add_block("transform", 1).register_function(consume, ["y"])
            pipeline.add_child_pipeline(child, 2)
            pipeline.add_block("tail", 3).register_function(consume_y, ["z"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("z"), 9)

            source.register_function(make_runtime_callable(), ["x"], forced=True)

            self.assertNotIn("y", child.para_value_dict)
            self.assertNotIn("z", pipeline.para_value_dict)
            self.assertNotIn("tail", pipeline.producer_outputs)

    def test_child_same_name_transform_erases_parent_consumer(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            source = pipeline.add_block("source", 1)
            source.register_function(make_runtime_callable(), ["x"])
            child = PipelineHandler("child", {}, tmp / "child")
            child.add_block("transform", 1).register_function(transform_x, ["x"])
            pipeline.add_child_pipeline(child, 2)
            pipeline.add_block("tail", 3).register_function(consume, ["z"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("z"), 9)

            source.register_function(make_runtime_callable(), ["x"], forced=True)

            self.assertNotIn("x", child.para_value_dict)
            self.assertNotIn("z", pipeline.para_value_dict)

    def test_multilevel_same_name_transform_erases_root_consumer(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            root = PipelineHandler("root", {}, tmp / "root")
            source = root.add_block("source", 0.5)
            source.register_function(make_runtime_callable(), ["x"])
            middle = PipelineHandler("middle", {}, tmp / "middle")
            leaf = PipelineHandler("leaf", {}, tmp / "leaf")
            leaf.add_block("transform", 1).register_function(transform_x, ["x"])
            middle.add_child_pipeline(leaf, 1)
            root.add_child_pipeline(middle, 1)
            root.add_block("tail", 2).register_function(consume, ["z"])
            root.run_all()
            self.assertEqual(root.get_value("z"), 9)

            source.register_function(make_runtime_callable(), ["x"], forced=True)

            self.assertNotIn("z", root.para_value_dict)
            self.assertNotIn("tail", root.producer_outputs)

    def test_descendant_override_does_not_mirror_inherited_same_name(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            pipeline.add_block("source", 1).register_function(produce, ["x"])
            child = PipelineHandler("child", {}, tmp / "child")
            transform = child.add_block("transform", 1)
            transform.register_function(make_runtime_transform(1), ["x"])
            pipeline.add_child_pipeline(child, 2)
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 2)

            transform.register_function(
                make_runtime_transform(2),
                ["x"],
                forced=True,
            )

            self.assertNotIn("child", pipeline.producer_outputs)
            self.assertEqual(pipeline.get_value("x"), 1)

    def test_intermediate_ancestor_producer_shields_root_consumer(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            root = PipelineHandler("root", {}, tmp / "root")
            middle = PipelineHandler("middle", {}, tmp / "middle")
            leaf = PipelineHandler("leaf", {}, tmp / "leaf")
            expression = leaf.add_block("expression", 1)
            expression.register_expression("print('log')")
            middle.add_child_pipeline(leaf, 1)
            middle.add_block("shield", 2).register_function(produce, ["x"])
            root.add_child_pipeline(middle, 1)
            root.add_block("consumer", 2).register_function(consume, ["y"])
            root.run_all()
            self.assertEqual(root.get_value("y"), 2)

            expression.register_expression("x = 5", forced=True)

            self.assertEqual(root.get_value("y"), 2)
            self.assertIn("consumer", root.producer_outputs)

    def test_unexecuted_same_name_producer_does_not_shield_consumer(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            source = pipeline.add_block("source", 1)
            source.register_function(make_runtime_callable(), ["x"])
            pipeline.add_block("shield", 2).register_function(produce, ["x"])
            pipeline.add_block("consumer", 3).register_function(consume, ["y"])
            pipeline.run_block("source")
            pipeline.run_block("consumer")
            self.assertEqual(pipeline.get_value("y"), 8)

            source.register_function(make_runtime_callable(), ["x"], forced=True)

            self.assertNotIn("y", pipeline.para_value_dict)
            self.assertNotIn("consumer", pipeline.producer_outputs)

    def test_invalid_function_override_preserves_old_registration_and_outputs(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "p",
                {},
                Path(temp_dir) / "p",
                strict_mode=True,
            )
            block = pipeline.add_block("producer", 1)
            original = block.register_function(produce, ["x"])
            pipeline.run_all()

            with self.assertRaises(RegistrationError):
                block.register_function(
                    produce,
                    ["x"],
                    save_to_disk=["not_x"],
                    forced=True,
                )

            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertIs(block.functions[0], original)

    def test_atom_override_erases_own_and_earliest_consumer_only(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            pipeline.create_atom_child_pipeline(
                "atom", 1, produce, output_variable_names=["x"]
            )
            pipeline.add_block("c", 2).register_function(produce_y, ["y"])
            pipeline.add_block("d", 3).register_function(consume, ["z"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertEqual(pipeline.get_value("y"), 9)
            self.assertEqual(pipeline.get_value("z"), 2)
            first_atom = pipeline.get_child_pipeline("atom")

            pipeline.create_atom_child_pipeline(
                "atom", 1, produce_y, output_variable_names=["x"]
            )

            self.assertNotIn("x", pipeline.para_value_dict)
            self.assertNotIn("d", pipeline.producer_outputs)
            self.assertNotIn("z", pipeline.para_value_dict)
            self.assertEqual(pipeline.get_value("y"), 9)
            new_atom = pipeline.get_child_pipeline("atom")
            self.assertIsNot(new_atom, first_atom)
            self.assertTrue(new_atom._is_atom)

    def test_identical_function_forced_is_noop(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            block.register_function(produce, ["x"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 1)

            block.register_function(produce, ["x"], forced=True)

            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertIn("x", pipeline.producer_outputs["b"])

    def test_redefined_runtime_callable_is_treated_as_different(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            first = make_runtime_callable()
            block.register_function(first, ["x"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 7)

            second = make_runtime_callable()
            block.register_function(second, ["x"], forced=True)

            self.assertNotIn("x", pipeline.para_value_dict)

    def test_import_path_identity_after_load_preserves_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            block = pipeline.add_block("b", 1)
            block.register_function(produce, ["x"])
            pipeline.run_all()
            pipeline.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(tmp / "bundle", forced_deleting=True)
            loaded_block = loaded.get_block("b")
            self.assertEqual(loaded.get_value("x"), 1)

            loaded_block.register_function(produce, ["x"], forced=True)

            self.assertEqual(loaded.get_value("x"), 1)
            self.assertIn("x", loaded.producer_outputs["b"])

    def test_identical_inline_partial_forced_is_noop(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            block.register_function(partial(scale, 3, factor=2), ["x"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 6)

            block.register_function(partial(scale, 3, factor=2), ["x"], forced=True)

            self.assertEqual(pipeline.get_value("x"), 6)
            self.assertIn("x", pipeline.producer_outputs["b"])

    def test_changed_partial_binding_forced_replaces(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            block.register_function(partial(scale, 3, factor=2), ["x"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 6)

            block.register_function(partial(scale, 3, factor=3), ["x"], forced=True)

            self.assertNotIn("x", pipeline.para_value_dict)

    def test_equal_dataclass_numpy_partial_binding_is_noop(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            first = block.register_function(
                partial(consume_spec, ArraySpec(np.array([1, 2]))),
                ["x"],
            )
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 3)

            second = block.register_function(
                partial(consume_spec, ArraySpec(np.array([1, 2]))),
                ["x"],
                forced=True,
            )

            self.assertIs(second, first)
            self.assertEqual(pipeline.get_value("x"), 3)

    def test_changed_dataclass_numpy_partial_binding_replaces(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            block.register_function(
                partial(consume_spec, ArraySpec(np.array([1, 2]))),
                ["x"],
            )
            pipeline.run_all()

            block.register_function(
                partial(consume_spec, ArraySpec(np.array([1, 3]))),
                ["x"],
                forced=True,
            )

            self.assertNotIn("x", pipeline.para_value_dict)

    def test_partial_of_redefined_runtime_callable_replaces(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            block.register_function(partial(make_runtime_callable()), ["x"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 7)

            block.register_function(
                partial(make_runtime_callable()), ["x"], forced=True
            )

            self.assertNotIn("x", pipeline.para_value_dict)

    def test_identical_partial_atom_reregistration_is_noop(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            pipeline.create_atom_child_pipeline(
                "atom", 1, partial(scale, 3, factor=2), output_variable_names=["x"]
            )
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 6)

            pipeline.create_atom_child_pipeline(
                "atom", 1, partial(scale, 3, factor=2), output_variable_names=["x"]
            )

            self.assertEqual(pipeline.get_value("x"), 6)

    def test_changed_partial_atom_reregistration_replaces(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            pipeline.create_atom_child_pipeline(
                "atom", 1, partial(scale, 3, factor=2), output_variable_names=["x"]
            )
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 6)

            pipeline.create_atom_child_pipeline(
                "atom", 1, partial(scale, 4, factor=2), output_variable_names=["x"]
            )

            self.assertNotIn("x", pipeline.para_value_dict)

    def test_partial_registration_round_trips_save_and_load(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            block = pipeline.add_block("b", 1)
            block.register_function(partial(scale, 3, factor=2), ["x"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 6)
            pipeline.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(tmp / "bundle", forced_deleting=True)
            self.assertEqual(loaded.get_value("x"), 6)

            loaded.get_block("b").register_function(
                partial(scale, 3, factor=2), ["x"], forced=True
            )

            self.assertEqual(loaded.get_value("x"), 6)
            self.assertIn("x", loaded.producer_outputs["b"])

    def test_partial_atom_round_trips_save_and_load(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            pipeline.create_atom_child_pipeline(
                "atom", 1, partial(scale, 3, factor=2), output_variable_names=["x"]
            )
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 6)
            pipeline.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(tmp / "bundle", forced_deleting=True)
            self.assertEqual(loaded.get_value("x"), 6)

            loaded.create_atom_child_pipeline(
                "atom", 1, partial(scale, 3, factor=2), output_variable_names=["x"]
            )

            self.assertEqual(loaded.get_value("x"), 6)

    def test_partial_wrapping_runtime_callable_round_trips(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            wrapped = bind_runtime_wrapped()
            pipeline = PipelineHandler("p", {}, tmp / "p")
            block = pipeline.add_block("b", 1)
            block.register_function(partial(wrapped, 5), ["x"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 50)
            pipeline.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(tmp / "bundle", forced_deleting=True)
            self.assertEqual(loaded.get_value("x"), 50)

    def test_partial_with_nested_placeholder_argument_fails_save(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            pipeline.add_block("b", 1).register_function(
                partial(consume_list, [lambda: 1]),
                ["x"],
            )

            with self.assertRaisesRegex(PersistenceError, "Partial argument"):
                pipeline.save_pipeline(tmp / "bundle")

    def test_missing_partial_callable_does_not_replace_working_tree(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            work_root = tmp / "work"
            bundle = tmp / "bundle"
            pipeline = PipelineHandler("p", {}, work_root)
            pipeline.add_block("b", 1).register_function(
                partial(scale, 3, factor=2),
                ["x"],
            )
            pipeline.save_pipeline(bundle)
            marker = work_root / "unsaved.txt"
            marker.write_text("preserve me", encoding="utf-8")
            state_path = bundle / "pipeline_state.pkl"
            with state_path.open("rb") as handle:
                payload = pickle.load(handle)
            partial_payload = payload["nodes"][0]["functions"][0]["partial"]
            partial_payload.pop("partial", None)
            partial_payload["func_import_path"] = None
            partial_payload["func_callable_name"] = "missing_partial_callable"
            with state_path.open("wb") as handle:
                pickle.dump(payload, handle)

            with self.assertRaisesRegex(PersistenceError, "missing_partial_callable"):
                PipelineHandler.load_pipeline(bundle, forced_deleting=True)

            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve me")

    def test_missing_bound_runtime_callable_does_not_replace_working_tree(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            work_root = tmp / "work"
            bundle = tmp / "bundle"
            exec(
                "def missing_bound_callback() -> int:\n    return 10\n",
                __main__.__dict__,
            )
            callback = getattr(__main__, "missing_bound_callback")
            try:
                pipeline = PipelineHandler("p", {}, work_root)
                pipeline.add_block("b", 1).register_function(
                    partial(call_callback, callback),
                    ["x"],
                )
                pipeline.save_pipeline(bundle)
                marker = work_root / "unsaved.txt"
                marker.write_text("preserve me", encoding="utf-8")
                delattr(__main__, "missing_bound_callback")

                with self.assertRaisesRegex(
                    PersistenceError,
                    "missing_bound_callback",
                ):
                    PipelineHandler.load_pipeline(bundle, forced_deleting=True)

                self.assertEqual(
                    marker.read_text(encoding="utf-8"),
                    "preserve me",
                )
            finally:
                if hasattr(__main__, "missing_bound_callback"):
                    delattr(__main__, "missing_bound_callback")

    def test_picklable_value_without_deepcopy_round_trips(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "work")
            pipeline.set_constant_value(
                "payload",
                PickleWithoutDeepcopy(7),
                copy=False,
            )
            pipeline.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(
                tmp / "bundle",
                forced_deleting=True,
            )

            self.assertEqual(loaded.get_constant_value("payload").value, 7)

    def test_artifact_cleanup_failure_does_not_abort_function_replacement(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            block = pipeline.add_block("b", 1)
            original = block.register_function(
                partial(scale, 3, factor=2),
                ["x"],
                save_to_disk=["x"],
            )
            pipeline.run_all()

            with patch.object(
                pipeline.artifact_store,
                "delete",
                side_effect=PermissionError("read-only artifact"),
            ):
                replacement = block.register_function(
                    partial(scale, 3, factor=3),
                    ["x"],
                    save_to_disk=["x"],
                    forced=True,
                )

            self.assertIsNot(replacement, original)
            self.assertIs(block.functions[0], replacement)
            self.assertNotIn("x", pipeline.para_value_dict)

    def test_artifact_store_refuses_delete_outside_artifact_root(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            outside = tmp / "outside.txt"
            outside.write_text("keep", encoding="utf-8")
            store = ArtifactStore(tmp / "project")
            record = ArtifactRecord(
                variable_name="x",
                serializer="pickle",
                file_path=str(outside),
                produced_by_block="b",
                produced_by_function="f",
                run_id="run",
            )

            with self.assertRaisesRegex(PersistenceError, "outside artifact root"):
                store.delete(record)

            self.assertEqual(outside.read_text(encoding="utf-8"), "keep")

    def test_artifact_store_ignores_missing_outside_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            store = ArtifactStore(tmp / "project")
            record = ArtifactRecord(
                variable_name="x",
                serializer="pickle",
                file_path=str(tmp / "missing.txt"),
                produced_by_block="b",
                produced_by_function="f",
                run_id="run",
            )

            store.delete(record)

    def test_changed_args_helper_prevents_function_noop(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
            pipeline.set_constant_value("a", 1)
            pipeline.set_constant_value("b", 2)
            block = pipeline.add_block("producer", 1)
            block.register_args("items", ["a"])
            block.register_function(
                consume_values,
                ["x"],
                var_pos_name="items",
            )
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 1)
            block.register_args("items", ["b"], forced=True)

            block.register_function(
                consume_values,
                ["x"],
                var_pos_name="items",
                forced=True,
            )

            self.assertNotIn("x", pipeline.para_value_dict)


class AtomPipelineEqualityTests(unittest.TestCase):
    def test_rejected_atom_over_block_preserves_block_and_output(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            existing = pipeline.add_block("atom", 1)
            existing.register_function(produce, ["old"])
            pipeline.run_all()
            container = PipelineHandler("container", {}, tmp / "container")
            duplicate = PipelineHandler("atom", {}, tmp / "duplicate")
            container.add_child_pipeline(duplicate, 1)
            pipeline.add_child_pipeline(container, 2)

            with self.assertRaisesRegex(RegistrationError, "Pipeline names must be unique"):
                pipeline.create_atom_child_pipeline(
                    "atom",
                    1,
                    produce,
                    output_variable_names=["new"],
                )

            self.assertIs(pipeline.get_block("atom"), existing)
            self.assertEqual(pipeline.get_value("old"), 1)
            self.assertIn("old", pipeline.producer_outputs["atom"])

    def test_equivalent_mutable_child_is_replaced_by_locked_atom(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            child = PipelineHandler(
                "atom",
                {},
                tmp / "child",
                execution_priority=1,
            )
            child.add_block("atom_block", 10).register_function(produce, ["x"])
            pipeline.add_child_pipeline(child, 1)

            pipeline.create_atom_child_pipeline(
                "atom",
                1,
                produce,
                output_variable_names=["x"],
            )

            atom = pipeline.get_child_pipeline("atom")
            self.assertIsNot(atom, child)
            self.assertTrue(atom._is_atom)

    def test_identical_atom_reregistration_is_noop(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            pipeline.create_atom_child_pipeline("atom", 1, produce, output_variable_names=["x"])
            pipeline.add_block("c", 2).register_function(consume, ["y"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertEqual(pipeline.get_value("y"), 2)
            first_atom = pipeline.get_child_pipeline("atom")

            pipeline.create_atom_child_pipeline(
                "atom", 1, produce, output_variable_names=["x"]
            )

            self.assertEqual(pipeline.get_value("x"), 1)
            self.assertEqual(pipeline.get_value("y"), 2)
            self.assertIs(pipeline.get_child_pipeline("atom"), first_atom)

    def test_identical_redefined_main_atom_reregistration_is_noop(self) -> None:
        source = (
            "def notebook_field_specs(param_definitions, field_info, "
            "existing_field_specs=None, delimiter='_'):\n"
            "    return [*param_definitions, *field_info, existing_field_specs, delimiter]\n"
        )
        exec(source, __main__.__dict__)
        try:
            with TemporaryDirectory() as temp_dir:
                pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
                pipeline.set_constant_value("definitions", [("weight", 1.0)])
                pipeline.set_constant_value("field_info", [("score", float)])
                pipeline.set_constant_value("delimiter", "_")
                pipeline.create_atom_child_pipeline(
                    "atom",
                    1,
                    getattr(__main__, "notebook_field_specs"),
                    output_variable_names="field_specs",
                    param_mapping={
                        "param_definitions": "definitions",
                        "field_info": "field_info",
                        "existing_field_specs": None,
                        "delimiter": "delimiter",
                    },
                )
                pipeline.add_block("consumer", 2).register_function(
                    consume_list,
                    ["field_count"],
                    param_mapping={"values": "field_specs"},
                )
                pipeline.run_all()
                first_atom = pipeline.get_child_pipeline("atom")

                exec(source, __main__.__dict__)
                pipeline.create_atom_child_pipeline(
                    "atom",
                    1,
                    getattr(__main__, "notebook_field_specs"),
                    output_variable_names="field_specs",
                    param_mapping={
                        "param_definitions": "definitions",
                        "field_info": "field_info",
                        "existing_field_specs": None,
                        "delimiter": "delimiter",
                    },
                )

                self.assertIs(pipeline.get_child_pipeline("atom"), first_atom)
                self.assertEqual(pipeline.get_value("field_count"), 4)
        finally:
            if hasattr(__main__, "notebook_field_specs"):
                delattr(__main__, "notebook_field_specs")

    def test_changed_redefined_main_atom_reregistration_replaces(self) -> None:
        first_source = "def notebook_increment(value: int) -> int:\n    return value + 1\n"
        second_source = "def notebook_increment(value: int) -> int:\n    return value + 2\n"
        exec(first_source, __main__.__dict__)
        try:
            with TemporaryDirectory() as temp_dir:
                pipeline = PipelineHandler("p", {}, Path(temp_dir) / "p")
                pipeline.set_constant_value("value", 1)
                pipeline.create_atom_child_pipeline(
                    "atom",
                    1,
                    getattr(__main__, "notebook_increment"),
                    output_variable_names="x",
                )
                pipeline.run_all()
                first_atom = pipeline.get_child_pipeline("atom")

                exec(second_source, __main__.__dict__)
                pipeline.create_atom_child_pipeline(
                    "atom",
                    1,
                    getattr(__main__, "notebook_increment"),
                    output_variable_names="x",
                )

                self.assertIsNot(pipeline.get_child_pipeline("atom"), first_atom)
                self.assertNotIn("x", pipeline.para_value_dict)
        finally:
            if hasattr(__main__, "notebook_increment"):
                delattr(__main__, "notebook_increment")

    def test_atom_override_brutal_erasure_crosses_ancestor_only_when_allowed(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            root = PipelineHandler("quant_pipeline", {}, tmp / "root")
            score = PipelineHandler("score_analysis_pipeline", {}, tmp / "score")
            score.create_atom_child_pipeline(
                "score_produce_median_params",
                10,
                produce,
                output_variable_names="score_best_params",
            )
            score.add_block("score_update_sc_config", 20).register_function(
                consume,
                ["score_sc_config"],
                param_mapping={"x": "score_best_params"},
            )
            root.add_child_pipeline(score, 40)
            root.add_block("produce_output_df", 50).register_function(
                produce_y,
                ["output_df"],
            )
            root.run_all()

            root.forbid_invalidate_objects()
            score.create_atom_child_pipeline(
                "score_produce_median_params",
                10,
                produce_y,
                output_variable_names="score_best_params",
            )

            self.assertNotIn("score_best_params", root.para_value_dict)
            self.assertEqual(root.get_value("score_sc_config"), 2)
            self.assertEqual(root.get_value("output_df"), 9)

            root.allow_invalidate_objects()
            root.run_all()
            score.create_atom_child_pipeline(
                "score_produce_median_params",
                10,
                produce,
                output_variable_names="score_best_params",
            )

            self.assertNotIn("score_best_params", root.para_value_dict)
            self.assertNotIn("score_sc_config", root.para_value_dict)
            self.assertNotIn("output_df", root.para_value_dict)

    def test_atom_priority_change_replaces(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            pipeline.create_atom_child_pipeline("atom", 1, produce, output_variable_names=["x"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 1)

            pipeline.create_atom_child_pipeline(
                "atom", 2, produce, output_variable_names=["x"]
            )

            self.assertNotIn("x", pipeline.para_value_dict)

    def test_atom_output_change_replaces(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            pipeline.create_atom_child_pipeline("atom", 1, produce, output_variable_names=["x"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 1)

            pipeline.create_atom_child_pipeline(
                "atom", 1, produce, output_variable_names=["y"]
            )

            self.assertNotIn("x", pipeline.para_value_dict)

    def test_atom_used_config_field_change_replaces(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {"base": 5}, tmp / "p")
            pipeline.create_atom_child_pipeline(
                "atom",
                1,
                produce_with_config,
                output_variable_names=["x"],
                param_mapping={"base": "base"},
            )
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 10)

            pipeline.set_config({"base": 7})
            pipeline.create_atom_child_pipeline(
                "atom",
                1,
                produce_with_config,
                output_variable_names=["x"],
                param_mapping={"base": "base"},
            )

            self.assertNotIn("x", pipeline.para_value_dict)

    def test_atom_direct_config_input_change_replaces(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {"base": 5}, tmp / "p")
            pipeline.create_atom_child_pipeline(
                "atom",
                1,
                produce_with_config,
                output_variable_names=["x"],
            )
            pipeline.run_all()
            first_atom = pipeline.get_child_pipeline("atom")
            self.assertEqual(pipeline.get_value("x"), 10)
            pipeline.set_config({"base": 7})

            pipeline.create_atom_child_pipeline(
                "atom",
                1,
                produce_with_config,
                output_variable_names=["x"],
            )

            self.assertIsNot(pipeline.get_child_pipeline("atom"), first_atom)
            self.assertNotIn("x", pipeline.para_value_dict)

    def test_atom_consumed_runtime_callable_change_replaces(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            exec(
                "def atom_runtime_callback() -> int:\n    return 1\n",
                __main__.__dict__,
            )
            try:
                pipeline = PipelineHandler(
                    "p",
                    {"callback": getattr(__main__, "atom_runtime_callback")},
                    tmp / "p",
                )
                pipeline.create_atom_child_pipeline(
                    "atom",
                    1,
                    call_callback,
                    output_variable_names=["x"],
                )
                pipeline.run_all()
                first_atom = pipeline.get_child_pipeline("atom")
                self.assertEqual(pipeline.get_value("x"), 1)
                exec(
                    "def atom_runtime_callback() -> int:\n    return 2\n",
                    __main__.__dict__,
                )
                pipeline.set_config(
                    {"callback": getattr(__main__, "atom_runtime_callback")}
                )

                pipeline.create_atom_child_pipeline(
                    "atom",
                    1,
                    call_callback,
                    output_variable_names=["x"],
                )

                self.assertIsNot(pipeline.get_child_pipeline("atom"), first_atom)
                self.assertNotIn("x", pipeline.para_value_dict)
            finally:
                if hasattr(__main__, "atom_runtime_callback"):
                    delattr(__main__, "atom_runtime_callback")

    def test_equal_dataclass_numpy_used_config_is_noop(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            pipeline.create_atom_child_pipeline(
                "atom",
                1,
                consume_spec,
                output_variable_names=["x"],
                param_mapping={"spec": "spec"},
                child_configuration={"spec": ArraySpec(np.array([1, 2]))},
            )
            pipeline.run_all()
            first_atom = pipeline.get_child_pipeline("atom")
            self.assertEqual(pipeline.get_value("x"), 3)

            pipeline.create_atom_child_pipeline(
                "atom",
                1,
                consume_spec,
                output_variable_names=["x"],
                param_mapping={"spec": "spec"},
                child_configuration={"spec": ArraySpec(np.array([1, 2]))},
            )

            self.assertIs(pipeline.get_child_pipeline("atom"), first_atom)
            self.assertEqual(pipeline.get_value("x"), 3)

    def test_invalid_atom_priority_change_preserves_existing_atom(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            pipeline.create_atom_child_pipeline(
                "atom",
                1,
                produce,
                output_variable_names=["x"],
            )
            pipeline.add_block("occupied", 2).register_function(produce_y, ["y"])
            pipeline.run_all()
            original = pipeline.get_child_pipeline("atom")

            with self.assertRaises(RegistrationError):
                pipeline.create_atom_child_pipeline(
                    "atom",
                    2,
                    produce,
                    output_variable_names=["x"],
                )

            self.assertIs(pipeline.get_child_pipeline("atom"), original)
            self.assertEqual(pipeline.get_value("x"), 1)

    def test_atom_used_config_missing_and_none_are_different(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            pipeline.create_atom_child_pipeline(
                "atom",
                1,
                produce_with_config,
                output_variable_names=["x"],
                param_mapping={"base": "base"},
            )
            first_atom = pipeline.get_child_pipeline("atom")

            pipeline.create_atom_child_pipeline(
                "atom",
                1,
                produce_with_config,
                output_variable_names=["x"],
                param_mapping={"base": "base"},
                child_configuration={"base": None},
            )

            self.assertIsNot(pipeline.get_child_pipeline("atom"), first_atom)

    def test_atom_unused_config_field_change_is_noop(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {"unused": 1}, tmp / "p")
            pipeline.create_atom_child_pipeline(
                "atom",
                1,
                produce_with_config,
                output_variable_names=["x"],
                param_mapping={"base": "base"},
            )
            pipeline.set_config({"unused": 1, "base": 5})
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 10)
            first_atom = pipeline.get_child_pipeline("atom")

            pipeline.set_config({"unused": 99, "base": 5})
            pipeline.create_atom_child_pipeline(
                "atom",
                1,
                produce_with_config,
                output_variable_names=["x"],
                param_mapping={"base": "base"},
            )

            self.assertEqual(pipeline.get_value("x"), 10)
            self.assertIs(pipeline.get_child_pipeline("atom"), first_atom)

    def test_atom_function_change_replaces(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            pipeline.create_atom_child_pipeline("atom", 1, produce, output_variable_names=["x"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 1)

            pipeline.create_atom_child_pipeline(
                "atom", 1, produce_y, output_variable_names=["x"]
            )

            self.assertNotIn("x", pipeline.para_value_dict)

    def test_atom_gate_change_replaces(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {"flag": True}, tmp / "p")
            pipeline.create_atom_child_pipeline(
                "atom",
                1,
                produce,
                output_variable_names=["x"],
                gate_config="flag",
                expected_value=True,
            )
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("x"), 1)

            pipeline.create_atom_child_pipeline(
                "atom",
                1,
                produce,
                output_variable_names=["x"],
                gate_config="flag",
                expected_value=False,
            )

            self.assertNotIn("x", pipeline.para_value_dict)


class AtomLockTests(unittest.TestCase):
    def _locked_atom(self, temp_dir: str) -> tuple[PipelineHandler, PipelineHandler]:
        tmp = Path(temp_dir)
        pipeline = PipelineHandler("p", {}, tmp / "p")
        pipeline.create_atom_child_pipeline(
            "atom", 1, produce, output_variable_names=["x"]
        )
        pipeline.run_all()
        return pipeline, pipeline.get_child_pipeline("atom")

    def test_atom_rejects_structural_mutations(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline, atom = self._locked_atom(temp_dir)
            with self.assertRaises(RegistrationError):
                atom.add_block("extra", 3)
            with self.assertRaises(RegistrationError):
                atom.add_child_pipeline(PipelineHandler("c2", {}, Path(temp_dir) / "c2"), 2)
            with self.assertRaises(RegistrationError):
                atom.create_atom_child_pipeline(
                    "nested", 1, produce, output_variable_names=["x"]
                )
            with self.assertRaises(RegistrationError):
                atom.add_gate_block("flag", expected_value=True)
            with self.assertRaises(RegistrationError):
                atom.set_gate_block("flag", expected_value=True)
            with self.assertRaises(RegistrationError):
                atom.reset_gate_block()
            with self.assertRaises(RegistrationError):
                atom.remove_block("atom_block")
            block = atom.get_block("atom_block")
            with self.assertRaises(RegistrationError):
                block.register_function(consume, ["z"])
            with self.assertRaises(RegistrationError):
                block.register_expression("z = 3")
            with self.assertRaises(RegistrationError):
                block.register_args("args", ["z"])
            with self.assertRaises(RegistrationError):
                block.register_kwargs("kwargs", {"z": "z"})
            with self.assertRaises(RegistrationError):
                block.remove_function("produce")

    def test_atom_allows_config_and_value_mutations(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline, atom = self._locked_atom(temp_dir)
            atom.set_config({"extra": 1})
            atom.set_constant_value("const", 3)
            self.assertEqual(atom.get_config_value("extra"), 1)
            self.assertEqual(atom.get_constant_value("const"), 3)

    def test_atom_lock_survives_save_and_load(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            pipeline.create_atom_child_pipeline(
                "atom", 1, produce, output_variable_names=["x"]
            )
            pipeline.run_all()
            pipeline.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(tmp / "bundle", forced_deleting=True)
            loaded_atom = loaded.get_child_pipeline("atom")

            self.assertTrue(loaded_atom._is_atom)
            with self.assertRaises(RegistrationError):
                loaded_atom.add_block("extra", 3)
            with self.assertRaises(RegistrationError):
                loaded_atom.get_block("atom_block").register_function(consume, ["z"])

    def test_identical_atom_reregistration_after_load_is_noop(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            pipeline.create_atom_child_pipeline(
                "atom", 1, produce, output_variable_names=["x"]
            )
            pipeline.add_block("c", 2).register_function(consume, ["y"])
            pipeline.run_all()
            pipeline.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(tmp / "bundle", forced_deleting=True)
            self.assertEqual(loaded.get_value("x"), 1)
            self.assertEqual(loaded.get_value("y"), 2)

            loaded.create_atom_child_pipeline(
                "atom", 1, produce, output_variable_names=["x"]
            )

            self.assertEqual(loaded.get_value("x"), 1)
            self.assertEqual(loaded.get_value("y"), 2)

    def test_non_atom_pipeline_remains_mutable(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("p", {}, tmp / "p")
            pipeline.add_block("b", 1).register_function(produce, ["x"])
            pipeline.run_all()
            self.assertFalse(pipeline._is_atom)
            pipeline.add_block("c", 2).register_function(consume, ["y"])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("y"), 2)


if __name__ == "__main__":
    unittest.main()
