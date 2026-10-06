from __future__ import annotations

import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from mlpipelineholder import ExecutionBlock, PersistenceError, PipelineHolder, RegistrationError


def produce_seed() -> int:
    return 3


def double_seed(seed: int) -> int:
    return seed * 2


class ExecutionPriorityTests(unittest.TestCase):
    def test_none_block_priority_is_rejected_before_replacing_outputs(self) -> None:
        invalid_priority: Any = None
        with TemporaryDirectory() as directory:
            pipeline = PipelineHolder("root", {}, Path(directory))
            block = pipeline.add_block("prepare", 0)
            block.register_function(produce_seed, ["seed"])
            pipeline.run_all()

            for register in (pipeline.add_block, pipeline._add_block_strict):
                with self.subTest(register=register.__name__):
                    with self.assertRaisesRegex(RegistrationError, "priority"):
                        register("prepare", invalid_priority)
                    self.assertIs(pipeline.get_block("prepare"), block)
                    self.assertEqual(pipeline.get_value("seed"), 3)

            with self.assertRaisesRegex(RegistrationError, "priority"):
                ExecutionBlock(pipeline, "invalid", invalid_priority)

            # The review's next-registration failure must no longer be possible.
            second = pipeline.add_block("second", 1)
            second.register_function(double_seed, ["doubled"], param_mapping={"seed": "seed"})
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("doubled"), 6)

    def test_none_child_priority_preserves_both_parents_and_child_files(self) -> None:
        invalid_priority: Any = None
        with TemporaryDirectory() as directory:
            base = Path(directory)
            parent = PipelineHolder("parent", {}, base / "parent")
            block = parent.add_block("existing", 0)
            block.register_function(produce_seed, ["seed"])
            parent.run_all()
            old_parent = PipelineHolder("old-parent", {}, base / "old-parent")
            child = PipelineHolder("child", {}, base / "child")
            self.assertIsNone(child.execution_priority)

            for attached in (False, True):
                with self.subTest(attached=attached):
                    if attached:
                        old_parent.add_child_pipeline(child, 1)
                    original_root = child.project_root
                    marker = original_root / "marker.txt"
                    marker.write_text("keep", encoding="utf-8")
                    with self.assertRaisesRegex(RegistrationError, "priority"):
                        parent.add_child_pipeline(
                            child, invalid_priority, registration_name="existing", forced=True
                        )
                    self.assertEqual(child.registration_name, "child")
                    self.assertEqual(child.project_root, original_root)
                    self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
                    self.assertIs(child.parent_pipeline, old_parent if attached else None)
                    if attached:
                        self.assertIs(old_parent.get_child_pipeline("child"), child)
                    self.assertIs(parent.get_block("existing"), block)
                    self.assertEqual(parent.get_value("seed"), 3)
                    self.assertFalse((parent.project_root / "children" / "existing").exists())

    def test_none_atom_priorities_are_rejected_before_creating_directories(self) -> None:
        invalid_priority: Any = None
        with TemporaryDirectory() as directory:
            parent = PipelineHolder("root", {}, Path(directory))
            for priority, block_priority in ((invalid_priority, 10), (1, invalid_priority)):
                with self.subTest(priority=priority, block_priority=block_priority):
                    with self.assertRaisesRegex(RegistrationError, "priority"):
                        parent.create_atom_child_pipeline(
                            "atom", priority, produce_seed,
                            output_variable_names="seed", block_priority=block_priority,
                        )
                    self.assertEqual(parent.nodes, [])
                    self.assertFalse((parent.project_root / "children" / "atom").exists())

    def test_load_rejects_none_node_priority_but_accepts_none_root_priority(self) -> None:
        for kind in ("block", "child"):
            with self.subTest(kind=kind), TemporaryDirectory() as directory:
                base = Path(directory)
                pipeline = PipelineHolder("root", {}, base / "project")
                if kind == "block":
                    pipeline.add_block("prepare", 0).register_function(produce_seed, ["seed"])
                else:
                    child = PipelineHolder("child", {}, base / "child")
                    child.add_block("prepare", 0).register_function(produce_seed, ["seed"])
                    pipeline.add_child_pipeline(child, 1)
                pipeline.run_all()
                bundle = base / "bundle"
                pipeline.save_pipeline(bundle)
                loaded = PipelineHolder.load_pipeline(bundle, forced_deleting=True, trust_project=True)
                self.assertIsNone(loaded.execution_priority)
                self.assertEqual(loaded.get_value("seed"), 3)

                state_path = bundle / "pipeline_state.pkl"
                with state_path.open("rb") as handle:
                    payload = pickle.load(handle)
                payload["nodes"][0]["execution_priority"] = None
                with state_path.open("wb") as handle:
                    pickle.dump(payload, handle)
                with self.assertRaises(PersistenceError):
                    PipelineHolder.load_pipeline(bundle, forced_deleting=True, trust_project=True)

    def test_fractional_priority_order_is_not_replaced_by_name_order(self) -> None:
        with TemporaryDirectory() as directory:
            pipeline = PipelineHolder("root", {}, Path(directory))
            pipeline.add_block("a_later", 5.9)
            pipeline.add_block("z_earlier", 5.1)
            self.assertEqual(
                pipeline.get_priority_group(5),
                (["z_earlier", "a_later"], "z_earlier"),
            )
