from __future__ import annotations

import __main__
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.mlpipelineholder import (
    PersistenceError,
    PipelineHandler,
    RegistrationError,
    ResolutionError,
)
from src.mlpipelineholder.backup_recovery import _VariableOwnershipInventory
from src.mlpipelineholder.models import ArtifactRecord


def current_callable(value: int) -> int:
    return value * 10


def produce_blob() -> dict[str, int]:
    return {"saved": 7}


@dataclass
class RecoveryConfig:
    factor: int
    label: str = "keep"


def bind_runtime_callable() -> object:
    exec(
        "def recovery_runtime_callable(value: int) -> int:\n    return value + 4\n",
        __main__.__dict__,
    )
    return getattr(__main__, "recovery_runtime_callable")


def recovery_produce() -> dict[str, int]:
    return {"v": 1}


def recovery_produce_later() -> dict[str, int]:
    return {"later": 2}


def recovery_produce_a() -> dict[str, str]:
    return {"from": "A"}


def recovery_produce_b() -> dict[str, str]:
    return {"from": "B"}


class BackupRecoveryTests(unittest.TestCase):
    def test_recover_variable_restores_root_manual_value_and_returns_none(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {"factor": 2})
            root.set_constant_value("threshold", 5)
            root.save_pipeline()
            root.set_constant_value("threshold", 99)

            result = root.recover_variable_from_backup(name="threshold")

            self.assertIsNone(result)
            self.assertEqual(root.get_constant_value("threshold"), 5)

    def test_recover_variable_on_child_requires_root(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            child = PipelineHandler("child", {}, Path(temp_dir) / "child")
            child.set_constant_value("shared", 2)
            root.add_child_pipeline(child, 1)
            root.save_pipeline()

            with self.assertRaisesRegex(RegistrationError, "root.*recover_variable"):
                child.recover_variable_from_backup(name="shared")

    def test_recover_variable_prompts_once_and_updates_independent_owners(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            root.set_constant_value("shared", "saved-root")
            child = PipelineHandler("child", {}, Path(temp_dir) / "child")
            child.set_constant_value("shared", "saved-child")
            root.add_child_pipeline(child, 1)
            root.save_pipeline()
            root.set_constant_value("shared", "current-root")
            child.set_constant_value("shared", "current-child")

            with patch("builtins.input", return_value=" y ") as mocked_input:
                result = root.recover_variable_from_backup(name="shared")

            self.assertIsNone(result)
            self.assertEqual(mocked_input.call_count, 1)
            prompt = mocked_input.call_args.args[0]
            self.assertIn("root", prompt)
            self.assertIn("root/child", prompt)
            self.assertEqual(root.get_constant_value("shared"), "saved-root")
            self.assertEqual(child.get_constant_value("shared"), "saved-root")

    def test_recover_variable_refusal_leaves_every_owner_unchanged(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            root.set_constant_value("shared", 1)
            child = PipelineHandler("child", {}, Path(temp_dir) / "child")
            child.set_constant_value("shared", 2)
            root.add_child_pipeline(child, 1)
            root.save_pipeline()
            root.set_constant_value("shared", 10)
            child.set_constant_value("shared", 20)

            with patch("builtins.input", return_value="no"):
                result = root.recover_variable_from_backup(name="shared")

            self.assertIsNone(result)
            self.assertEqual(root.get_constant_value("shared"), 10)
            self.assertEqual(child.get_constant_value("shared"), 20)

    def test_recover_variable_commit_failure_restores_all_owner_slots(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            root.set_constant_value("shared", "saved-root")
            child = PipelineHandler("child", {}, Path(temp_dir) / "child")
            child.set_constant_value("shared", "saved-child")
            root.add_child_pipeline(child, 1)
            root.save_pipeline()
            root.set_constant_value("shared", "current-root")
            child.set_constant_value("shared", "current-child")

            def fail_after_one_assignment(
                inventory: _VariableOwnershipInventory, value: object
            ) -> None:
                first_slot = inventory.owners[0].update_slots[0]
                first_slot.mapping[first_slot.key] = value
                raise OSError("injected commit failure")

            with patch("builtins.input", return_value="yes"), patch(
                "src.mlpipelineholder.backup_recovery_service._assign_inventory",
                side_effect=fail_after_one_assignment,
            ):
                with self.assertRaises(PersistenceError):
                    root.recover_variable_from_backup(name="shared")

            self.assertEqual(root.get_constant_value("shared"), "current-root")
            self.assertEqual(child.get_constant_value("shared"), "current-child")

    def test_recover_variable_resolves_runtime_callable_for_partial(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            previous = getattr(__main__, "recovery_runtime_callable", None)
            had_previous = hasattr(__main__, "recovery_runtime_callable")
            runtime_callable = bind_runtime_callable()
            try:
                root.set_constant_value("target_callable", runtime_callable)
                root.save_pipeline()
                root.set_constant_value("target_callable", current_callable)

                root.recover_variable_from_backup(name="target_callable")

                restored = root.get_constant_value("target_callable")
                self.assertTrue(callable(restored))
                self.assertEqual(partial(restored, 3)(), 7)
            finally:
                if had_previous:
                    setattr(__main__, "recovery_runtime_callable", previous)
                else:
                    delattr(__main__, "recovery_runtime_callable")

    def test_recover_variable_missing_runtime_callable_preserves_current_value(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            previous = getattr(__main__, "recovery_runtime_callable", None)
            had_previous = hasattr(__main__, "recovery_runtime_callable")
            runtime_callable = bind_runtime_callable()
            try:
                root.set_constant_value("target_callable", runtime_callable)
                root.save_pipeline()
                root.set_constant_value("target_callable", current_callable)
                delattr(__main__, "recovery_runtime_callable")

                with self.assertRaises(PersistenceError):
                    root.recover_variable_from_backup(name="target_callable")

                self.assertIs(root.get_constant_value("target_callable"), current_callable)
            finally:
                if had_previous:
                    setattr(__main__, "recovery_runtime_callable", previous)
                elif hasattr(__main__, "recovery_runtime_callable"):
                    delattr(__main__, "recovery_runtime_callable")

    def test_recover_variable_clones_disk_artifact_into_live_project(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, work, backup = self._saved_root(temp_dir, {})
            block = root.add_block("blob", 1)
            block.register_function(produce_blob, ["blob"], save_to_disk=["blob"])
            root.run_all()
            root.save_pipeline()
            root.update_value("blob", {"current": 9})

            root.recover_variable_from_backup(name="blob")
            recovered_record = root.para_value_dict["blob"]
            shutil.rmtree(backup)

            self.assertEqual(root.get_value("blob"), {"saved": 7})
            self.assertTrue(Path(recovered_record.file_path).is_relative_to(work))

    def test_recover_config_restores_receiver_local_field_and_inheritance(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {"root_only": 1})
            child = PipelineHandler("child", {"choice": "saved"}, Path(temp_dir) / "child")
            inheriting = PipelineHandler("inheriting", {}, Path(temp_dir) / "inheriting")
            overriding = PipelineHandler(
                "overriding", {"choice": "override"}, Path(temp_dir) / "overriding"
            )
            child.add_child_pipeline(inheriting, 1)
            child.add_child_pipeline(overriding, 2)
            root.add_child_pipeline(child, 1)
            root.save_pipeline()
            child.update_config({"choice": "current"})

            result = child.recover_config_from_backup(name="choice")

            self.assertIsNone(result)
            self.assertEqual(child.get_config_value("choice"), "saved")
            self.assertEqual(inheriting.get_config_value("choice"), "saved")
            self.assertEqual(overriding.get_config_value("choice"), "override")
            self.assertEqual(root.get_config_value("root_only"), 1)

    def test_recover_config_preserves_dataclass_type_and_unrelated_fields(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            root = PipelineHandler(
                "root",
                RecoveryConfig(factor=2),
                root_path / "work",
                pipeline_backup_directory=root_path / "backup",
            )
            root.save_pipeline()
            previous_config = root.config
            root.update_config({"factor": 9})

            root.recover_config_from_backup(name="factor")

            self.assertIsInstance(root.config, RecoveryConfig)
            self.assertIsNot(root.config, previous_config)
            self.assertEqual(root.get_config_value("factor"), 2)
            self.assertEqual(root.get_config_value("label"), "keep")

    def test_recover_config_rejects_inherited_only_field(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {"shared": "saved"})
            child = PipelineHandler("child", {}, Path(temp_dir) / "child")
            root.add_child_pipeline(child, 1)
            root.save_pipeline()

            with self.assertRaises(ResolutionError):
                child.recover_config_from_backup(name="shared")

    def test_recover_config_resolves_runtime_callable_for_partial(self) -> None:
        with TemporaryDirectory() as temp_dir:
            previous = getattr(__main__, "recovery_runtime_callable", None)
            had_previous = hasattr(__main__, "recovery_runtime_callable")
            runtime_callable = bind_runtime_callable()
            try:
                root, _, _ = self._saved_root(
                    temp_dir, {"target_callable": runtime_callable}
                )
                root.save_pipeline()
                root.update_config({"target_callable": current_callable})

                root.recover_config_from_backup(name="target_callable")

                restored = root.get_config_value("target_callable")
                self.assertTrue(callable(restored))
                self.assertEqual(partial(restored, 2)(), 6)
            finally:
                if had_previous:
                    setattr(__main__, "recovery_runtime_callable", previous)
                else:
                    delattr(__main__, "recovery_runtime_callable")

    def test_recovery_requires_complete_configured_backup_and_existing_name(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = PipelineHandler("root", {}, Path(temp_dir) / "work")
            root.set_constant_value("present", 1)
            with self.assertRaises(PersistenceError):
                root.recover_variable_from_backup(name="present")

        with TemporaryDirectory() as temp_dir:
            root, _, backup = self._saved_root(temp_dir, {})
            root.set_constant_value("present", 1)
            root.save_pipeline()
            (backup / "config.pkl").unlink()
            with self.assertRaises(PersistenceError):
                root.recover_variable_from_backup(name="present")

        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            root.set_constant_value("only_current", 1)
            root.save_pipeline()
            root.set_constant_value("missing_from_backup", 2)
            with self.assertRaises(ResolutionError):
                root.recover_variable_from_backup(name="missing_from_backup")

    def test_update_value_on_child_output_syncs_parent_and_allows_unrelated_recovery(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            root.set_constant_value("unrelated", "hello")
            child = PipelineHandler("child", {}, Path(temp_dir) / "child")
            block = child.add_block("producer", 1)
            block.register_function(produce_blob, ["blob"], save_to_disk=["blob"])
            root.add_child_pipeline(child, 1)
            root.run_all()
            root.save_pipeline()

            child.update_value("blob", {"current": 9})

            # Disk-backed outputs stay disk-backed when updated: the new value
            # is saved as an ArtifactRecord and propagates through the tree.
            self.assertIsInstance(root.para_value_dict["blob"], ArtifactRecord)
            self.assertIsInstance(
                root.producer_outputs["child"]["blob"], ArtifactRecord
            )
            self.assertIn("blob", root.artifact_registry)

            root.save_pipeline()
            root.recover_variable_from_backup(name="unrelated")
            self.assertEqual(root.get_constant_value("unrelated"), "hello")

    def test_recover_variable_ignores_missing_artifact_of_other_value(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, work, backup = self._saved_root(temp_dir, {})
            root.set_constant_value("unrelated", "hello")
            child = PipelineHandler("child", {}, Path(temp_dir) / "child")
            block = child.add_block("producer", 1)
            block.register_function(produce_blob, ["blob"], save_to_disk=["blob"])
            root.add_child_pipeline(child, 1)
            root.run_all()
            root.save_pipeline()

            record = child.para_value_dict["blob"]
            relative_path = Path(record.file_path).relative_to(work)
            (backup / relative_path).unlink()

            root.recover_variable_from_backup(name="unrelated")
            self.assertEqual(root.get_constant_value("unrelated"), "hello")

    def test_recover_variable_rejects_missing_artifact_of_selected_value(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, work, backup = self._saved_root(temp_dir, {})
            block = root.add_block("producer", 1)
            block.register_function(produce_blob, ["blob"], save_to_disk=["blob"])
            root.run_all()
            root.save_pipeline()

            record = root.para_value_dict["blob"]
            relative_path = Path(record.file_path).relative_to(work)
            (backup / relative_path).unlink()

            with self.assertRaises(PersistenceError):
                root.recover_variable_from_backup(name="blob")

    def test_recover_declared_but_unproduced_value(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            child = PipelineHandler("child", {}, Path(temp_dir) / "child")
            block = child.add_block("producer", 1)
            block.register_function(recovery_produce, ["saved_blob"])
            root.add_child_pipeline(child, 1)
            root.run_all()
            root.save_pipeline()
            root._invalidate_from_priority(0)
            self.assertNotIn("saved_blob", root.para_value_dict)

            root.recover_variable_from_backup(name="saved_blob")

            self.assertEqual(child.get_value("saved_blob"), {"v": 1})
            self.assertEqual(root.get_value("saved_blob"), {"v": 1})

    def test_recover_declared_with_pipeline_name_targets_scope(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            child_a = PipelineHandler("child_a", {}, Path(temp_dir) / "a")
            child_b = PipelineHandler("child_b", {}, Path(temp_dir) / "b")
            ba = child_a.add_block("ba", 1)
            ba.register_function(recovery_produce_a, ["shared_value"])
            bb = child_b.add_block("bb", 1)
            bb.register_function(recovery_produce_b, ["shared_value"])
            root.add_child_pipeline(child_a, 10.0)
            root.add_child_pipeline(child_b, 20.0)
            root.run_all()
            root.save_pipeline()
            root._invalidate_from_priority(0)

            root.recover_variable_from_backup(
                pipeline_name="child_b", name="shared_value"
            )

            self.assertEqual(child_b.get_value("shared_value"), {"from": "B"})

    def test_recover_declared_atom_output_with_pipeline_name_targets_atom(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            score = PipelineHandler(
                "score_analysis_pipeline",
                {},
                Path(temp_dir) / "score",
            )
            score.create_atom_child_pipeline(
                "score_optimise_parameters",
                41.0,
                recovery_produce,
                output_variable_names="score_best_params",
            )
            root.add_child_pipeline(score, 1)
            root.run_all()
            root.save_pipeline()
            root._invalidate_from_priority(0)

            root.recover_variable_from_backup(
                pipeline_name="score_analysis_pipeline",
                name="score_best_params",
            )

            atom = score.get_child_pipeline("score_optimise_parameters")
            self.assertEqual(atom.get_value("score_best_params"), {"v": 1})
            self.assertEqual(score.get_value("score_best_params"), {"v": 1})
            self.assertEqual(root.get_value("score_best_params"), {"v": 1})

    def test_recover_declared_default_injects_first_declaring_pipeline(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            child_a = PipelineHandler("child_a", {}, Path(temp_dir) / "a")
            child_b = PipelineHandler("child_b", {}, Path(temp_dir) / "b")
            ba = child_a.add_block("ba", 1)
            ba.register_function(recovery_produce_a, ["shared_value"])
            bb = child_b.add_block("bb", 1)
            bb.register_function(recovery_produce_b, ["shared_value"])
            root.add_child_pipeline(child_a, 10.0)
            root.add_child_pipeline(child_b, 20.0)
            root.run_all()
            root.save_pipeline()
            root._invalidate_from_priority(0)

            root.recover_variable_from_backup(name="shared_value")

            self.assertEqual(child_a.get_value("shared_value"), {"from": "A"})

    def test_recover_declared_unknown_pipeline_name_raises(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            child = PipelineHandler("child", {}, Path(temp_dir) / "child")
            block = child.add_block("producer", 1)
            block.register_function(recovery_produce, ["saved_blob"])
            root.add_child_pipeline(child, 1)
            root.run_all()
            root.save_pipeline()
            root._invalidate_from_priority(0)

            with self.assertRaisesRegex(ResolutionError, "Unknown pipeline"):
                root.recover_variable_from_backup(
                    pipeline_name="nope", name="saved_blob"
                )

    def test_recover_unknown_value_still_raises(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            child = PipelineHandler("child", {}, Path(temp_dir) / "child")
            block = child.add_block("producer", 1)
            block.register_function(recovery_produce, ["saved_blob"])
            root.add_child_pipeline(child, 1)
            root.run_all()
            root.save_pipeline()
            root._invalidate_from_priority(0)

            with self.assertRaisesRegex(ResolutionError, "Unknown pipeline value"):
                root.recover_variable_from_backup(name="no_such_value")

    def test_recover_scoped_pipeline_name_ignores_descendant_owner(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            parent = PipelineHandler("parent", {}, Path(temp_dir) / "parent")
            leaf = PipelineHandler("leaf", {}, Path(temp_dir) / "leaf")
            parent.set_constant_value("shared", "saved-parent")
            leaf.set_constant_value("shared", "saved-leaf")
            parent.add_child_pipeline(leaf, 1)
            root.add_child_pipeline(parent, 1)
            root.save_pipeline()
            parent.set_constant_value("shared", "current-parent")
            leaf.set_constant_value("shared", "current-leaf")

            with patch(
                "builtins.input",
                side_effect=AssertionError("scoped recovery must not prompt"),
            ):
                root.recover_variable_from_backup(
                    pipeline_name="parent", name="shared"
                )

            self.assertEqual(parent.get_constant_value("shared"), "saved-parent")
            self.assertEqual(leaf.get_constant_value("shared"), "current-leaf")

    def test_recover_scoped_raises_when_only_descendant_declares(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            parent = PipelineHandler("parent", {}, Path(temp_dir) / "parent")
            leaf = PipelineHandler("leaf", {}, Path(temp_dir) / "leaf")
            block = leaf.add_block("producer", 1)
            block.register_function(recovery_produce, ["leaf_only"])
            parent.add_child_pipeline(leaf, 1)
            root.add_child_pipeline(parent, 1)
            root.run_all()
            root.save_pipeline()
            root._invalidate_from_priority(0)

            with self.assertRaisesRegex(ResolutionError, "Unknown pipeline value"):
                root.recover_variable_from_backup(
                    pipeline_name="parent", name="leaf_only"
                )

    def test_recover_declared_preserves_later_parent_artifact(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            child = PipelineHandler("child", {}, Path(temp_dir) / "child")
            child.add_block("producer", 1).register_function(
                recovery_produce, ["saved_blob"]
            )
            root.add_child_pipeline(child, 1)
            root.add_block("later", 2).register_function(
                recovery_produce_later,
                ["later_blob"],
                save_to_disk=["later_blob"],
            )
            root.run_all()
            root.save_pipeline()
            child._invalidate_all_outputs()
            root.producer_outputs.pop("child", None)
            root._rebuild_visible_state({})
            later_record = root.producer_outputs["later"]["later_blob"]
            self.assertIsInstance(later_record, ArtifactRecord)
            later_path = Path(later_record.file_path)

            root.recover_variable_from_backup(name="saved_blob")

            self.assertIs(root.producer_outputs["later"]["later_blob"], later_record)
            self.assertTrue(later_path.exists())
            self.assertEqual(root.get_value("later_blob"), {"later": 2})

    def test_recover_declared_commit_failure_preserves_later_parent_artifact(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            child = PipelineHandler("child", {}, Path(temp_dir) / "child")
            child.add_block("producer", 1).register_function(
                recovery_produce, ["saved_blob"]
            )
            root.add_child_pipeline(child, 1)
            root.add_block("later", 2).register_function(
                recovery_produce_later,
                ["later_blob"],
                save_to_disk=["later_blob"],
            )
            root.run_all()
            root.save_pipeline()
            child._invalidate_all_outputs()
            root.producer_outputs.pop("child", None)
            root._rebuild_visible_state({})
            later_record = root.producer_outputs["later"]["later_blob"]
            self.assertIsInstance(later_record, ArtifactRecord)
            later_path = Path(later_record.file_path)

            with patch(
                "src.mlpipelineholder.backup_recovery_service._ArtifactRecoveryTransaction.commit",
                side_effect=OSError("injected commit failure"),
            ):
                with self.assertRaises(PersistenceError):
                    root.recover_variable_from_backup(name="saved_blob")

            self.assertIs(root.producer_outputs["later"]["later_blob"], later_record)
            self.assertTrue(later_path.exists())
            self.assertEqual(root.get_value("later_blob"), {"later": 2})

    def test_recover_declared_synchronizes_all_ancestor_mirrors(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            parent = PipelineHandler("parent", {}, Path(temp_dir) / "parent")
            leaf = PipelineHandler("leaf", {}, Path(temp_dir) / "leaf")
            leaf.add_block("producer", 1).register_function(
                recovery_produce, ["saved_blob"]
            )
            parent.add_child_pipeline(leaf, 1)
            root.add_child_pipeline(parent, 1)
            root.run_all()
            root.save_pipeline()
            leaf._invalidate_all_outputs()
            parent.producer_outputs.pop("leaf", None)
            parent._rebuild_visible_state(parent._incoming_parent_outputs())
            root.producer_outputs.pop("parent", None)
            root._rebuild_visible_state({})

            root.recover_variable_from_backup(
                pipeline_name="leaf", name="saved_blob"
            )

            value = leaf.producer_outputs["producer"]["saved_blob"]
            self.assertIs(leaf.para_value_dict["saved_blob"], value)
            self.assertIs(parent.producer_outputs["leaf"]["saved_blob"], value)
            self.assertIs(parent.para_value_dict["saved_blob"], value)
            self.assertIs(root.producer_outputs["parent"]["saved_blob"], value)
            self.assertIs(root.para_value_dict["saved_blob"], value)

    def test_recover_declared_commit_failure_restores_absence(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = self._saved_root(temp_dir, {})
            child = PipelineHandler("child", {}, Path(temp_dir) / "child")
            block = child.add_block("producer", 1)
            block.register_function(recovery_produce, ["saved_blob"])
            root.add_child_pipeline(child, 1)
            root.run_all()
            root.save_pipeline()
            root._invalidate_from_priority(0)
            self.assertNotIn("saved_blob", root.para_value_dict)

            with patch(
                "src.mlpipelineholder.backup_recovery_service._ArtifactRecoveryTransaction.commit",
                side_effect=OSError("injected commit failure"),
            ):
                with self.assertRaises(PersistenceError):
                    root.recover_variable_from_backup(name="saved_blob")

            self.assertNotIn("saved_blob", child.para_value_dict)
            self.assertNotIn("saved_blob", root.para_value_dict)
            self.assertNotIn("saved_blob", child.producer_outputs.get("producer", {}))
            self.assertNotIn("child", root.producer_outputs)
            with self.assertRaises(ResolutionError):
                root.get_value("saved_blob")

    @staticmethod
    def _saved_root(
        temp_dir: str, config: dict[str, object]
    ) -> tuple[PipelineHandler, Path, Path]:
        root_path = Path(temp_dir)
        work = root_path / "work"
        backup = root_path / "backup"
        root = PipelineHandler(
            "root",
            config,
            work,
            pipeline_backup_directory=backup,
        )
        return root, work, backup


if __name__ == "__main__":
    unittest.main()
