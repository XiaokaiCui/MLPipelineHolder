from __future__ import annotations

from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import unittest
from unittest.mock import patch

from src.mlpipelineholder.artifact_store import ArtifactStore
from src.mlpipelineholder.exceptions import PersistenceError
from src.mlpipelineholder.models import ArtifactRecord


class ArtifactRecoveryBaselineTests(unittest.TestCase):
    def test_artifact_store_saves_loads_and_deletes_file_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            store = ArtifactStore(project_root)

            artifact = store.save(
                variable_name="value_blob",
                value={"value": 3},
                block_name="block",
                function_name="writer",
                run_id="run-1",
            )

            self.assertEqual(store.load(artifact), {"value": 3})
            self.assertTrue(Path(artifact.file_path).is_file())

            store.delete(artifact)

            self.assertFalse(Path(artifact.file_path).exists())

    def test_artifact_store_deletes_directory_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            artifact_dir = project_root / "artifacts" / "block" / "dataset.parquet"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "part-000.parquet").write_text("data", encoding="utf-8")
            artifact = ArtifactRecord(
                variable_name="dataset",
                serializer="parquet",
                file_path=str(artifact_dir),
                produced_by_block="block",
                produced_by_function="writer",
                run_id="run-1",
            )

            ArtifactStore(project_root).delete(artifact)

            self.assertFalse(artifact_dir.exists())


class ArtifactRecoveryTransactionTests(unittest.TestCase):
    def _build_directory_artifact(self, project_root: Path) -> ArtifactRecord:
        pd = import_module("pandas")

        artifact_dir = project_root / "artifacts" / "block" / "dataset.parquet"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"value": [1, 2, 3]}).to_parquet(artifact_dir / "part-000.parquet")
        return ArtifactRecord(
            variable_name="dataset",
            serializer="parquet",
            file_path=str(artifact_dir),
            produced_by_block="block",
            produced_by_function="writer",
            run_id="run-1",
        )

    def _build_torch_artifact(self, project_root: Path):
        from src.mlpipelineholder.models import TorchStateArtifactRecord

        torch = import_module("torch")

        artifact_path = project_root / "artifacts" / "block" / "optimizer.pt"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": {0: {"step": torch.tensor(1)}}}, artifact_path)
        return TorchStateArtifactRecord(
            variable_name="optimizer_obj",
            file_path=str(artifact_path),
            object_kind="torch_optimizer_state",
            metadata={"linked_model_variable": "model_obj"},
        )

    def _make_backup(self, saved_root: Path, backup_root: Path) -> None:
        shutil.copytree(saved_root, backup_root)

    def _artifact_recovery_module(self):
        return import_module("src.mlpipelineholder.artifact_recovery")

    def test_clone_value_copies_file_directory_and_torch_records(self) -> None:
        artifact_recovery = self._artifact_recovery_module()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            saved_root = temp_path / "saved-project"
            backup_root = temp_path / "backup"
            live_root = temp_path / "live-project"
            store = ArtifactStore(saved_root)
            file_record = store.save("value_blob", {"value": 7}, "block", "writer", "run-1")
            directory_record = self._build_directory_artifact(saved_root)
            torch_record = self._build_torch_artifact(saved_root)
            self._make_backup(saved_root, backup_root)

            transaction = artifact_recovery._ArtifactRecoveryTransaction(saved_root, backup_root, live_root)
            cloned = transaction.clone_value(
                {
                    "file": file_record,
                    "file_alias": file_record,
                    "items": [file_record, directory_record, torch_record],
                }
            )
            transaction.commit()
            shutil.rmtree(backup_root)

            self.assertIs(cloned["file"], cloned["file_alias"])
            self.assertIs(cloned["file"], cloned["items"][0])
            self.assertEqual(ArtifactStore(live_root).load(cloned["file"]), {"value": 7})
            self.assertEqual(list(ArtifactStore(live_root).load(cloned["items"][1])["value"]), [1, 2, 3])
            self.assertTrue(Path(cloned["items"][2].file_path).exists())
            self.assertEqual(cloned["items"][2].metadata["linked_model_variable"], "model_obj")
            self.assertTrue(cloned["file"].file_path.startswith(str(live_root / "artifacts")))
            self.assertFalse(cloned["file"].file_path.startswith(str(backup_root)))

    def test_clone_value_rejects_path_outside_saved_project_root(self) -> None:
        artifact_recovery = self._artifact_recovery_module()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            saved_root = temp_path / "saved-project"
            backup_root = temp_path / "backup"
            live_root = temp_path / "live-project"
            saved_root.mkdir(parents=True, exist_ok=True)
            backup_root.mkdir(parents=True, exist_ok=True)
            record = ArtifactRecord(
                variable_name="value_blob",
                serializer="json",
                file_path=str(temp_path / "elsewhere" / "value.json"),
                produced_by_block="block",
                produced_by_function="writer",
                run_id="run-1",
            )

            transaction = artifact_recovery._ArtifactRecoveryTransaction(saved_root, backup_root, live_root)

            with self.assertRaises(PersistenceError):
                transaction.clone_value(record)

    def test_clone_value_rejects_symlink_source(self) -> None:
        artifact_recovery = self._artifact_recovery_module()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            saved_root = temp_path / "saved-project"
            backup_root = temp_path / "backup"
            live_root = temp_path / "live-project"
            target = saved_root / "artifacts" / "block" / "value.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('{"value": 7}', encoding="utf-8")
            record = ArtifactRecord(
                variable_name="value_blob",
                serializer="json",
                file_path=str(target),
                produced_by_block="block",
                produced_by_function="writer",
                run_id="run-1",
            )
            self._make_backup(saved_root, backup_root)
            backup_target = backup_root / "artifacts" / "block" / "value.json"
            backup_target.unlink()
            backup_target.symlink_to(temp_path / "outside.json")

            transaction = artifact_recovery._ArtifactRecoveryTransaction(saved_root, backup_root, live_root)

            with self.assertRaises(PersistenceError):
                transaction.clone_value(record)

    def test_clone_value_rolls_back_staging_when_copy_fails(self) -> None:
        artifact_recovery = self._artifact_recovery_module()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            saved_root = temp_path / "saved-project"
            backup_root = temp_path / "backup"
            live_root = temp_path / "live-project"
            live_old = live_root / "artifacts" / "keep.txt"
            live_old.parent.mkdir(parents=True, exist_ok=True)
            live_old.write_text("keep", encoding="utf-8")
            file_record = ArtifactStore(saved_root).save(
                "value_blob",
                {"value": 7},
                "block",
                "writer",
                "run-1",
            )
            self._make_backup(saved_root, backup_root)
            transaction = artifact_recovery._ArtifactRecoveryTransaction(saved_root, backup_root, live_root)

            with patch("src.mlpipelineholder.artifact_recovery.shutil.copy2", side_effect=OSError("copy failed")):
                with self.assertRaises(PersistenceError):
                    transaction.clone_value(file_record)

            self.assertTrue(live_old.exists())
            self.assertEqual(sorted(path.name for path in (live_root / "artifacts").iterdir()), ["keep.txt"])

    def test_commit_rolls_back_finalized_paths_when_rename_fails(self) -> None:
        artifact_recovery = self._artifact_recovery_module()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            saved_root = temp_path / "saved-project"
            backup_root = temp_path / "backup"
            live_root = temp_path / "live-project"
            live_old = live_root / "artifacts" / "keep.txt"
            live_old.parent.mkdir(parents=True, exist_ok=True)
            live_old.write_text("keep", encoding="utf-8")
            store = ArtifactStore(saved_root)
            first_record = store.save("first", {"value": 1}, "block", "writer", "run-1")
            second_record = store.save("second", {"value": 2}, "block", "writer", "run-1")
            self._make_backup(saved_root, backup_root)
            transaction = artifact_recovery._ArtifactRecoveryTransaction(saved_root, backup_root, live_root)
            cloned = transaction.clone_value([first_record, second_record])
            rename_count = {"value": 0}

            def fail_on_second_rename(source: Path, target: Path) -> None:
                rename_count["value"] += 1
                if rename_count["value"] == 2:
                    raise OSError("rename failed")
                source.rename(target)

            with patch("src.mlpipelineholder.artifact_recovery._rename_path", side_effect=fail_on_second_rename):
                with self.assertRaises(PersistenceError):
                    transaction.commit()

            self.assertTrue(live_old.exists())
            self.assertFalse(Path(cloned[0].file_path).exists())
            self.assertFalse(Path(cloned[1].file_path).exists())
            self.assertEqual(sorted(path.name for path in (live_root / "artifacts").iterdir()), ["keep.txt"])

    def test_delete_unreferenced_artifact_only_removes_globally_unreferenced_paths(self) -> None:
        artifact_recovery = self._artifact_recovery_module()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            kept_path = temp_path / "artifacts" / "keep.json"
            dropped_path = temp_path / "artifacts" / "drop.json"
            kept_path.parent.mkdir(parents=True, exist_ok=True)
            kept_path.write_text('{"value": 1}', encoding="utf-8")
            dropped_path.write_text('{"value": 2}', encoding="utf-8")
            kept_record = ArtifactRecord(
                variable_name="keep",
                serializer="json",
                file_path=str(kept_path),
                produced_by_block="block",
                produced_by_function="writer",
                run_id="run-1",
            )
            dropped_record = ArtifactRecord(
                variable_name="drop",
                serializer="json",
                file_path=str(dropped_path),
                produced_by_block="block",
                produced_by_function="writer",
                run_id="run-1",
            )

            artifact_recovery._delete_unreferenced_artifact(kept_record, [{"still_live": kept_record}])
            artifact_recovery._delete_unreferenced_artifact(dropped_record, [{"still_live": kept_record}])

            self.assertTrue(kept_path.exists())
            self.assertFalse(dropped_path.exists())
