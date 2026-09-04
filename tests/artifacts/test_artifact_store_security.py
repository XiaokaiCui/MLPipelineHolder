from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.mlpipelineholder.artifact_store import ArtifactStore
from src.mlpipelineholder.exceptions import PersistenceError
from src.mlpipelineholder.models import ArtifactRecord


def make_record(file_path: Path) -> ArtifactRecord:
    return ArtifactRecord(
        variable_name="value",
        serializer="json",
        file_path=str(file_path),
        produced_by_block="block",
        produced_by_function="function",
        run_id="run",
    )


class ArtifactStoreContainmentTests(unittest.TestCase):
    def test_transfer_refuses_source_outside_artifact_root(self) -> None:
        # Given: a store rooted in one directory and a real file beside it.
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = ArtifactStore(root / "store")
            outside_file = root / "outside.json"
            outside_file.write_text("payload", encoding="utf-8")
            record = make_record(outside_file)

            # When / Then: transfer must refuse before moving anything.
            with self.assertRaises(PersistenceError):
                store.transfer(record, "promoted_block")
            self.assertTrue(outside_file.is_file())
            self.assertEqual(outside_file.read_text(encoding="utf-8"), "payload")

    def test_transfer_refuses_symlink_resolving_outside_artifact_root(self) -> None:
        # Given: a symlink inside the artifact tree pointing at an outside file.
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = ArtifactStore(root / "store")
            outside_file = root / "outside.json"
            outside_file.write_text("payload", encoding="utf-8")
            link = store.artifact_root / "sneaky.json"
            link.symlink_to(outside_file)
            record = make_record(link)

            # When / Then: the resolved target is outside, so transfer refuses.
            with self.assertRaises(PersistenceError):
                store.transfer(record, "promoted_block")
            self.assertTrue(outside_file.is_file())
            self.assertTrue(link.is_symlink())

    def test_transfer_moves_managed_artifact_within_root(self) -> None:
        # Given: a real artifact file inside the store.
        with TemporaryDirectory() as temp_dir:
            store = ArtifactStore(Path(temp_dir) / "store")
            source = store.artifact_root / "managed.json"
            source.write_text("payload", encoding="utf-8")
            record = make_record(source)

            # When: the store transfers the managed artifact.
            moved = store.transfer(record, "promoted_block")

            # Then: the file moved to a managed location and the source is gone.
            self.assertFalse(source.exists())
            moved_path = Path(moved.file_path)
            self.assertTrue(moved_path.is_file())
            self.assertIn(store.artifact_root.resolve(), moved_path.resolve().parents)

    def test_delete_refuses_source_outside_artifact_root(self) -> None:
        # Given: a record whose file lives outside the store.
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = ArtifactStore(root / "store")
            outside_file = root / "outside.json"
            outside_file.write_text("payload", encoding="utf-8")
            record = make_record(outside_file)

            # When / Then: delete must refuse and leave the file untouched.
            with self.assertRaises(PersistenceError):
                store.delete(record)
            self.assertTrue(outside_file.is_file())


if __name__ == "__main__":
    unittest.main()
