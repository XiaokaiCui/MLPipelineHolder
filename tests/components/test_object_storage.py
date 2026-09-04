from __future__ import annotations

import pickle
import tempfile
import unittest
import warnings
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from sys import modules
from unittest.mock import patch
from uuid import UUID

import pandas as pd

from mlpipelineholder import PipelineHandler
from mlpipelineholder.exceptions import PersistenceError, RegistrationError, ResolutionError
from mlpipelineholder.models import ArtifactRecord
from mlpipelineholder.object_storage import record_from_payload


class _PickleStoredValue:
    def __init__(self, value: str) -> None:
        self.value: str = value


class _UnpicklableStoredValue:
    def __init__(self) -> None:
        self.callback: object = lambda: None


class TestObjectStorage(unittest.TestCase):
    def test_save_list_and_get_keep_objects_out_of_pipeline_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "storage-root",
                configuration={"shared_name": "config-value"},
                local_folder_path=Path(tmp) / "project",
            )

            hash_id = pipeline.save_to_storage(
                "shared_name",
                {"source": "storage"},
                object_description="independent object",
            )

            self.assertEqual(pipeline.get_config("shared_name"), "config-value")
            self.assertNotIn(hash_id, pipeline.para_value_dict)
            self.assertEqual(
                pipeline.get_from_storage(hash_id=hash_id),
                {"source": "storage"},
            )
            frame = pipeline.list_stored_objects()
            self.assertIsInstance(frame, pd.DataFrame)
            self.assertEqual(
                frame.columns.tolist(),
                [
                    "hash_id",
                    "object_name",
                    "object_type",
                    "object_description",
                    "created_at_utc",
                    "last_modified_at_utc",
                ],
            )
            self.assertEqual(frame.loc[0, "hash_id"], hash_id)
            self.assertEqual(frame.loc[0, "object_name"], "shared_name")
            self.assertEqual(frame.loc[0, "object_type"], "dict")
            self.assertEqual(frame.loc[0, "object_description"], "independent object")
            self.assertIsNotNone(frame.loc[0, "created_at_utc"].tzinfo)
            self.assertIsNotNone(frame.loc[0, "last_modified_at_utc"].tzinfo)

    def test_hash_lookup_takes_precedence_and_duplicate_names_are_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", local_folder_path=Path(tmp) / "project")
            first_id = pipeline.save_to_storage("duplicate", 1)
            pipeline.save_to_storage("duplicate", 2)

            self.assertEqual(
                pipeline.get_from_storage(hash_id=first_id, object_name="wrong-name"),
                1,
            )
            with self.assertRaisesRegex(ResolutionError, "multiple stored objects"):
                pipeline.get_from_storage(object_name="duplicate")
            with self.assertRaisesRegex(ResolutionError, "multiple stored objects"):
                pipeline.remove_from_storage(object_name="duplicate")
            with self.assertRaisesRegex(ResolutionError, "hash_id or object_name"):
                pipeline.get_from_storage()
            with self.assertRaisesRegex(ResolutionError, "No stored object"):
                pipeline.get_from_storage(hash_id="missing")

    def test_update_persist_and_remove_storage_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            pipeline = PipelineHandler("root", local_folder_path=project_root)
            hash_id = pipeline.save_to_storage("before", {"version": 1})

            pipeline.update_storage(
                hash_id,
                {"version": 2},
                object_name="after",
                object_description="updated",
                to_disk=True,
            )

            self.assertEqual(
                pipeline.get_from_storage(object_name="after"),
                {"version": 2},
            )
            frame = pipeline.list_stored_objects()
            self.assertEqual(frame.loc[0, "object_description"], "updated")
            self.assertGreaterEqual(
                frame.loc[0, "last_modified_at_utc"],
                frame.loc[0, "created_at_utc"],
            )
            storage_artifacts = list((project_root / "artifacts" / "storage").glob("*"))
            self.assertEqual(len(storage_artifacts), 1)

            pipeline.remove_from_storage(hash_id=hash_id)

            self.assertTrue(pipeline.list_stored_objects().empty)
            self.assertFalse(storage_artifacts[0].exists())

    def test_repeated_immediate_updates_retire_previous_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            pipeline = PipelineHandler("root", local_folder_path=project_root)
            hash_id = pipeline.save_to_storage("versioned", {"version": 1})
            pipeline.update_storage(hash_id, {"version": 2}, to_disk=True)

            pipeline.update_storage(hash_id, {"version": 3}, to_disk=True)

            storage_artifacts = list((project_root / "artifacts" / "storage").glob("*"))
            self.assertEqual(len(storage_artifacts), 1)
            self.assertEqual(
                pipeline.get_from_storage(hash_id=hash_id),
                {"version": 3},
            )

    def test_pipeline_save_load_persists_known_and_pickle_fallback_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            pipeline = PipelineHandler("root", local_folder_path=project_root)
            json_id = pipeline.save_to_storage("mapping", {"answer": 42})
            with self.assertWarnsRegex(UserWarning, "falling back to pickle"):
                pickle_id = pipeline.save_to_storage(
                    "custom",
                    _PickleStoredValue("persisted"),
                )

            _ = pipeline.save_pipeline()
            loaded = PipelineHandler.load_pipeline(project_root)

            self.assertEqual(loaded.get_from_storage(hash_id=json_id), {"answer": 42})
            restored = loaded.get_from_storage(hash_id=pickle_id)
            self.assertIsInstance(restored, _PickleStoredValue)
            self.assertEqual(restored.value, "persisted")

    def test_temporary_root_relocation_preserves_storage_and_cleans_old_generations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "relocated"
            pipeline = PipelineHandler("root")
            hash_id = pipeline.save_to_storage("relocated-object", {"version": 1})

            _ = pipeline.save_pipeline(project_root)
            first_artifact = next((project_root / "artifacts" / "storage").glob("*"))
            pipeline.update_storage(hash_id, {"version": 2}, to_disk=True)
            _ = pipeline.save_pipeline()

            storage_artifacts = list((project_root / "artifacts" / "storage").glob("*"))
            self.assertEqual(len(storage_artifacts), 1)
            self.assertFalse(first_artifact.exists())
            loaded = PipelineHandler.load_pipeline(project_root)
            self.assertEqual(
                loaded.get_from_storage(hash_id=hash_id),
                {"version": 2},
            )

    def test_failed_pickle_warns_without_breaking_pipeline_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            pipeline = PipelineHandler("root", local_folder_path=project_root)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                hash_id = pipeline.save_to_storage(
                    "unpicklable",
                    _UnpicklableStoredValue(),
                )
                _ = pipeline.save_pipeline()

            messages = [str(item.message) for item in caught]
            self.assertTrue(any("falling back to pickle" in item for item in messages))
            self.assertTrue(any("could not be pickled" in item for item in messages))
            loaded = PipelineHandler.load_pipeline(project_root)
            with self.assertRaisesRegex(ResolutionError, "No stored object"):
                loaded.get_from_storage(hash_id=hash_id)

    def test_attached_pipeline_apis_raise_and_storage_merges_to_ultimate_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = PipelineHandler("root", local_folder_path=tmp_path / "root")
            middle = PipelineHandler("middle", local_folder_path=tmp_path / "middle")
            leaf = PipelineHandler("leaf", local_folder_path=tmp_path / "leaf")
            stored_id = leaf.save_to_storage("leaf-object", [1, 2, 3])
            leaf.update_storage(stored_id, [4, 5, 6], to_disk=True)

            root.add_child_pipeline(middle, 1)
            middle.add_child_pipeline(leaf, 1)

            self.assertEqual(root.get_from_storage(hash_id=stored_id), [4, 5, 6])
            self.assertEqual(
                root.list_stored_objects()["object_name"].tolist(),
                ["leaf-object"],
            )
            api_calls = (
                leaf.list_stored_objects,
                lambda: leaf.save_to_storage("forbidden", 1),
                lambda: leaf.update_storage(stored_id, 2),
                lambda: leaf.get_from_storage(hash_id=stored_id),
                lambda: leaf.remove_from_storage(hash_id=stored_id),
            )
            for api_call in api_calls:
                with self.subTest(api_call=api_call):
                    with self.assertRaisesRegex(RegistrationError, "root pipeline"):
                        api_call()

    def test_parent_wins_hash_collision_and_child_artifact_is_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = PipelineHandler("root", local_folder_path=tmp_path / "root")
            child = PipelineHandler("child", local_folder_path=tmp_path / "child")
            collision_id = "00000000000000000000000000000001"
            with patch(
                "mlpipelineholder.object_storage.uuid4",
                return_value=UUID(int=1),
            ):
                root.save_to_storage("parent-copy", "parent")
                child.save_to_storage("child-copy", "child")
            child.update_storage(collision_id, "child", to_disk=True)

            root.add_child_pipeline(child, 1)

            self.assertEqual(root.get_from_storage(hash_id=collision_id), "parent")
            self.assertEqual(
                root.list_stored_objects()["object_name"].tolist(),
                ["parent-copy"],
            )
            child_storage_root = root.project_root / "children" / "child" / "artifacts" / "storage"
            self.assertEqual(list(child_storage_root.glob("*")), [])

    def test_malformed_storage_record_is_rejected_before_tree_restore(self) -> None:
        payload = {
            "registration_name": "root",
            "config": {},
            "nodes": [],
            "object_storage": {"deadbeef": "not-a-record"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            with (project_root / "pipeline_state.pkl").open("wb") as handle:
                pickle.dump(payload, handle)

            with patch.object(
                PipelineHandler,
                "_restore_working_tree_if_needed",
                side_effect=AssertionError("tree restore must not run"),
            ) as restore:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with self.assertRaisesRegex(PersistenceError, "invalid record"):
                        PipelineHandler.load_pipeline(project_root)

            restore.assert_not_called()

    def test_saved_storage_record_fields_are_not_coerced(self) -> None:
        now = datetime.now(UTC)
        artifact = ArtifactRecord(
            variable_name="stored",
            serializer="json",
            file_path="/tmp/stored.json",
            produced_by_block="storage",
            produced_by_function="object",
            run_id="stored",
        )
        valid_payload = {
            "hash_id": "deadbeef",
            "object_name": "stored",
            "object_type": "dict",
            "object_description": None,
            "created_at_utc": now,
            "last_modified_at_utc": now,
            "artifact": artifact,
        }
        invalid_fields = (
            ("hash_id", 123),
            ("object_name", 123),
            ("object_name", "   "),
            ("object_type", 123),
            ("object_type", ""),
            ("object_description", 123),
            ("created_at_utc", "now"),
            ("created_at_utc", datetime.now()),
            (
                "last_modified_at_utc",
                datetime.now(timezone(timedelta(hours=1))),
            ),
            ("artifact", "not-an-artifact"),
        )

        for field_name, invalid_value in invalid_fields:
            with self.subTest(field_name=field_name, invalid_value=invalid_value):
                malformed_payload = {**valid_payload, field_name: invalid_value}
                with self.assertRaises(PersistenceError):
                    record_from_payload(malformed_payload)

        without_description = dict(valid_payload)
        del without_description["object_description"]
        with self.assertRaises(PersistenceError):
            record_from_payload(without_description)

    def test_list_stored_objects_raises_import_error_without_pandas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", local_folder_path=Path(tmp) / "project")

            with patch.dict(modules, {"pandas": None}):
                with self.assertRaisesRegex(ImportError, "requires pandas"):
                    pipeline.list_stored_objects()


if __name__ == "__main__":
    unittest.main()
