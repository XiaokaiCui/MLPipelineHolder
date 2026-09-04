from __future__ import annotations

import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from src.mlpipelineholder import PipelineHandler
from src.mlpipelineholder.function_registry import _values_equal
from src.mlpipelineholder.models import ArtifactRecord, TorchStateArtifactRecord


def produce_blob() -> dict[str, int]:
    return {"saved": 7}


def produce_leaf() -> int:
    return 3


def legacy_artifact_record(record: ArtifactRecord) -> ArtifactRecord:
    legacy = object.__new__(ArtifactRecord)
    for name in (
        "variable_name",
        "serializer",
        "file_path",
        "produced_by_block",
        "produced_by_function",
        "run_id",
    ):
        object.__setattr__(legacy, name, getattr(record, name))
    return legacy


def strip_new_fields(value: Any) -> Any:
    if isinstance(value, ArtifactRecord):
        return legacy_artifact_record(value)
    if isinstance(value, dict):
        return {key: strip_new_fields(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strip_new_fields(item) for item in value]
    return value


class BackwardCompatibleLoadTests(unittest.TestCase):
    def test_legacy_artifact_record_pickle_restores_new_fields_with_defaults(self) -> None:
        # Given: an instance shaped like a pickle from a version without the
        # created_at, torch_load_weights_only, or metadata fields.
        legacy = object.__new__(ArtifactRecord)
        for name in (
            "variable_name",
            "serializer",
            "file_path",
            "produced_by_block",
            "produced_by_function",
            "run_id",
        ):
            object.__setattr__(legacy, name, name)

        # When: the pickle is loaded with the current class definition.
        restored = pickle.loads(pickle.dumps(legacy))

        # Then: fields added after the save are filled with their defaults.
        self.assertIsInstance(restored, ArtifactRecord)
        self.assertTrue(restored.created_at)
        self.assertFalse(restored.torch_load_weights_only)
        self.assertEqual(restored.metadata, {})

    def test_legacy_torch_record_pickle_restores_metadata_default(self) -> None:
        # Given: a torch artifact record without the newer metadata field.
        legacy = object.__new__(TorchStateArtifactRecord)
        object.__setattr__(legacy, "variable_name", "weights")
        object.__setattr__(legacy, "file_path", "/tmp/weights.pt")
        object.__setattr__(legacy, "object_kind", "state_dict")

        # When: the pickle is loaded with the current class definition.
        restored = pickle.loads(pickle.dumps(legacy))

        # Then: the metadata field is default-filled.
        self.assertEqual(restored.metadata, {})

    def test_values_equal_tolerates_dataclass_instances_with_missing_fields(self) -> None:
        # Given: a current record and a legacy-shaped record without new fields.
        full = ArtifactRecord("v", "pickle", "/tmp/v", "b", "f", "r")
        legacy = legacy_artifact_record(full)

        # When: the deep equality comparison runs on those instances.
        # Then: it never raises; missing on both sides compares equal.
        self.assertTrue(_values_equal(legacy, legacy))
        self.assertFalse(_values_equal(legacy, full))
        self.assertFalse(_values_equal(full, legacy))
        self.assertTrue(_values_equal(full, full))

    def test_loading_pipeline_with_legacy_artifact_records_succeeds(self) -> None:
        # Given: a saved pipeline whose disk-backed output was recorded before
        # the new ArtifactRecord fields existed, plus a gated child pipeline
        # so the gate input snapshot contains that record.
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            work = tmp / "work"
            root = PipelineHandler("root", {}, work)
            producer = root.add_block("producer", 1)
            if producer is None:
                self.fail("add_block should return the producer block")
            producer.register_function(produce_blob, ["blob"], save_to_disk=["blob"])
            child = PipelineHandler("child", {"gate_on": True}, tmp / "child")
            child.set_gate_block("gate_on")
            grandchild = PipelineHandler("grandchild", {}, tmp / "grandchild")
            leaf = grandchild.add_block("leaf", 1)
            if leaf is None:
                self.fail("add_block should return the leaf block")
            leaf.register_function(produce_leaf, ["leaf_value"])
            child.add_child_pipeline(grandchild, 1)
            root.add_child_pipeline(child, 2)
            _ = root.run_all()
            _ = root.save_pipeline()

            state_path = work / "pipeline_state.pkl"
            payload = strip_new_fields(pickle.loads(state_path.read_bytes()))
            state_path.write_bytes(pickle.dumps(payload))

            # When: the backup is loaded with the current library version.
            loaded = PipelineHandler.load_pipeline(work)

            # Then: the legacy record is repaired and the value materializes.
            record = loaded.producer_outputs["producer"]["blob"]
            if not isinstance(record, ArtifactRecord):
                self.fail("blob should remain a disk-backed artifact record")
            self.assertEqual(record.metadata, {})
            self.assertEqual(loaded.get_value("blob"), {"saved": 7})


if __name__ == "__main__":
    _ = unittest.main()
