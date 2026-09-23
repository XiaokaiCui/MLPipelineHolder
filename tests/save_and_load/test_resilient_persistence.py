from __future__ import annotations

import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import optuna

from mlpipelineholder import PersistenceError, PipelineHandler
from mlpipelineholder.core.models import ArtifactRecord
from mlpipelineholder.state.output_pointers import OutputAddress, OutputPointer


def produce_study() -> optuna.study.Study:
    study = optuna.create_study(study_name="legacy-nested", direction="minimize")
    study.add_trial(optuna.trial.create_trial(value=3.0))
    return study


def produce_artifacts() -> tuple[dict[str, int], dict[str, int]]:
    return {"bad": 1}, {"good": 2}


def produce_number() -> int:
    return 1


class ResilientPersistenceTests(unittest.TestCase):
    def test_nested_legacy_study_pointer_graph_is_normalized_during_load(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = PipelineHandler("root", {}, base / "project")
            child = PipelineHandler("child", {"gate_on": True}, base / "child")
            child.set_gate_block("gate_on")
            grandchild = PipelineHandler("grandchild", {}, base / "grandchild")
            producer = grandchild.add_block("producer", 1)
            if producer is None:
                raise AssertionError("add_block should return a block")
            producer.register_function(produce_study, ["study"])
            child.add_child_pipeline(grandchild, 1)
            root.add_child_pipeline(child, 1)
            root.run_all()
            root.save_pipeline()

            state_path = root.project_root / "pipeline_state.pkl"
            payload = pickle.loads(state_path.read_bytes())
            child_payload = payload["nodes"][0]["payload"]
            grandchild_payload = child_payload["nodes"][0]["payload"]
            artifact = payload["producer_outputs"]["child"]["study"]
            if not isinstance(artifact, ArtifactRecord):
                raise AssertionError("root mirror should hold the Study artifact")
            artifact.metadata.pop("study_owner_kind", None)
            artifact.metadata.pop("study_owner_key", None)
            child_payload["producer_outputs"]["grandchild"]["study"] = OutputPointer(
                OutputAddress("root", "child", "study")
            )
            grandchild_payload["producer_outputs"]["producer"]["study"] = OutputPointer(
                OutputAddress("child", "grandchild", "study")
            )
            state_path.write_bytes(pickle.dumps(payload))

            loaded = PipelineHandler.load_pipeline(root.project_root, trust_project=True)
            loaded_grandchild = (
                loaded.get_child_pipeline("child").get_child_pipeline("grandchild")
            )
            restored = loaded_grandchild.get_node_output("producer", "study")
            restored_record = loaded_grandchild.producer_outputs["producer"]["study"]

            self.assertEqual([trial.value for trial in restored.trials], [3.0])
            self.assertIsInstance(restored_record, ArtifactRecord)
            self.assertEqual(
                restored_record.metadata["study_owner_key"],
                "grandchild.producer.study",
            )
            loaded._validate_runtime_output_pointers()

    def test_invalid_pointer_graph_is_rejected_before_save_creates_target(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            pipeline = PipelineHandler("root", {}, base / "project")
            producer = pipeline.add_block("producer", 1)
            if producer is None:
                raise AssertionError("add_block should return a block")
            producer.register_function(produce_number, ["value"])
            pipeline.run_all()
            pipeline.producer_outputs["producer"]["value"] = OutputPointer(
                OutputAddress("root", "missing", "value")
            )
            target = base / "invalid-save"

            with self.assertRaisesRegex(
                PersistenceError,
                "Saved output pointer graph is invalid",
            ):
                pipeline.save_pipeline(target)

            self.assertFalse(target.exists())

    def test_invalid_persisted_values_load_as_none_without_losing_valid_values(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "project"
            pipeline = PipelineHandler("root", {}, project_root)
            producer = pipeline.add_block("producer", 1)
            if producer is None:
                raise AssertionError("add_block should return a block")
            producer.register_function(
                produce_artifacts,
                ["bad_output", "good_output"],
                save_to_disk=["bad_output", "good_output"],
            )
            pipeline.run_all()
            pipeline.set_constant_value("bad_constant", {"bad": 3}, to_disk=True)
            pipeline.set_constant_value("good_constant", {"good": 4}, to_disk=True)
            bad_storage_hash = pipeline.save_to_storage("bad_storage", {"bad": 5})
            good_storage_hash = pipeline.save_to_storage("good_storage", {"good": 6})
            pipeline.save_pipeline()

            bad_output = pipeline.producer_outputs["producer"]["bad_output"]
            bad_constant = pipeline.manual_values["bad_constant"]
            bad_storage = pipeline._stored_objects[bad_storage_hash].artifact
            for artifact in (bad_output, bad_constant, bad_storage):
                if not isinstance(artifact, ArtifactRecord):
                    raise AssertionError("test value should be disk-backed")
                Path(artifact.file_path).unlink()

            loaded = PipelineHandler.load_pipeline(project_root, trust_project=True)

            self.assertIsNone(loaded.get_value("bad_output"))
            self.assertEqual(loaded.get_value("good_output"), {"good": 2})
            self.assertIsNone(loaded.get_constant_value("bad_constant"))
            self.assertEqual(
                loaded.get_constant_value("good_constant"),
                {"good": 4},
            )
            self.assertIsNone(loaded.get_from_storage(hash_id=bad_storage_hash))
            self.assertEqual(
                loaded.get_from_storage(hash_id=good_storage_hash),
                {"good": 6},
            )


    def test_dangling_nested_study_pointer_recovers_from_orphan_mirror_artifact(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = PipelineHandler("root", {}, base / "project")
            child = PipelineHandler("child", {"gate_on": True}, base / "child")
            child.set_gate_block("gate_on")
            grandchild = PipelineHandler("grandchild", {}, base / "grandchild")
            producer = grandchild.add_block("producer", 1)
            if producer is None:
                raise AssertionError("add_block should return a block")
            producer.register_function(produce_study, ["study"])
            child.add_child_pipeline(grandchild, 1)
            root.add_child_pipeline(child, 1)
            root.run_all()
            root.save_pipeline()

            state_path = root.project_root / "pipeline_state.pkl"
            payload = pickle.loads(state_path.read_bytes())
            child_payload = payload["nodes"][0]["payload"]
            grandchild_payload = child_payload["nodes"][0]["payload"]
            artifact = payload["producer_outputs"]["child"]["study"]
            if not isinstance(artifact, ArtifactRecord):
                raise AssertionError("root mirror should hold the Study artifact")
            artifact.metadata.pop("study_owner_kind", None)
            artifact.metadata.pop("study_owner_key", None)
            child_payload["producer_outputs"].pop("grandchild")
            grandchild_payload["producer_outputs"]["producer"]["study"] = OutputPointer(
                OutputAddress("child", "grandchild", "study")
            )
            state_path.write_bytes(pickle.dumps(payload))

            loaded = PipelineHandler.load_pipeline(root.project_root, trust_project=True)
            loaded_grandchild = (
                loaded.get_child_pipeline("child").get_child_pipeline("grandchild")
            )
            restored = loaded_grandchild.get_node_output("producer", "study")
            restored_record = loaded_grandchild.producer_outputs["producer"]["study"]

            self.assertEqual([trial.value for trial in restored.trials], [3.0])
            self.assertIsInstance(restored_record, ArtifactRecord)
            self.assertEqual(
                restored_record.metadata["study_owner_key"],
                "grandchild.producer.study",
            )
            loaded._validate_runtime_output_pointers()

    def test_dangling_output_pointer_without_recoverable_artifact_loads_as_none(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            pipeline = PipelineHandler("root", {}, base / "project")
            producer = pipeline.add_block("producer", 1)
            if producer is None:
                raise AssertionError("add_block should return a block")
            producer.register_function(produce_number, ["value"])
            pipeline.run_all()
            pipeline.save_pipeline()

            state_path = pipeline.project_root / "pipeline_state.pkl"
            payload = pickle.loads(state_path.read_bytes())
            payload["producer_outputs"]["producer"]["value"] = OutputPointer(
                OutputAddress("root", "missing", "value")
            )
            state_path.write_bytes(pickle.dumps(payload))

            loaded = PipelineHandler.load_pipeline(pipeline.project_root, trust_project=True)

            self.assertIsNone(loaded.get_value("value"))
            loaded._validate_runtime_output_pointers()


    def test_same_group_atom_study_slots_override_without_pointers_after_load(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = PipelineHandler("root", {}, base / "project")
            child = PipelineHandler("child", {"gate_on": True}, base / "child")
            child.set_gate_block("gate_on")
            grandchild = PipelineHandler("grandchild", {}, base / "grandchild")
            loser_atom = grandchild.create_atom_child_pipeline(
                "optimise",
                41.0,
                produce_study,
                output_variable_names="study",
            )
            winner_atom = grandchild.create_atom_child_pipeline(
                "optimise_from_previous",
                41.1,
                produce_study,
                output_variable_names="study",
            )
            if (
                grandchild.get_child_pipeline("optimise") is None
                or grandchild.get_child_pipeline("optimise_from_previous") is None
            ):
                raise AssertionError("atom creation should succeed")
            child.add_child_pipeline(grandchild, 1)
            root.add_child_pipeline(child, 1)
            root.run_all()
            root.save_pipeline()

            state_path = root.project_root / "pipeline_state.pkl"
            payload = pickle.loads(state_path.read_bytes())
            child_payload = payload["nodes"][0]["payload"]
            grandchild_payload = child_payload["nodes"][0]["payload"]
            artifact = payload["producer_outputs"]["child"]["study"]
            if not isinstance(artifact, ArtifactRecord):
                raise AssertionError("root mirror should hold the Study artifact")
            artifact.produced_by_block = (
                "root/child/grandchild/optimise_from_previous/"
                "optimise_from_previous_block"
            )
            artifact.metadata.pop("study_owner_kind", None)
            artifact.metadata.pop("study_owner_key", None)
            loser_atom_payload = grandchild_payload["nodes"][0]["payload"]
            loser_atom_payload["producer_outputs"]["optimise_block"]["study"] = (
                OutputPointer(OutputAddress("grandchild", "optimise", "study"))
            )
            grandchild_payload["producer_outputs"]["optimise"]["study"] = (
                OutputPointer(OutputAddress("child", "grandchild", "study"))
            )
            grandchild_payload["producer_outputs"]["optimise_from_previous"] = {
                "study": artifact
            }
            child_payload["producer_outputs"].pop("grandchild")
            state_path.write_bytes(pickle.dumps(payload))

            loaded = PipelineHandler.load_pipeline(root.project_root, trust_project=True)
            loaded_grandchild = (
                loaded.get_child_pipeline("child").get_child_pipeline("grandchild")
            )

            self.assertIsNone(loaded_grandchild.producer_outputs["optimise"]["study"])
            winner_slot = loaded_grandchild.producer_outputs[
                "optimise_from_previous"
            ]["study"]
            self.assertIsInstance(winner_slot, ArtifactRecord)
            self.assertNotIsInstance(winner_slot, OutputPointer)
            restored = loaded_grandchild.get_value("study")
            self.assertEqual([trial.value for trial in restored.trials], [3.0])
            loaded._validate_runtime_output_pointers()

    def test_same_group_study_atoms_override_without_pointer_creation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            first_atom = pipeline.create_atom_child_pipeline(
                "atom_first",
                41.0,
                produce_study,
                output_variable_names="study",
            )
            second_atom = pipeline.create_atom_child_pipeline(
                "atom_second",
                41.1,
                produce_study,
                output_variable_names="study",
            )
            if (
                pipeline.get_child_pipeline("atom_first") is None
                or pipeline.get_child_pipeline("atom_second") is None
            ):
                raise AssertionError("atom creation should succeed")
            pipeline.run_all()
            pipeline.forbid_invalidate_objects()
            pipeline.run_block("atom_second")

            for atom_name in ("atom_first", "atom_second"):
                atom = pipeline.get_child_pipeline(atom_name)
                for outputs in atom.producer_outputs.values():
                    self.assertNotIsInstance(outputs.get("study"), OutputPointer)
            pipeline._validate_runtime_output_pointers()
            pipeline.save_pipeline()
            loaded = PipelineHandler.load_pipeline(
                pipeline.project_root,
                forced_deleting=True,
                trust_project=True,
            )
            loaded._validate_runtime_output_pointers()


if __name__ == "__main__":
    _ = unittest.main()
