from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import optuna

from mlpipelineholder import PipelineHandler
from mlpipelineholder.models import ArtifactRecord
from mlpipelineholder.output_pointers import OutputAddress, OutputPointer


def _study_for(owner: str, value: float) -> optuna.study.Study:
    study = optuna.create_study(study_name="shared-study", direction="minimize")
    study.add_trial(optuna.trial.create_trial(value=value))
    study.set_user_attr("owner", owner)
    return study


def first_study() -> optuna.study.Study:
    return _study_for("first", 1.0)


def second_study() -> optuna.study.Study:
    return _study_for("second", 2.0)


def third_study() -> optuna.study.Study:
    return _study_for("third", 3.0)


def first_number() -> int:
    return 1


def second_number() -> int:
    return 2


class OptunaOutputGroupTests(unittest.TestCase):
    def _pipeline(self, project_root: Path) -> PipelineHandler:
        pipeline = PipelineHandler("root", {}, project_root)
        for priority, name, function in (
            (1, "first", first_study),
            (2, "second", second_study),
            (3, "third", third_study),
        ):
            block = pipeline.add_block(name, priority)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_function(function, ["study"])
        return pipeline

    def assert_study_terminal(
        self,
        pipeline: PipelineHandler,
        terminal_name: str,
    ) -> None:
        terminal = OutputAddress("root", terminal_name, "study")
        raw_values = {
            node_name: outputs["study"]
            for node_name, outputs in pipeline.producer_outputs.items()
        }
        self.assertEqual(
            sum(isinstance(value, ArtifactRecord) for value in raw_values.values()),
            1,
        )
        self.assertIsInstance(raw_values[terminal_name], ArtifactRecord)
        for node_name, value in raw_values.items():
            if node_name == terminal_name:
                continue
            self.assertEqual(value, OutputPointer(terminal))
        pipeline._validate_runtime_output_pointers()

    def test_latest_initial_producer_is_sole_study_terminal(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given three nodes produce the same Study output name
            pipeline = self._pipeline(Path(temp_dir) / "project")

            # When the complete pipeline runs
            pipeline.run_all()

            # Then only the latest producer remains concrete
            self.assert_study_terminal(pipeline, "third")
            for node_name in ("first", "second", "third"):
                restored = pipeline.get_node_output(node_name, "study")
                self.assertEqual(restored.study_name, "shared-study")
                self.assertEqual(restored.user_attrs, {"owner": "third"})

    def test_attached_child_study_remains_owned_by_child_producer(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given a child pipeline whose block produces a Study
            base = Path(temp_dir)
            parent = PipelineHandler("parent", {}, base / "parent")
            child = PipelineHandler("child", {}, base / "child")
            producer = child.add_block("producer", 1)
            if producer is None:
                raise AssertionError("add_block should return a block")
            producer.register_function(first_study, ["study"])
            parent.add_child_pipeline(child, 1)

            # When the parent pipeline runs the attached child
            parent.run_all()

            # Then the child producer remains the concrete artifact owner
            child_value = child.producer_outputs["producer"]["study"]
            parent_mirror = parent.producer_outputs["child"]["study"]
            self.assertIsInstance(child_value, ArtifactRecord)
            self.assertEqual(parent_mirror, child_value)
            self.assertTrue(Path(child_value.file_path).is_file())

    def test_normal_rerun_restores_invalidated_study_peers(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given a complete three-node Study group
            pipeline = self._pipeline(Path(temp_dir) / "project")
            pipeline.run_all()

            # When the first node reruns with normal cascade invalidation
            pipeline.run_block("first")

            # Then all produced-once peers point directly to the new terminal
            self.assert_study_terminal(pipeline, "first")
            restored = pipeline.get_value("study")
            self.assertEqual(restored.user_attrs, {"owner": "first"})

    def test_forbidden_rerun_rewires_all_study_peers(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given a complete Study group with invalidation forbidden
            pipeline = self._pipeline(Path(temp_dir) / "project")
            pipeline.run_all()
            pipeline.forbid_invalidate_objects()

            # When the first node reruns
            pipeline.run_block("first")

            # Then it becomes the sole terminal without a pointer cycle
            self.assert_study_terminal(pipeline, "first")
            restored = pipeline.get_value("study")
            self.assertEqual(restored.user_attrs, {"owner": "first"})

    def test_non_study_same_name_outputs_remain_concrete(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given ordinary nodes produce the same integer output name
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            first = pipeline.add_block("first", 1)
            second = pipeline.add_block("second", 2)
            if first is None or second is None:
                raise AssertionError("add_block should return blocks")
            first.register_function(first_number, ["value"])
            second.register_function(second_number, ["value"])

            # When the complete pipeline runs
            pipeline.run_all()

            # Then the existing non-Study output slots remain concrete
            self.assertEqual(pipeline.producer_outputs["first"]["value"], 1)
            self.assertEqual(pipeline.producer_outputs["second"]["value"], 2)
            self.assertEqual(pipeline.get_value("value"), 2)

    def test_study_group_round_trips_before_normal_rotation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given a persisted three-node Study group
            project_root = Path(temp_dir) / "project"
            pipeline = self._pipeline(project_root)
            pipeline.run_all()
            pipeline.save_pipeline()
            loaded = PipelineHandler.load_pipeline(project_root, forced_deleting=True)

            # When an earlier producer runs after loading
            loaded.run_block("first")

            # Then restored group ownership rotates without losing peer slots
            self.assert_study_terminal(loaded, "first")
            restored = loaded.get_value("study")
            self.assertEqual(restored.user_attrs, {"owner": "first"})

    def test_different_output_names_receive_independent_studies(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given two nodes return the same literal Study name under different outputs
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            first = pipeline.add_block("first", 1)
            second = pipeline.add_block("second", 2)
            if first is None or second is None:
                raise AssertionError("add_block should return blocks")
            first.register_function(first_study, ["study_a"])
            second.register_function(second_study, ["study_b"])

            # When the complete pipeline runs
            pipeline.run_all()

            # Then each output group has an independently readable physical Study
            study_a = pipeline.get_node_output("first", "study_a")
            study_b = pipeline.get_node_output("second", "study_b")
            self.assertEqual(study_a.study_name, "shared-study")
            self.assertEqual(study_b.study_name, "shared-study_1")
            self.assertEqual(study_a.user_attrs, {"owner": "first"})
            self.assertEqual(study_b.user_attrs, {"owner": "second"})

    def test_removed_terminal_is_not_resurrected_by_later_rotation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given a Study group whose latest terminal is removed
            pipeline = self._pipeline(Path(temp_dir) / "project")
            pipeline.run_all()
            pipeline.remove_block("third")
            self.assert_study_terminal(pipeline, "second")

            # When an earlier surviving producer reruns
            pipeline.run_block("first")

            # Then the removed node stays absent and ownership rotates among survivors
            self.assertNotIn("third", pipeline.producer_outputs)
            self.assert_study_terminal(pipeline, "first")

    def test_study_group_save_as_rebases_database_without_changing_name(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given a complete Study group in one project root
            temporary_root = Path(temp_dir)
            pipeline = self._pipeline(temporary_root / "source")
            pipeline.run_all()

            # When it is saved elsewhere and restored into its recorded working root
            saved_root = pipeline.save_pipeline(temporary_root / "saved")
            loaded = PipelineHandler.load_pipeline(saved_root, forced_deleting=True)

            # Then the sole terminal and literal name survive with a rebased database
            self.assert_study_terminal(loaded, "third")
            terminal = loaded.producer_outputs["third"]["study"]
            self.assertIsInstance(terminal, ArtifactRecord)
            self.assertEqual(terminal.metadata["study_name"], "shared-study")
            self.assertEqual(
                Path(terminal.metadata["db_path"]),
                loaded.optuna_studies_db_path,
            )
            self.assertEqual(
                loaded.get_value("study").user_attrs,
                {"owner": "third"},
            )


if __name__ == "__main__":
    _ = unittest.main()
