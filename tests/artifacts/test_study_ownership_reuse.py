from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import optuna

from mlpipelineholder import PipelineHandler, RegistrationError


def _study_for(study_name: str, value: float) -> optuna.study.Study:
    study = optuna.create_study(study_name=study_name, direction="minimize")
    study.add_trial(optuna.trial.create_trial(value=value))
    return study


def _output_study() -> optuna.study.Study:
    return _study_for("output-study", 9.0)


def _trial_values(study: optuna.study.Study) -> list[float | None]:
    return [trial.value for trial in study.trials]


def _attached_pipelines(project_root: Path) -> tuple[PipelineHandler, PipelineHandler]:
    quant_pipeline = PipelineHandler("quant_pipeline", {}, project_root / "quant")
    joint_analysis_pipeline = PipelineHandler(
        "joint_analysis_pipeline",
        {},
        project_root / "joint_analysis",
    )
    quant_pipeline.add_child_pipeline(joint_analysis_pipeline, 1)
    return quant_pipeline, joint_analysis_pipeline


class StudyOwnershipReuseTests(unittest.TestCase):
    def test_storage_to_constant_copy_avoids_optuna_trial_insertion_api(self) -> None:
        with TemporaryDirectory() as temp_dir:
            quant_pipeline, joint_analysis_pipeline = _attached_pipelines(
                Path(temp_dir)
            )
            source = optuna.create_study(
                study_name="fast-copy",
                direction="minimize",
                sampler=optuna.samplers.RandomSampler(seed=17),
            )
            source.set_user_attr("study-owner", "source")
            source.add_trial(
                optuna.trial.create_trial(
                    params={"depth": 3},
                    distributions={
                        "depth": optuna.distributions.IntDistribution(1, 5)
                    },
                    value=2.0,
                    intermediate_values={1: 2.5},
                    user_attrs={"trial-owner": "source"},
                    system_attrs={"origin": "fixture"},
                )
            )
            quant_pipeline.save_to_storage("source", source)
            stored = quant_pipeline.get_from_storage(object_name="source")

            with (
                patch.object(
                    optuna.study.Study,
                    "add_trial",
                    side_effect=AssertionError("add_trial must not be used"),
                ),
                patch.object(
                    optuna.study.Study,
                    "add_trials",
                    side_effect=AssertionError("add_trials must not be used"),
                ),
            ):
                joint_analysis_pipeline.set_constant_value("copied", stored)

            copied = joint_analysis_pipeline.get_constant_value("copied")
            copied_trial = copied.trials[0]
            self.assertEqual(copied.user_attrs, {"study-owner": "source"})
            self.assertEqual(copied_trial.params, {"depth": 3})
            self.assertEqual(copied_trial.intermediate_values, {1: 2.5})
            self.assertEqual(copied_trial.user_attrs, {"trial-owner": "source"})
            self.assertEqual(copied_trial.system_attrs, {"origin": "fixture"})
            self.assertEqual(copied_trial.state, optuna.trial.TrialState.COMPLETE)
            copied.add_trial(optuna.trial.create_trial(value=4.0))
            unchanged = quant_pipeline.get_from_storage(object_name="source")
            self.assertEqual(_trial_values(copied), [2.0, 4.0])
            self.assertEqual(_trial_values(unchanged), [2.0])

    def test_storage_owned_study_can_seed_constant_with_independent_copy(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given root storage owns a Study artifact
            quant_pipeline, joint_analysis_pipeline = _attached_pipelines(
                Path(temp_dir)
            )
            source = _study_for("stored-seed", 2.0)
            hash_id = quant_pipeline.save_to_storage("joint_study_tpe_2", source)
            stored_record = quant_pipeline._stored_objects[hash_id]
            stored_artifact = stored_record.artifact
            if stored_artifact is None:
                raise AssertionError("stored Study should have an artifact")
            artifact_path = Path(stored_artifact.file_path)

            # When the attached child seeds a constant from that stored Study
            joint_analysis_pipeline.set_constant_value(
                "seed",
                quant_pipeline.get_from_storage(object_name="joint_study_tpe_2"),
            )

            # Then each pool retains an independently owned Study with the same trials
            constant_study = joint_analysis_pipeline.get_constant_value("seed")
            stored_study = quant_pipeline.get_from_storage(
                object_name="joint_study_tpe_2"
            )
            self.assertIsInstance(constant_study, optuna.study.Study)
            self.assertEqual(_trial_values(constant_study), _trial_values(source))
            self.assertEqual(
                quant_pipeline._managed_study_owner(constant_study),
                ("constant", "joint_analysis_pipeline.seed"),
            )
            self.assertEqual(_trial_values(stored_study), _trial_values(source))
            self.assertTrue(artifact_path.is_file())

    def test_reported_three_line_sequence_succeeds(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given the two stored Studies used by the reported workflow
            quant_pipeline, joint_analysis_pipeline = _attached_pipelines(
                Path(temp_dir)
            )
            tpe_2 = _study_for("joint-study", 2.0)
            tpe_5 = _study_for("joint-study", 5.0)
            quant_pipeline.save_to_storage("joint_study_tpe_2", tpe_2)
            quant_pipeline.save_to_storage("joint_study_tpe_5", tpe_5)

            # When the user's exact set, clear, set sequence runs
            joint_analysis_pipeline.set_constant_value(
                "joint_reused_joint_study",
                quant_pipeline.get_from_storage(object_name="joint_study_tpe_2"),
            )
            joint_analysis_pipeline.set_constant_value(
                "joint_reused_joint_study",
                None,
            )
            joint_analysis_pipeline.set_constant_value(
                "joint_reused_joint_study",
                quant_pipeline.get_from_storage(object_name="joint_study_tpe_5"),
            )

            # Then the constant resolves the second stored Study's trials
            restored = joint_analysis_pipeline.get_constant_value(
                "joint_reused_joint_study"
            )
            self.assertEqual(_trial_values(restored), _trial_values(tpe_5))

    def test_constant_owned_study_can_be_saved_to_storage_with_independent_copy(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given the child constant owns a Study
            quant_pipeline, joint_analysis_pipeline = _attached_pipelines(
                Path(temp_dir)
            )
            source = _study_for("constant-seed", 3.0)
            joint_analysis_pipeline.set_constant_value("seed", source)
            constant_study = joint_analysis_pipeline.get_constant_value("seed")

            # When root storage archives the managed constant Study
            hash_id = quant_pipeline.save_to_storage("archived", constant_study)

            # Then storage has the trials and the source constant remains unchanged
            stored_study = quant_pipeline.get_from_storage(hash_id=hash_id)
            unchanged_constant = joint_analysis_pipeline.get_constant_value("seed")
            self.assertEqual(_trial_values(stored_study), _trial_values(source))
            self.assertEqual(_trial_values(unchanged_constant), _trial_values(source))
            self.assertEqual(
                quant_pipeline._managed_study_owner(unchanged_constant),
                ("constant", "joint_analysis_pipeline.seed"),
            )

    def test_constant_owned_study_can_replace_storage_slot_via_update_storage(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given root storage and the child constant own different Studies
            quant_pipeline, joint_analysis_pipeline = _attached_pipelines(
                Path(temp_dir)
            )
            hash_id = quant_pipeline.save_to_storage(
                "archived",
                _study_for("old-storage", 1.0),
            )
            replacement = _study_for("constant-replacement", 4.0)
            joint_analysis_pipeline.set_constant_value("seed", replacement)

            # When the constant Study replaces the existing storage slot
            quant_pipeline.update_storage(
                hash_id,
                joint_analysis_pipeline.get_constant_value("seed"),
            )

            # Then that slot resolves the replacement trials
            restored = quant_pipeline.get_from_storage(hash_id=hash_id)
            self.assertEqual(_trial_values(restored), _trial_values(replacement))

    def test_output_owned_study_still_assignable_to_constant_and_storage(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given a block on the attached child produces a Study
            quant_pipeline, joint_analysis_pipeline = _attached_pipelines(
                Path(temp_dir)
            )
            producer = joint_analysis_pipeline.add_block("producer", 1)
            if producer is None:
                raise AssertionError("add_block should return a block")
            producer.register_function(_output_study, ["study"])
            quant_pipeline.run_all()
            output_study = joint_analysis_pipeline.get_node_output("producer", "study")

            # When that output seeds both destination pools
            joint_analysis_pipeline.set_constant_value("seed", output_study)
            hash_id = quant_pipeline.save_to_storage("archived", output_study)

            # Then both destinations resolve the output trials
            constant_study = joint_analysis_pipeline.get_constant_value("seed")
            stored_study = quant_pipeline.get_from_storage(hash_id=hash_id)
            self.assertEqual(_trial_values(constant_study), _trial_values(output_study))
            self.assertEqual(_trial_values(stored_study), _trial_values(output_study))

    def test_constant_study_reuse_under_other_constant_is_rejected_with_actionable_message(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given one child constant owns a Study
            _, joint_analysis_pipeline = _attached_pipelines(Path(temp_dir))
            joint_analysis_pipeline.set_constant_value(
                "seed",
                _study_for("constant-seed", 6.0),
            )
            managed = joint_analysis_pipeline.get_constant_value("seed")

            # When another constant attempts to reuse that managed Study
            # Then the error explains the constant-pool rule and original-source fix
            with self.assertRaisesRegex(
                RegistrationError,
                "already managed by constant.*seed the new constant from the original source",
            ):
                joint_analysis_pipeline.set_constant_value("duplicate", managed)

    def test_storage_duplicate_save_is_rejected_with_actionable_message(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given the same storage-owned Study has been materialized twice
            quant_pipeline, _ = _attached_pipelines(Path(temp_dir))
            quant_pipeline.save_to_storage(
                "stored",
                _study_for("storage-seed", 7.0),
            )
            quant_pipeline.get_from_storage(object_name="stored")
            managed = quant_pipeline.get_from_storage(object_name="stored")

            # When a duplicate storage save is attempted
            # Then the error explains the storage-pool rule
            with self.assertRaisesRegex(
                RegistrationError,
                "already managed by stored object.*Storage cannot hold duplicate Studies",
            ):
                quant_pipeline.save_to_storage("duplicate", managed)

    def test_update_storage_cross_slot_is_rejected_with_actionable_message(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given two storage slots own different Studies
            quant_pipeline, _ = _attached_pipelines(Path(temp_dir))
            hash_a = quant_pipeline.save_to_storage(
                "study_a",
                _study_for("storage-a", 8.0),
            )
            quant_pipeline.save_to_storage(
                "study_b",
                _study_for("storage-b", 9.0),
            )

            # When slot A attempts to reuse slot B's managed Study
            # Then the error explains the storage-pool rule
            with self.assertRaisesRegex(
                RegistrationError,
                "already managed by stored object.*Storage cannot hold duplicate Studies",
            ):
                quant_pipeline.update_storage(
                    hash_a,
                    quant_pipeline.get_from_storage(object_name="study_b"),
                )


if __name__ == "__main__":
    _ = unittest.main()
