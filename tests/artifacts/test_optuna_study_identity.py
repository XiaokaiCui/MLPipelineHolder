from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import optuna

from mlpipelineholder import PipelineHandler
from mlpipelineholder.core.models import ArtifactRecord


def build_named_study(study_name: str) -> optuna.study.Study:
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=31),
    )
    study.add_trial(
        optuna.trial.create_trial(
            value=0.25,
            params={"value": 0.5},
            distributions={"value": optuna.distributions.FloatDistribution(-1.0, 1.0)},
            user_attrs={"trial-owner": "source"},
        )
    )
    study.set_user_attr("study-owner", "source")
    return study


def build_output_study() -> optuna.study.Study:
    return build_named_study("output_study")


def build_colliding_output_study() -> optuna.study.Study:
    study = build_named_study("abc_study_1")
    study.set_user_attr("study-owner", "output")
    return study


class OptunaStudyIdentityTests(unittest.TestCase):
    def test_new_constant_clones_exact_nested_study_name(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given an external Study whose literal name already ends in an integer
            source = build_named_study("abc_study_1")
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")

            # When it is registered as a new constant
            pipeline.set_constant_value("search", source)

            # Then only the Study path creates an independent, nested-suffix artifact
            restored = pipeline.get_constant_value("search")
            self.assertIsInstance(restored, optuna.study.Study)
            self.assertEqual(restored.study_name, "abc_study_1_1")
            self.assertEqual(restored.user_attrs, {"study-owner": "source"})
            self.assertEqual(source.study_name, "abc_study_1")
            self.assertEqual(len(source.trials), 1)

    def test_new_storage_record_clones_study_eagerly(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given an external Study and an empty object store
            source = build_named_study("abc_study")
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")

            # When it is added to object storage
            hash_id = pipeline.save_to_storage("search", source)

            # Then the stored owner has an independent Study artifact immediately
            restored = pipeline.get_from_storage(hash_id=hash_id)
            self.assertIsInstance(restored, optuna.study.Study)
            self.assertEqual(restored.study_name, "abc_study_1")
            self.assertEqual(restored.user_attrs, {"study-owner": "source"})
            self.assertEqual(source.study_name, "abc_study")
            self.assertEqual(len(source.trials), 1)

    def test_non_study_constant_keeps_existing_snapshot_behavior(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given an ordinary mutable value
            source = ["original"]
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")

            # When it is registered as a constant with the default copy behavior
            pipeline.set_constant_value("ordinary", source)
            source.append("changed")

            # Then the existing non-Study snapshot path remains unchanged
            self.assertEqual(pipeline.get_constant_value("ordinary"), ["original"])

    def test_storage_copy_uses_next_global_suffix_after_constant(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given a constant already owns the first independent Study copy
            source = build_named_study("abc_study")
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            pipeline.set_constant_value("constant_search", source)

            # When the same external Study is copied into object storage
            hash_id = pipeline.save_to_storage("stored_search", source)

            # Then storage receives the next globally free literal suffix
            stored = pipeline.get_from_storage(hash_id=hash_id)
            self.assertIsInstance(stored, optuna.study.Study)
            self.assertEqual(stored.study_name, "abc_study_2")

    def test_output_allocates_after_constant_physical_name_collision(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given a constant owns the output Study's literal source name
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            pipeline.set_constant_value(
                "constant_search",
                build_named_study("abc_study"),
            )
            block = pipeline.add_block("search", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_function(build_colliding_output_study, ["study"])

            # When the new output group is persisted
            pipeline.run_all()

            # Then it allocates a nested suffix without replacing the constant
            constant = pipeline.get_constant_value("constant_search")
            output = pipeline.get_value("study")
            self.assertEqual(constant.study_name, "abc_study_1")
            self.assertEqual(constant.user_attrs, {"study-owner": "source"})
            self.assertEqual(output.study_name, "abc_study_1_1")
            self.assertEqual(output.user_attrs, {"study-owner": "output"})

    def test_output_allocates_after_storage_physical_name_collision(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given object storage owns the output Study's literal source name
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            hash_id = pipeline.save_to_storage(
                "stored_search",
                build_named_study("abc_study"),
            )
            block = pipeline.add_block("search", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_function(build_colliding_output_study, ["study"])

            # When the new output group is persisted
            pipeline.run_all()

            # Then it allocates a nested suffix without replacing object storage
            stored = pipeline.get_from_storage(hash_id=hash_id)
            output = pipeline.get_value("study")
            self.assertEqual(stored.study_name, "abc_study_1")
            self.assertEqual(stored.user_attrs, {"study-owner": "source"})
            self.assertEqual(output.study_name, "abc_study_1_1")
            self.assertEqual(output.user_attrs, {"study-owner": "output"})

    def test_established_study_survives_population_failure(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given a constant owns a populated Study and sampler artifact
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            pipeline.set_constant_value("search", build_named_study("abc_study"))
            previous_artifact = pipeline.manual_values["search"]
            self.assertIsInstance(previous_artifact, ArtifactRecord)
            previous_sampler_path = Path(previous_artifact.file_path)
            original_add_trials = optuna.study.Study.add_trials
            failed = False

            def fail_once(
                study: optuna.study.Study,
                trials: list[optuna.trial.FrozenTrial],
            ) -> None:
                nonlocal failed
                if not failed:
                    failed = True
                    raise RuntimeError("injected Study population failure")
                original_add_trials(study, trials)

            # When replacement population fails after the existing name is claimed
            with patch.object(optuna.study.Study, "add_trials", new=fail_once):
                with self.assertRaisesRegex(RuntimeError, "injected Study"):
                    pipeline.set_constant_value(
                        "search",
                        build_named_study("replacement"),
                    )

            # Then the previous Study and sampler remain complete and referenced
            restored = pipeline.get_constant_value("search")
            self.assertIs(pipeline.manual_values["search"], previous_artifact)
            self.assertTrue(previous_sampler_path.is_file())
            self.assertEqual(restored.study_name, "abc_study_1")
            self.assertEqual(restored.user_attrs, {"study-owner": "source"})
            self.assertEqual(len(restored.trials), 1)

    def test_resetting_study_constant_retains_owned_name(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given a constant owns an independently persisted Study
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            pipeline.set_constant_value("search", build_named_study("abc_study"))

            # When that same constant key is reset from a differently named Study
            pipeline.set_constant_value("search", build_named_study("replacement"))

            # Then the existing owner name is overwritten rather than reallocated
            restored = pipeline.get_constant_value("search")
            self.assertIsInstance(restored, optuna.study.Study)
            self.assertEqual(restored.study_name, "abc_study_1")
            self.assertEqual(len(restored.trials), 1)

    def test_updating_stored_study_retains_owned_name(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given a storage record owns an independently persisted Study
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            hash_id = pipeline.save_to_storage(
                "search",
                build_named_study("abc_study"),
            )

            # When that same hash ID is updated from a differently named Study
            pipeline.update_storage(hash_id, build_named_study("replacement"))

            # Then the storage owner keeps and overwrites its literal Study name
            restored = pipeline.get_from_storage(hash_id=hash_id)
            self.assertIsInstance(restored, optuna.study.Study)
            self.assertEqual(restored.study_name, "abc_study_1")
            self.assertEqual(len(restored.trials), 1)

    def test_constant_owned_study_can_create_storage_owner(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given a Study handle materialized from a managed constant
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            pipeline.set_constant_value("search", build_named_study("abc_study"))
            managed = pipeline.get_constant_value("search")

            # When it is copied into object storage
            hash_id = pipeline.save_to_storage("archive", managed)

            # Then storage owns an independent suffixed copy and the constant is intact
            restored = pipeline.get_from_storage(hash_id=hash_id)
            self.assertIsInstance(restored, optuna.study.Study)
            self.assertEqual(restored.study_name, "abc_study_1_1")
            self.assertEqual(restored.user_attrs, managed.user_attrs)
            self.assertEqual(
                pipeline._managed_study_owner(restored),
                ("storage", hash_id),
            )
            self.assertEqual(
                pipeline._managed_study_owner(pipeline.get_constant_value("search")),
                ("constant", "root.search"),
            )

    def test_untracked_same_text_name_receives_nested_suffix(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given a managed copy and an unrelated external Study with its name
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            pipeline.set_constant_value("search", build_named_study("abc_study"))
            external = build_named_study("abc_study_1")

            # When the untracked external Study is added to storage
            hash_id = pipeline.save_to_storage("external", external)

            # Then name equality alone does not imply managed provenance
            restored = pipeline.get_from_storage(hash_id=hash_id)
            self.assertEqual(restored.study_name, "abc_study_1_1")

    def test_storage_owned_study_can_create_constant_owner(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given a Study handle materialized from managed object storage
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            hash_id = pipeline.save_to_storage(
                "search",
                build_named_study("abc_study"),
            )
            managed = pipeline.get_from_storage(hash_id=hash_id)

            # When it is copied into a pipeline constant
            pipeline.set_constant_value("duplicate", managed)

            # Then the constant owns an independent copy and storage is intact
            constant = pipeline.get_constant_value("duplicate")
            self.assertIsInstance(constant, optuna.study.Study)
            self.assertEqual(constant.user_attrs, managed.user_attrs)
            self.assertEqual(
                pipeline._managed_study_owner(constant),
                ("constant", "root.duplicate"),
            )
            self.assertEqual(
                pipeline._managed_study_owner(pipeline.get_from_storage(hash_id=hash_id)),
                ("storage", hash_id),
            )

    def test_output_artifact_can_create_independent_storage_copy(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given an eagerly persisted Study output artifact
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            block = pipeline.add_block("search", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_function(build_output_study, ["study"])
            pipeline.run_all()
            output_artifact = pipeline.producer_outputs["search"]["study"]
            self.assertIsInstance(output_artifact, ArtifactRecord)

            # When the artifact itself is copied into object storage
            hash_id = pipeline.save_to_storage("copy", output_artifact)

            # Then storage owns an independent suffixed Study
            restored = pipeline.get_from_storage(hash_id=hash_id)
            self.assertIsInstance(restored, optuna.study.Study)
            self.assertEqual(restored.study_name, "output_study_1")

    def test_constant_and_storage_studies_survive_repeated_save_and_load(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Given independent constant and storage Study owners
            project_root = Path(temp_dir) / "project"
            pipeline = PipelineHandler("root", {}, project_root)
            source = build_named_study("abc_study")
            pipeline.set_constant_value("constant", source)
            hash_id = pipeline.save_to_storage("stored", source)

            # When the pipeline is saved twice and loaded
            pipeline.save_pipeline()
            pipeline.save_pipeline()
            loaded = PipelineHandler.load_pipeline(project_root, forced_deleting=True, trust_project=True)

            # Then both fresh handles retain their independent identities and data
            constant = loaded.get_constant_value("constant")
            stored = loaded.get_from_storage(hash_id=hash_id)
            self.assertEqual(constant.study_name, "abc_study_1")
            self.assertEqual(stored.study_name, "abc_study_2")
            self.assertEqual(len(constant.trials), 1)
            self.assertEqual(len(stored.trials), 1)


if __name__ == "__main__":
    _ = unittest.main()
