from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import optuna

from mlpipelineholder import PersistenceError, PipelineHandler
from mlpipelineholder.models import ArtifactRecord


def _objective(trial: optuna.trial.Trial) -> float:
    value = trial.suggest_float("value", -1.0, 1.0)
    return value * value


def build_study() -> optuna.study.Study:
    study = optuna.create_study(
        study_name="pipeline-study",
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=19),
    )
    study.optimize(_objective, n_trials=3)
    study.set_user_attr("owner", "pipeline")
    return study


def build_sampler() -> optuna.samplers.BaseSampler:
    return optuna.samplers.TPESampler(seed=23)


def passthrough(value: int) -> int:
    return value


class OptunaArtifactTests(unittest.TestCase):
    def test_db_path_is_lazy_and_atoms_do_not_expose_it(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            pipeline = PipelineHandler("root", {"value": 1}, root)

            self.assertEqual(
                pipeline.optuna_studies_db_path,
                root / "optuna_studies.db",
            )
            self.assertIsInstance(pipeline.optuna_studies_db_path, Path)
            self.assertFalse(pipeline.optuna_studies_db_path.exists())

            pipeline.save_pipeline()

            self.assertFalse(pipeline.optuna_studies_db_path.exists())
            pipeline.create_atom_child_pipeline(
                "atom",
                1,
                passthrough,
                output_variable_names="result",
            )
            atom = pipeline.get_child_pipeline("atom")
            self.assertFalse(hasattr(atom, "optuna_studies_db_path"))

    def test_sampler_uses_pickle_without_creating_study_db(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            pipeline = PipelineHandler("sampler", {}, root)
            block = pipeline.add_block("search", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_function(
                build_sampler,
                ["sampler"],
                save_to_disk=["sampler"],
            )

            pipeline.run_all()

            record = pipeline.para_value_dict["sampler"]
            self.assertIsInstance(record, ArtifactRecord)
            self.assertEqual(record.serializer, "pickle")
            self.assertFalse(pipeline.optuna_studies_db_path.exists())
            restored = pipeline.get_value("sampler")
            self.assertIsInstance(restored, optuna.samplers.BaseSampler)
            self.assertIsInstance(restored, optuna.samplers.TPESampler)

    def test_disk_backed_study_round_trips_name_trials_attrs_and_sampler(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            pipeline = PipelineHandler("study", {}, root)
            block = pipeline.add_block("search", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_function(
                build_study,
                ["study"],
                save_to_disk=["study"],
            )

            pipeline.run_all()

            record = pipeline.para_value_dict["study"]
            self.assertIsInstance(record, ArtifactRecord)
            self.assertEqual(record.serializer, "optuna-study")
            self.assertEqual(record.metadata["study_name"], "pipeline-study")
            self.assertEqual(
                Path(record.metadata["db_path"]),
                pipeline.optuna_studies_db_path,
            )
            self.assertTrue(pipeline.optuna_studies_db_path.is_file())
            self.assertTrue(Path(record.file_path).is_file())

            saved = pipeline.save_pipeline()
            loaded = PipelineHandler.load_pipeline(saved, forced_deleting=True)
            restored = loaded.get_value("study")

            self.assertIsInstance(restored, optuna.study.Study)
            self.assertEqual(restored.study_name, "pipeline-study")
            self.assertEqual(len(restored.trials), 3)
            self.assertEqual(restored.user_attrs, {"owner": "pipeline"})
            self.assertIsInstance(restored.sampler, optuna.samplers.RandomSampler)

    def test_saving_pipeline_persists_live_study_and_sampler_as_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            pipeline = PipelineHandler("live", {}, root)
            block = pipeline.add_block("search", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_function(build_study, ["study"])
            block.register_function(build_sampler, ["sampler"])
            pipeline.run_all()
            self.assertFalse(pipeline.optuna_studies_db_path.exists())

            pipeline.save_pipeline()
            loaded = PipelineHandler.load_pipeline(root, forced_deleting=True)

            study_record = loaded.para_value_dict["study"]
            sampler_record = loaded.para_value_dict["sampler"]
            self.assertIsInstance(study_record, ArtifactRecord)
            self.assertEqual(study_record.serializer, "optuna-study")
            self.assertIsInstance(sampler_record, ArtifactRecord)
            self.assertEqual(sampler_record.serializer, "pickle")
            self.assertIsInstance(loaded.get_value("study"), optuna.study.Study)
            self.assertIsInstance(
                loaded.get_value("sampler"),
                optuna.samplers.TPESampler,
            )

    def test_loading_missing_study_db_does_not_create_empty_database(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            pipeline = PipelineHandler("missing", {}, root)
            block = pipeline.add_block("search", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_function(
                build_study,
                ["study"],
                save_to_disk=["study"],
            )
            pipeline.run_all()
            pipeline.optuna_studies_db_path.unlink()

            with self.assertRaises(PersistenceError):
                pipeline.get_value("study")

            self.assertFalse(pipeline.optuna_studies_db_path.exists())

    def test_attaching_pipeline_rebases_existing_study_storage(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            old_child_root = tmp / "standalone-child"
            child = PipelineHandler("child", {}, old_child_root)
            block = child.add_block("search", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_function(
                build_study,
                ["study"],
                save_to_disk=["study"],
            )
            child.run_all()
            self.assertTrue(child.optuna_studies_db_path.is_file())

            parent = PipelineHandler("parent", {}, tmp / "parent")
            parent.add_child_pipeline(child, 1)

            expected_path = parent.project_root / "children" / "child" / "optuna_studies.db"
            self.assertEqual(child.optuna_studies_db_path, expected_path)
            self.assertTrue(expected_path.is_file())
            self.assertFalse(old_child_root.exists())
            record = child.para_value_dict["study"]
            self.assertEqual(Path(record.metadata["db_path"]), expected_path)
            restored = child.get_value("study")
            self.assertEqual(len(restored.trials), 3)
            self.assertIsInstance(restored.sampler, optuna.samplers.RandomSampler)


if __name__ == "__main__":
    unittest.main()
