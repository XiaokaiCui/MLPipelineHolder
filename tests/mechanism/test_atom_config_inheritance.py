from __future__ import annotations

import inspect
import unittest
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

from src.mlpipelineholder import PipelineHandler, RegistrationError


def select_parameter_branch(joint_use_median_param: bool = True) -> str:
    return "median" if joint_use_median_param else "best"


class AtomConfigInheritanceTests(unittest.TestCase):
    def test_parent_config_update_reaches_default_atom_during_run_from(self) -> None:
        # Given: an atom was created while the root selection flag was True.
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "joint_analysis_pipeline",
                {"joint_use_median_param": True},
                Path(temp_dir) / "pipeline",
            )
            pipeline.create_atom_child_pipeline(
                "joint_select_param",
                26,
                select_parameter_branch,
                output_variable_names="selected_branch",
            )

            # When: the root flag changes before the atom is run.
            pipeline.set_configs({"joint_use_median_param": False})
            _ = pipeline.run_from("joint_select_param")

            # Then: the atom resolves the current inherited root value.
            self.assertEqual(pipeline.get_value("selected_branch"), "best")
            self.assertFalse(
                pipeline.get_child_pipeline("joint_select_param").get_config_value(
                    "joint_use_median_param"
                )
            )

    def test_atom_references_parent_config(self) -> None:
        # Given: an atom is attached to a configured parent.
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "joint_analysis_pipeline",
                {"joint_use_median_param": False},
                Path(temp_dir) / "pipeline",
            )
            pipeline.create_atom_child_pipeline(
                "joint_select_param",
                26,
                select_parameter_branch,
                output_variable_names="selected_branch",
            )
            atom = pipeline.get_child_pipeline("joint_select_param")

            # When: callers inspect the atom's configuration object.
            atom_config = atom.config

            # Then: the atom exposes the parent's object rather than owning a copy.
            self.assertIs(atom_config, pipeline.config)

    def test_atom_set_config_is_rejected(self) -> None:
        # Given: an atom is attached to a configured parent.
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "joint_analysis_pipeline",
                {"joint_use_median_param": True},
                Path(temp_dir) / "pipeline",
            )
            pipeline.create_atom_child_pipeline(
                "joint_select_param",
                26,
                select_parameter_branch,
                output_variable_names="selected_branch",
            )
            atom = pipeline.get_child_pipeline("joint_select_param")

            # When/Then: atom-level config mutation is rejected.
            with self.assertRaises(RegistrationError):
                atom.set_config("joint_use_median_param", False)

    def test_atom_creation_has_no_configuration_argument(self) -> None:
        # Given: callers inspect the public atom factory signature.
        parameters = inspect.signature(
            PipelineHandler.create_atom_child_pipeline
        ).parameters

        # Then: atom-local configuration cannot be supplied.
        self.assertNotIn("child_configuration", parameters)
        self.assertNotIn("default_config_value", parameters)

    def test_atom_creation_rejects_default_config_value_argument(self) -> None:
        # Given: a parent pipeline with the gate config already visible.
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "joint_analysis_pipeline",
                {"joint_use_median_param": True},
                Path(temp_dir) / "pipeline",
            )
            method_name = "create_atom_child_pipeline"
            factory: Callable[..., None] = getattr(pipeline, method_name)
            invalid_kwargs = {"default_config_value": False}

            # When/Then: removed atom-local config arguments fail with Python's normal signature error.
            with self.assertRaises(TypeError):
                factory(
                    "joint_select_param",
                    26,
                    select_parameter_branch,
                    gate_config="joint_use_median_param",
                    output_variable_names="selected_branch",
                    **invalid_kwargs,
                )

    def test_default_atom_config_inheritance_survives_save_and_load(self) -> None:
        # Given: a default-config atom has been saved and loaded.
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline = PipelineHandler(
                "joint_analysis_pipeline",
                {"joint_use_median_param": True},
                root / "pipeline",
            )
            pipeline.create_atom_child_pipeline(
                "joint_select_param",
                26,
                select_parameter_branch,
                output_variable_names="selected_branch",
            )
            saved = pipeline.save_pipeline(root / "saved")
            loaded = PipelineHandler.load_pipeline(saved, forced_deleting=True)

            # When: the loaded root flag changes before the atom runs.
            loaded.set_configs({"joint_use_median_param": False})
            _ = loaded.run_from("joint_select_param")

            # Then: the loaded atom still resolves current parent configuration.
            self.assertEqual(loaded.get_value("selected_branch"), "best")
            self.assertIs(
                loaded.get_child_pipeline("joint_select_param").config,
                loaded.config,
            )


if __name__ == "__main__":
    _ = unittest.main()
