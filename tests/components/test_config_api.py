from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.mlpipelineholder import PipelineHandler, RegistrationError, ResolutionError


def read_factor(factor: int) -> int:
    return factor


class ConfigApiTests(unittest.TestCase):
    def test_set_config_changes_one_configuration(self) -> None:
        # Given: a pipeline without the requested configuration field.
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "config",
                {},
                Path(temp_dir) / "pipeline",
            )

            # When: one field is changed through the singular API.
            pipeline.set_config("factor", 7)

        # Then: the requested field is created with the new value.
        self.assertEqual(pipeline.get_config("factor"), 7)

    def test_set_configs_changes_multiple_configurations(self) -> None:
        # Given: a pipeline has multiple configuration fields.
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "config",
                {"factor": 2},
                Path(temp_dir) / "pipeline",
            )

            # When: fields are changed through the plural API.
            pipeline.set_configs({"factor": 7, "offset": 3})

            # Then: every requested field has its new value.
            self.assertEqual(pipeline.get_full_config(), {"factor": 7, "offset": 3})

    def test_update_config_changes_one_existing_configuration(self) -> None:
        # Given: a pipeline has an existing configuration field.
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "config",
                {"factor": 2},
                Path(temp_dir) / "pipeline",
            )

            # When: one field is changed through the singular update API.
            pipeline.update_config("factor", 7)

            # Then: the existing field has the new value.
            self.assertEqual(pipeline.get_config("factor"), 7)

    def test_update_configs_changes_multiple_existing_configurations(self) -> None:
        # Given: a pipeline has multiple existing configuration fields.
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "config",
                {"factor": 2, "offset": 1},
                Path(temp_dir) / "pipeline",
            )

            # When: fields are changed through the plural update API.
            pipeline.update_configs({"factor": 7, "offset": 3})

            # Then: every existing field has its new value.
            self.assertEqual(pipeline.get_full_config(), {"factor": 7, "offset": 3})

    def test_update_config_rejects_unknown_configuration(self) -> None:
        # Given: a pipeline without the requested configuration field.
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "config",
                {},
                Path(temp_dir) / "pipeline",
            )

            # When/Then: singular update refuses to create the field.
            with self.assertRaisesRegex(ResolutionError, "Unknown config field: factor"):
                pipeline.update_config("factor", 7)

    def test_get_config_mirrors_get_config_value(self) -> None:
        # Given: a configured pipeline.
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "config",
                {"factor": 2},
                Path(temp_dir) / "pipeline",
            )

            # When: the same field is read through both public APIs.
            value = pipeline.get_config("factor")

            # Then: get_config follows get_config_value semantics.
            self.assertEqual(value, pipeline.get_config_value("factor"))

    def test_atom_rejects_set_config(self) -> None:
        # Given: an atom that delegates configuration to its parent.
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "config",
                {"factor": 2},
                Path(temp_dir) / "pipeline",
            )
            pipeline.create_atom_child_pipeline(
                "read_factor",
                1,
                read_factor,
                output_variable_names="result",
            )
            atom = pipeline.get_child_pipeline("read_factor")

            # When/Then: singular atom configuration mutation is rejected.
            with self.assertRaises(RegistrationError):
                atom.set_config("factor", 7)

    def test_atom_rejects_set_configs(self) -> None:
        # Given: an atom that delegates configuration to its parent.
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "config",
                {"factor": 2},
                Path(temp_dir) / "pipeline",
            )
            pipeline.create_atom_child_pipeline(
                "read_factor",
                1,
                read_factor,
                output_variable_names="result",
            )
            atom = pipeline.get_child_pipeline("read_factor")

            # When/Then: plural atom configuration mutation is rejected.
            with self.assertRaises(RegistrationError):
                atom.set_configs({"factor": 7})


if __name__ == "__main__":
    _ = unittest.main()
