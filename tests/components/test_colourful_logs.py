from __future__ import annotations

from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from mlpipelineholder import PipelineHandler


def produce_value() -> int:
    return 1


class ColourfulLogsTests(unittest.TestCase):
    def test_logger_console_is_uncoloured_by_default(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("plain-logs", {}, Path(temp_dir))
            output = StringIO()

            with patch("mlpipelineholder.presentation.logger.sys_stdout", output):
                pipeline.logger.debug("debug")
                pipeline.logger.info("info")
                pipeline.logger.warning("warning")
                pipeline.logger.error("error")
                pipeline.logger.critical("critical")
                pipeline.logger.result("result")

            self.assertNotIn("\x1b[", output.getvalue())

    def test_logger_console_is_coloured_when_enabled(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "colourful-logs",
                {},
                Path(temp_dir),
                colourful_logs=True,
            )
            output = StringIO()

            with patch("mlpipelineholder.presentation.logger.sys_stdout", output):
                pipeline.logger.info("info")

            self.assertIn("\x1b[", output.getvalue())

    def test_pipeline_structure_remains_coloured_when_logs_are_plain(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("colourful-structure", {}, Path(temp_dir))
            block = pipeline.add_block("build", 1)
            block.register_function(produce_value, ["result"])

            self.assertIn("\x1b[", pipeline.describe_pipeline())

    def test_attached_subtree_inherits_root_colourful_logs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            root = PipelineHandler(
                "root",
                {},
                root_path / "root",
                colourful_logs=True,
            )
            child = PipelineHandler("child", {}, root_path / "child")
            grandchild = PipelineHandler(
                "grandchild",
                {},
                root_path / "grandchild",
            )
            child.add_child_pipeline(grandchild, 1)

            root.add_child_pipeline(child, 1)

            self.assertTrue(child.colourful_logs)
            self.assertTrue(grandchild.colourful_logs)
            self.assertIs(child.logger, root.logger)
            self.assertIs(grandchild.logger, root.logger)

    def test_atom_inherits_root_colourful_logs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "root",
                {},
                Path(temp_dir),
                colourful_logs=True,
            )

            pipeline.create_atom_child_pipeline(
                "atom",
                1,
                produce_value,
                output_variable_names="result",
            )

            self.assertTrue(pipeline.get_child_pipeline("atom").colourful_logs)

    def test_colourful_logs_round_trips_through_save_and_load(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            pipeline = PipelineHandler(
                "root",
                {},
                root_path / "project",
                colourful_logs=True,
            )
            child = PipelineHandler("child", {}, root_path / "child")
            pipeline.add_child_pipeline(child, 1)
            bundle = root_path / "bundle"
            pipeline.save_pipeline(bundle)

            loaded = PipelineHandler.load_pipeline(bundle, forced_deleting=True)

            self.assertTrue(loaded.colourful_logs)
            self.assertTrue(loaded.get_child_pipeline("child").colourful_logs)


if __name__ == "__main__":
    unittest.main()
