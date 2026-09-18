from __future__ import annotations

from inspect import signature
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mlpipelineholder import PipelineHandler


def produce_one() -> int:
    return 1


class ConvenientDefaultsTests(unittest.TestCase):
    def test_pipeline_defaults_reuse_existing_root(self) -> None:
        with TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "project"
            project_root.mkdir()
            marker = project_root / "marker.txt"
            marker.write_text("keep", encoding="utf-8")

            pipeline = PipelineHandler("root", {}, project_root)

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            parameters = signature(PipelineHandler).parameters
            self.assertIs(parameters["forced"].default, True)
            self.assertIs(parameters["_allow_existing_root"].default, True)
            self.assertEqual(pipeline.project_root, project_root)

    def test_add_block_replaces_same_named_block_by_default(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            original = pipeline.add_block("build", 1)
            original.register_function(produce_one, ["value"])

            replacement = pipeline.add_block("build", 1)

            self.assertIsNot(original, replacement)
            self.assertIs(pipeline.get_block("build"), replacement)

    def test_add_child_pipeline_replaces_conflict_by_default(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            parent = PipelineHandler("parent", {}, root_path / "parent")
            original = PipelineHandler("child", {}, root_path / "original")
            replacement = PipelineHandler("child", {}, root_path / "replacement")
            parent.add_child_pipeline(original, 1)

            attached = parent.add_child_pipeline(replacement, 1)

            self.assertIs(attached, replacement)
            self.assertIs(parent.get_child_pipeline("child"), replacement)

    def test_register_expression_replaces_expression_by_default(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            block = pipeline.add_block("expression", 1)
            block.register_expression("value = 1")

            registration = block.register_expression("value = 2")

            self.assertEqual(registration.code, "value = 2")
            self.assertEqual(len(block.functions), 1)

    def test_atom_factory_convenience_defaults(self) -> None:
        parameters = signature(PipelineHandler.create_atom_child_pipeline).parameters

        self.assertIs(parameters["allow_existing_root"].default, True)
        self.assertIs(parameters["forced"].default, True)
        self.assertEqual(parameters["block_priority"].default, 10.0)


if __name__ == "__main__":
    _ = unittest.main()
