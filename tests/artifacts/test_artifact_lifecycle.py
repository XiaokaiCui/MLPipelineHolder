from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mlpipelineholder import PipelineHandler
from mlpipelineholder.models import ArtifactRecord


def produce_blob(seed: int) -> str:
    return f"blob={seed}"


class ArtifactLifecycleTests(unittest.TestCase):
    def test_invalidation_deletes_unreferenced_disk_artifact(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("inv", {"seed": 3}, tmp / "project")
            writer = pipeline.add_block("writer", 1)
            if writer is None:
                raise AssertionError("add_block should return a block")
            writer.register_function(produce_blob, ["blob"], save_to_disk=["blob"])
            pipeline.run_all()

            artifact = pipeline.para_value_dict["blob"]
            self.assertIsInstance(artifact, ArtifactRecord)
            artifact_path = Path(artifact.file_path)
            self.assertTrue(artifact_path.is_file())

            pipeline.remove_block("writer")

            self.assertFalse(artifact_path.exists())

    def test_override_retention_survives_downstream_invalidation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("retain", {"seed": 3}, tmp / "project")
            first = pipeline.add_block("first", 1)
            if first is None:
                raise AssertionError("add_block should return a block")
            first.register_function(produce_blob, ["shared"], save_to_disk=["shared"])
            second = pipeline.add_block("second", 2)
            if second is None:
                raise AssertionError("add_block should return a block")
            second.register_function(produce_blob, ["shared"])
            pipeline.run_all()

            first_artifact = pipeline.producer_outputs["first"]["shared"]
            self.assertIsInstance(first_artifact, ArtifactRecord)
            artifact_path = Path(first_artifact.file_path)
            self.assertTrue(artifact_path.is_file())
            self.assertEqual(pipeline.get_node_output("first", "shared"), "blob=3")
            self.assertEqual(pipeline.get_value("shared"), "blob=3")

            pipeline.remove_block("second")

            self.assertTrue(artifact_path.exists())
            self.assertEqual(pipeline.get_node_output("first", "shared"), "blob=3")
            self.assertEqual(pipeline.get_value("shared"), "blob=3")


if __name__ == "__main__":
    unittest.main()
