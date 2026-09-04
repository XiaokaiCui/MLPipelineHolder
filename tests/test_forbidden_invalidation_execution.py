from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.mlpipelineholder import PipelineHandler
from src.mlpipelineholder.models import ArtifactRecord


def produce_early_value() -> int:
    return 1


def produce_later_value() -> int:
    return 2


class ForbiddenInvalidationExecutionTests(unittest.TestCase):
    def test_child_run_and_save_preserve_later_sibling_artifact(self) -> None:
        # Given: a saved root whose later child owns a disk-backed output.
        with TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "root"
            root = PipelineHandler("root", {}, project_root)
            early = PipelineHandler("early", {}, Path(temp_dir) / "early")
            early_block = early.add_block("early_block", 1)
            if early_block is None:
                raise AssertionError("add_block should return the early block")
            early_block.register_function(
                produce_early_value,
                ["early_value"],
            )
            later = PipelineHandler("later", {}, Path(temp_dir) / "later")
            later_block = later.add_block("later_block", 1)
            if later_block is None:
                raise AssertionError("add_block should return the later block")
            later_block.register_function(
                produce_later_value,
                ["later_value"],
                save_to_disk=["later_value"],
            )
            root.add_child_pipeline(early, 1)
            root.add_child_pipeline(later, 2)
            _ = root.run_all()
            _ = root.save_pipeline()

            loaded = PipelineHandler.load_pipeline(project_root)
            loaded_later = loaded.get_child_pipeline("later")
            stored = loaded_later.producer_outputs["later_block"]["later_value"]
            if not isinstance(stored, ArtifactRecord):
                self.fail("later_value should be disk-backed")
            artifact_path = Path(stored.file_path)

            # When: invalidation is forbidden and only the earlier child is rerun.
            loaded.forbid_invalidate_objects()
            _ = loaded.get_child_pipeline("early").run_all()
            self.assertTrue(artifact_path.exists())
            _ = loaded.save_pipeline()

            # Then: the unrelated later output remains persisted and loadable.
            self.assertTrue(artifact_path.exists())
            self.assertEqual(loaded_later.get_value("later_value"), 2)
            reloaded = PipelineHandler.load_pipeline(project_root)
            self.assertEqual(
                reloaded.get_child_pipeline("later").get_value("later_value"),
                2,
            )


if __name__ == "__main__":
    _ = unittest.main()
