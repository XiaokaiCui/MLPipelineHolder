from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.mlpipelineholder import PersistenceError, PipelineHandler


@dataclass
class SaveConfig:
    value: int


def importable(value: int) -> int:
    return value + 1


class SaveLoadTests(unittest.TestCase):
    def local_callable(self, value):
        return value + 1

    def test_non_importable_callables_cannot_be_saved(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("persist", SaveConfig(value=1), tmp_path)
            block = pipeline.add_block("block", 1)

            block.register_function(self.local_callable, ["result"])

            with self.assertRaises(PersistenceError):
                pipeline.save_project(tmp_path / "bundle")

    def test_importable_callable_round_trips(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("persist", SaveConfig(value=2), tmp_path / "project")
            block = pipeline.add_block("block", 1)
            block.register_function(importable, ["result"])
            pipeline.run_all()

            save_dir = tmp_path / "bundle"
            pipeline.save_project(save_dir)
            loaded = PipelineHandler.load_project(save_dir)

            self.assertEqual(loaded.para_value_dict["result"], 3)
