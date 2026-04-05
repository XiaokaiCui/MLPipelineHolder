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


def mapped_variadic(obj: int, *more_values: int, scale: int = 1, **extra_values: int) -> int:
    return (obj + sum(more_values) + sum(extra_values.values())) * scale


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

    def test_mapping_metadata_round_trips_for_variadic_function(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler(
                "persist-mapped",
                {
                    "payload": 2,
                    "scale_value": 3,
                    "extra_args": [4, 5],
                    "extra_kwargs": {"bonus": 6},
                },
                tmp_path / "project",
            )
            block = pipeline.add_block("block", 1)
            block.register_function(
                mapped_variadic,
                ["result"],
                param_mapping={"obj": "payload", "scale": "scale_value"},
                var_pos_name="extra_args",
                var_kw_name="extra_kwargs",
            )
            pipeline.run_all()

            save_dir = tmp_path / "bundle"
            pipeline.save_project(save_dir)
            loaded = PipelineHandler.load_project(save_dir)
            loaded.run_all()

            self.assertEqual(loaded.get_value("result"), 51)
