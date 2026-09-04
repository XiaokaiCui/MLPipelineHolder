from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol
import unittest

from src.mlpipelineholder import PipelineHandler


class DynamicConfig(Protocol):
    weight: float


class DynamicConfigFactory(Protocol):
    def __call__(self, *, weight: float = 1.0) -> DynamicConfig: ...


def create_dynamic_config_class() -> DynamicConfigFactory:
    @dataclass
    class GeneratedScoreConfig:
        weight: float = 1.0

    return GeneratedScoreConfig


def instantiate_dynamic_config(
    config_cls: DynamicConfigFactory,
) -> DynamicConfig:
    return config_cls(weight=2.5)


class DynamicDataclassRecoveryTests(unittest.TestCase):
    def test_instance_reconnects_to_pipeline_produced_class_after_load(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline = PipelineHandler("pipeline", {}, root / "pipeline")
            class_block = pipeline.add_block("build_config_class", 1.0)
            assert class_block is not None
            class_block.register_function(
                create_dynamic_config_class,
                ["config_cls"],
            )
            instance_block = pipeline.add_block("build_config", 2.0)
            assert instance_block is not None
            instance_block.register_function(
                instantiate_dynamic_config,
                ["config"],
                param_mapping={"config_cls": "config_cls"},
            )
            _ = pipeline.run_all()
            _ = pipeline.save_pipeline(root / "bundle")

            loaded = PipelineHandler.load_pipeline(
                root / "bundle",
                forced_deleting=True,
            )

            loaded_class = loaded.get_value("config_cls")
            loaded_config = loaded.get_value("config")
            self.assertTrue(is_dataclass(loaded_class))
            self.assertIs(type(loaded_config), loaded_class)
            self.assertEqual(loaded_config.weight, 2.5)
