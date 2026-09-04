from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.mlpipelineholder import PipelineHandler


def consume(value: int) -> int:
    return value + 1


class ForcedExpressionReRegistrationTests(unittest.TestCase):
    def test_unchanged_definition_after_load_preserves_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline = PipelineHandler("pipeline", {}, root / "pipeline")
            block = pipeline.add_block("build_value", 25.0)
            assert block is not None
            block.register_expression("value = 1")
            pipeline.create_atom_child_pipeline(
                "consume_value",
                44.0,
                consume,
                output_variable_names="result",
                param_mapping={"value": "value"},
            )
            _ = pipeline.run_all()
            _ = pipeline.save_pipeline(root / "bundle")

            loaded = PipelineHandler.load_pipeline(
                root / "bundle",
                forced_deleting=True,
            )

            rebuilt = loaded.add_block("build_value", 25.0, forced=True)
            assert rebuilt is not None
            rebuilt.register_expression("value = 1", forced=True)

            self.assertEqual(loaded.get_value("value"), 1)
            self.assertEqual(loaded.get_value("result"), 2)

    def test_changed_definition_after_load_invalidates_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline = PipelineHandler("pipeline", {}, root / "pipeline")
            block = pipeline.add_block("build_value", 25.0)
            assert block is not None
            block.register_expression("value = 1")
            pipeline.create_atom_child_pipeline(
                "consume_value",
                44.0,
                consume,
                output_variable_names="result",
                param_mapping={"value": "value"},
            )
            _ = pipeline.run_all()
            _ = pipeline.save_pipeline(root / "bundle")

            loaded = PipelineHandler.load_pipeline(
                root / "bundle",
                forced_deleting=True,
            )

            rebuilt = loaded.add_block("build_value", 25.0, forced=True)
            assert rebuilt is not None
            rebuilt.register_expression("value = 2", forced=True)

            self.assertNotIn("value", loaded.para_value_dict)
            self.assertNotIn("result", loaded.para_value_dict)
