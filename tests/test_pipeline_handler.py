from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unittest

from src.mlpipelineholder import PipelineHandler, RegistrationError, ResolutionError
from src.mlpipelineholder.models import ArtifactRecord


@dataclass
class DemoConfig:
    base: int
    factor: int = 2


def produce_seed(base: int) -> int:
    return base + 1


def multiply(seed: int, factor: int) -> int:
    return seed * factor


def branch_left(seed: int) -> int:
    return seed + 10


def branch_right(seed: int) -> int:
    return seed + 20


def combine(left: int, right: int) -> int:
    return left + right


def save_text(seed: int) -> str:
    return f"value={seed}"


def read_text(saved_blob: str) -> str:
    return saved_blob.upper()


def pair(seed: int) -> tuple[int, int]:
    return seed, seed + 1


def needs_missing(missing: int) -> int:
    return missing


class PipelineHandlerTests(unittest.TestCase):
    def test_pipeline_runs_full_and_partial_flow(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("demo", DemoConfig(base=3, factor=4), tmp_path)
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])

            branch = pipeline.add_block("branch", 2)
            branch.register_function(branch_left, ["left"])
            branch.register_function(branch_right, ["right"])

            final = pipeline.add_block("final", 3)
            final.register_function(combine, ["total"])
            final.register_function(multiply, ["scaled_total"])

            run = pipeline.run_all()

            self.assertEqual(run.status, "success")
            self.assertEqual(pipeline.para_value_dict["seed"], 4)
            self.assertEqual(pipeline.para_value_dict["left"], 14)
            self.assertEqual(pipeline.para_value_dict["right"], 24)
            self.assertEqual(pipeline.para_value_dict["total"], 38)
            self.assertEqual(pipeline.para_value_dict["scaled_total"], 16)

            pipeline.run_block("setup", overrides={"base": 10})
            self.assertEqual(pipeline.para_value_dict["seed"], 11)
            self.assertNotIn("left", pipeline.para_value_dict)
            self.assertNotIn("total", pipeline.para_value_dict)

            pipeline.run_from("branch")
            self.assertEqual(pipeline.para_value_dict["left"], 21)
            self.assertEqual(pipeline.para_value_dict["right"], 31)
            self.assertEqual(pipeline.para_value_dict["total"], 52)

    def test_disk_artifact_is_saved_and_loaded_for_downstream_use(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("disk", DemoConfig(base=2), tmp_path)
            setup = pipeline.add_block("setup", 1)
            setup.register_function(produce_seed, ["seed"])
            disk_block = pipeline.add_block("disk_write", 2)
            disk_block.register_function(save_text, ["saved_blob"], save_to_disk=["saved_blob"])
            consumer = pipeline.add_block("consumer", 3)
            consumer.register_function(read_text, ["upper_blob"])

            pipeline.run_all()

            artifact = pipeline.para_value_dict["saved_blob"]
            self.assertIsInstance(artifact, ArtifactRecord)
            self.assertTrue(Path(artifact.file_path).exists())
            self.assertEqual(pipeline.para_value_dict["upper_blob"], "VALUE=3")

    def test_project_can_be_saved_and_loaded(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            project_dir = tmp_path / "project"
            pipeline = PipelineHandler("persisted", DemoConfig(base=5), project_dir)
            block = pipeline.add_block("setup", 1)
            block.register_function(produce_seed, ["seed"])
            pipeline.run_all()

            save_dir = tmp_path / "save_bundle"
            pipeline.save_project(save_dir)
            loaded = PipelineHandler.load_project(save_dir)

            self.assertEqual(loaded.registration_name, "persisted")
            self.assertEqual(loaded.para_value_dict["seed"], 6)
            self.assertEqual(list(loaded.blocks_by_name), ["setup"])

    def test_duplicate_outputs_are_rejected_across_blocks(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("dup", DemoConfig(base=1), tmp_path)
            first = pipeline.add_block("first", 1)
            second = pipeline.add_block("second", 2)

            first.register_function(produce_seed, ["seed"])
            with self.assertRaises(RegistrationError):
                second.register_function(produce_seed, ["seed"])

    def test_multiple_outputs_require_matching_return_arity(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("pair", DemoConfig(base=2), tmp_path)
            first = pipeline.add_block("first", 1)
            first.register_function(produce_seed, ["seed"])
            second = pipeline.add_block("second", 2)
            second.register_function(pair, ["first_value", "second_value"])

            pipeline.run_all()
            self.assertEqual(pipeline.para_value_dict["first_value"], 3)
            self.assertEqual(pipeline.para_value_dict["second_value"], 4)

    def test_missing_argument_raises_resolution_error(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("missing", DemoConfig(base=1), tmp_path)
            block = pipeline.add_block("broken", 1)
            block.register_function(needs_missing, ["x"])

            with self.assertRaises(ResolutionError):
                pipeline.run_all()
