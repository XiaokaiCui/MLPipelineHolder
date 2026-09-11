from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from src.mlpipelineholder import PipelineHandler, RegistrationError
from src.mlpipelineholder.execution.atom_pipeline import AtomPipeline


def produce_value(value: str = "value") -> str:
    return value


class AtomPipelineSealedClassTests(unittest.TestCase):
    def _create_atom(self, pipeline: PipelineHandler, temp_dir: str) -> Any:
        pipeline.create_atom_child_pipeline(
            "atom",
            10,
            produce_value,
            output_variable_names="result",
        )
        atom = pipeline.get_child_pipeline("atom")
        self.assertIs(type(atom), AtomPipeline)
        return atom

    def test_atom_uses_sealed_class_and_class_level_marker(self) -> None:
        # Given/Then: the marker is structural, not an instance flag.
        self.assertTrue(AtomPipeline._is_atom)
        self.assertFalse(PipelineHandler._is_atom)

        with TemporaryDirectory() as temp_dir:
            # When: an atom is created through the public factory.
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "pipeline")
            atom = self._create_atom(pipeline, temp_dir)

            # Then: construction completes sealed.
            self.assertTrue(atom._sealed)

    def test_sealed_atom_rejects_extending_structure(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "pipeline")
            atom = self._create_atom(pipeline, temp_dir)

            # When/Then: every extension entry point rejects the sealed atom.
            with self.assertRaisesRegex(
                RegistrationError, "immutable and cannot accept new blocks"
            ):
                atom.add_block("late_block", 20)
            with self.assertRaisesRegex(
                RegistrationError, "immutable and cannot accept child pipelines"
            ):
                atom.add_child_pipeline(
                    PipelineHandler("donor", {}, Path(temp_dir) / "donor"),
                    20,
                )
            with self.assertRaisesRegex(
                RegistrationError, "immutable and cannot accept child pipelines"
            ):
                atom.create_atom_child_pipeline("nested", 20, produce_value)
            with self.assertRaisesRegex(
                RegistrationError, "immutable and cannot remove blocks"
            ):
                atom.remove_block("atom_block")

    def test_sealed_atom_rejects_gate_change(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", {"flag": True}, Path(temp_dir) / "pipeline")
            atom = self._create_atom(pipeline, temp_dir)

            with self.assertRaisesRegex(
                RegistrationError, "immutable and cannot change its gate"
            ):
                atom.set_gate_block("flag")
            with self.assertRaisesRegex(
                RegistrationError, "immutable and cannot change its gate"
            ):
                atom.add_gate_block("flag")
            with self.assertRaisesRegex(
                RegistrationError, "immutable and cannot change its gate"
            ):
                atom.reset_gate_block()

    def test_sealed_atom_rejects_block_mutation_and_config(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", {"flag": True}, Path(temp_dir) / "pipeline")
            atom = self._create_atom(pipeline, temp_dir)
            block = atom.get_block("atom_block")

            with self.assertRaisesRegex(
                RegistrationError, "immutable and cannot accept new functions"
            ):
                block.register_function(produce_value, ["extra"])
            with self.assertRaisesRegex(
                RegistrationError, "immutable and cannot accept new expressions"
            ):
                block.register_expression("1 + 1", output_variable_name="expr_out")
            with self.assertRaisesRegex(
                RegistrationError, "immutable and cannot accept new args helpers"
            ):
                block.register_args("more_args", ["value"])
            with self.assertRaisesRegex(
                RegistrationError, "immutable and cannot accept new kwargs helpers"
            ):
                block.register_kwargs("more_kwargs", {"value": "value"})
            with self.assertRaisesRegex(
                RegistrationError, "immutable and cannot remove functions"
            ):
                block.remove_function("produce_value")

            with self.assertRaisesRegex(RegistrationError, "does not own configuration"):
                atom.set_configs({"flag": False})
            with self.assertRaisesRegex(RegistrationError, "does not own configuration"):
                atom.update_configs({"flag": False})


    def test_loaded_atom_arrives_sealed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline = PipelineHandler("root", {}, root / "pipeline")
            pipeline.create_atom_child_pipeline(
                "atom",
                10,
                produce_value,
                output_variable_names="result",
            )
            saved = pipeline.save_pipeline(root / "saved")
            loaded = PipelineHandler.load_pipeline(saved, forced_deleting=True)
            atom = loaded.get_child_pipeline("atom")

            self.assertIs(type(atom), AtomPipeline)
            self.assertTrue(atom._sealed)
            with self.assertRaisesRegex(
                RegistrationError, "immutable and cannot accept new blocks"
            ):
                atom.add_block("late_block", 20)


if __name__ == "__main__":
    _ = unittest.main()
