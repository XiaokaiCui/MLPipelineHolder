from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.mlpipelineholder import (
    ExecutionBlock,
    ExecutionError,
    PersistenceError,
    PipelineHandler,
    ResolutionError,
)
from src.mlpipelineholder.output_pointers import OutputAddress, OutputPointer


def make_value(seed: int) -> list[int]:
    return [seed]


def append_two(value: list[int]) -> list[int]:
    value.append(2)
    return value


def add_block(
    pipeline: PipelineHandler,
    registration_name: str,
    execution_priority: float,
) -> ExecutionBlock:
    block = pipeline.add_block(registration_name, execution_priority)
    assert block is not None
    return block


class OutputPointerInvalidationTests(unittest.TestCase):
    def _pipeline(self, root: Path) -> PipelineHandler:
        pipeline = PipelineHandler("root", {"seed": 1}, root)
        first = add_block(pipeline, "first", 1)
        first.register_function(make_value, ["value"])
        second = add_block(pipeline, "second", 2)
        second.register_function(
            append_two,
            ["value"],
            overridden_outputs={"value": ("root", "first")},
        )
        third = add_block(pipeline, "third", 3)
        third.register_expression(
            "value = value + [3]",
            overridden_outputs={"value": ("root", "second")},
        )
        return pipeline

    def test_normal_tail_rerun_promotes_value_to_latest_survivor(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            pipeline = self._pipeline(Path(temp_dir))
            pipeline.run_all()

            # When
            pipeline.run_block("third")

            # Then
            self.assertEqual(pipeline.get_value("value"), [1, 2, 3, 3])
            self.assertIsInstance(
                pipeline.producer_outputs["first"]["value"],
                OutputPointer,
            )

    def test_forbidden_middle_removal_splices_surviving_chain(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            pipeline = self._pipeline(Path(temp_dir))
            pipeline.run_all()
            terminal = pipeline.get_node_output("third", "value")
            pipeline.forbid_invalidate_objects()

            # When
            pipeline.remove_block("second")

            # Then
            self.assertIs(pipeline.get_node_output("first", "value"), terminal)
            self.assertIs(pipeline.get_node_output("third", "value"), terminal)

    def test_terminal_removal_promotes_value_to_latest_survivor(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            pipeline = self._pipeline(Path(temp_dir))
            pipeline.run_all()

            # When
            pipeline.remove_block("third")

            # Then
            terminal = pipeline.get_node_output("second", "value")
            self.assertEqual(terminal, [1, 2, 3])
            self.assertIs(pipeline.get_node_output("first", "value"), terminal)

    def test_forced_registration_replacement_removes_old_pointer(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            pipeline = self._pipeline(Path(temp_dir))
            pipeline.run_all()
            second = pipeline.get_block("second")

            # When
            registration = second.register_function(
                append_two,
                ["value"],
                forced=True,
            )

            # Then
            self.assertEqual(registration.overridden_outputs, {})
            self.assertEqual(pipeline.get_node_output("first", "value"), [1, 2, 3])
            with self.assertRaises(ResolutionError):
                pipeline.get_node_output("second", "value")


class OutputPointerRestartTests(unittest.TestCase):
    def _pipeline(self, root: Path, *, disk_backed: bool = False) -> PipelineHandler:
        pipeline = PipelineHandler("root", {"seed": 1}, root)
        first = add_block(pipeline, "first", 1)
        first.register_function(
            make_value,
            ["value"],
            save_to_disk=["value"] if disk_backed else None,
        )
        second = add_block(pipeline, "second", 2)
        second.register_function(
            append_two,
            ["value"],
            save_to_disk=["value"] if disk_backed else None,
            overridden_outputs={"value": ("root", "first")},
        )
        third = add_block(pipeline, "third", 3)
        third.register_expression(
            "value = value + [3]",
            save_to_disk=disk_backed,
            overridden_outputs={"value": ("root", "second")},
        )
        return pipeline

    def test_run_from_pointer_target_uses_previous_terminal_value(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            pipeline = self._pipeline(Path(temp_dir))
            pipeline.run_all()

            # When
            pipeline.run_from("second")

            # Then
            self.assertEqual(pipeline.get_value("value"), [1, 2, 3, 2, 3])

    def test_run_from_disk_backed_pointer_target_keeps_artifact_valid(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "project"
            pipeline = self._pipeline(project_root, disk_backed=True)
            pipeline.run_all()

            # When
            pipeline.run_from("second")

            # Then
            self.assertEqual(pipeline.get_value("value"), [1, 2, 3, 2, 3])
            artifact_files = [
                path
                for path in (project_root / "artifacts").rglob("*")
                if path.is_file()
            ]
            self.assertEqual(len(artifact_files), 1)

    def test_nested_run_from_pointer_target_uses_previous_terminal_value(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler("parent", {"seed": 1}, root / "parent")
            child = PipelineHandler("child", {"seed": 1}, root / "child")
            first = add_block(child, "first", 1)
            first.register_function(make_value, ["value"])
            second = add_block(child, "second", 2)
            second.register_function(
                append_two,
                ["value"],
                overridden_outputs={"value": ("child", "first")},
            )
            third = add_block(child, "third", 3)
            third.register_expression(
                "value = value + [3]",
                overridden_outputs={"value": ("child", "second")},
            )
            parent.add_child_pipeline(child, 1)
            parent.run_all()

            # When
            parent.run_from("child", "second")

            # Then
            self.assertEqual(parent.get_value("value"), [1, 2, 3, 2, 3])


class OutputPointerPersistenceTests(unittest.TestCase):
    def test_active_chain_round_trips(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline = PipelineHandler("root", {"seed": 1}, root / "working")
            first = add_block(pipeline, "first", 1)
            first.register_function(make_value, ["value"])
            second = add_block(pipeline, "second", 2)
            second.register_function(
                append_two,
                ["value"],
                overridden_outputs={"value": ("root", "first")},
            )
            pipeline.run_all()
            pipeline.save_pipeline(root / "saved")

            # When
            loaded = PipelineHandler.load_pipeline(root / "saved", forced_deleting=True)

            # Then
            self.assertEqual(loaded.get_node_output("first", "value"), [1, 2])
            self.assertIsInstance(
                loaded.producer_outputs["first"]["value"],
                OutputPointer,
            )
            second_registration = loaded.blocks_by_name["second"].functions[0]
            self.assertEqual(
                second_registration.overridden_outputs,
                {"value": OutputAddress("root", "first", "value")},
            )
            loaded.run_all()
            self.assertIsInstance(
                loaded.producer_outputs["first"]["value"],
                OutputPointer,
            )

    def test_runtime_cycle_is_rejected_during_load(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline = PipelineHandler("root", {}, root / "working")
            first = add_block(pipeline, "first", 1)
            first.register_expression("value = [1]")
            second = add_block(pipeline, "second", 2)
            second.register_expression("value = value")
            first_address = OutputAddress("root", "first", "value")
            second_address = OutputAddress("root", "second", "value")
            pipeline.producer_outputs = {
                "first": {"value": OutputPointer(second_address)},
                "second": {"value": OutputPointer(first_address)},
            }
            pipeline._rebuild_visible_state()
            pipeline.save_pipeline(root / "saved")

            # When / Then
            with self.assertRaisesRegex(PersistenceError, "pointer"):
                PipelineHandler.load_pipeline(root / "saved", forced_deleting=True)


class OutputPointerArtifactTests(unittest.TestCase):
    def test_complete_disk_chain_keeps_one_artifact_after_tail_rerun(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "project"
            pipeline = PipelineHandler("root", {"seed": 1}, project_root)
            first = add_block(pipeline, "first", 1)
            first.register_function(
                make_value,
                ["value"],
                save_to_disk=["value"],
            )
            second = add_block(pipeline, "second", 2)
            second.register_function(
                append_two,
                ["value"],
                save_to_disk=["value"],
                overridden_outputs={"value": ("root", "first")},
            )
            third = add_block(pipeline, "third", 3)
            third.register_expression(
                "value = value + [3]",
                save_to_disk=True,
                overridden_outputs={"value": ("root", "second")},
            )
            pipeline.run_all()

            # When
            pipeline.run_block("third")

            # Then
            artifact_files = [
                path
                for path in (project_root / "artifacts").rglob("*")
                if path.is_file()
            ]
            self.assertEqual(len(artifact_files), 1)
            self.assertEqual(pipeline.get_value("value"), [1, 2, 3, 3])

    def test_disk_terminal_removal_transfers_artifact_to_survivor(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "project"
            pipeline = OutputPointerRestartTests()._pipeline(
                project_root,
                disk_backed=True,
            )
            pipeline.run_all()

            # When
            pipeline.remove_block("third")

            # Then
            self.assertEqual(pipeline.get_node_output("second", "value"), [1, 2, 3])
            artifact_files = [
                path
                for path in (project_root / "artifacts").rglob("*")
                if path.is_file()
            ]
            self.assertEqual(len(artifact_files), 1)

    def test_invalidated_pointer_target_cleans_failed_disk_output(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "project"
            pipeline = OutputPointerRestartTests()._pipeline(
                project_root,
                disk_backed=True,
            )
            pipeline.run_all()
            registration = pipeline.get_block("second").functions[0]
            registration.overridden_outputs["value"] = OutputAddress(
                "root",
                "missing",
                "value",
            )

            # When / Then
            with self.assertRaisesRegex(ExecutionError, "no longer valid"):
                pipeline.run_block("second")
            self.assertEqual(pipeline.get_node_output("first", "value"), [1, 2, 3])
            artifact_files = [
                path
                for path in (project_root / "artifacts").rglob("*")
                if path.is_file()
            ]
            self.assertEqual(len(artifact_files), 1)


if __name__ == "__main__":
    unittest.main()
