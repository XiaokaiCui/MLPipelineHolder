from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock

from src.mlpipelineholder import PipelineHandler
from src.mlpipelineholder.backup_recovery import (
    ImpactConfirmation,
    OwnerKind,
    SlotKind,
    confirm_recovery_impact,
    discover_owned_variable_slots,
)
from src.mlpipelineholder.models import ArtifactRecord


@dataclass
class DemoConfig:
    base: int


def produce_seed(base: int) -> int:
    return base + 1


def produce_late_seed(seed: int) -> int:
    return seed + 100


def produce_shared_text(base: int) -> str:
    del base
    return "shared"


def _snapshot_pipeline_state(pipelines, variable_name: str):
    snapshot = []
    for pipeline in pipelines:
        producer_snapshot = {
            owner_name: dict(outputs)
            for owner_name, outputs in pipeline.producer_outputs.items()
            if variable_name in outputs
        }
        snapshot.append(
            (
                pipeline.full_path(),
                {
                    "manual": dict(pipeline.manual_values),
                    "para": dict(pipeline.para_value_dict),
                    "artifact": dict(pipeline.artifact_registry),
                },
                producer_snapshot,
            )
        )
    return snapshot


def _slot_summary(inventory):
    return [
        (slot.slot_kind.value, slot.pipeline_path, slot.producer_name)
        for owner in inventory.owners
        for slot in owner.update_slots
    ] + [
        (slot.slot_kind.value, slot.pipeline_path, slot.producer_name)
        for slot in inventory.mirror_slots
    ]


class BackupRecoveryOwnershipBaselineTests(unittest.TestCase):
    def test_update_value_rewrites_manual_and_visible_slots(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", DemoConfig(base=1), Path(temp_dir) / "root")
            pipeline.set_value("manual_value", 5)

            pipeline.update_value("manual_value", 9)

            self.assertEqual(pipeline.manual_values["manual_value"], 9)
            self.assertEqual(pipeline.para_value_dict["manual_value"], 9)
            self.assertNotIn("manual_value", pipeline.artifact_registry)

    def test_update_value_rewrites_only_latest_local_producer_slot(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", DemoConfig(base=1), Path(temp_dir) / "root")
            first = pipeline.add_block("first", 1)
            first.register_function(produce_seed, ["seed"])
            second = pipeline.add_block("second", 2)
            second.register_function(produce_late_seed, ["seed"])
            _ = pipeline.run_all()

            self.assertEqual(pipeline.producer_outputs["first"]["seed"], 2)
            self.assertEqual(pipeline.producer_outputs["second"]["seed"], 102)

            pipeline.update_value("seed", 42)

            self.assertEqual(pipeline.producer_outputs["first"]["seed"], 2)
            self.assertEqual(pipeline.producer_outputs["second"]["seed"], 42)
            self.assertEqual(pipeline.para_value_dict["seed"], 42)


class BackupRecoveryOwnershipTests(unittest.TestCase):
    def test_discover_owned_variable_slots_for_manual_value_needs_no_prompt(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = PipelineHandler("root", DemoConfig(base=1), Path(temp_dir) / "root")
            root.set_value("manual_value", 5)

            inventory = discover_owned_variable_slots(root, "manual_value")
            confirmation = confirm_recovery_impact(
                inventory,
                root.logger,
                input_func=lambda prompt: self.fail(prompt),
            )

            self.assertEqual([owner.pipeline_path for owner in inventory.owners], ["root"])
            self.assertEqual(inventory.owners[0].owner_kind, OwnerKind.MANUAL)
            self.assertEqual(
                _slot_summary(inventory),
                [
                    (SlotKind.PARA.value, "root", None),
                    (SlotKind.MANUAL.value, "root", None),
                    (SlotKind.ARTIFACT_REGISTRY.value, "root", None),
                ],
            )
            self.assertEqual(
                confirmation,
                ImpactConfirmation(
                    authorized=True,
                    prompted=False,
                    affected_paths=("root",),
                ),
            )

    def test_discover_owned_variable_slots_for_latest_local_producer_ignores_shadowed_output(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = PipelineHandler("root", DemoConfig(base=1), Path(temp_dir) / "root")
            first = root.add_block("first", 1)
            first.register_function(produce_seed, ["seed"])
            second = root.add_block("second", 2)
            second.register_function(produce_late_seed, ["seed"])
            _ = root.run_all()

            inventory = discover_owned_variable_slots(root, "seed")

            self.assertEqual([owner.pipeline_path for owner in inventory.owners], ["root"])
            self.assertEqual(inventory.owners[0].owner_kind, OwnerKind.LATEST_PRODUCER)
            self.assertEqual(
                _slot_summary(inventory),
                [
                    (SlotKind.PARA.value, "root", None),
                    (SlotKind.LATEST_PRODUCER.value, "root", "second"),
                    (SlotKind.ARTIFACT_REGISTRY.value, "root", None),
                ],
            )

    def test_discover_owned_variable_slots_for_fallback_para_slot_tracks_registry(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = PipelineHandler("root", DemoConfig(base=1), Path(temp_dir) / "root")
            artifact = ArtifactRecord(
                variable_name="saved_blob",
                serializer="json",
                file_path=str(Path(temp_dir) / "artifact.json"),
                produced_by_block="root/setup",
                produced_by_function="save",
                run_id="run-1",
            )
            root.para_value_dict["saved_blob"] = artifact
            root.artifact_registry["saved_blob"] = artifact

            inventory = discover_owned_variable_slots(root, "saved_blob")

            self.assertEqual([owner.pipeline_path for owner in inventory.owners], ["root"])
            self.assertEqual(inventory.owners[0].owner_kind, OwnerKind.FALLBACK_PARA)
            self.assertEqual(
                _slot_summary(inventory),
                [
                    (SlotKind.PARA.value, "root", None),
                    (SlotKind.ARTIFACT_REGISTRY.value, "root", None),
                ],
            )

    def test_confirmation_counts_same_object_reference_owners_separately(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = PipelineHandler("root", DemoConfig(base=1), Path(temp_dir) / "root")
            child = PipelineHandler("child", DemoConfig(base=2), Path(temp_dir) / "child")
            root.add_child_pipeline(child, 1)
            shared_value: list[str] = ["same-object"]
            root.set_value("shared", shared_value)
            child.set_value("shared", shared_value)
            inventory = discover_owned_variable_slots(root, "shared")
            prompts: list[str] = []

            confirmation = confirm_recovery_impact(
                inventory,
                root.logger,
                input_func=lambda prompt: prompts.append(prompt) or " y ",
            )

            self.assertEqual([owner.pipeline_path for owner in inventory.owners], ["root", "root/child"])
            self.assertTrue(confirmation.authorized)
            self.assertTrue(confirmation.prompted)
            self.assertEqual(len(prompts), 1)
            self.assertLess(prompts[0].find("root"), prompts[0].find("root/child"))
            self.assertEqual(prompts[0].count("root\n"), 1)
            self.assertEqual(prompts[0].count("root/child\n"), 1)

    def test_confirmation_counts_same_interned_immutable_owners_separately(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = PipelineHandler("root", DemoConfig(base=1), Path(temp_dir) / "root")
            child = PipelineHandler("child", DemoConfig(base=2), Path(temp_dir) / "child")
            block = child.add_block("shared", 1)
            block.register_function(produce_shared_text, ["shared"])
            root.add_child_pipeline(child, 1)
            root.set_value("shared", "shared")
            _ = root.run_all()

            inventory = discover_owned_variable_slots(root, "shared")
            prompts: list[str] = []

            confirmation = confirm_recovery_impact(
                inventory,
                root.logger,
                input_func=lambda prompt: prompts.append(prompt) or "Y",
            )

            self.assertEqual([owner.pipeline_path for owner in inventory.owners], ["root", "root/child"])
            self.assertTrue(confirmation.authorized)
            self.assertEqual(len(prompts), 1)

    def test_discover_owned_variable_slots_treats_child_and_ancestor_views_as_mirrors(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = PipelineHandler("root", DemoConfig(base=1), Path(temp_dir) / "root")
            middle = PipelineHandler("middle", DemoConfig(base=2), Path(temp_dir) / "middle")
            producer = PipelineHandler("producer", DemoConfig(base=3), Path(temp_dir) / "producer")
            block = producer.add_block("seed_block", 1)
            block.register_function(produce_seed, ["seed"])
            middle.add_child_pipeline(producer, 1)
            root.add_child_pipeline(middle, 1)
            _ = root.run_all()

            inventory = discover_owned_variable_slots(root, "seed")

            self.assertEqual([owner.pipeline_path for owner in inventory.owners], ["root/middle/producer"])
            self.assertEqual(
                _slot_summary(inventory),
                [
                    (SlotKind.PARA.value, "root/middle/producer", None),
                    (SlotKind.LATEST_PRODUCER.value, "root/middle/producer", "seed_block"),
                    (SlotKind.ARTIFACT_REGISTRY.value, "root/middle/producer", None),
                    (SlotKind.CHILD_OUTPUT_MIRROR.value, "root/middle", "producer"),
                    (SlotKind.ANCESTOR_VISIBLE_MIRROR.value, "root/middle", None),
                    (SlotKind.CHILD_OUTPUT_MIRROR.value, "root", "middle"),
                    (SlotKind.ANCESTOR_VISIBLE_MIRROR.value, "root", None),
                ],
            )

    def test_confirmation_refuses_on_non_yes_input_and_logs_once_without_state_change(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = PipelineHandler("root", DemoConfig(base=1), Path(temp_dir) / "root")
            child = PipelineHandler("child", DemoConfig(base=2), Path(temp_dir) / "child")
            root.add_child_pipeline(child, 1)
            root.set_value("shared", 1)
            child.set_value("shared", 2)
            inventory = discover_owned_variable_slots(root, "shared")
            warning = MagicMock()
            root.logger.warning = warning
            before = _snapshot_pipeline_state([root, child], "shared")

            confirmation = confirm_recovery_impact(
                inventory,
                root.logger,
                input_func=lambda prompt: "nope",
            )
            after = _snapshot_pipeline_state([root, child], "shared")

            self.assertFalse(confirmation.authorized)
            self.assertTrue(confirmation.prompted)
            self.assertEqual(before, after)
            self.assertEqual(warning.call_count, 1)

    def test_confirmation_refuses_on_eof_and_logs_once_without_state_change(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = PipelineHandler("root", DemoConfig(base=1), Path(temp_dir) / "root")
            child = PipelineHandler("child", DemoConfig(base=2), Path(temp_dir) / "child")
            root.add_child_pipeline(child, 1)
            root.set_value("shared", 1)
            child.set_value("shared", 2)
            inventory = discover_owned_variable_slots(root, "shared")
            warning = MagicMock()
            root.logger.warning = warning
            before = _snapshot_pipeline_state([root, child], "shared")

            confirmation = confirm_recovery_impact(
                inventory,
                root.logger,
                input_func=lambda prompt: (_ for _ in ()).throw(EOFError),
            )
            after = _snapshot_pipeline_state([root, child], "shared")

            self.assertFalse(confirmation.authorized)
            self.assertTrue(confirmation.prompted)
            self.assertEqual(before, after)
            self.assertEqual(warning.call_count, 1)

    def test_confirmation_propagates_keyboard_interrupt(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = PipelineHandler("root", DemoConfig(base=1), Path(temp_dir) / "root")
            child = PipelineHandler("child", DemoConfig(base=2), Path(temp_dir) / "child")
            root.add_child_pipeline(child, 1)
            root.set_value("shared", 1)
            child.set_value("shared", 2)
            inventory = discover_owned_variable_slots(root, "shared")
            warning = MagicMock()
            root.logger.warning = warning

            with self.assertRaises(KeyboardInterrupt):
                confirm_recovery_impact(
                    inventory,
                    root.logger,
                    input_func=lambda prompt: (_ for _ in ()).throw(KeyboardInterrupt),
                )

            self.assertEqual(warning.call_count, 0)

    def test_confirmation_propagates_system_exit(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = PipelineHandler("root", DemoConfig(base=1), Path(temp_dir) / "root")
            child = PipelineHandler("child", DemoConfig(base=2), Path(temp_dir) / "child")
            root.add_child_pipeline(child, 1)
            root.set_value("shared", 1)
            child.set_value("shared", 2)
            inventory = discover_owned_variable_slots(root, "shared")
            warning = MagicMock()
            root.logger.warning = warning

            with self.assertRaises(SystemExit):
                confirm_recovery_impact(
                    inventory,
                    root.logger,
                    input_func=lambda prompt: (_ for _ in ()).throw(SystemExit(7)),
                )

            self.assertEqual(warning.call_count, 0)


if __name__ == "__main__":
    unittest.main()
