from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from mlpipelineholder import PipelineHandler, RegistrationError, ResolutionError
from mlpipelineholder.persistence.artifacts.store import ArtifactStore


def seed() -> int:
    return 3


def increment(value: int) -> int:
    return value + 1


def increment_optional(value: int = 0) -> int:
    return value + 1


def increment_pair(value: int, counter: int = 0) -> tuple[int, int]:
    return value + 1, counter + 1


class SelfOutputExclusionTests(unittest.TestCase):
    def test_frozenset_exclusion_is_accepted(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHandler("root", {}, Path(tmp) / "root")
            root.add_block("upstream", 1).register_function(seed, ["value"])
            block = root.add_block("target", 2, exclude_self_outputs=frozenset({"value"}))
            block.register_function(increment, ["value"], param_mapping={"value": "value"})
            root.run_all()
            root.run_block("target")
            self.assertEqual(root.get_value("value"), 4)

    def test_targeted_reruns_use_upstream_instead_of_previous_self_output(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHandler("root", {}, Path(tmp) / "root")
            root.add_block("upstream", 1).register_function(seed, ["value"])
            block = root.add_block("target", 2, exclude_self_outputs={"value"})
            block.register_function(increment, ["value"], param_mapping={"value": "value"})
            root.run_all()
            self.assertEqual(root.get_value("value"), 4)
            self.assertEqual(block.exclude_self_outputs, frozenset({"value"}))

            resolved = root.inspect_node(node_name="target", resolve_only=True)
            self.assertEqual(resolved.arguments["value"], 3)
            self.assertFalse(resolved.sources["value"].same_node_previous_output)
            self.assertEqual(block.inspect(), 4)
            root.run_block("target")
            self.assertEqual(root.get_value("value"), 4)
            root.run_from("target")
            self.assertEqual(root.get_value("value"), 4)
            root.run_block("target", overrides={"value": 10})
            self.assertEqual(root.get_value("value"), 11)
            self.assertEqual(block.inspect(overrides={"value": 20}), 21)
            root.run_block("target")
            self.assertEqual(root.get_value("value"), 4)

    def test_strict_mode_uses_explicit_upstream_mapping(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHandler("root", {}, Path(tmp) / "root", strict_mode=True)
            root.add_block("upstream", 1).register_function(seed, ["value"])
            block = root.add_block("target", 2, exclude_self_outputs=("value",))
            block.register_function(increment, ["value"], param_mapping={"value": "value"})
            root.run_all()
            root.run_block("target")
            self.assertEqual(root.get_value("value"), 4)
            self.assertFalse(block.inspect(resolve_only=True).sources["value"].same_node_previous_output)

    def test_exclusion_is_per_name_and_other_self_outputs_remain_available(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHandler("root", {}, Path(tmp) / "root")
            root.add_block("upstream", 1).register_function(seed, ["value"])
            block = root.add_block("target", 2, exclude_self_outputs=["value"])
            block.register_function(increment_pair, ["value", "counter"], param_mapping={"value": "value"})
            root.run_all()
            self.assertEqual((root.get_value("value"), root.get_value("counter")), (4, 1))
            resolved = block.inspect(resolve_only=True)
            self.assertEqual(resolved.arguments, {"value": 3, "counter": 1})
            self.assertFalse(resolved.sources["value"].same_node_previous_output)
            self.assertTrue(resolved.sources["counter"].same_node_previous_output)
            root.run_block("target")
            self.assertEqual((root.get_value("value"), root.get_value("counter")), (4, 2))

    def test_default_behavior_and_saved_block_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHandler("root", {}, Path(tmp) / "root")
            root.add_block("upstream", 1).register_function(seed, ["value"])
            block = root.add_block("target", 2)
            block.register_function(increment, ["value"], param_mapping={"value": "value"})
            root.run_all()
            root.run_block("target")
            self.assertEqual(root.get_value("value"), 5)
            self.assertTrue(block.inspect(resolve_only=True).sources["value"].same_node_previous_output)

            replaced = root.add_block("target", 2, exclude_self_outputs=["value"])
            replaced.register_function(increment, ["value"], param_mapping={"value": "value"})
            root.run_block("target")
            self.assertEqual(root.get_value("value"), 4)
            root.save_pipeline(Path(tmp) / "bundle")
            loaded = PipelineHandler.load_pipeline(Path(tmp) / "bundle", forced_deleting=True, trust_project=True)
            self.assertEqual(loaded.get_block("target").exclude_self_outputs, frozenset({"value"}))
            loaded.run_block("target")
            self.assertEqual(loaded.get_value("value"), 4)

    def test_missing_upstream_uses_normal_fallback_or_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHandler("root", {}, Path(tmp) / "root")
            optional = root.add_block("optional", 1, exclude_self_outputs=["value"])
            optional.register_function(increment_optional, ["value"])
            root.run_all()
            root.run_block("optional")
            self.assertEqual(root.get_value("value"), 1)
            self.assertEqual(optional.inspect(resolve_only=True).arguments["value"], 0)

        with TemporaryDirectory() as tmp:
            root = PipelineHandler("root", {}, Path(tmp) / "root")
            required = root.add_block("required", 1, exclude_self_outputs=["value"])
            required.register_function(increment, ["value"])
            root.run_all(overrides={"value": 3})
            with self.assertRaises(ResolutionError):
                root.run_block("required")
            with self.assertRaises(ResolutionError):
                root.get_value("value")
            self.assertEqual(required.inspect(overrides={"value": 5}), 6)

    def test_expression_uses_upstream_value_after_targeted_rerun(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHandler("root", {}, Path(tmp) / "root")
            root.add_block("upstream", 1).register_function(seed, ["value"])
            block = root.add_block("expression", 2, exclude_self_outputs={"value"})
            block.register_expression("value = value + 1")
            root.run_all()
            root.run_block("expression")
            self.assertEqual(root.get_value("value"), 4)
            self.assertFalse(block.inspect(resolve_only=True).sources["value"].same_node_previous_output)

    def test_excluded_disk_output_is_not_materialized_for_a_rerun(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHandler("root", {}, Path(tmp) / "root")
            root.add_block("upstream", 1).register_function(seed, ["value"])
            block = root.add_block("target", 2, exclude_self_outputs=["value"])
            block.register_function(increment, ["value"], save_to_disk=["value"], param_mapping={"value": "value"})
            root.run_all()
            with patch.object(ArtifactStore, "load", side_effect=AssertionError("loaded old output")):
                root.run_block("target")
            self.assertEqual(block.inspect(resolve_only=True).arguments["value"], 3)

    def test_atom_recreation_and_saved_blocks_preserve_exclusion(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHandler("root", {}, Path(tmp) / "root")
            root.add_block("upstream", 1).register_function(seed, ["value"])
            root.create_atom_child_pipeline(
                "atom", 2, increment, output_variable_names="value",
                param_mapping={"value": "value"}, exclude_self_outputs=["value"],
            )
            atom = root.get_child_pipeline("atom")
            root.run_all()
            self.assertEqual(root.get_value("value"), 4)
            self.assertFalse(root.inspect_node(node_name="atom", resolve_only=True).sources["value"].same_node_previous_output)
            root.run_block("atom")
            self.assertEqual(root.get_value("value"), 4)
            root.create_atom_child_pipeline(
                "atom", 2, increment, output_variable_names="value",
                param_mapping={"value": "value"}, exclude_self_outputs=["value"],
            )
            self.assertIs(root.get_child_pipeline("atom"), atom)
            root.save_pipeline(Path(tmp) / "bundle")
            loaded = PipelineHandler.load_pipeline(Path(tmp) / "bundle", forced_deleting=True, trust_project=True)
            self.assertEqual(loaded.get_child_pipeline("atom").nodes[0].exclude_self_outputs, frozenset({"value"}))
            loaded.run_block("atom")
            self.assertEqual(loaded.get_value("value"), 4)
            loaded_atom = loaded.get_child_pipeline("atom")
            loaded.create_atom_child_pipeline(
                "atom", 2, increment, output_variable_names="value",
                param_mapping={"value": "value"}, exclude_self_outputs=[],
            )
            self.assertIsNot(loaded.get_child_pipeline("atom"), loaded_atom)
            self.assertEqual(loaded.get_child_pipeline("atom").nodes[0].exclude_self_outputs, frozenset())

    def test_invalid_names_do_not_replace_existing_block(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHandler("root", {}, Path(tmp) / "root")
            block = root.add_block("target", 1)
            for invalid in ("value", [""], ["_"], [1]):
                with self.subTest(invalid=invalid), self.assertRaises(RegistrationError):
                    root.add_block("target", 1, exclude_self_outputs=invalid)  # pyright: ignore[reportArgumentType]
                self.assertIs(root.get_block("target"), block)
