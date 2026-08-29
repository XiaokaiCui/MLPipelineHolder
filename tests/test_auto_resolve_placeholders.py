from __future__ import annotations

import threading
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from mlpipelineholder import PipelineHandler
from mlpipelineholder.exceptions import ResolutionError
from mlpipelineholder.models import RuntimeValueReference


@dataclass
class Holder:
    value: int
    fn: Any = field(default=None)


@dataclass
class PipelineSettings:
    learning_rate: float = 0.01
    batch_size: int = 32


@dataclass
class ConfigWithDerived:
    value: int
    derived: int = field(init=False, default=7)


_CALL_COUNTERS: dict[str, int] = {}


class Lock:
    """Unpicklable stand-in for ``threading.Lock``.

    ``threading.Lock`` is a factory function, not a class, before Python 3.13,
    so it cannot be used as the second argument of ``isinstance`` on 3.11/3.12.
    This class reproduces the behaviour the tests need on every supported
    version: instances are not picklable and not deep-copyable, so the pipeline
    saves them as placeholders and recovers them by re-running their block.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()


def produce_lock() -> Lock:
    return Lock()


def produce_lock_then_raise() -> Lock:
    calls = _CALL_COUNTERS.setdefault("produce_lock_then_raise", 0)
    _CALL_COUNTERS["produce_lock_then_raise"] = calls + 1
    if calls >= 1:
        raise RuntimeError("boom on re-run")
    return Lock()


def consume_lock_and_produce(lock: Any) -> Lock:
    del lock
    return Lock()


def produce_holder() -> Holder:
    return Holder(value=5, fn=lambda x: x)


def produce_local_dataclass() -> Any:
    @dataclass
    class LocalHolder:
        value: int
        fn: Any = field(default=None)

    return LocalHolder(value=7, fn=lambda x: x)


def produce_plain() -> str:
    return "plain-value"


def consume_plain_and_produce_lock(plain: str) -> Lock:
    del plain
    return Lock()


def consume_lock_produce_plain(lock: Any) -> str:
    del lock
    return "plain-value"


def consume_config_and_produce_lock(cfg_name: str) -> Lock:
    del cfg_name
    return Lock()


def declare_x() -> Lock:
    return Lock()


def counting_true_gate() -> bool:
    calls = _CALL_COUNTERS.setdefault("counting_true_gate", 0)
    _CALL_COUNTERS["counting_true_gate"] = calls + 1
    return True


def counting_false_gate() -> bool:
    calls = _CALL_COUNTERS.setdefault("counting_false_gate", 0)
    _CALL_COUNTERS["counting_false_gate"] = calls + 1
    return False


def raising_gate() -> bool:
    raise ValueError("gate boom")


def produce_flag() -> bool:
    return True


def gate_reads_flag(flag: bool) -> bool:
    calls = _CALL_COUNTERS.setdefault("gate_reads_flag", 0)
    _CALL_COUNTERS["gate_reads_flag"] = calls + 1
    return flag


def gate_reads_items(items: list[int]) -> bool:
    calls = _CALL_COUNTERS.setdefault("gate_reads_items", 0)
    _CALL_COUNTERS["gate_reads_items"] = calls + 1
    return bool(items)


def produce_items() -> list[int]:
    return [1]


def clear_items_and_produce_lock(items: list[int]) -> Lock:
    items.clear()
    return Lock()


def declare_x_real() -> str:
    return "real-x"


class AutoResolvePlaceholderTests(unittest.TestCase):
    def _log_text(self, project: Path) -> str:
        return (project / "metadata" / "pipeline.log").read_text(encoding="utf-8")

    def test_placeholder_output_recovered_by_default(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("ar", {}, tmp / "project")
            block = pipeline.add_block("producer", 1)
            block.register_function(produce_lock, ["out"])
            pipeline.run_all()
            pipeline.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(tmp / "bundle", forced_deleting=True)

            self.assertIsInstance(loaded.get_value("out"), Lock)
            self.assertTrue(
                any(
                    record.mode == "auto_resolve_placeholder:producer"
                    for record in loaded.run_history
                )
            )
            self.assertTrue(
                any(
                    record.mode == "auto_resolve_placeholder:producer"
                    and record.status == "success"
                    for record in loaded.run_history
                )
            )

    def test_flag_false_keeps_placeholder(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("ar", {}, tmp / "project")
            block = pipeline.add_block("producer", 1)
            block.register_function(produce_lock, ["out"])
            pipeline.run_all()
            pipeline.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(
                tmp / "bundle",
                forced_deleting=True,
                auto_resolve_placeholders=False,
            )

            with self.assertRaises(ResolutionError):
                loaded.get_value("out")
            self.assertFalse(
                any(
                    record.mode.startswith("auto_resolve_placeholder:")
                    for record in loaded.run_history
                )
            )

    def test_placeholder_constant_input_blocks_recovery(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("ar", {}, tmp / "project")
            pipeline.set_constant_value("bad_const", Lock())
            block = pipeline.add_block("consumer", 1)
            block.register_function(
                consume_lock_and_produce,
                ["out"],
                param_mapping={"lock": "bad_const"},
            )
            pipeline.run_all()
            pipeline.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(
                tmp / "bundle", forced_deleting=True, verbose=True
            )
            log_text = self._log_text(tmp / "project")

            with self.assertRaises(ResolutionError):
                loaded.get_value("out")
            self.assertIn("required input 'bad_const' is a placeholder", log_text)
            self.assertIn("Constant 'bad_const' was saved as a placeholder", log_text)

    def test_unresolvable_input_warns_only_when_verbose(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("ar", {}, tmp / "project")
            block = pipeline.add_block("consumer", 1)
            block.register_function(consume_plain_and_produce_lock, ["out"])
            block.declared_outputs()
            pipeline.producer_outputs["consumer"] = {
                "out": RuntimeValueReference(
                    type_name="Lock",
                    repr_text="lock",
                    reason="test",
                )
            }
            pipeline._rebuild_visible_state({})

            pipeline._auto_resolve_placeholder_outputs(verbose=False)
            self.assertNotIn(
                "cannot be resolved",
                self._log_text(tmp / "project"),
            )

            pipeline.producer_outputs["consumer"] = {
                "out": RuntimeValueReference(
                    type_name="Lock",
                    repr_text="lock",
                    reason="test",
                )
            }
            pipeline._rebuild_visible_state({})
            pipeline._auto_resolve_placeholder_outputs(verbose=True)
            self.assertIn("cannot be resolved", self._log_text(tmp / "project"))
            self.assertIsInstance(
                pipeline.producer_outputs["consumer"]["out"],
                RuntimeValueReference,
            )

    def test_execution_exception_warns_unconditionally(self) -> None:
        _CALL_COUNTERS.pop("produce_lock_then_raise", None)
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("ar", {}, tmp / "project")
            block = pipeline.add_block("producer", 1)
            block.register_function(produce_lock_then_raise, ["out"])
            pipeline.run_all()
            pipeline.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(tmp / "bundle", forced_deleting=True)
            log_text = self._log_text(tmp / "project")

            self.assertIn("not recoverable", log_text)
            self.assertIn("re-running block 'producer' failed", log_text)
            self.assertIn("boom on re-run", log_text)
            with self.assertRaises(ResolutionError):
                loaded.get_value("out")

    def test_gate_off_block_placeholder_skipped_silently(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("ar", {"flag": False}, tmp / "project")
            pipeline.add_gate_block("flag", expected_value=True)
            block = pipeline.add_block("gated", 1)
            block.register_function(declare_x, ["x"])
            pipeline.producer_outputs["gated"] = {
                "x": RuntimeValueReference(
                    type_name="Lock",
                    repr_text="lock",
                    reason="test",
                )
            }
            pipeline._rebuild_visible_state({})

            pipeline._auto_resolve_placeholder_outputs(verbose=True)
            log_text = self._log_text(tmp / "project")

            self.assertNotIn("not recoverable", log_text)
            self.assertIsInstance(
                pipeline.producer_outputs["gated"]["x"],
                RuntimeValueReference,
            )

    def test_gate_off_unique_input_blocks_recovery_with_reason(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            root = PipelineHandler("ar", {"flag": False}, tmp / "project")
            child = PipelineHandler("gated_child", {}, tmp / "child_root")
            child.add_gate_block("flag", expected_value=True)
            gated = child.add_block("gated", 1)
            gated.register_function(declare_x, ["x"])
            consumer = root.add_block("consumer", 3)
            consumer.register_function(
                consume_lock_and_produce,
                ["out"],
                param_mapping={"lock": "x"},
            )
            root.add_child_pipeline(child, execution_priority=2)
            child.producer_outputs["gated"] = {
                "x": RuntimeValueReference(
                    type_name="Lock",
                    repr_text="lock",
                    reason="test",
                )
            }
            child._rebuild_visible_state({})
            root.producer_outputs["consumer"] = {
                "out": RuntimeValueReference(
                    type_name="Lock",
                    repr_text="lock",
                    reason="test",
                )
            }
            root._rebuild_visible_state({})

            root._auto_resolve_placeholder_outputs(verbose=True)
            log_text = self._log_text(tmp / "project")

            self.assertIn("gated off by config", log_text)
            self.assertIn("'gated'", log_text)
            self.assertIsInstance(
                root.producer_outputs["consumer"]["out"],
                RuntimeValueReference,
            )

    def test_gate_off_input_with_same_name_alternative_recovers(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            root = PipelineHandler("ar", {"flag": False}, tmp / "project")
            child = PipelineHandler("gated_child", {}, tmp / "child_root")
            child.add_gate_block("flag", expected_value=True)
            gated = child.add_block("gated", 1)
            gated.register_function(declare_x, ["x"])
            alternative = root.add_block("alternative", 2)
            alternative.register_function(declare_x_real, ["x"])
            consumer = root.add_block("consumer", 3)
            consumer.register_function(
                consume_lock_and_produce,
                ["out"],
                param_mapping={"lock": "x"},
            )
            root.add_child_pipeline(child, execution_priority=1)
            child.producer_outputs["gated"] = {
                "x": RuntimeValueReference(
                    type_name="Lock",
                    repr_text="lock",
                    reason="test",
                )
            }
            child._rebuild_visible_state({})
            root.producer_outputs["alternative"] = {"x": "real-x"}
            root.producer_outputs["consumer"] = {
                "out": RuntimeValueReference(
                    type_name="Lock",
                    repr_text="lock",
                    reason="test",
                )
            }
            root._rebuild_visible_state({})

            root._auto_resolve_placeholder_outputs(verbose=True)
            log_text = self._log_text(tmp / "project")

            self.assertNotIn("gated off by config", log_text)
            self.assertIsInstance(
                root.producer_outputs["consumer"]["out"],
                Lock,
            )

    def test_downstream_values_untouched_after_recovery(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("ar", {}, tmp / "project")
            producer = pipeline.add_block("producer", 1)
            producer.register_function(produce_lock, ["lock_out"])
            downstream = pipeline.add_block("downstream", 2)
            downstream.register_function(
                consume_lock_produce_plain,
                ["plain_out"],
                param_mapping={"lock": "lock_out"},
            )
            pipeline.run_all()
            pipeline.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(tmp / "bundle", forced_deleting=True)

            self.assertIsInstance(loaded.get_value("lock_out"), Lock)
            self.assertEqual(loaded.get_value("plain_out"), "plain-value")
            recovery_modes = [
                record.mode
                for record in loaded.run_history
                if record.mode.startswith("auto_resolve_placeholder:")
            ]
            self.assertEqual(recovery_modes, ["auto_resolve_placeholder:producer"])

    def test_dataclass_placeholder_reconstructed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("ar", {}, tmp / "project")
            block = pipeline.add_block("producer", 1)
            block.register_function(produce_holder, ["out"])
            pipeline.run_all()
            pipeline.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(tmp / "bundle", forced_deleting=True)

            value = loaded.get_value("out")
            self.assertIsInstance(value, Holder)
            self.assertEqual(value.value, 5)

    def test_dataclass_unimportable_falls_back_to_namespace(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("ar", {}, tmp / "project")
            block = pipeline.add_block("producer", 1)
            block.register_function(produce_local_dataclass, ["out"])
            pipeline.run_all()
            pipeline.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(
                tmp / "bundle", forced_deleting=True, verbose=True
            )
            log_text = self._log_text(tmp / "project")

            value = loaded.get_value("out")
            self.assertIsInstance(value, SimpleNamespace)
            self.assertEqual(value.value, 7)
            self.assertIn("could not be reconstructed", log_text)

    def test_config_dataclass_reconstructed_regardless_of_flag(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler(
                "ar", PipelineSettings(learning_rate=0.5), tmp / "project"
            )
            pipeline.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(
                tmp / "bundle",
                forced_deleting=True,
                auto_resolve_placeholders=False,
            )

            self.assertIsInstance(loaded.config, PipelineSettings)
            self.assertEqual(loaded.get_config_value("learning_rate"), 0.5)
            self.assertEqual(loaded.get_config_value("batch_size"), 32)

    def test_unregistered_producer_placeholder_warns_verbose(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("ar", {}, tmp / "project")
            pipeline.para_value_dict["orphan"] = RuntimeValueReference(
                type_name="Lock",
                repr_text="lock",
                reason="test",
            )

            pipeline._auto_resolve_placeholder_outputs(verbose=True)
            log_text = self._log_text(tmp / "project")

            self.assertIn("producing block is not registered", log_text)

    def test_dataclass_reconstruction_preserves_mirror_identity(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            root = PipelineHandler("ar", {}, tmp / "project")
            child = PipelineHandler("child", {}, tmp / "child")
            block = child.add_block("producer", 1)
            block.register_function(produce_holder, ["out"])
            root.add_child_pipeline(child, 1)
            root.run_all()
            root.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(tmp / "bundle", forced_deleting=True)
            loaded_child = loaded.get_child_pipeline("child")

            self.assertIs(
                loaded_child.producer_outputs["producer"]["out"],
                loaded_child.para_value_dict["out"],
            )
            self.assertIs(
                loaded_child.para_value_dict["out"],
                loaded.producer_outputs["child"]["out"],
            )
            self.assertIs(
                loaded.producer_outputs["child"]["out"],
                loaded.para_value_dict["out"],
            )

            loaded_child.update_value("out", Holder(value=99), copy=False)
            self.assertEqual(loaded_child.get_value("out").value, 99)
            self.assertEqual(loaded.get_value("out").value, 99)

    def test_legacy_runtime_reference_dataclass_reconstructed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("ar", {}, tmp / "project")
            pipeline.para_value_dict["settings"] = RuntimeValueReference(
                type_name="PipelineSettings",
                repr_text="PipelineSettings(learning_rate=0.01, batch_size=32)",
                reason="legacy save without dataclass metadata",
            )
            pipeline.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(
                tmp / "bundle",
                forced_deleting=True,
                auto_resolve_placeholders=False,
            )

            value = loaded.get_value("settings")
            self.assertIsInstance(value, PipelineSettings)
            self.assertEqual(value.learning_rate, 0.01)
            self.assertEqual(value.batch_size, 32)

    def test_legacy_runtime_reference_unconstructible_stays_placeholder(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("ar", {}, tmp / "project")
            pipeline.para_value_dict["holder"] = RuntimeValueReference(
                type_name="Holder",
                repr_text="Holder(value=5, fn=<lambda>)",
                reason="legacy save without dataclass metadata",
            )
            pipeline.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(
                tmp / "bundle",
                forced_deleting=True,
                auto_resolve_placeholders=False,
                verbose=True,
            )
            log_text = self._log_text(tmp / "project")

            self.assertIsInstance(loaded.para_value_dict["holder"], RuntimeValueReference)
            self.assertIn("could not be reconstructed", log_text)
            with self.assertRaises(ResolutionError):
                loaded.get_value("holder")

    def test_flag_false_warns_placeholder_when_verbose(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("ar", {}, tmp / "project")
            block = pipeline.add_block("producer", 1)
            block.register_function(produce_lock, ["out"])
            pipeline.run_all()
            pipeline.save_pipeline(tmp / "bundle")

            PipelineHandler.load_pipeline(
                tmp / "bundle",
                forced_deleting=True,
                auto_resolve_placeholders=False,
                verbose=True,
            )
            self.assertIn(
                "rather than a real value",
                self._log_text(tmp / "project"),
            )

            PipelineHandler.load_pipeline(
                tmp / "bundle",
                forced_deleting=True,
                auto_resolve_placeholders=False,
            )
            self.assertNotIn(
                "rather than a real value",
                self._log_text(tmp / "project"),
            )

    def test_gate_config_none_passes_recovery(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            root = PipelineHandler("ar", {"flag": None}, tmp / "project")
            child = PipelineHandler("child", {}, tmp / "child")
            child.add_gate_block("flag", expected_value=True)
            block = child.add_block("producer", 1)
            block.register_function(produce_lock, ["out"])
            root.add_child_pipeline(child, 1)
            child.producer_outputs["producer"] = {
                "out": RuntimeValueReference(
                    type_name="Lock",
                    repr_text="lock",
                    reason="test",
                )
            }
            child._rebuild_visible_state({})
            root.producer_outputs["child"] = {
                "out": child.para_value_dict["out"]
            }
            root._rebuild_visible_state({})
            root.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(tmp / "bundle", forced_deleting=True)

            self.assertIsInstance(loaded.get_value("out"), Lock)

    def test_gate_evaluation_error_warns_verbose_and_keeps_placeholder(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            root = PipelineHandler("ar", {}, tmp / "project")
            child = PipelineHandler("child", {}, tmp / "child")
            child.add_gate_block("missing_gate_config", expected_value=True)
            block = child.add_block("producer", 1)
            block.register_function(produce_lock, ["out"])
            root.add_child_pipeline(child, 1)
            child.producer_outputs["producer"] = {
                "out": RuntimeValueReference(
                    type_name="Lock",
                    repr_text="lock",
                    reason="test",
                )
            }
            child._rebuild_visible_state({})
            root.producer_outputs["child"] = {
                "out": child.para_value_dict["out"]
            }
            root._rebuild_visible_state({})
            root.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(
                tmp / "bundle",
                forced_deleting=True,
                verbose=True,
            )
            log_text = self._log_text(tmp / "project")

            self.assertIn("gate could not be evaluated", log_text)
            self.assertIsInstance(
                loaded.get_child_pipeline("child").producer_outputs["producer"]["out"],
                RuntimeValueReference,
            )

    def test_auto_resolve_grandchild_placeholder_synchronizes_root_mirror(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            root = PipelineHandler("ar", {}, tmp / "project")
            parent = PipelineHandler("parent", {}, tmp / "parent")
            leaf = PipelineHandler("leaf", {}, tmp / "leaf")
            leaf.add_block("producer", 1).register_function(produce_lock, ["out"])
            parent.add_child_pipeline(leaf, 1)
            root.add_child_pipeline(parent, 1)
            root.run_all()
            root.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(
                tmp / "bundle", forced_deleting=True
            )
            loaded_parent = loaded.get_child_pipeline("parent")
            loaded_leaf = loaded_parent.get_child_pipeline("leaf")

            value = loaded_leaf.producer_outputs["producer"]["out"]
            self.assertIsInstance(value, Lock)
            self.assertIs(loaded_parent.producer_outputs["leaf"]["out"], value)
            self.assertIs(loaded_parent.para_value_dict["out"], value)
            self.assertIs(loaded.producer_outputs["parent"]["out"], value)
            self.assertIs(loaded.para_value_dict["out"], value)
            self.assertIs(loaded.get_value("out"), value)

    def test_auto_resolve_ancestor_sync_preserves_intermediate_override(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            root = PipelineHandler("ar", {}, tmp / "project")
            parent = PipelineHandler("parent", {}, tmp / "parent")
            leaf = PipelineHandler("leaf", {}, tmp / "leaf")
            leaf.add_block("producer", 1).register_function(produce_lock, ["out"])
            parent.add_child_pipeline(leaf, 1)
            parent.add_block("later", 2).register_function(produce_plain, ["out"])
            root.add_child_pipeline(parent, 1)
            root.run_all()
            root.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(
                tmp / "bundle", forced_deleting=True
            )
            loaded_parent = loaded.get_child_pipeline("parent")
            loaded_leaf = loaded_parent.get_child_pipeline("leaf")

            self.assertIsInstance(
                loaded_leaf.producer_outputs["producer"]["out"], Lock
            )
            self.assertEqual(loaded_parent.para_value_dict["out"], "plain-value")
            self.assertEqual(
                loaded.producer_outputs["parent"]["out"], "plain-value"
            )
            self.assertEqual(loaded.get_value("out"), "plain-value")

    def test_gate_error_without_placeholders_does_not_warn(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler("ar", {}, tmp / "project")
            pipeline.add_gate_block("missing_gate_config", expected_value=True)
            pipeline.add_block("producer", 1).register_function(
                produce_plain, ["out"]
            )
            warning = MagicMock()
            pipeline.logger.warning = warning

            pipeline._auto_resolve_placeholder_outputs(verbose=True)

            warning.assert_not_called()

    def test_callable_gate_false_skips_child_recovery_silently(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            root = PipelineHandler("ar", {}, tmp / "project")
            child = PipelineHandler("gated_child", {}, tmp / "child_root")
            child.set_gate_block(counting_false_gate)
            gated = child.add_block("gated", 1)
            gated.register_function(declare_x, ["x"])
            root.add_child_pipeline(child, execution_priority=1)
            child.producer_outputs["gated"] = {
                "x": RuntimeValueReference(
                    type_name="Lock",
                    repr_text="lock",
                    reason="test",
                )
            }
            child._rebuild_visible_state({})
            root._rebuild_visible_state({})

            root._auto_resolve_placeholder_outputs(verbose=True)
            log_text = self._log_text(tmp / "project")

            self.assertNotIn("not recoverable", log_text)
            self.assertIsInstance(
                child.producer_outputs["gated"]["x"],
                RuntimeValueReference,
            )

    def test_callable_gate_error_blocks_recovery_with_warning(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            root = PipelineHandler("ar", {}, tmp / "project")
            child = PipelineHandler("gated_child", {}, tmp / "child_root")
            child.set_gate_block(raising_gate)
            gated = child.add_block("gated", 1)
            gated.register_function(declare_x, ["x"])
            root.add_child_pipeline(child, execution_priority=1)
            child.producer_outputs["gated"] = {
                "x": RuntimeValueReference(
                    type_name="Lock",
                    repr_text="lock",
                    reason="test",
                )
            }
            child._rebuild_visible_state({})
            root._rebuild_visible_state({})

            root._auto_resolve_placeholder_outputs(verbose=True)
            log_text = self._log_text(tmp / "project")

            self.assertIn("gate could not be evaluated", log_text)
            self.assertIsInstance(
                child.producer_outputs["gated"]["x"],
                RuntimeValueReference,
            )

    def test_callable_gate_recovery_recovers_multiple_gated_children(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            _CALL_COUNTERS.pop("counting_true_gate", None)
            root = PipelineHandler("ar", {}, tmp / "project")
            parent = PipelineHandler("parent", {}, tmp / "parent")
            parent.set_gate_block(counting_true_gate)
            for leaf_name, priority in (("leaf_a", 1), ("leaf_b", 2)):
                leaf = PipelineHandler(leaf_name, {}, tmp / leaf_name)
                leaf.add_block("producer", 1).register_function(produce_lock, ["out"])
                parent.add_child_pipeline(leaf, priority)
                leaf.producer_outputs["producer"] = {
                    "out": RuntimeValueReference(
                        type_name="Lock",
                        repr_text="lock",
                        reason="test",
                    )
                }
                leaf._rebuild_visible_state({})
            root.add_child_pipeline(parent, 1)

            root._auto_resolve_placeholder_outputs(verbose=True)

            recovered_parent = root.get_child_pipeline("parent")
            for leaf_name in ("leaf_a", "leaf_b"):
                leaf = recovered_parent.get_child_pipeline(leaf_name)
                self.assertIsInstance(leaf.producer_outputs["producer"]["out"], Lock)
            self.assertEqual(_CALL_COUNTERS.get("counting_true_gate", 0), 1)

    def test_gate_reevaluates_when_upstream_input_changes_between_passes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            _CALL_COUNTERS.pop("gate_reads_flag", None)
            root = PipelineHandler("ar", {}, tmp / "project")
            flag_block = root.add_block("flag_block", 1)
            flag_block.register_function(produce_flag, ["flag"])
            child = PipelineHandler("child", {}, tmp / "child")
            child.set_gate_block(gate_reads_flag)
            producer = child.add_block("producer", 1)
            producer.register_function(produce_lock, ["out"])
            root.add_child_pipeline(child, 2)
            child.producer_outputs["producer"] = {
                "out": RuntimeValueReference(
                    type_name="Lock",
                    repr_text="lock",
                    reason="test",
                )
            }
            child._rebuild_visible_state({})
            root.producer_outputs["flag_block"] = {"flag": True}
            root._rebuild_visible_state({})

            root._auto_resolve_placeholder_outputs(verbose=True)
            self.assertIsInstance(child.producer_outputs["producer"]["out"], Lock)
            self.assertEqual(_CALL_COUNTERS["gate_reads_flag"], 1)

            root.producer_outputs["flag_block"] = {"flag": False}
            root._rebuild_visible_state({})
            child.producer_outputs["producer"] = {
                "out": RuntimeValueReference(
                    type_name="Lock",
                    repr_text="lock",
                    reason="test",
                )
            }
            child._rebuild_visible_state({})

            root._auto_resolve_placeholder_outputs(verbose=True)
            self.assertEqual(_CALL_COUNTERS["gate_reads_flag"], 2)
            self.assertIsInstance(
                child.producer_outputs["producer"]["out"],
                RuntimeValueReference,
            )

    def test_gate_reevaluates_after_in_place_input_mutation_during_recovery(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            _CALL_COUNTERS.pop("gate_reads_items", None)
            root = PipelineHandler("ar", {}, tmp / "project")
            items_block = root.add_block("items_block", 1)
            assert items_block is not None
            items_block.register_function(produce_items, ["items"])
            parent = PipelineHandler("parent", {}, tmp / "parent")
            parent.set_gate_block(gate_reads_items)

            first = PipelineHandler("first", {}, tmp / "first")
            first_block = first.add_block("producer", 1)
            assert first_block is not None
            first_block.register_function(
                clear_items_and_produce_lock,
                ["first_out"],
            )
            second = PipelineHandler("second", {}, tmp / "second")
            second_block = second.add_block("producer", 1)
            assert second_block is not None
            second_block.register_function(
                produce_lock,
                ["second_out"],
            )
            parent.add_child_pipeline(first, 1)
            parent.add_child_pipeline(second, 2)
            root.add_child_pipeline(parent, 2)

            first.producer_outputs["producer"] = {
                "first_out": RuntimeValueReference(
                    type_name="Lock",
                    repr_text="lock",
                    reason="test",
                )
            }
            first._rebuild_visible_state({})
            second.producer_outputs["producer"] = {
                "second_out": RuntimeValueReference(
                    type_name="Lock",
                    repr_text="lock",
                    reason="test",
                )
            }
            second._rebuild_visible_state({})
            root.producer_outputs["items_block"] = {"items": [1]}
            root._rebuild_visible_state({})

            root._auto_resolve_placeholder_outputs(verbose=True)

            self.assertIsInstance(first.get_value("first_out"), Lock)
            self.assertEqual(root.get_value("items"), [])
            self.assertEqual(_CALL_COUNTERS["gate_reads_items"], 2)
            self.assertIsInstance(
                second.producer_outputs["producer"]["second_out"],
                RuntimeValueReference,
            )

    def test_dataclass_with_init_false_field_reconstructs_real_class(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            pipeline = PipelineHandler(
                "ar", ConfigWithDerived(value=3), tmp / "project"
            )
            pipeline.save_pipeline(tmp / "bundle")

            loaded = PipelineHandler.load_pipeline(
                tmp / "bundle",
                forced_deleting=True,
                auto_resolve_placeholders=False,
            )

            self.assertIsInstance(loaded.config, ConfigWithDerived)
            self.assertEqual(loaded.get_config_value("value"), 3)
            self.assertEqual(loaded.get_config_value("derived"), 7)


if __name__ == "__main__":
    unittest.main()
