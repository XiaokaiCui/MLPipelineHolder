from __future__ import annotations

import sys
import threading
import unittest
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import pandas as pd

from mlpipelineholder import (
    ExecutionError,
    InspectionCopyError,
    InspectionMemoryError,
    PipelineHandler,
    RegistrationError,
    ResolutionError,
    ResolvedInspectionCall,
)
from mlpipelineholder.core.models import ArtifactRecord
from mlpipelineholder.execution.inspection import (
    InspectionCopyCandidate,
    _InspectionMemoryEstimator,
    preflight_protected_inspection,
)
from mlpipelineholder.integrations.optuna.support import OPTUNA_STUDY_SERIALIZER
from mlpipelineholder.persistence.artifacts.store import ArtifactStore
from mlpipelineholder.state.output_pointers import (
    OutputAddress,
    OutputPointer,
    PointerResolutionError,
)


def mutate_items(items: list[object]) -> list[object]:
    items.append("inspected")
    return items


def select_data(data: object) -> object:
    return data


def rolling_value(value: int = 0) -> int:
    return value + 1


def choose_threshold(threshold: float = 0.5) -> float:
    return threshold


def collect_values(
    prefix: str = "default",
    *values: int,
    **named: int,
) -> tuple[str, tuple[int, ...], dict[str, int]]:
    return prefix, values, named


def produce_shared() -> int:
    return 1


def consume_shared(shared: int) -> int:
    return shared + 1


def arbitrary_raw_result():
    return "raw"


def return_logger(logger: object) -> object:
    return logger


def use_callback(callback: object) -> object:
    return callback


def mutate_default(options: list[str] = []) -> list[str]:
    options.append("inspection")
    return options


def print_value(value: int) -> int:
    print(f"inspection-value={value}")
    return value


INVOCATIONS: list[str] = []


def record_invocation(value: int) -> int:
    INVOCATIONS.append("called")
    return value


class Uncopyable:
    def __init__(self) -> None:
        self.lock = threading.Lock()


def return_uncopyable(value: object) -> object:
    return value


def combine_arrays(left: object, right: object) -> int:
    return 0


def mapped_default(a: int, b: int = 5) -> int:
    return a


def invoke_callable(callback: object, value: int) -> int:
    return callback(value)  # type: ignore[operator]


def variadic_target(a: int, *args: int, **kwargs: int) -> int:
    return a


def log_and_return(value: int, logger: object) -> int:
    logger.info(f"inspection-logger={value}")
    return value


class CallableCounter:
    def __init__(self, start: int = 0) -> None:
        self.count = start

    def __call__(self, value: int) -> int:
        self.count += 1
        return value + self.count


class UncopyableCallable:
    def __init__(self) -> None:
        self.lock = threading.Lock()

    def __call__(self, value: int) -> int:
        return value


class MemoryErrorOnCopy:
    def __deepcopy__(self, memo: object) -> object:
        raise MemoryError("simulated allocation failure")


def return_object(value: object) -> object:
    return value


def sum_variadic_items(*items: int) -> int:
    return sum(items)


def sum_variadic_named(**items: int) -> int:
    return sum(items.values())


def fail_with_memory_error() -> None:
    raise MemoryError("simulated execution allocation failure")


class LargeBoundMethodOwner:
    def __init__(self, size: int) -> None:
        self.data: np.ndarray = np.zeros(size, dtype=np.float64)

    def read(self) -> int:
        return len(self.data)


def invoke_without_arguments(callback: Callable[[], object]) -> object:
    return callback()


class ExecutionInspectionTests(unittest.TestCase):
    def test_block_inspect_returns_raw_result_without_committing_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root")
            block = pipeline.add_block("raw", 1)
            block.register_function(
                arbitrary_raw_result,
                ["first", "second"],
                save_to_disk=["first"],
            )
            before_history = list(pipeline.run_history)

            result = block.inspect()

            self.assertEqual(result, "raw")
            self.assertEqual(pipeline.producer_outputs, {})
            self.assertEqual(pipeline.para_value_dict, {})
            self.assertEqual(pipeline.artifact_registry, {})
            self.assertEqual(pipeline.run_history, before_history)
            self.assertFalse(any(pipeline.artifact_store.artifact_root.rglob("*")))

    def test_protected_inspection_copies_pipeline_value_and_preserves_aliases(self) -> None:
        with TemporaryDirectory() as tmp:
            shared: list[object] = [1]
            pipeline = PipelineHandler(
                "root",
                {"left": shared, "right": shared},
                Path(tmp) / "root",
            )

            def compare_and_mutate(left: list[object], right: list[object]) -> bool:
                left.append(2)
                return left is right

            block = pipeline.add_block("compare", 1)
            block.register_function(compare_and_mutate, ["same"])

            self.assertTrue(block.inspect(allow_mutable_objects=False))
            self.assertEqual(pipeline.get_config("left"), [1])
            self.assertIs(pipeline.get_config("left"), pipeline.get_config("right"))

    def test_protected_inspection_preserves_cycles(self) -> None:
        with TemporaryDirectory() as tmp:
            cyclic: list[object] = []
            cyclic.append(cyclic)
            pipeline = PipelineHandler(
                "root",
                {"data": cyclic},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("cycle", 1)
            block.register_function(select_data, ["result"])

            result = block.inspect(allow_mutable_objects=False)

            self.assertIs(result[0], result)
            self.assertIsNot(result, pipeline.get_config("data"))

    def test_override_is_not_copied_and_shared_mode_uses_pipeline_object(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline_value: list[object] = [1]
            pipeline = PipelineHandler(
                "root",
                {"items": pipeline_value},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("mutate", 1)
            block.register_function(mutate_items, ["result"])

            override: list[object] = [10]
            returned = block.inspect(overrides={"items": override})
            self.assertIs(returned, override)
            self.assertEqual(override, [10, "inspected"])

            shared_returned = block.inspect(allow_mutable_objects=True)
            self.assertIs(shared_returned, pipeline.get_config("items"))
            self.assertEqual(pipeline.get_config("items"), [1, "inspected"])

    def test_uncopyable_pipeline_value_fails_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            value = Uncopyable()
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root")
            pipeline.set_constant_value("value", value, copy=False)
            block = pipeline.add_block("use", 1)
            block.register_function(return_uncopyable, ["result"])

            with self.assertRaisesRegex(InspectionCopyError, "Cannot isolate parameter 'value'"):
                block.inspect(allow_mutable_objects=False)
            self.assertIs(
                block.inspect(allow_mutable_objects=True),
                pipeline.get_constant_value("value"),
            )
            replacement = Uncopyable()
            self.assertIs(block.inspect(overrides={"value": replacement}), replacement)

    def test_functions_and_real_logger_are_identity_passthrough(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "root",
                {"callback": mutate_items},
                Path(tmp) / "root",
            )
            callback_block = pipeline.add_block("callback", 1)
            callback_block.register_function(use_callback, ["result"])
            logger_block = pipeline.add_block("logger", 2)
            logger_block.register_function(return_logger, ["logger_result"])

            callback_resolved = callback_block.inspect(
                resolve_only=True,
                allow_mutable_objects=False,
            )
            logger_resolved = logger_block.inspect(
                resolve_only=True,
                allow_mutable_objects=False,
            )

            self.assertIs(callback_resolved.arguments["callback"], mutate_items)
            self.assertTrue(callback_resolved.sources["callback"].identity_passthrough)
            self.assertIs(logger_resolved.arguments["logger"], pipeline.logger)
            self.assertTrue(logger_resolved.sources["logger"].identity_passthrough)
            custom_logger = object()
            self.assertIs(
                logger_block.inspect(overrides={"logger": custom_logger}),
                custom_logger,
            )

    def test_mutable_callable_default_is_copied(self) -> None:
        with TemporaryDirectory() as tmp:
            original_default = mutate_default.__defaults__[0]
            original_default.clear()
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root", strict_mode=True)
            block = pipeline.add_block("default", 1)
            block.register_function(mutate_default, ["result"])

            self.assertEqual(
                block.inspect(allow_mutable_objects=False),
                ["inspection"],
            )
            self.assertEqual(original_default, [])
            resolved = block.inspect(
                resolve_only=True,
                allow_mutable_objects=False,
            )
            self.assertTrue(resolved.sources["options"].copied)

    def test_previous_same_node_output_matches_run_block_input_visibility(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root")
            block = pipeline.add_block("rolling", 1)
            block.register_function(rolling_value, ["value"])
            pipeline.run_all()

            resolved = block.inspect(resolve_only=True)

            self.assertEqual(resolved.arguments["value"], 1)
            self.assertTrue(resolved.sources["value"].same_node_previous_output)
            self.assertEqual(block.inspect(), 2)
            self.assertEqual(pipeline.get_value("value"), 1)

    def test_previous_disk_output_is_materialized_without_writing_an_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root")
            block = pipeline.add_block("rolling", 1)
            block.register_function(
                rolling_value,
                ["value"],
                save_to_disk=["value"],
            )
            pipeline.run_all()
            artifact_paths = set(pipeline.artifact_store.artifact_root.rglob("*"))

            resolved = block.inspect(resolve_only=True)

            self.assertEqual(resolved.arguments["value"], 1)
            self.assertTrue(resolved.sources["value"].materialized)
            self.assertEqual(block.inspect(), 2)
            self.assertEqual(
                set(pipeline.artifact_store.artifact_root.rglob("*")),
                artifact_paths,
            )

    def test_strict_mode_can_be_temporarily_overridden(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "root",
                {"threshold": 0.8},
                Path(tmp) / "root",
                strict_mode=True,
            )
            block = pipeline.add_block("choose", 1)
            block.register_function(choose_threshold, ["result"])

            self.assertEqual(block.inspect(), 0.5)
            self.assertEqual(block.inspect(strict_mode=False), 0.8)
            self.assertTrue(pipeline.strict_mode)
            self.assertEqual(
                block.inspect(overrides={"threshold": 0.2}, strict_mode=True),
                0.2,
            )

    def test_resolve_only_preserves_variadic_call_shape_and_sources(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "root",
                {"first": 1, "second": 2, "bonus": 3},
                Path(tmp) / "root",
                strict_mode=True,
            )
            block = pipeline.add_block("collect", 1)
            block.register_args("args", ["first", "second"])
            block.register_kwargs("kwargs", {"bonus": "bonus"})
            block.register_function(
                collect_values,
                ["prefix", "values", "named"],
                var_pos_name="args",
                var_kw_name="kwargs",
            )

            resolved = block.inspect(resolve_only=True)

            self.assertIsInstance(resolved, ResolvedInspectionCall)
            self.assertEqual(resolved.args, ("default", 1, 2))
            self.assertNotIn("prefix", resolved.kwargs)
            self.assertEqual(resolved.kwargs["bonus"], 3)
            self.assertEqual(resolved.arguments["values"], (1, 2))
            self.assertEqual(resolved.arguments["named"], {"bonus": 3})
            self.assertEqual(resolved.sources["values"][0].kind, "registered_args")
            self.assertEqual(resolved.sources["named"]["bonus"].kind, "registered_kwargs")

            self.assertEqual(
                block.inspect(overrides={"values": (8,), "named": {"extra": 9}}),
                ("default", (8,), {"extra": 9}),
            )

    def test_duplicate_registration_bindings_can_be_neutralized_by_override(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "root",
                {"left": "L", "right": "R"},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("duplicates", 1)
            block._register_function_strict(
                select_data,
                ["left_result"],
                param_mapping={"data": "left"},
            )
            block._register_function_strict(
                select_data,
                ["right_result"],
                param_mapping={"data": "right"},
            )

            with self.assertRaisesRegex(RegistrationError, "different input bindings"):
                block.inspect(function_name="select_data")
            self.assertEqual(
                block.inspect(
                    function_name="select_data",
                    overrides={"data": "override"},
                ),
                "override",
            )

    def test_selected_registration_keeps_same_block_dependency_validation(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root")
            block = pipeline.add_block("parallel", 1)
            block.register_function(produce_shared, ["shared"])
            block.register_function(consume_shared, ["result"])

            with self.assertRaisesRegex(ExecutionError, "same block"):
                block.inspect(function_name="consume_shared")

    def test_expression_inspection_supports_overrides_copying_and_raw_result(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "root",
                {"items": [1]},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("expression", 1)
            block.register_expression("result = items + [2]")

            resolved = block.inspect(
                resolve_only=True,
                strict_mode=True,
                allow_mutable_objects=False,
            )
            self.assertEqual(resolved.arguments["items"], [1])
            self.assertTrue(resolved.sources["items"].copied)
            self.assertEqual(block.inspect(), [1, 2])
            self.assertEqual(
                block.inspect(overrides={"items": [10]}, strict_mode=False),
                [10, 2],
            )
            with self.assertRaisesRegex(ResolutionError, "Unknown inspection override"):
                block.inspect(overrides={"missing": 1})

    def test_resolve_only_does_not_invoke_and_prints_are_captured(self) -> None:
        with TemporaryDirectory() as tmp:
            INVOCATIONS.clear()
            pipeline = PipelineHandler("root", {"value": 4}, Path(tmp) / "root")
            record_block = pipeline.add_block("record", 1)
            record_block.register_function(record_invocation, ["result"])
            print_block = pipeline.add_block("print", 2)
            print_block.register_function(print_value, ["printed"])

            record_block.inspect(resolve_only=True)
            self.assertEqual(INVOCATIONS, [])
            self.assertEqual(print_block.inspect(), 4)
            self.assertIn(
                "inspection-value=4",
                pipeline.logger.log_file_path.read_text(encoding="utf-8"),
            )

    def test_atom_inspection_bypasses_gate_and_uses_previous_atom_output(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "root",
                {"run_atom": False},
                Path(tmp) / "root",
            )
            pipeline.create_atom_child_pipeline(
                "atom",
                5.1,
                rolling_value,
                gate_config="run_atom",
                output_variable_names="value",
            )
            atom = pipeline.get_child_pipeline("atom")

            self.assertEqual(atom.inspect(), 1)
            pipeline.set_config("run_atom", True)
            pipeline.run_all()
            pipeline.set_config("run_atom", False)
            resolved = atom.inspect(resolve_only=True)
            self.assertEqual(resolved.arguments["value"], 1)
            self.assertTrue(resolved.sources["value"].same_node_previous_output)
            self.assertEqual(atom.inspect(), 2)
            self.assertEqual(pipeline.get_value("value"), 1)

    def test_pipeline_inspect_node_uses_exact_priority_and_rejects_children(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHandler("root", {}, Path(tmp) / "root")
            first = root.add_block("first", 5.1)
            first.register_function(arbitrary_raw_result, ["first_value"])
            second = root.add_block("second", 5.2)
            second.register_function(choose_threshold, ["second_value"])

            self.assertEqual(root.inspect_node(priority=5.1), "raw")
            self.assertEqual(
                root.inspect_node(node_name="second", priority=5.2),
                0.5,
            )
            with self.assertRaisesRegex(ResolutionError, "exact priority 5"):
                root.inspect_node(priority=5)
            with self.assertRaisesRegex(ResolutionError, "different nodes"):
                root.inspect_node(node_name="first", priority=5.2)

            child = PipelineHandler("child", {}, Path(tmp) / "child")
            child.add_block("inside", 1).register_function(
                arbitrary_raw_result,
                ["child_value"],
            )
            root.add_child_pipeline(child, 6)
            with self.assertRaisesRegex(RegistrationError, "ordinary child pipeline"):
                root.inspect_node(node_name="child")

    def test_unknown_override_and_invocation_failure_are_reported(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", {"data": 1}, Path(tmp) / "root")
            block = pipeline.add_block("select", 1)
            block.register_function(select_data, ["result"])

            with self.assertRaisesRegex(ResolutionError, "Unknown inspection override"):
                block.inspect(overrides={"selected_df": 2})

            def fail(data: int) -> int:
                raise ValueError(f"bad-{data}")

            failing = pipeline.add_block("failing", 2)
            failing.register_function(fail, ["failed"])
            with self.assertRaisesRegex(ExecutionError, "failed during inspection") as caught:
                failing.inspect()
            self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_pandas_numpy_and_optional_torch_values_are_isolated(self) -> None:
        with TemporaryDirectory() as tmp:
            frame = pd.DataFrame({"value": [[1]]})
            array = np.array([1, 2])
            pipeline = PipelineHandler(
                "root",
                {"frame": frame, "array": array},
                Path(tmp) / "root",
            )

            def mutate_data(frame: pd.DataFrame, array: np.ndarray) -> tuple[object, object]:
                frame.iat[0, 0].append(2)
                array[0] = 9
                return frame, array

            block = pipeline.add_block("data", 1)
            block.register_function(mutate_data, ["frame_result", "array_result"])
            copied_frame, copied_array = block.inspect(
                allow_mutable_objects=False,
            )
            self.assertEqual(copied_frame.iat[0, 0], [1, 2])
            self.assertEqual(frame.iat[0, 0], [1])
            self.assertEqual(copied_array.tolist(), [9, 2])
            self.assertEqual(array.tolist(), [1, 2])

    @unittest.skipUnless(find_spec("torch") is not None, "torch is not installed")
    def test_torch_tensor_is_isolated(self) -> None:
        import torch

        with TemporaryDirectory() as tmp:
            tensor = torch.tensor([1.0])
            pipeline = PipelineHandler(
                "root",
                {"tensor": tensor},
                Path(tmp) / "root",
            )

            def mutate_tensor(tensor):
                tensor.add_(1)
                return tensor

            block = pipeline.add_block("tensor", 1)
            block.register_function(mutate_tensor, ["result"])
            result = block.inspect(allow_mutable_objects=False)
            self.assertEqual(result.item(), 2.0)
            self.assertEqual(tensor.item(), 1.0)

    @unittest.skipUnless(find_spec("dask.dataframe") is not None, "dask is not installed")
    def test_dask_dataframe_is_copied(self) -> None:
        import dask.dataframe as dd

        with TemporaryDirectory() as tmp:
            frame = dd.from_pandas(pd.DataFrame({"value": [1, 2]}), npartitions=1)
            pipeline = PipelineHandler(
                "root",
                {"data": frame},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("dask", 1)
            block.register_function(select_data, ["result"])

            resolved = block.inspect(
                resolve_only=True,
                allow_mutable_objects=False,
            )

            self.assertIsNot(resolved.arguments["data"], pipeline.get_config("data"))
            self.assertTrue(resolved.sources["data"].copied)
            self.assertEqual(
                resolved.arguments["data"].compute().to_dict("list"),
                {"value": [1, 2]},
            )

    @unittest.skipUnless(find_spec("optuna") is not None, "optuna is not installed")
    def test_optuna_sampler_is_shared_by_default_and_rejected_when_protected(self) -> None:
        import optuna

        with TemporaryDirectory() as tmp:
            sampler = optuna.samplers.RandomSampler()
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root")
            pipeline.set_constant_value("data", sampler, copy=False)
            block = pipeline.add_block("sampler", 1)
            block.register_function(select_data, ["result"])

            self.assertIs(
                block.inspect(),
                pipeline.get_constant_value("data"),
            )
            with self.assertRaisesRegex(InspectionCopyError, "Optuna state"):
                block.inspect(allow_mutable_objects=False)

    def test_default_inspection_shares_objects_for_block_atom_and_node(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "root",
                {"items": [1]},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("block", 1)
            block.register_function(mutate_items, ["result"])
            self.assertTrue(block.inspect(resolve_only=True).allow_mutable_objects)

            pipeline.create_atom_child_pipeline(
                "atom",
                2,
                mutate_items,
                output_variable_names="atom_result",
            )
            atom = pipeline.get_child_pipeline("atom")
            self.assertTrue(atom.inspect(resolve_only=True).allow_mutable_objects)
            self.assertTrue(
                pipeline.inspect_node(
                    node_name="atom",
                    resolve_only=True,
                ).allow_mutable_objects
            )

    def test_default_inspection_passes_original_object_by_reference(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline_value: list[object] = [1]
            pipeline = PipelineHandler(
                "root",
                {"items": pipeline_value},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("mutate", 1)
            block.register_function(mutate_items, ["result"])

            result = block.inspect()

            self.assertIs(result, pipeline_value)
            self.assertEqual(pipeline_value, [1, "inspected"])
            self.assertIs(result, pipeline.get_config("items"))

    def test_override_sources_report_not_copied_in_both_modes(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "root",
                {"items": [1]},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("mutate", 1)
            block.register_function(mutate_items, ["result"])

            protected_override: list[object] = [10]
            protected = block.inspect(
                overrides={"items": protected_override},
                resolve_only=True,
                allow_mutable_objects=False,
            )
            self.assertIs(protected.arguments["items"], protected_override)
            self.assertFalse(protected.sources["items"].copied)

            shared_override: list[object] = [20]
            shared = block.inspect(
                overrides={"items": shared_override},
                resolve_only=True,
            )
            self.assertIs(shared.arguments["items"], shared_override)
            self.assertFalse(shared.sources["items"].copied)

    def test_protected_inspection_copies_nested_mutable_mappings(self) -> None:
        with TemporaryDirectory() as tmp:
            shared = {"values": [1]}
            pipeline = PipelineHandler(
                "root",
                {"data": shared},
                Path(tmp) / "root",
            )

            def mutate_mapping(data: dict[str, list[int]]) -> dict[str, list[int]]:
                data["values"].append(2)
                return data

            block = pipeline.add_block("nested", 1)
            block.register_function(mutate_mapping, ["result"])

            result = block.inspect(allow_mutable_objects=False)

            self.assertEqual(result, {"values": [1, 2]})
            self.assertEqual(shared, {"values": [1]})
            self.assertIsNot(result, shared)
            self.assertIsNot(result["values"], shared["values"])

    def test_callable_instance_value_is_copied_or_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            counter = CallableCounter()
            pipeline = PipelineHandler(
                "root",
                {"callback": counter, "value": 1},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("callable", 1)
            block.register_function(invoke_callable, ["result"])

            self.assertEqual(block.inspect(allow_mutable_objects=False), 2)
            self.assertEqual(counter.count, 0)
            self.assertEqual(block.inspect(), 2)
            self.assertEqual(counter.count, 1)

            uncopyable = UncopyableCallable()
            failing = PipelineHandler(
                "failing",
                {"value": 1},
                Path(tmp) / "failing",
            )
            failing.set_constant_value("callback", uncopyable, copy=False)
            failing_block = failing.add_block("callable", 1)
            failing_block.register_function(invoke_callable, ["result"])
            with self.assertRaisesRegex(InspectionCopyError, "Cannot isolate"):
                failing_block.inspect(allow_mutable_objects=False)
            self.assertIs(uncopyable, failing.get_constant_value("callback"))

    def test_real_logger_writes_are_visible_in_both_modes(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", {"value": 3}, Path(tmp) / "root")
            block = pipeline.add_block("logger", 1)
            block.register_function(log_and_return, ["result"])

            self.assertEqual(block.inspect(), 3)
            self.assertEqual(block.inspect(allow_mutable_objects=False), 3)

            log_text = pipeline.logger.log_file_path.read_text(encoding="utf-8")
            self.assertEqual(log_text.count("inspection-logger=3"), 2)

    def test_non_strict_inspection_matches_mapped_callable_defaults(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root")
            block = pipeline.add_block("mapped", 1)
            block.register_function(
                mapped_default,
                ["result"],
                param_mapping={"a": "b"},
            )

            self.assertEqual(block.inspect(), 5)
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("result"), 5)

        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", {"b": 7}, Path(tmp) / "root")
            block = pipeline.add_block("mapped", 1)
            block.register_function(
                mapped_default,
                ["result"],
                param_mapping={"a": "b"},
            )

            self.assertEqual(block.inspect(), 7)
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("result"), 7)

        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root")
            block = pipeline.add_block("mapped", 1)
            block.register_function(
                mapped_default,
                ["result"],
                param_mapping={"a": "b"},
            )

            with self.assertRaisesRegex(ResolutionError, "Cannot resolve argument 'b'"):
                block.inspect(strict_mode=True)

    def test_same_block_dependency_can_be_removed_by_original_parameter_override(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root")
            block = pipeline.add_block("parallel", 1)
            block.register_function(produce_shared, ["shared"])
            block.register_function(consume_shared, ["result"])

            with self.assertRaisesRegex(ExecutionError, "same block"):
                pipeline.run_all()
            with self.assertRaisesRegex(ExecutionError, "same block"):
                block.inspect(function_name="consume_shared")

            self.assertEqual(
                block.inspect(
                    function_name="consume_shared",
                    overrides={"shared": 5},
                ),
                6,
            )

    def test_signature_binding_failure_reports_resolution_error(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", {"a": 1}, Path(tmp) / "root")
            block = pipeline.add_block("binding", 1)
            block.register_function(
                variadic_target,
                ["result"],
                var_kw_name="kwargs",
            )

            with self.assertRaisesRegex(
                ResolutionError,
                "Cannot bind resolved arguments for function 'variadic_target'",
            ):
                block.inspect(overrides={"kwargs": {"a": 3}})

    def test_protected_preflight_rejects_before_materializing_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root")
            block = pipeline.add_block("rolling", 1)
            block.register_function(
                rolling_value,
                ["value"],
                save_to_disk=["value"],
            )
            pipeline.run_all()

            outputs_before = {
                name: dict(outputs)
                for name, outputs in pipeline.producer_outputs.items()
            }
            values_before = dict(pipeline.para_value_dict)
            registry_before = dict(pipeline.artifact_registry)
            history_before = list(pipeline.run_history)
            artifact_paths = set(pipeline.artifact_store.artifact_root.rglob("*"))

            real_load = ArtifactStore.load
            with (
                mock.patch(
                    "mlpipelineholder.execution.inspection._available_memory_bytes",
                    return_value=(0, "test RAM", ("0 B via test RAM",)),
                ),
                mock.patch.object(ArtifactStore, "load", wraps=real_load) as load_mock,
            ):
                with self.assertRaises(InspectionMemoryError) as caught:
                    block.inspect(
                        resolve_only=True,
                        allow_mutable_objects=False,
                    )

            self.assertFalse(caught.exception.copy_started)
            self.assertIn("value", caught.exception.inputs)
            load_mock.assert_not_called()
            self.assertEqual(pipeline.producer_outputs, outputs_before)
            self.assertEqual(pipeline.para_value_dict, values_before)
            self.assertEqual(pipeline.artifact_registry, registry_before)
            self.assertEqual(list(pipeline.run_history), history_before)
            self.assertEqual(
                set(pipeline.artifact_store.artifact_root.rglob("*")),
                artifact_paths,
            )

            resolved = block.inspect(
                resolve_only=True,
                allow_mutable_objects=False,
            )
            self.assertEqual(resolved.arguments["value"], 1)
            self.assertTrue(resolved.sources["value"].materialized)

    def test_shared_mode_skips_memory_preflight(self) -> None:
        with TemporaryDirectory() as tmp:
            array = np.zeros(64, dtype=np.float64)
            pipeline = PipelineHandler(
                "root",
                {"data": array},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("shared", 1)
            block.register_function(select_data, ["result"])

            with (
                mock.patch(
                    "mlpipelineholder.execution.inspection._available_memory_bytes",
                    side_effect=AssertionError("preflight must not run"),
                ),
                mock.patch(
                    "mlpipelineholder.execution.inspection._InspectionMemoryEstimator",
                    side_effect=AssertionError("estimator must not run"),
                ),
            ):
                result = block.inspect(resolve_only=True)

            self.assertIs(result.arguments["data"], array)

    def test_preflight_combines_distinct_inputs(self) -> None:
        with TemporaryDirectory() as tmp:
            first = np.zeros(100_000, dtype=np.float64)
            second = np.zeros(100_000, dtype=np.float64)
            pipeline = PipelineHandler(
                "root",
                {"left": first, "right": second},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("pair", 1)
            block.register_function(combine_arrays, ["result"])

            with mock.patch(
                "mlpipelineholder.execution.inspection._available_memory_bytes",
                return_value=(1_500_000, "test RAM", ("1.4 MiB via test RAM",)),
            ):
                with self.assertRaisesRegex(
                    InspectionMemoryError,
                    "exceeds available memory",
                ):
                    block.inspect(
                        resolve_only=True,
                        allow_mutable_objects=False,
                    )

            with mock.patch(
                "mlpipelineholder.execution.inspection._available_memory_bytes",
                return_value=(3_000_000, "test RAM", ("2.9 MiB via test RAM",)),
            ):
                block.inspect(
                    resolve_only=True,
                    allow_mutable_objects=False,
                )

    def test_preflight_does_not_double_count_shared_identities(self) -> None:
        with TemporaryDirectory() as tmp:
            array = np.zeros(100_000, dtype=np.float64)
            pipeline = PipelineHandler(
                "root",
                {"left": array, "right": array},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("pair", 1)
            block.register_function(combine_arrays, ["result"])

            with mock.patch(
                "mlpipelineholder.execution.inspection._available_memory_bytes",
                return_value=(1_250_000, "test RAM", ("1.2 MiB via test RAM",)),
            ):
                block.inspect(
                    resolve_only=True,
                    allow_mutable_objects=False,
                )

    def test_preflight_rejects_unreliable_estimates(self) -> None:
        with TemporaryDirectory() as tmp:
            record = ArtifactRecord(
                variable_name="data",
                serializer="mystery",
                file_path=str(Path(tmp) / "missing.bin"),
                produced_by_block="b",
                produced_by_function="f",
                run_id="r",
            )
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root")
            pipeline.set_constant_value("data", record, copy=False)
            block = pipeline.add_block("uncertain", 1)
            block.register_function(select_data, ["result"])

            with self.assertRaisesRegex(
                InspectionMemoryError,
                "could not be established reliably",
            ):
                block.inspect(allow_mutable_objects=False)

    def test_preflight_prompts_and_continues_when_confirmed(self) -> None:
        with TemporaryDirectory() as tmp:
            array = np.zeros(100_000, dtype=np.float64)
            pipeline = PipelineHandler(
                "root",
                {"data": array},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("prompt", 1)
            block.register_function(select_data, ["result"])

            with (
                mock.patch(
                    "mlpipelineholder.execution.inspection._available_memory_bytes",
                    return_value=(1_000, "test RAM", ("1000 B via test RAM",)),
                ),
                mock.patch(
                    "mlpipelineholder.execution.inspection._interactive_stdin_available",
                    return_value=True,
                ),
                mock.patch("builtins.input", return_value="y") as prompt,
            ):
                resolved = block.inspect(
                    resolve_only=True,
                    allow_mutable_objects=False,
                )

            self.assertIsNot(resolved.arguments["data"], array)
            self.assertTrue(resolved.sources["data"].copied)
            prompt.assert_called_once()
            prompt_text = prompt.call_args.args[0]
            self.assertIn("Proceed with protected copying", prompt_text)
            self.assertIn("allow_mutable_objects=True", prompt_text)

    def test_preflight_prompt_decline_and_noninteractive_raise(self) -> None:
        with TemporaryDirectory() as tmp:
            array = np.zeros(100_000, dtype=np.float64)
            pipeline = PipelineHandler(
                "root",
                {"data": array},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("prompt", 1)
            block.register_function(select_data, ["result"])

            low_memory = mock.patch(
                "mlpipelineholder.execution.inspection._available_memory_bytes",
                return_value=(1_000, "test RAM", ("1000 B via test RAM",)),
            )
            interactive = mock.patch(
                "mlpipelineholder.execution.inspection._interactive_stdin_available",
                return_value=True,
            )

            for answer in ("n", "", "no"):
                with (
                    low_memory,
                    interactive,
                    mock.patch("builtins.input", return_value=answer),
                ):
                    with self.assertRaises(InspectionMemoryError):
                        block.inspect(
                            resolve_only=True,
                            allow_mutable_objects=False,
                        )

            with (
                low_memory,
                interactive,
                mock.patch("builtins.input", side_effect=EOFError),
            ):
                with self.assertRaises(InspectionMemoryError):
                    block.inspect(
                        resolve_only=True,
                        allow_mutable_objects=False,
                    )

            with (
                low_memory,
                mock.patch(
                    "mlpipelineholder.execution.inspection._interactive_stdin_available",
                    return_value=False,
                ),
                mock.patch(
                    "builtins.input",
                    side_effect=AssertionError("must not prompt"),
                ),
            ):
                with self.assertRaises(InspectionMemoryError):
                    block.inspect(
                        resolve_only=True,
                        allow_mutable_objects=False,
                    )

    def test_preflight_rejects_unmeasurable_available_memory(self) -> None:
        with TemporaryDirectory() as tmp:
            array = np.zeros(100_000, dtype=np.float64)
            pipeline = PipelineHandler(
                "root",
                {"data": array},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("unknown", 1)
            block.register_function(select_data, ["result"])

            with mock.patch(
                "mlpipelineholder.execution.inspection._available_memory_bytes",
                return_value=(None, "", ()),
            ):
                with self.assertRaisesRegex(
                    InspectionMemoryError,
                    "could not be determined reliably",
                ):
                    block.inspect(
                        resolve_only=True,
                        allow_mutable_objects=False,
                    )

    def test_memory_safety_margin_is_configurable_and_validated(self) -> None:
        with TemporaryDirectory() as tmp:
            array = np.zeros(100_000, dtype=np.float64)
            pipeline = PipelineHandler(
                "root",
                {"data": array},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("margin", 1)
            block.register_function(select_data, ["result"])

            with mock.patch(
                "mlpipelineholder.execution.inspection._available_memory_bytes",
                return_value=(1_000_000, "test RAM", ("0.9 MiB via test RAM",)),
            ):
                block.inspect(
                    resolve_only=True,
                    allow_mutable_objects=False,
                    memory_safety_margin=0.1,
                )
                with self.assertRaisesRegex(
                    InspectionMemoryError,
                    "exceeds available memory",
                ):
                    block.inspect(
                        resolve_only=True,
                        allow_mutable_objects=False,
                        memory_safety_margin=0.5,
                    )

            for invalid in (
                -0.1,
                float("nan"),
                float("inf"),
                True,
                10**1000,
                1e308,
            ):
                with self.assertRaisesRegex(TypeError, "memory_safety_margin"):
                    block.inspect(
                        resolve_only=True,
                        allow_mutable_objects=False,
                        memory_safety_margin=invalid,
                    )

    def test_recoverable_memory_error_during_copy_is_reported(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root")
            pipeline.set_constant_value("value", MemoryErrorOnCopy(), copy=False)
            block = pipeline.add_block("use", 1)
            block.register_function(return_object, ["result"])

            with self.assertRaises(InspectionMemoryError) as caught:
                block.inspect(allow_mutable_objects=False)

            self.assertTrue(caught.exception.copy_started)
            self.assertIsInstance(caught.exception.__cause__, MemoryError)
            self.assertIn("value", caught.exception.inputs)

    def test_memory_estimator_accounts_for_pandas_object_cells(self) -> None:
        array = np.zeros(100_000, dtype=np.float64)
        nested = [array]
        frame = pd.DataFrame({"data": [nested, nested]})
        estimator = _InspectionMemoryEstimator()
        estimator.add("data", frame)

        baseline = int(frame.memory_usage(index=True, deep=False).sum())
        self.assertGreaterEqual(
            estimator.total_bytes,
            baseline + sys.getsizeof(nested) + array.nbytes,
        )
        self.assertLess(estimator.total_bytes, baseline + 2 * array.nbytes)
        self.assertEqual(estimator.uncertain, ())

    @unittest.skipUnless(
        find_spec("dask.dataframe") is not None,
        "dask is not installed",
    )
    def test_preflight_estimates_lazy_dask_graph_not_dataset(self) -> None:
        import dask.dataframe as dd

        with TemporaryDirectory() as tmp:
            frame = pd.DataFrame({"value": np.arange(200_000, dtype=np.float64)})
            collection = dd.from_pandas(frame, npartitions=4)
            pipeline = PipelineHandler(
                "root",
                {"data": collection},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("dask", 1)
            block.register_function(select_data, ["result"])

            with mock.patch(
                "mlpipelineholder.execution.inspection._available_memory_bytes",
                return_value=(1 << 20, "test RAM", ("1.0 MiB via test RAM",)),
            ):
                block.inspect(
                    resolve_only=True,
                    allow_mutable_objects=False,
                )

    @unittest.skipUnless(find_spec("torch") is not None, "torch is not installed")
    def test_cuda_preflight_uses_gpu_memory_availability(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        with TemporaryDirectory() as tmp:
            tensor = torch.zeros(1024, device="cuda")
            pipeline = PipelineHandler(
                "root",
                {"data": tensor},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("cuda", 1)
            block.register_function(select_data, ["result"])

            with mock.patch(
                "mlpipelineholder.execution.inspection._available_cuda_bytes",
                return_value=(1, "test VRAM"),
            ):
                with self.assertRaisesRegex(
                    InspectionMemoryError,
                    "CUDA memory",
                ):
                    block.inspect(
                        resolve_only=True,
                        allow_mutable_objects=False,
                    )

            with mock.patch(
                "mlpipelineholder.execution.inspection._available_cuda_bytes",
                return_value=(1 << 30, "test VRAM"),
            ):
                block.inspect(
                    resolve_only=True,
                    allow_mutable_objects=False,
                )

    def test_preflight_failure_covers_all_selected_call_modes(self) -> None:
        with TemporaryDirectory() as tmp:
            array = np.zeros(100_000, dtype=np.float64)
            pipeline = PipelineHandler(
                "root",
                {"data": array},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("expression", 1)
            block.register_expression("result = data")

            with mock.patch(
                "mlpipelineholder.execution.inspection._available_memory_bytes",
                return_value=(0, "test RAM", ("0 B via test RAM",)),
            ):
                with self.assertRaises(InspectionMemoryError):
                    block.inspect(resolve_only=True, allow_mutable_objects=False)
                with self.assertRaises(InspectionMemoryError):
                    block.inspect(allow_mutable_objects=False)

    def test_lightweight_estimator_does_not_sample_numeric_dataframe_cells(self) -> None:
        frame = pd.DataFrame(
            {
                "left": np.arange(100_000, dtype=np.float64),
                "right": np.arange(100_000, dtype=np.int64),
            }
        )
        estimator = _InspectionMemoryEstimator()
        estimator.add("frame", frame)

        self.assertGreaterEqual(
            estimator.total_bytes,
            int(frame.memory_usage(index=True, deep=False).sum()),
        )
        self.assertEqual(estimator.uncertain, ())

    def test_truncated_container_is_uncertain_even_when_prefix_is_immutable(self) -> None:
        value = [*range(32), np.zeros(100_000, dtype=np.float64)]
        estimator = _InspectionMemoryEstimator()

        estimator.add("value", value)

        self.assertTrue(estimator.uncertain)
        self.assertIn("container contents were sampled", estimator.uncertain[0][1])

    def test_repeated_artifact_bindings_use_peak_materialization_budget(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root")
            record = pipeline.artifact_store.save(
                variable_name="data",
                value=np.zeros(100_000, dtype=np.float64),
                block_name="producer",
                function_name="produce",
                run_id="run",
            )
            pipeline.set_constant_value("left", record, copy=False)
            pipeline.set_constant_value("right", record, copy=False)
            block = pipeline.add_block("pair", 1)
            block.register_function(combine_arrays, ["result"])

            with (
                mock.patch(
                    "mlpipelineholder.execution.inspection._available_memory_bytes",
                    return_value=(4_000_000, "test RAM", ("3.8 MiB via test RAM",)),
                ),
                mock.patch.object(
                    ArtifactStore,
                    "load",
                    side_effect=[
                        np.zeros(100_000, dtype=np.float64),
                        np.zeros(100_000, dtype=np.float64),
                    ],
                ) as load_mock,
            ):
                block.inspect(resolve_only=True, allow_mutable_objects=False)

            self.assertEqual(load_mock.call_count, 2)

    def test_generic_artifact_materializations_are_retained_in_budget(self) -> None:
        with TemporaryDirectory() as tmp:
            first_path = Path(tmp) / "first.json"
            second_path = Path(tmp) / "second.json"
            first_path.write_bytes(b"0" * 100)
            second_path.write_bytes(b"0" * 200)
            records = [
                ArtifactRecord(
                    variable_name=name,
                    serializer="json",
                    file_path=str(path),
                    produced_by_block="producer",
                    produced_by_function="produce",
                    run_id="run",
                )
                for name, path in (("first", first_path), ("second", second_path))
            ]
            estimator = _InspectionMemoryEstimator()

            for record in records:
                estimator.add(record.variable_name, record, materialize=True)

            self.assertEqual(estimator.materialization_bytes, 600)

    def test_dask_estimation_does_not_swallow_memory_error(self) -> None:
        class FailingCollection:
            def __dask_graph__(self) -> object:
                raise MemoryError("simulated graph allocation failure")

        estimator = _InspectionMemoryEstimator()

        with self.assertRaises(MemoryError):
            estimator._estimate_dask_collection(FailingCollection())

    def test_artifact_estimate_is_order_independent_when_also_nested(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.npy"
            np.save(path, np.zeros(100_000, dtype=np.float64))
            record = ArtifactRecord(
                variable_name="data",
                serializer="numpy",
                file_path=str(path),
                produced_by_block="producer",
                produced_by_function="produce",
                run_id="run",
            )

            nested_first = _InspectionMemoryEstimator()
            nested_first.add("container", [record])
            nested_first.add("item", record, materialize=True)
            direct_first = _InspectionMemoryEstimator()
            direct_first.add("item", record, materialize=True)
            direct_first.add("container", [record])

            self.assertEqual(nested_first.total_bytes, direct_first.total_bytes)
            self.assertEqual(nested_first.uncertain, direct_first.uncertain)

    def test_bound_method_owner_storage_is_included_in_preflight(self) -> None:
        with TemporaryDirectory() as tmp:
            owner = LargeBoundMethodOwner(100_000)
            pipeline = PipelineHandler(
                "root",
                {"callback": owner.read},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("callback", 1)
            block.register_function(invoke_without_arguments, ["result"])

            with mock.patch(
                "mlpipelineholder.execution.inspection._available_memory_bytes",
                return_value=(100_000, "test RAM", ("97.7 KiB via test RAM",)),
            ):
                with self.assertRaisesRegex(
                    InspectionMemoryError,
                    "exceeds available memory",
                ):
                    block.inspect(resolve_only=True, allow_mutable_objects=False)

    def test_unresolvable_pointer_uses_uncertainty_confirmation_path(self) -> None:
        pointer = OutputPointer(OutputAddress("root", "missing", "value"))

        def fail_pointer(_: OutputPointer) -> object:
            raise PointerResolutionError("missing destination")

        with mock.patch(
            "mlpipelineholder.execution.inspection._interactive_stdin_available",
            return_value=False,
        ):
            with self.assertRaisesRegex(
                InspectionMemoryError,
                "output pointer is unresolvable",
            ):
                preflight_protected_inspection(
                    [InspectionCopyCandidate("value", pointer)],
                    node_name="node",
                    function_name="function",
                    safety_margin=0.25,
                    pointer_reader=fail_pointer,
                )

    def test_uncertain_estimate_requires_explicit_user_decision(self) -> None:
        with TemporaryDirectory() as tmp:
            frame = pd.DataFrame({"value": [[index] for index in range(64)]})
            pipeline = PipelineHandler(
                "root",
                {"data": frame},
                Path(tmp) / "root",
            )
            block = pipeline.add_block("sample", 1)
            block.register_function(select_data, ["result"])
            interactive = mock.patch(
                "mlpipelineholder.execution.inspection._interactive_stdin_available",
                return_value=True,
            )

            with interactive, mock.patch("builtins.input", return_value="yes") as prompt:
                resolved = block.inspect(
                    resolve_only=True,
                    allow_mutable_objects=False,
                )

            self.assertIsNot(resolved.arguments["data"], frame)
            self.assertIn("terminate the Python or Jupyter process", prompt.call_args.args[0])
            self.assertIn("lose unsaved in-memory work", prompt.call_args.args[0])

            with interactive, mock.patch("builtins.input", return_value="no"):
                with self.assertRaises(InspectionMemoryError):
                    block.inspect(
                        resolve_only=True,
                        allow_mutable_objects=False,
                    )

    def test_resolution_memory_error_reports_that_copy_has_not_started(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root")
            record = pipeline.artifact_store.save(
                variable_name="data",
                value=np.zeros(8, dtype=np.float64),
                block_name="producer",
                function_name="produce",
                run_id="run",
            )
            pipeline.set_constant_value("data", record, copy=False)
            block = pipeline.add_block("load", 1)
            block.register_function(select_data, ["result"])

            with mock.patch.object(
                ArtifactStore,
                "load",
                side_effect=MemoryError("simulated load failure"),
            ):
                with self.assertRaises(InspectionMemoryError) as caught:
                    block.inspect(resolve_only=True, allow_mutable_objects=False)

            self.assertFalse(caught.exception.copy_started)
            self.assertIn("materialising", caught.exception.reason or "")

    def test_estimation_memory_error_is_stage_aware(self) -> None:
        record = ArtifactRecord(
            variable_name="data",
            serializer="numpy",
            file_path="unused.npy",
            produced_by_block="producer",
            produced_by_function="produce",
            run_id="run",
        )

        with mock.patch.object(
            _InspectionMemoryEstimator,
            "_estimate_artifact_record",
            side_effect=MemoryError("simulated estimation failure"),
        ):
            with self.assertRaises(InspectionMemoryError) as caught:
                preflight_protected_inspection(
                    [InspectionCopyCandidate("data", record)],
                    node_name="node",
                    function_name="function",
                    safety_margin=0.25,
                )

        self.assertFalse(caught.exception.copy_started)
        self.assertIn("estimating protected-copy memory", caught.exception.reason or "")

    def test_resolution_memory_error_reports_an_earlier_started_copy(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root")
            record = pipeline.artifact_store.save(
                variable_name="right",
                value=np.zeros(8, dtype=np.float64),
                block_name="producer",
                function_name="produce",
                run_id="run",
            )
            pipeline.manual_values["left"] = np.zeros(8, dtype=np.float64)
            pipeline.manual_values["right"] = record
            block = pipeline.add_block("load", 1)
            block.register_function(combine_arrays, ["result"])

            with mock.patch.object(
                ArtifactStore,
                "load",
                side_effect=MemoryError("simulated load failure"),
            ):
                with self.assertRaises(InspectionMemoryError) as caught:
                    block.inspect(resolve_only=True, allow_mutable_objects=False)

            self.assertTrue(caught.exception.copy_started)
            self.assertIn("materialising", caught.exception.reason or "")

    def test_execution_memory_error_is_classified_separately(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root")
            block = pipeline.add_block("execute", 1)
            block.register_function(fail_with_memory_error, [])

            with self.assertRaises(InspectionMemoryError) as caught:
                block.inspect(allow_mutable_objects=False)

            self.assertFalse(caught.exception.copy_started)
            self.assertIn(
                "executing the inspected function",
                caught.exception.reason or "",
            )

    def test_disk_backed_variadic_aggregates_work_in_protected_mode(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            positional = PipelineHandler("positional", {}, root / "positional")
            positional_record = positional.artifact_store.save(
                variable_name="items",
                value=[2, 3],
                block_name="producer",
                function_name="produce",
                run_id="run",
            )
            positional.set_constant_value("items", positional_record, copy=False)
            positional_block = positional.add_block("sum", 1)
            positional_block.register_function(sum_variadic_items, ["result"])

            resolved_positional = positional_block.inspect(
                resolve_only=True,
                allow_mutable_objects=False,
            )
            self.assertEqual(resolved_positional.args, (2, 3))
            self.assertEqual(positional_block.inspect(allow_mutable_objects=False), 5)

            keyword = PipelineHandler("keyword", {}, root / "keyword")
            keyword_record = keyword.artifact_store.save(
                variable_name="items",
                value={"left": 2, "right": 3},
                block_name="producer",
                function_name="produce",
                run_id="run",
            )
            keyword.set_constant_value("items", keyword_record, copy=False)
            keyword_block = keyword.add_block("sum", 1)
            keyword_block.register_function(sum_variadic_named, ["result"])

            resolved_keyword = keyword_block.inspect(
                resolve_only=True,
                allow_mutable_objects=False,
            )
            self.assertEqual(resolved_keyword.kwargs, {"left": 2, "right": 3})
            self.assertEqual(keyword_block.inspect(allow_mutable_objects=False), 5)

    def test_known_optuna_artifact_is_rejected_without_loading(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root")
            record = ArtifactRecord(
                variable_name="study",
                serializer=OPTUNA_STUDY_SERIALIZER,
                file_path=str(Path(tmp) / "sampler.pkl"),
                produced_by_block="producer",
                produced_by_function="produce",
                run_id="run",
            )
            pipeline.manual_values["data"] = record
            block = pipeline.add_block("study", 1)
            block.register_function(select_data, ["result"])

            with mock.patch.object(ArtifactStore, "load") as load_mock:
                with self.assertRaisesRegex(InspectionCopyError, "Optuna state"):
                    block.inspect(resolve_only=True, allow_mutable_objects=False)

            load_mock.assert_not_called()

    @unittest.skipUnless(find_spec("optuna") is not None, "optuna is not installed")
    def test_disk_backed_optuna_sampler_is_rejected_without_loading(self) -> None:
        import optuna

        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root")
            record = pipeline.artifact_store.save(
                variable_name="sampler",
                value=optuna.samplers.RandomSampler(),
                block_name="producer",
                function_name="produce",
                run_id="run",
            )
            pipeline.set_constant_value("data", record, copy=False)
            block = pipeline.add_block("sampler", 1)
            block.register_function(select_data, ["result"])

            with mock.patch.object(ArtifactStore, "load") as load_mock:
                with self.assertRaisesRegex(InspectionCopyError, "Optuna state"):
                    block.inspect(resolve_only=True, allow_mutable_objects=False)

            load_mock.assert_not_called()

    def test_protected_numpy_view_never_mutates_original_storage(self) -> None:
        with TemporaryDirectory() as tmp:
            original = np.arange(8, dtype=np.int64)
            view = original[2:6]
            pipeline = PipelineHandler(
                "root",
                {"data": view},
                Path(tmp) / "root",
            )

            def mutate_view(data: np.ndarray) -> np.ndarray:
                data[:] = -1
                return data

            block = pipeline.add_block("view", 1)
            block.register_function(mutate_view, ["result"])
            inspected = block.inspect(allow_mutable_objects=False)

            self.assertTrue(np.all(inspected == -1))
            np.testing.assert_array_equal(original, np.arange(8, dtype=np.int64))

    @unittest.skipUnless(find_spec("torch") is not None, "torch is not installed")
    def test_protected_torch_view_never_mutates_original_storage(self) -> None:
        import torch

        with TemporaryDirectory() as tmp:
            original = torch.arange(8)
            view = original[2:6]
            pipeline = PipelineHandler(
                "root",
                {"data": view},
                Path(tmp) / "root",
            )

            def mutate_view(data: torch.Tensor) -> torch.Tensor:
                data.fill_(-1)
                return data

            block = pipeline.add_block("view", 1)
            block.register_function(mutate_view, ["result"])
            inspected = block.inspect(allow_mutable_objects=False)

            self.assertTrue(torch.all(inspected == -1))
            self.assertTrue(torch.equal(original, torch.arange(8)))

    def test_pipeline_inspect_requires_boolean_compute(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "root",
                {"value": 1},
                Path(tmp) / "root",
            )

            for invalid in (1, "yes", object()):
                with self.subTest(value=invalid):
                    with self.assertRaisesRegex(TypeError, "compute"):
                        pipeline.inspect(
                            "value",
                            compute=invalid,  # pyright: ignore[reportArgumentType]
                        )


if __name__ == "__main__":
    unittest.main()
