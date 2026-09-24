from __future__ import annotations

import threading
import unittest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from mlpipelineholder import (
    ExecutionError,
    InspectionCopyError,
    PipelineHandler,
    RegistrationError,
    ResolutionError,
    ResolvedInspectionCall,
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

            self.assertTrue(block.inspect())
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

            result = block.inspect()

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
                block.inspect()
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

            callback_resolved = callback_block.inspect(resolve_only=True)
            logger_resolved = logger_block.inspect(resolve_only=True)

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

            self.assertEqual(block.inspect(), ["inspection"])
            self.assertEqual(original_default, [])
            resolved = block.inspect(resolve_only=True)
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

            resolved = block.inspect(resolve_only=True, strict_mode=True)
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
            copied_frame, copied_array = block.inspect()
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
            result = block.inspect()
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

            resolved = block.inspect(resolve_only=True)

            self.assertIsNot(resolved.arguments["data"], pipeline.get_config("data"))
            self.assertTrue(resolved.sources["data"].copied)
            self.assertEqual(
                resolved.arguments["data"].compute().to_dict("list"),
                {"value": [1, 2]},
            )

    @unittest.skipUnless(find_spec("optuna") is not None, "optuna is not installed")
    def test_optuna_sampler_requires_explicit_shared_mutability(self) -> None:
        import optuna

        with TemporaryDirectory() as tmp:
            sampler = optuna.samplers.RandomSampler()
            pipeline = PipelineHandler("root", {}, Path(tmp) / "root")
            pipeline.set_constant_value("data", sampler, copy=False)
            block = pipeline.add_block("sampler", 1)
            block.register_function(select_data, ["result"])

            with self.assertRaisesRegex(InspectionCopyError, "Optuna state"):
                block.inspect()
            self.assertIs(
                block.inspect(allow_mutable_objects=True),
                pipeline.get_constant_value("data"),
            )


if __name__ == "__main__":
    unittest.main()
