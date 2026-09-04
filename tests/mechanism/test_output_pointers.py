from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.mlpipelineholder import (
    ExecutionBlock,
    PipelineHandler,
    RegistrationError,
    ResolutionError,
)
from src.mlpipelineholder.output_pointers import (
    OutputAddress,
    OutputPointer,
    PointerResolutionError,
    resolve_pointer_chain,
)


def make_value(seed: int) -> list[int]:
    return [seed]


def append_two(value: list[int]) -> list[int]:
    value.append(2)
    return value


def append_three(value: list[int]) -> list[int]:
    value.append(3)
    return value


def identity(value: list[int]) -> list[int]:
    return value


def produce_none(seed: int) -> None:
    return None


def preserve_none(value: None) -> None:
    return value


def add_block(
    pipeline: PipelineHandler,
    registration_name: str,
    execution_priority: float,
) -> ExecutionBlock:
    block = pipeline.add_block(registration_name, execution_priority)
    assert block is not None
    return block


class OutputPointerDomainTests(unittest.TestCase):
    def test_chain_resolves_to_latest_address(self) -> None:
        # Given
        first = OutputAddress("root", "first", "value")
        second = OutputAddress("root", "second", "value")
        third = OutputAddress("root", "third", "value")
        values = {
            first: OutputPointer(second),
            second: OutputPointer(third),
            third: [1, 2, 3],
        }

        # When
        owner, value = resolve_pointer_chain(first, values.__getitem__)

        # Then
        self.assertEqual(owner, third)
        self.assertIs(value, values[third])

    def test_chain_rejects_cycle(self) -> None:
        # Given
        first = OutputAddress("root", "first", "value")
        second = OutputAddress("root", "second", "value")
        values = {
            first: OutputPointer(second),
            second: OutputPointer(first),
        }

        # When / Then
        with self.assertRaises(PointerResolutionError):
            resolve_pointer_chain(first, values.__getitem__)


class OutputPointerRegistrationTests(unittest.TestCase):
    def test_function_accepts_declared_unrun_upstream_target(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "root",
                {"seed": 1},
                Path(temp_dir),
                strict_mode=True,
            )
            first = add_block(pipeline, "first", 1)
            first.register_function(make_value, ["value"])
            second = add_block(pipeline, "second", 2)

            # When
            registration = second.register_function(
                append_two,
                ["value"],
                overridden_outputs={"value": ("root", "first")},
            )

            # Then
            self.assertEqual(
                registration.overridden_outputs,
                {"value": OutputAddress("root", "first", "value")},
            )

    def test_registration_rejects_downstream_target(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "root",
                {"seed": 1},
                Path(temp_dir),
                strict_mode=True,
            )
            first = add_block(pipeline, "first", 1)
            second = add_block(pipeline, "second", 2)
            second.register_function(append_two, ["value"])

            # When / Then
            with self.assertRaises(RegistrationError):
                first.register_function(
                    make_value,
                    ["value"],
                    overridden_outputs={"value": ("root", "second")},
                )

    def test_registration_rejects_unknown_output_key(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "root",
                {"seed": 1},
                Path(temp_dir),
                strict_mode=True,
            )
            first = add_block(pipeline, "first", 1)
            first.register_function(make_value, ["value"])
            second = add_block(pipeline, "second", 2)

            # When / Then
            with self.assertRaisesRegex(RegistrationError, "declared outputs"):
                second.register_function(
                    append_two,
                    ["value"],
                    overridden_outputs={"unknown": ("root", "first")},
                )

    def test_registration_rejects_unknown_pipeline_or_node(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "root",
                {"seed": 1},
                Path(temp_dir),
                strict_mode=True,
            )
            first = add_block(pipeline, "first", 1)
            first.register_function(make_value, ["value"])

            # When / Then
            for priority, block_name, target, message in (
                (2, "unknown_pipeline", ("missing", "first"), "one attached pipeline"),
                (3, "unknown_node", ("root", "missing"), "does not exist"),
            ):
                with self.subTest(block_name=block_name):
                    block = add_block(pipeline, block_name, priority)
                    with self.assertRaisesRegex(RegistrationError, message):
                        block.register_function(
                            append_two,
                            ["value"],
                            overridden_outputs={"value": target},
                        )

    def test_registration_rejects_target_without_output(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "root",
                {"seed": 1},
                Path(temp_dir),
                strict_mode=True,
            )
            first = add_block(pipeline, "first", 1)
            first.register_function(make_value, ["source"])
            second = add_block(pipeline, "second", 2)

            # When / Then
            with self.assertRaisesRegex(RegistrationError, "does not declare"):
                second.register_function(
                    append_two,
                    ["value"],
                    param_mapping={"value": "source"},
                    overridden_outputs={"value": ("root", "first")},
                )

    def test_registration_rejects_target_in_same_priority_group(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "root",
                {"seed": 1},
                Path(temp_dir),
                strict_mode=True,
            )
            first = add_block(pipeline, "first", 1.1)
            first.register_function(make_value, ["value"])
            second = add_block(pipeline, "second", 1.2)

            # When / Then
            with self.assertRaisesRegex(RegistrationError, "globally upstream"):
                second.register_function(
                    append_two,
                    ["value"],
                    overridden_outputs={"value": ("root", "first")},
                )

    def test_attaching_ambiguous_nested_pipeline_name_is_rejected(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = PipelineHandler("parent", {}, root / "parent")
            left = PipelineHandler("left", {}, root / "left")
            left_leaf = PipelineHandler("shared", {}, root / "left_leaf")
            left.add_child_pipeline(left_leaf, 1)
            parent.add_child_pipeline(left, 1)
            right = PipelineHandler("right", {}, root / "right")
            right_leaf = PipelineHandler("shared", {}, root / "right_leaf")
            right.add_child_pipeline(right_leaf, 1)

            # When / Then
            with self.assertRaisesRegex(RegistrationError, "unique"):
                parent.add_child_pipeline(right, 2)


class OutputPointerRuntimeTests(unittest.TestCase):
    def test_function_expression_chain_resolves_terminal_object(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", {"seed": 1}, Path(temp_dir))
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
            consumer = add_block(pipeline, "consumer", 4)
            consumer.register_function(identity, ["observed"], param_mapping={"value": "value"})

            # When
            pipeline.run_all()

            # Then
            terminal = pipeline.get_node_output("third", "value")
            self.assertIs(pipeline.get_node_output("first", "value"), terminal)
            self.assertIs(pipeline.get_node_output("second", "value"), terminal)
            self.assertIs(pipeline.get_value("value"), terminal)
            self.assertIs(pipeline.get_value("observed"), terminal)
            self.assertEqual(terminal, [1, 2, 3])
            self.assertIsInstance(
                pipeline.producer_outputs["first"]["value"],
                OutputPointer,
            )

    def test_corrupt_runtime_cycle_fails_safely(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", {}, Path(temp_dir))
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

            # When / Then
            with self.assertRaises(ResolutionError):
                pipeline.get_node_output("first", "value")

    def test_atom_can_override_upstream_expression(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", {"seed": 1}, Path(temp_dir))
            first = add_block(pipeline, "first", 1)
            first.register_function(make_value, ["value"])
            second = add_block(pipeline, "second", 2)
            second.register_expression(
                "value = value + [2]",
                overridden_outputs={"value": ("root", "first")},
            )
            pipeline.create_atom_child_pipeline(
                "third",
                3,
                append_three,
                output_variable_names=["value"],
                overridden_outputs={"value": ("root", "second")},
            )

            # When
            pipeline.run_all()

            # Then
            terminal = pipeline.get_node_output("third", "value")
            self.assertEqual(terminal, [1, 2, 3])
            self.assertIs(pipeline.get_node_output("first", "value"), terminal)
            self.assertIs(pipeline.get_node_output("second", "value"), terminal)

    def test_set_value_rejects_pointer_and_identifies_real_owner(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            root = PipelineHandler("root", {"seed": 1}, Path(temp_dir) / "root")
            child = PipelineHandler("child", {"seed": 1}, Path(temp_dir) / "child")
            first = add_block(child, "first", 1)
            first.register_function(make_value, ["value"])
            root.add_child_pipeline(child, 1)
            second = add_block(root, "second", 2)
            second.register_function(
                append_two,
                ["value"],
                overridden_outputs={"value": ("child", "first")},
            )
            root.run_all()

            # When / Then
            with self.assertRaisesRegex(
                ResolutionError,
                r"root\.second\.value.*set_value.*pipeline 'root'",
            ):
                child.set_value("value", [99])

    def test_none_terminal_value_resolves_through_pointer(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", {"seed": 1}, Path(temp_dir))
            first = add_block(pipeline, "first", 1)
            first.register_function(produce_none, ["value"])
            second = add_block(pipeline, "second", 2)
            second.register_function(
                preserve_none,
                ["value"],
                overridden_outputs={"value": ("root", "first")},
            )

            # When
            pipeline.run_all()

            # Then
            self.assertIsNone(pipeline.get_node_output("first", "value"))
            self.assertIsNone(pipeline.get_node_output("second", "value"))
            self.assertIsNone(pipeline.get_value("value"))

    def test_pointer_override_remains_visible_in_conflict_report(self) -> None:
        # Given
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", {"seed": 1}, Path(temp_dir))
            first = add_block(pipeline, "first", 1)
            first.register_function(make_value, ["value"])
            second = add_block(pipeline, "second", 2)
            second.register_function(
                append_two,
                ["value"],
                overridden_outputs={"value": ("root", "first")},
            )

            # When
            conflicts = pipeline.get_output_conflicts()

            # Then
            self.assertEqual(
                conflicts["value"],
                {
                    "created_by": "root/first",
                    "overridden_by": ["root/second"],
                },
            )


if __name__ == "__main__":
    unittest.main()
