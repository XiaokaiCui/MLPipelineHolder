from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest

from mlpipelineholder import PipelineHandler, ResolutionError


def use_default(value: str = "function-default") -> str:
    return value


def require_value(value: str) -> str:
    return value


def identify_logger(logger) -> int:
    return id(logger)


def collect_explicit_variadics(
    prefix: str = "function-default",
    *values: int,
    **named: int,
) -> tuple[str, int]:
    return prefix, sum(values) + sum(named.values())


def collect_implicit_variadics(*values: int, **named: int) -> tuple[int, int]:
    return len(values), len(named)


def config_gate(flag: bool) -> bool:
    return flag


def produce_value() -> int:
    return 1


class UnpicklableResult:
    def __init__(self, source: str) -> None:
        self.source = source
        self.lock = threading.Lock()


def build_unpicklable(value: str = "function-default") -> UnpicklableResult:
    return UnpicklableResult(value)


class StrictArgumentResolutionTests(unittest.TestCase):
    def test_unmapped_default_uses_callable_default_without_advisory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            pipeline = PipelineHandler(
                "strict-default",
                {"value": "visible-value"},
                project,
                strict_mode=True,
            )
            block = pipeline.add_block("build", 1)

            registration = block.register_function(use_default, ["result"])
            pipeline.run_all()

            self.assertEqual(registration.input_names, [])
            self.assertEqual(pipeline.get_value("result"), "function-default")
            self.assertNotIn(
                "may be resolved implicitly",
                pipeline.logger.log_file_path.read_text(encoding="utf-8"),
            )

    def test_unmapped_required_parameter_does_not_use_visible_value(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "strict-required",
                {"value": "visible-value"},
                Path(temp_dir),
                strict_mode=True,
            )
            block = pipeline.add_block("build", 1)
            block.register_function(require_value, ["result"])

            with self.assertRaisesRegex(ResolutionError, "Cannot resolve argument 'value'"):
                pipeline.run_all()

    def test_explicit_same_name_mapping_uses_visible_value(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "strict-mapped",
                {"value": "visible-value"},
                Path(temp_dir),
                strict_mode=True,
            )
            block = pipeline.add_block("build", 1)

            registration = block.register_function(
                require_value,
                ["result"],
                param_mapping={"value": "value"},
            )
            pipeline.run_all()

            self.assertEqual(registration.input_names, ["value"])
            self.assertEqual(pipeline.get_value("result"), "visible-value")

    def test_unmapped_logger_receives_pipeline_logger(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "strict-logger",
                {},
                Path(temp_dir),
                strict_mode=True,
            )
            block = pipeline.add_block("build", 1)

            registration = block.register_function(identify_logger, ["logger_id"])
            pipeline.run_all()

            self.assertEqual(registration.input_names, ["logger"])
            self.assertEqual(pipeline.get_value("logger_id"), id(pipeline.logger))

    def test_registered_variadics_resolve_while_regular_parameter_uses_default(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "strict-variadics",
                {
                    "prefix": "visible-prefix",
                    "first": 2,
                    "second": 3,
                    "bonus": 5,
                },
                Path(temp_dir),
                strict_mode=True,
            )
            block = pipeline.add_block("build", 1)
            block.register_args("selected_args", ["first", "second"])
            block.register_kwargs("selected_kwargs", {"bonus": "bonus"})
            registration = block.register_function(
                collect_explicit_variadics,
                ["prefix_result", "total"],
                var_pos_name="selected_args",
                var_kw_name="selected_kwargs",
            )

            pipeline.run_all()

            self.assertEqual(registration.input_names, ["first", "second", "bonus"])
            self.assertEqual(pipeline.get_value("prefix_result"), "function-default")
            self.assertEqual(pipeline.get_value("total"), 10)

    def test_unregistered_variadics_do_not_use_same_named_visible_values(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "strict-empty-variadics",
                {"values": [1, 2], "named": {"bonus": 3}},
                Path(temp_dir),
                strict_mode=True,
            )
            block = pipeline.add_block("build", 1)
            registration = block.register_function(
                collect_implicit_variadics,
                ["positional_count", "keyword_count"],
            )

            pipeline.run_all()

            self.assertEqual(registration.input_names, [])
            self.assertEqual(pipeline.get_value("positional_count"), 0)
            self.assertEqual(pipeline.get_value("keyword_count"), 0)

    def test_switching_strict_mode_refreshes_resolution_and_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "strict-transition",
                {"value": "visible-value"},
                Path(temp_dir),
            )
            block = pipeline.add_block("build", 1)
            registration = block.register_function(use_default, ["result"])

            pipeline.set_strict_mode(True)
            pipeline.run_all()

            self.assertEqual(registration.input_names, [])
            self.assertEqual(pipeline.get_value("result"), "function-default")

    def test_callable_gate_keeps_existing_implicit_resolution(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "strict-gate",
                {"flag": True},
                Path(temp_dir),
                strict_mode=True,
            )
            pipeline.add_gate_block(config_gate)
            block = pipeline.add_block("build", 1)
            block.register_function(produce_value, ["result"])

            pipeline.run_all()

            self.assertEqual(pipeline.get_value("result"), 1)

    def test_placeholder_recovery_keeps_strict_callable_default(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline = PipelineHandler(
                "strict-recovery",
                {"value": "visible-value"},
                root / "project",
                strict_mode=True,
            )
            block = pipeline.add_block("build", 1)
            block.register_function(build_unpicklable, ["result"])
            pipeline.run_all()
            bundle = root / "bundle"
            pipeline.save_pipeline(bundle)

            loaded = PipelineHandler.load_pipeline(bundle, forced_deleting=True)
            result = loaded.get_value("result")

            self.assertIsInstance(result, UnpicklableResult)
            self.assertEqual(result.source, "function-default")
            self.assertTrue(
                any(
                    record.mode == "auto_resolve_placeholder:build"
                    and record.status == "success"
                    for record in loaded.run_history
                )
            )

    def test_non_strict_mode_keeps_implicit_resolution_and_advisory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "non-strict",
                {"value": "visible-value"},
                Path(temp_dir),
            )
            block = pipeline.add_block("build", 1)
            block.register_function(use_default, ["result"])
            pipeline.run_all()

            self.assertEqual(pipeline.get_value("result"), "visible-value")
            self.assertIn(
                "may be resolved implicitly",
                pipeline.logger.log_file_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
