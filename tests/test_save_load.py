from __future__ import annotations

import __main__
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import pickle
import shutil
from tempfile import TemporaryDirectory
import unittest

from src.mlpipelineholder import PersistenceError, PipelineHandler
from src.mlpipelineholder.models import RuntimeCallableReference, RuntimeValueReference


@dataclass
class SaveConfig:
    value: int


@dataclass
class NotebookMainConfig:
    value: int


@dataclass
class LegacyMainConfig:
    value: int


@dataclass
class LegacyMainRuntimeHelper:
    name: str


class CallableContainer:
    @staticmethod
    def static_increment(value: int) -> int:
        return value + 1


def importable(value: int) -> int:
    return value + 1


def mapped_variadic(obj: int, *more_values: int, scale: int = 1, **extra_values: int) -> int:
    return (obj + sum(more_values) + sum(extra_values.values())) * scale


def call_with_value(target_callable, value: int) -> int:
    return target_callable(value)


class SaveLoadTests(unittest.TestCase):
    def local_callable(self, value):
        return value + 1

    def test_runtime_registered_callable_must_be_available_during_load(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler(
                "persist",
                SaveConfig(value=1),
                tmp_path / "project",
            )
            block = pipeline.add_block("block", 1)

            block.register_function(self.local_callable, ["result"])

            save_dir = tmp_path / "bundle"
            pipeline.save_project(save_dir)

            with self.assertRaisesRegex(
                PersistenceError,
                "local_callable.*__main__.*before loading",
            ):
                PipelineHandler.load_project(save_dir, forced_deleting=True)

    def test_importable_callable_round_trips(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("persist", SaveConfig(value=2), tmp_path / "project")
            block = pipeline.add_block("block", 1)
            block.register_function(importable, ["result"])
            pipeline.run_all()

            save_dir = tmp_path / "bundle"
            pipeline.save_project(save_dir)
            loaded = PipelineHandler.load_project(save_dir, forced_deleting=True)

            self.assertEqual(loaded.para_value_dict["result"], 3)

    def test_explicit_save_path_preserves_disk_backed_output_without_original_tree(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            project_dir = tmp_path / "project"
            pipeline = PipelineHandler(
                "persist-artifact-backup",
                SaveConfig(value=2),
                project_dir,
            )
            block = pipeline.add_block("produce_output", 1)
            block.register_function(
                importable,
                ["output_df"],
                save_to_disk=["output_df"],
            )
            pipeline.run_all()

            backup_dir = tmp_path / "backup"
            pipeline.save_pipeline(backup_dir)
            shutil.rmtree(project_dir)

            loaded = PipelineHandler.load_pipeline(backup_dir, forced_deleting=True)

            self.assertEqual(loaded.get_value("output_df"), 3)

    def test_explicit_save_path_rejects_overlap_with_project_tree(self) -> None:
        with TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "project"
            pipeline = PipelineHandler(
                "persist-overlap",
                SaveConfig(value=2),
                project_dir,
            )

            with self.assertRaisesRegex(PersistenceError, "overlapping directory"):
                pipeline.save_pipeline(project_dir / "backup")

    def test_registered_partial_round_trips_from_loading_runtime_and_executes(self) -> None:
        existing_partial = getattr(__main__, "partial", None)
        had_existing_partial = hasattr(__main__, "partial")
        setattr(__main__, "partial", partial)

        try:
            with TemporaryDirectory() as temp_dir:
                tmp_path = Path(temp_dir)
                pipeline = PipelineHandler(
                    "persist-runtime-partial",
                    SaveConfig(value=2),
                    tmp_path / "project",
                )
                pipeline.set_value("target_callable", importable)
                pipeline.create_atom_child_pipeline(
                    child_name="bind_runtime_callable",
                    execution_priority=1,
                    target_function=partial,
                    output_variable_names="bound_callable",
                    param_mapping={"func": "target_callable"},
                    kwargs_dct={"value": "value"},
                )

                save_dir = tmp_path / "bundle"
                pipeline.save_pipeline(save_dir)
                with (save_dir / "pipeline_state.pkl").open("rb") as handle:
                    payload = pickle.load(handle)
                function_payload = payload["nodes"][0]["payload"]["nodes"][0][
                    "functions"
                ][0]
                self.assertIsNone(function_payload["import_path"])
                self.assertEqual(
                    function_payload["runtime_callable_reference"],
                    RuntimeCallableReference(callable_name="partial"),
                )
                self.assertNotIn("callable_obj", function_payload)

                loaded = PipelineHandler.load_pipeline(save_dir, forced_deleting=True)
                loaded.run_all()

                bound_callable = loaded.get_value("bound_callable")
                self.assertTrue(callable(bound_callable))
                self.assertEqual(bound_callable(), 3)
        finally:
            if had_existing_partial:
                setattr(__main__, "partial", existing_partial)
            elif hasattr(__main__, "partial"):
                delattr(__main__, "partial")

    def test_mapping_metadata_round_trips_for_variadic_function(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler(
                "persist-mapped",
                {
                    "payload": 2,
                    "scale_value": 3,
                    "extra_args": [4, 5],
                    "extra_kwargs": {"bonus": 6},
                },
                tmp_path / "project",
            )
            block = pipeline.add_block("block", 1)
            block.register_function(
                mapped_variadic,
                ["result"],
                param_mapping={"obj": "payload", "scale": "scale_value"},
                var_pos_name="extra_args",
                var_kw_name="extra_kwargs",
            )
            pipeline.run_all()

            save_dir = tmp_path / "bundle"
            pipeline.save_project(save_dir)
            loaded = PipelineHandler.load_project(save_dir, forced_deleting=True)
            loaded.run_all()

            self.assertEqual(loaded.get_value("result"), 51)

    def test_expression_registration_round_trips(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("persist-expression", SaveConfig(value=2), tmp_path / "project")
            block = pipeline.add_block("block", 1)
            block.register_expression("result = value + 5", save_to_disk=True)
            pipeline.run_all()

            save_dir = tmp_path / "bundle"
            pipeline.save_project(save_dir)
            loaded = PipelineHandler.load_project(save_dir, forced_deleting=True)
            loaded.run_all()

            self.assertEqual(loaded.get_value("result"), 7)

    def test_load_project_does_not_require_predeclared_main_config_class_for_new_save(self) -> None:
        NotebookMainConfig.__module__ = "__main__"
        setattr(__main__, "NotebookMainConfig", NotebookMainConfig)

        try:
            with TemporaryDirectory() as temp_dir:
                tmp_path = Path(temp_dir)
                pipeline = PipelineHandler(
                    "persist-main-config",
                    NotebookMainConfig(value=2),
                    tmp_path / "project",
                )
                block = pipeline.add_block("block", 1)
                block.register_function(importable, ["result"])
                pipeline.run_all()

                save_dir = tmp_path / "bundle"
                pipeline.save_project(save_dir)
                delattr(__main__, "NotebookMainConfig")

                loaded = PipelineHandler.load_project(save_dir, forced_deleting=True)

                self.assertEqual(loaded.get_value("result"), 3)
                self.assertEqual(loaded.get_config_value("value"), 2)
                return
        finally:
            if hasattr(__main__, "NotebookMainConfig"):
                delattr(__main__, "NotebookMainConfig")

    def test_load_project_supports_old_payload_without_predeclared_main_config_class(self) -> None:
        LegacyMainConfig.__module__ = "__main__"
        setattr(__main__, "LegacyMainConfig", LegacyMainConfig)

        try:
            with TemporaryDirectory() as temp_dir:
                tmp_path = Path(temp_dir)
                pipeline = PipelineHandler(
                    "persist-legacy-main-config",
                    LegacyMainConfig(value=2),
                    tmp_path / "project",
                )
                block = pipeline.add_block("block", 1)
                block.register_function(importable, ["result"])
                pipeline.run_all()

                save_dir = tmp_path / "bundle"
                save_dir.mkdir(parents=True, exist_ok=True)
                legacy_config = LegacyMainConfig(value=2)
                payload = pipeline._serialize_payload()
                payload["config"] = legacy_config
                with (save_dir / "pipeline_state.pkl").open("wb") as handle:
                    pickle.dump(payload, handle)
                with (save_dir / "pipeline_meta.pkl").open("wb") as handle:
                    pickle.dump({"pipeline_directory": str(save_dir)}, handle)
                delattr(__main__, "LegacyMainConfig")

                loaded = PipelineHandler.load_project(save_dir, forced_deleting=True)

                self.assertEqual(loaded.get_value("result"), 3)
                self.assertEqual(loaded.get_config_value("value"), 2)
                return
        finally:
            if hasattr(__main__, "LegacyMainConfig"):
                delattr(__main__, "LegacyMainConfig")

    def test_load_project_supports_old_nested_child_payload_without_predeclared_main_config_class(self) -> None:
        LegacyMainConfig.__module__ = "__main__"
        setattr(__main__, "LegacyMainConfig", LegacyMainConfig)

        try:
            with TemporaryDirectory() as temp_dir:
                tmp_path = Path(temp_dir)
                parent = PipelineHandler(
                    "persist-legacy-parent",
                    SaveConfig(value=1),
                    tmp_path / "parent-project",
                )
                child = PipelineHandler(
                    "child",
                    LegacyMainConfig(value=2),
                    tmp_path / "child-project",
                )
                block = child.add_block("block", 1)
                block.register_function(importable, ["result"])
                parent.add_child_pipeline(child, 1)

                save_dir = tmp_path / "bundle"
                save_dir.mkdir(parents=True, exist_ok=True)
                payload = parent._serialize_payload()
                payload["nodes"][0]["payload"]["config"] = LegacyMainConfig(value=2)
                with (save_dir / "pipeline_state.pkl").open("wb") as handle:
                    pickle.dump(payload, handle)
                with (save_dir / "pipeline_meta.pkl").open("wb") as handle:
                    pickle.dump({"pipeline_directory": str(save_dir)}, handle)
                delattr(__main__, "LegacyMainConfig")

                loaded = PipelineHandler.load_project(save_dir, forced_deleting=True)
                loaded_child = loaded.get_child_pipeline("child")

                self.assertEqual(loaded_child.get_config_value("value"), 2)
                return
        finally:
            if hasattr(__main__, "LegacyMainConfig"):
                delattr(__main__, "LegacyMainConfig")

    def test_load_project_rejects_missing_main_class_outside_config(self) -> None:
        LegacyMainConfig.__module__ = "__main__"
        LegacyMainRuntimeHelper.__module__ = "__main__"
        setattr(__main__, "LegacyMainConfig", LegacyMainConfig)
        setattr(__main__, "LegacyMainRuntimeHelper", LegacyMainRuntimeHelper)

        try:
            with TemporaryDirectory() as temp_dir:
                tmp_path = Path(temp_dir)
                pipeline = PipelineHandler(
                    "persist-legacy-runtime-helper",
                    LegacyMainConfig(value=2),
                    tmp_path / "project",
                )
                block = pipeline.add_block("block", 1)
                block.register_function(importable, ["result"])
                pipeline.set_value("runtime_helper", LegacyMainRuntimeHelper(name="helper"))

                save_dir = tmp_path / "bundle"
                save_dir.mkdir(parents=True, exist_ok=True)
                payload = pipeline._serialize_payload()
                payload["config"] = LegacyMainConfig(value=2)
                payload["manual_values"] = {"runtime_helper": LegacyMainRuntimeHelper(name="helper")}
                with (save_dir / "pipeline_state.pkl").open("wb") as handle:
                    pickle.dump(payload, handle)
                with (save_dir / "pipeline_meta.pkl").open("wb") as handle:
                    pickle.dump({"pipeline_directory": str(save_dir)}, handle)
                delattr(__main__, "LegacyMainConfig")
                delattr(__main__, "LegacyMainRuntimeHelper")

                with self.assertRaises(PersistenceError):
                    PipelineHandler.load_project(save_dir, forced_deleting=True)
                return
        finally:
            if hasattr(__main__, "LegacyMainConfig"):
                delattr(__main__, "LegacyMainConfig")
            if hasattr(__main__, "LegacyMainRuntimeHelper"):
                delattr(__main__, "LegacyMainRuntimeHelper")

    def test_expression_runtime_round_trips_through_save_load(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("persist-runtime", SaveConfig(value=9), tmp_path / "project")
            pipeline.define_expression_runtime("from math import sqrt")
            block = pipeline.add_block("block", 1)
            block.register_expression("result = sqrt(value)")
            pipeline.run_all()

            save_dir = tmp_path / "bundle"
            pipeline.save_project(save_dir)
            loaded = PipelineHandler.load_project(save_dir, forced_deleting=True)
            loaded.run_all()

            self.assertEqual(loaded.get_value("result"), 3.0)
            self.assertEqual(loaded.get_expression_runtime_code(), "from math import sqrt")

    def test_importable_callable_value_round_trips_as_live_callable(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("persist-callable-value", SaveConfig(value=2), tmp_path / "project")
            pipeline.set_value("callable_value", importable)
            block = pipeline.add_block("block", 1)
            block.register_function(
                call_with_value,
                ["result"],
                param_mapping={
                    "target_callable": "callable_value",
                    "value": "value",
                },
            )
            pipeline.run_all()

            save_dir = tmp_path / "bundle"
            pipeline.save_project(save_dir)
            loaded = PipelineHandler.load_project(save_dir, forced_deleting=True)
            loaded_callable = loaded.get_value("callable_value")
            loaded.run_all()

            self.assertTrue(callable(loaded_callable))
            self.assertEqual(loaded_callable(2), 3)
            self.assertEqual(loaded.get_value("result"), 3)

    def test_non_importable_callable_value_loads_as_reference_placeholder(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("persist-nonimportable-callable", SaveConfig(value=2), tmp_path / "project")
            pipeline.set_value("callable_value", self.local_callable)

            save_dir = tmp_path / "bundle"
            pipeline.save_project(save_dir)
            loaded = PipelineHandler.load_project(save_dir, forced_deleting=True)

            self.assertIsInstance(loaded.get_value("callable_value"), RuntimeValueReference)

    def test_main_callable_value_loads_as_reference_placeholder(self) -> None:
        namespace: dict[str, object] = {}
        exec(
            "def main_increment(value: int) -> int:\n"
            "    return value + 1\n",
            __main__.__dict__,
            namespace,
        )
        setattr(__main__, "main_increment", namespace["main_increment"])

        try:
            with TemporaryDirectory() as temp_dir:
                tmp_path = Path(temp_dir)
                pipeline = PipelineHandler("persist-main-callable", SaveConfig(value=2), tmp_path / "project")
                pipeline.set_value("callable_value", __main__.main_increment)

                save_dir = tmp_path / "bundle"
                pipeline.save_project(save_dir)
                loaded = PipelineHandler.load_project(save_dir, forced_deleting=True)

                self.assertIsInstance(loaded.get_value("callable_value"), RuntimeValueReference)
        finally:
            if hasattr(__main__, "main_increment"):
                delattr(__main__, "main_increment")

    def test_static_method_callable_value_loads_as_reference_placeholder_when_not_round_trippable(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("persist-static-callable", SaveConfig(value=2), tmp_path / "project")
            pipeline.set_value("callable_value", CallableContainer.static_increment)

            save_dir = tmp_path / "bundle"
            pipeline.save_project(save_dir)
            loaded = PipelineHandler.load_project(save_dir, forced_deleting=True)

            self.assertIsInstance(loaded.get_value("callable_value"), RuntimeValueReference)
