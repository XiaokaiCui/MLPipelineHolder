from __future__ import annotations

import __main__
from dataclasses import dataclass
from functools import partial, wraps
from pathlib import Path
import pickle
import shutil
from tempfile import TemporaryDirectory
from typing import Any
import unittest
from unittest.mock import patch

from mlpipelineholder import (
    PersistenceError,
    PipelineHandler,
    ResolutionError,
    rename_args,
)
from mlpipelineholder.core.models import RunRecord, RuntimeCallableReference, RuntimeValueReference


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


def importable_outputs_with_ignored_slots(value: int) -> tuple[str, int, str, float]:
    return "unused-first", value, "unused-third", value / 2


def mapped_variadic(obj: int, *more_values: int, scale: int = 1, **extra_values: int) -> int:
    return (obj + sum(more_values) + sum(extra_values.values())) * scale


def call_with_value(target_callable, value: int) -> int:
    return target_callable(value)


def raw_increment(seed: int = 0, step: int = 1) -> int:
    return seed + step


class SaveLoadTests(unittest.TestCase):
    def local_callable(self, value):
        return value + 1

    def test_rename_args_round_trips_as_registration_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline = PipelineHandler(
                "rename",
                {"raw_value": 10},
                root / "project",
            )
            block = pipeline.add_block("increment", 1)
            block.register_function(
                rename_args(raw_increment, {"seed": "raw_value"}),
                ["result"],
            )
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("result"), 11)

            bundle = root / "bundle"
            pipeline.save_pipeline(bundle)
            loaded = PipelineHandler.load_pipeline(
                bundle,
                forced_deleting=True,
                trust_project=True,
            )
            loaded.run_all()

            registration = loaded.get_block("increment").functions[0]
            self.assertEqual(registration.param_mapping, {"seed": "raw_value"})
            self.assertEqual(loaded.get_value("result"), 11)

    def test_nested_rename_args_composes_and_round_trips(self) -> None:
        nested = rename_args(
            rename_args(raw_increment, {"seed": "first_name"}),
            {"first_name": "second_name"},
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline = PipelineHandler(
                "nested-rename",
                {"second_name": 10},
                root / "project",
            )
            block = pipeline.add_block("increment", 1)
            registration = block.register_function(nested, ["result"])

            self.assertEqual(nested(second_name=4), 5)
            self.assertEqual(registration.callable_obj, raw_increment)
            self.assertEqual(registration.param_mapping, {"seed": "second_name"})

            pipeline.run_all()
            self.assertEqual(pipeline.get_value("result"), 11)

            bundle = root / "bundle"
            pipeline.save_pipeline(bundle)
            loaded = PipelineHandler.load_pipeline(
                bundle,
                forced_deleting=True,
                trust_project=True,
            )
            loaded_registration = loaded.get_block("increment").functions[0]
            self.assertEqual(
                loaded_registration.param_mapping,
                {"seed": "second_name"},
            )
            loaded.run_all()
            self.assertEqual(loaded.get_value("result"), 11)

    def test_rename_args_preserves_explicit_identity_mapping_in_strict_mode(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "rename-strict",
                {"seed": 10},
                Path(temp_dir),
                strict_mode=True,
            )
            block = pipeline.add_block("increment", 1)
            registration = block.register_function(
                rename_args(raw_increment, {"seed": "raw_value"}),
                ["result"],
                param_mapping={"raw_value": "seed"},
            )

            pipeline.run_all()

            self.assertEqual(registration.param_mapping, {"seed": "seed"})
            self.assertEqual(pipeline.get_value("result"), 11)

    def test_outer_decorator_around_rename_args_is_not_discarded(self) -> None:
        renamed = rename_args(raw_increment, {"seed": "raw_value"})

        @wraps(renamed)
        def doubled(*args: Any, **kwargs: Any) -> int:
            return 2 * renamed(*args, **kwargs)

        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler(
                "rename-decorated",
                {"raw_value": 10},
                Path(temp_dir),
            )
            block = pipeline.add_block("increment", 1)
            block.register_function(doubled, ["result"])

            pipeline.run_all()

            self.assertEqual(pipeline.get_value("result"), 22)

    def test_loading_requires_explicit_trust_before_accessing_project(self) -> None:
        loaders: tuple[Any, ...] = (
            PipelineHandler.load_pipeline,
            PipelineHandler.load_project,
        )
        untrusted_values = (False, None, 1, "yes")
        with TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "does-not-exist"
            for loader in loaders:
                with self.subTest(loader=loader.__name__, trust_project="omitted"):
                    with self.assertRaisesRegex(TypeError, "trust_project"):
                        loader(project)
                for trust_project in untrusted_values:
                    with self.subTest(
                        loader=loader.__name__,
                        trust_project=trust_project,
                    ):
                        with patch.object(Path, "open") as open_file:
                            with self.assertRaisesRegex(
                                PersistenceError,
                                "only self-created or fully trusted pipeline projects",
                            ):
                                loader(project, trust_project=trust_project)
                            open_file.assert_not_called()

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
                PipelineHandler.load_project(save_dir, forced_deleting=True, trust_project=True)

    def test_importable_callable_round_trips(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("persist", SaveConfig(value=2), tmp_path / "project")
            block = pipeline.add_block("block", 1)
            block.register_function(importable, ["result"])
            pipeline.run_all()

            save_dir = tmp_path / "bundle"
            pipeline.save_project(save_dir)
            loaded = PipelineHandler.load_project(save_dir, forced_deleting=True, trust_project=True)

            self.assertEqual(loaded.para_value_dict["result"], 3)

    def test_ignored_output_slots_round_trip(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("persist-ignored", SaveConfig(value=8), tmp_path / "project")
            block = pipeline.add_block("block", 1)
            block.register_function(
                importable_outputs_with_ignored_slots,
                ["_", "o2e_root", "_", "bce"],
                save_to_disk=["o2e_root"],
            )
            pipeline.run_all()

            save_dir = tmp_path / "bundle"
            pipeline.save_project(save_dir)
            loaded = PipelineHandler.load_project(save_dir, forced_deleting=True, trust_project=True)

            loaded_registration = loaded.get_block("block").functions[0]
            self.assertEqual(
                loaded_registration.output_names,
                ["_", "o2e_root", "_", "bce"],
            )
            self.assertEqual(loaded_registration.produced_output_names, ["o2e_root", "bce"])
            self.assertEqual(loaded.list_declared_outputs(), {"o2e_root", "bce"})
            self.assertEqual(loaded.get_value("o2e_root"), 8)
            self.assertEqual(loaded.get_value("bce"), 4.0)
            self.assertNotIn("_", loaded.para_value_dict)

    def test_atom_ignored_output_slots_round_trip(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler(
                "persist-atom-ignored",
                SaveConfig(value=8),
                tmp_path / "project",
            )
            pipeline.create_atom_child_pipeline(
                child_name="metrics",
                execution_priority=1,
                target_function=importable_outputs_with_ignored_slots,
                output_variable_names=["_", "o2e_root", "_", "bce"],
                save_to_disk_lst=["o2e_root"],
                forced=True,
            )
            _ = pipeline.run_all()

            save_dir = tmp_path / "bundle"
            _ = pipeline.save_project(save_dir)
            loaded = PipelineHandler.load_project(save_dir, forced_deleting=True, trust_project=True)
            child = loaded.get_child_pipeline("metrics")
            registration = child.blocks[0].functions[0]

            self.assertEqual(
                registration.output_names,
                ["_", "o2e_root", "_", "bce"],
            )
            self.assertEqual(registration.produced_output_names, ["o2e_root", "bce"])
            self.assertEqual(loaded.list_declared_outputs(), {"o2e_root", "bce"})
            self.assertEqual(loaded.get_value("o2e_root"), 8)
            self.assertEqual(loaded.get_value("bce"), 4.0)
            self.assertNotIn("_", loaded.para_value_dict)

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

            loaded = PipelineHandler.load_pipeline(backup_dir, forced_deleting=True, trust_project=True)

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
                pipeline.set_constant_value("target_callable", importable)
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

                loaded = PipelineHandler.load_pipeline(save_dir, forced_deleting=True, trust_project=True)
                loaded.run_all()

                bound_callable = loaded.get_value("bound_callable")
                self.assertTrue(callable(bound_callable))
                self.assertEqual(bound_callable(), 3)
        finally:
            if had_existing_partial:
                setattr(__main__, "partial", existing_partial)
            elif hasattr(__main__, "partial"):
                delattr(__main__, "partial")

    def test_registered_partial_restores_main_callable_value_before_execution(self) -> None:
        namespace: dict[str, object] = {}
        exec(
            "def runtime_increment(value: int) -> int:\n"
            "    return value + 1\n",
            __main__.__dict__,
            namespace,
        )
        runtime_increment = namespace["runtime_increment"]
        existing_partial = getattr(__main__, "partial", None)
        had_existing_partial = hasattr(__main__, "partial")
        setattr(__main__, "partial", partial)
        setattr(__main__, "runtime_increment", runtime_increment)

        try:
            with TemporaryDirectory() as temp_dir:
                tmp_path = Path(temp_dir)
                pipeline = PipelineHandler(
                    "persist-main-partial-value",
                    SaveConfig(value=2),
                    tmp_path / "project",
                )
                pipeline.set_constant_value("target_callable", runtime_increment)
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
                loaded = PipelineHandler.load_pipeline(save_dir, forced_deleting=True, trust_project=True)
                loaded.run_all()

                self.assertIs(loaded.get_constant_value("target_callable"), runtime_increment)
                self.assertEqual(loaded.get_value("bound_callable")(), 3)
        finally:
            if had_existing_partial:
                setattr(__main__, "partial", existing_partial)
            elif hasattr(__main__, "partial"):
                delattr(__main__, "partial")
            if hasattr(__main__, "runtime_increment"):
                delattr(__main__, "runtime_increment")

    def test_registered_partial_restores_main_callable_config_before_execution(self) -> None:
        namespace: dict[str, object] = {}
        exec(
            "def runtime_increment(value: int) -> int:\n"
            "    return value + 1\n",
            __main__.__dict__,
            namespace,
        )
        runtime_increment = namespace["runtime_increment"]
        existing_partial = getattr(__main__, "partial", None)
        had_existing_partial = hasattr(__main__, "partial")
        setattr(__main__, "partial", partial)
        setattr(__main__, "runtime_increment", runtime_increment)

        try:
            with TemporaryDirectory() as temp_dir:
                tmp_path = Path(temp_dir)
                pipeline = PipelineHandler(
                    "persist-main-partial-config",
                    {
                        "value": 2,
                        "target_callable": runtime_increment,
                    },
                    tmp_path / "project",
                )
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
                loaded = PipelineHandler.load_pipeline(save_dir, forced_deleting=True, trust_project=True)
                loaded.run_all()

                self.assertIs(
                    loaded.get_config_value("target_callable"),
                    runtime_increment,
                )
                self.assertEqual(loaded.get_value("bound_callable")(), 3)
        finally:
            if had_existing_partial:
                setattr(__main__, "partial", existing_partial)
            elif hasattr(__main__, "partial"):
                delattr(__main__, "partial")
            if hasattr(__main__, "runtime_increment"):
                delattr(__main__, "runtime_increment")

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
            loaded = PipelineHandler.load_project(save_dir, forced_deleting=True, trust_project=True)
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
            loaded = PipelineHandler.load_project(save_dir, forced_deleting=True, trust_project=True)
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

                loaded = PipelineHandler.load_project(save_dir, forced_deleting=True, trust_project=True)

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

                loaded = PipelineHandler.load_project(save_dir, forced_deleting=True, trust_project=True)

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

                loaded = PipelineHandler.load_project(save_dir, forced_deleting=True, trust_project=True)
                loaded_child = loaded.get_child_pipeline("child")

                self.assertEqual(loaded_child.get_config_value("value"), 2)
                return
        finally:
            if hasattr(__main__, "LegacyMainConfig"):
                delattr(__main__, "LegacyMainConfig")

    def test_load_project_replaces_missing_main_runtime_value_with_none(self) -> None:
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
                pipeline.set_constant_value("runtime_helper", LegacyMainRuntimeHelper(name="helper"))

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

                loaded = PipelineHandler.load_project(
                    save_dir,
                    forced_deleting=True,
                    trust_project=True,
                )

                self.assertEqual(loaded.get_config_value("value"), 2)
                self.assertIsNone(loaded.get_constant_value("runtime_helper"))
                return
        finally:
            if hasattr(__main__, "LegacyMainConfig"):
                delattr(__main__, "LegacyMainConfig")
            if hasattr(__main__, "LegacyMainRuntimeHelper"):
                delattr(__main__, "LegacyMainRuntimeHelper")

    def test_source_package_loader_resolves_installed_package_pickle_to_active_class(self) -> None:
        # Given: a pickle written by the installed package namespace.
        payload = b"cmlpipelineholder.core.models\nRunRecord\n."

        # When: it is loaded through the directly imported source package.
        loaded_class = PipelineHandler._load_pickle_with_missing_class_fallback(payload)

        # Then: class resolution stays within the active source package.
        self.assertIs(loaded_class, RunRecord)

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
            loaded = PipelineHandler.load_project(save_dir, forced_deleting=True, trust_project=True)
            loaded.run_all()

            self.assertEqual(loaded.get_value("result"), 3.0)
            self.assertEqual(loaded.get_expression_runtime_code(), "from math import sqrt")

    def test_importable_callable_value_round_trips_as_live_callable(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("persist-callable-value", SaveConfig(value=2), tmp_path / "project")
            pipeline.set_constant_value("callable_value", importable)
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
            loaded = PipelineHandler.load_project(save_dir, forced_deleting=True, trust_project=True)
            loaded_callable = loaded.get_constant_value("callable_value")
            loaded.run_all()

            self.assertTrue(callable(loaded_callable))
            self.assertEqual(loaded_callable(2), 3)
            self.assertEqual(loaded.get_value("result"), 3)

    def test_non_importable_callable_value_loads_as_reference_placeholder(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("persist-nonimportable-callable", SaveConfig(value=2), tmp_path / "project")
            pipeline.set_constant_value("callable_value", self.local_callable)

            save_dir = tmp_path / "bundle"
            pipeline.save_project(save_dir)
            loaded = PipelineHandler.load_project(save_dir, forced_deleting=True, trust_project=True)

            with self.assertRaises(ResolutionError):
                loaded.get_constant_value("callable_value")

    def test_main_callable_value_restores_from_loading_runtime(self) -> None:
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
                pipeline.set_constant_value("callable_value", __main__.main_increment)

                save_dir = tmp_path / "bundle"
                pipeline.save_project(save_dir)
                loaded = PipelineHandler.load_project(save_dir, forced_deleting=True, trust_project=True)

                self.assertIs(loaded.get_constant_value("callable_value"), namespace["main_increment"])
        finally:
            if hasattr(__main__, "main_increment"):
                delattr(__main__, "main_increment")

    def test_main_callable_value_must_be_available_during_load(self) -> None:
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
                pipeline = PipelineHandler(
                    "persist-main-callable",
                    SaveConfig(value=2),
                    tmp_path / "project",
                )
                pipeline.set_constant_value("callable_value", __main__.main_increment)
                save_dir = tmp_path / "bundle"
                pipeline.save_project(save_dir)
                delattr(__main__, "main_increment")

                with self.assertRaisesRegex(
                    PersistenceError,
                    "main_increment.*pipeline value 'callable_value'.*__main__.*before loading",
                ):
                    PipelineHandler.load_project(save_dir, forced_deleting=True, trust_project=True)
        finally:
            if hasattr(__main__, "main_increment"):
                delattr(__main__, "main_increment")

    def test_static_method_callable_value_round_trips(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pipeline = PipelineHandler("persist-static-callable", SaveConfig(value=2), tmp_path / "project")
            pipeline.set_constant_value("callable_value", CallableContainer.static_increment)

            save_dir = tmp_path / "bundle"
            pipeline.save_project(save_dir)
            loaded = PipelineHandler.load_project(save_dir, forced_deleting=True, trust_project=True)

            restored = loaded.get_constant_value("callable_value")
            self.assertTrue(callable(restored))
            self.assertEqual(restored(41), 42)

    def test_save_pipeline_archives_timestamped_log_snapshot_in_history_logs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            project_dir = tmp_path / "project"
            pipeline = PipelineHandler("persist", SaveConfig(value=2), project_dir)
            block = pipeline.add_block("block", 1)
            block.register_function(importable, ["result"])
            pipeline.run_all()
            pipeline.save_pipeline()

            history_root = project_dir / "history_logs"
            self.assertTrue(history_root.is_dir())
            snapshots = list(history_root.glob("*.log"))
            self.assertEqual(len(snapshots), 1)
            self.assertRegex(
                snapshots[0].name,
                r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3}\.log$",
            )
            content = snapshots[0].read_text(encoding="utf-8")
            self.assertIn(" INFO ", content)
            self.assertIn("Pipeline has been saved to project root", content)

    def test_save_pipeline_archives_one_snapshot_per_save(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            project_dir = tmp_path / "project"
            pipeline = PipelineHandler("persist", SaveConfig(value=2), project_dir)
            block = pipeline.add_block("block", 1)
            block.register_function(importable, ["result"])
            pipeline.run_all()

            pipeline.save_pipeline()
            pipeline.save_pipeline()

            snapshots = list((project_dir / "history_logs").glob("*.log"))
            self.assertEqual(len(snapshots), 2)
            self.assertEqual(len({snapshot.name for snapshot in snapshots}), 2)

    def test_save_pipeline_to_new_path_archives_snapshot_in_project_root(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            project_dir = tmp_path / "project"
            pipeline = PipelineHandler("persist", SaveConfig(value=2), project_dir)
            block = pipeline.add_block("block", 1)
            block.register_function(importable, ["result"])
            pipeline.run_all()

            save_dir = tmp_path / "bundle"
            pipeline.save_pipeline(save_dir)

            snapshots = list((project_dir / "history_logs").glob("*.log"))
            self.assertEqual(len(snapshots), 1)

    def test_load_restores_history_logs_from_saved_bundle(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            project_dir = tmp_path / "project"
            pipeline = PipelineHandler("persist", SaveConfig(value=2), project_dir)
            block = pipeline.add_block("block", 1)
            block.register_function(importable, ["result"])
            pipeline.run_all()
            pipeline.save_pipeline()

            save_dir = tmp_path / "bundle"
            pipeline.save_pipeline(save_dir)

            bundle_snapshots = sorted((save_dir / "history_logs").glob("*.log"))
            self.assertTrue(bundle_snapshots)

            loaded = PipelineHandler.load_pipeline(save_dir, forced_deleting=True, trust_project=True)
            loaded_snapshots = sorted((loaded.project_root / "history_logs").glob("*.log"))
            self.assertEqual(
                [snapshot.name for snapshot in loaded_snapshots],
                [snapshot.name for snapshot in bundle_snapshots],
            )
