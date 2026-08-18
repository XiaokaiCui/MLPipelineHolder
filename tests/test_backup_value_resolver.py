from __future__ import annotations

import __main__
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace, TracebackType
import unittest
from typing import final

from src.mlpipelineholder import PersistenceError, PipelineHandler, ResolutionError
from src.mlpipelineholder.backup_value_resolver import (
    resolve_saved_config_field,
    resolve_saved_root_variable,
)
from src.mlpipelineholder.function_registry import resolve_callable
from src.mlpipelineholder.models import CallableValueReference, RuntimeCallableReference, RuntimeValueReference


@dataclass
class SaveConfig:
    value: int


def importable(value: int) -> int:
    return value + 1


def local_scale(value: int) -> int:
    return value * 10


def _callable_reference_for(target_callable: Callable[[int], int]) -> CallableValueReference:
    _, import_path, callable_name = resolve_callable(target_callable)
    if import_path is None:
        raise AssertionError("expected an importable callable")
    return CallableValueReference(callable_name=callable_name, import_path=import_path)


def _serialized_config(data: dict[str, object]) -> object:
    return {
        "__pipeline_serialized_config__": True,
        "kind": "dict",
        "data": data,
    }


def _child_node(child_payload: dict[str, object], execution_priority: int = 1) -> dict[str, object]:
    return {
        "kind": "pipeline",
        "registration_name": str(child_payload["registration_name"]),
        "execution_priority": execution_priority,
        "payload": child_payload,
    }


def _saved_payload(
    registration_name: str,
    *,
    para_value_dict: dict[str, object] | None = None,
    config: object | None = None,
    nodes: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "registration_name": registration_name,
        "config": _serialized_config({}) if config is None else config,
        "execution_priority": None,
        "expression_runtime_code": None,
        "historical_result_log_path": None,
        "gate": None,
        "nodes": [] if nodes is None else nodes,
        "producer_outputs": {},
        "para_value_dict": {} if para_value_dict is None else para_value_dict,
        "artifact_registry": {},
        "run_history": [],
    }


@final
class _FakeMissingMainPlaceholder:
    pass


def _missing_main_placeholder_instance() -> object:
    return _FakeMissingMainPlaceholder()


def _is_missing_main_placeholder(value: object) -> bool:
    return isinstance(value, _FakeMissingMainPlaceholder)


@final
class MainBindingScope:
    _name: str
    _value: object
    _had_existing: bool
    _existing: object | None

    def __init__(self, name: str, value: object) -> None:
        self._name = name
        self._value = value
        self._had_existing = hasattr(__main__, name)
        self._existing = getattr(__main__, name, None)

    def __enter__(self) -> object:
        setattr(__main__, self._name, self._value)
        return self._value

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._had_existing:
            setattr(__main__, self._name, self._existing)
        elif hasattr(__main__, self._name):
            delattr(__main__, self._name)


class BackupValueResolverBaselineTests(unittest.TestCase):
    def test_baseline_save_load_restores_runtime_callable_value_for_partial_execution(self) -> None:
        namespace: dict[str, object] = {}
        function_source = "def runtime_increment(value: int) -> int:\n    return value + 1\n"
        exec(function_source, __main__.__dict__, namespace)
        runtime_increment = namespace["runtime_increment"]
        if not callable(runtime_increment):
            self.fail("runtime_increment should be callable")
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
                _ = pipeline.save_pipeline(save_dir)
                loaded = PipelineHandler.load_pipeline(save_dir, forced_deleting=True)
                _ = loaded.run_all()

                restored_callable = loaded.get_constant_value("target_callable")
                if restored_callable is not runtime_increment:
                    self.fail("loaded callable should match the current runtime binding")
                bound_callable = loaded.get_value("bound_callable")
                if not callable(bound_callable):
                    self.fail("bound_callable should be callable")
                self.assertEqual(bound_callable(), 3)
        finally:
            if had_existing_partial:
                setattr(__main__, "partial", existing_partial)
            elif hasattr(__main__, "partial"):
                delattr(__main__, "partial")
            if hasattr(__main__, "runtime_increment"):
                delattr(__main__, "runtime_increment")


class BackupValueResolverTests(unittest.TestCase):
    def test_resolve_saved_root_variable_prefers_root_value_before_descendants(self) -> None:
        child_payload = _saved_payload(
            "child",
            para_value_dict={"selected": "child-value"},
        )
        root_payload = _saved_payload(
            "root",
            para_value_dict={"selected": "root-value"},
            nodes=[_child_node(child_payload)],
        )

        resolved = resolve_saved_root_variable(
            root_payload,
            "selected",
            is_missing_main_placeholder=_is_missing_main_placeholder,
        )

        self.assertEqual(resolved, "root-value")

    def test_resolve_saved_root_variable_transforms_only_selected_graph(self) -> None:
        namespace: dict[str, object] = {}
        function_source = "def runtime_increment(value: int) -> int:\n    return value + 1\n"
        exec(function_source, __main__.__dict__, namespace)
        runtime_increment = namespace["runtime_increment"]
        if not callable(runtime_increment):
            self.fail("runtime_increment should be callable")

        shared_list = [RuntimeCallableReference(callable_name="runtime_increment")]
        cycle_list: list[object] = []
        cycle_list.append(cycle_list)
        selected_value = {
            importable: "callable-key",
            "real": importable,
            "importable_ref": _callable_reference_for(local_scale),
            "runtime_ref": RuntimeCallableReference(callable_name="runtime_increment"),
            "list_alias_a": shared_list,
            "list_alias_b": shared_list,
            "tuple_value": (shared_list, shared_list),
            "set_value": {importable},
            "frozenset_value": frozenset({local_scale}),
            "namespace": {
                "__pipeline_serialized_config__": True,
                "kind": "namespace",
                "class_name": "RecoveredConfig",
                "data": {
                    "callback": RuntimeCallableReference(callable_name="runtime_increment"),
                    "real": importable,
                },
            },
            "cycle": cycle_list,
        }
        root_payload = _saved_payload(
            "root",
            para_value_dict={
                "selected": selected_value,
                "broken_elsewhere": RuntimeCallableReference(callable_name="missing_elsewhere"),
            },
            nodes=[
                _child_node(
                    _saved_payload(
                        "child",
                        para_value_dict={"still_ignored": RuntimeValueReference("Lock", "<lock>", "not serializable")},
                    )
                )
            ],
        )

        with MainBindingScope("runtime_increment", runtime_increment):
            resolved_object = resolve_saved_root_variable(
                root_payload,
                "selected",
                is_missing_main_placeholder=_is_missing_main_placeholder,
            )

        if not isinstance(resolved_object, dict):
            self.fail("resolved root variable should be a dict")
        resolved = resolved_object
        list_alias_a = resolved.get("list_alias_a")
        list_alias_b = resolved.get("list_alias_b")
        tuple_value = resolved.get("tuple_value")
        namespace_value = resolved.get("namespace")
        cycle_value = resolved.get("cycle")
        runtime_ref = resolved.get("runtime_ref")
        self.assertIs(resolved.get("real"), importable)
        self.assertIs(resolved.get("importable_ref"), local_scale)
        self.assertIs(runtime_ref, runtime_increment)
        self.assertIs(resolved.get(importable), "callable-key")
        self.assertIs(list_alias_a, list_alias_b)
        if not isinstance(tuple_value, tuple):
            self.fail("tuple_value should be a tuple")
        self.assertIs(tuple_value[0], list_alias_a)
        self.assertEqual(resolved.get("set_value"), {importable})
        self.assertEqual(resolved.get("frozenset_value"), frozenset({local_scale}))
        if not isinstance(namespace_value, SimpleNamespace):
            self.fail("namespace should resolve to a SimpleNamespace")
        self.assertIs(namespace_value.callback, runtime_increment)
        self.assertIs(namespace_value.real, importable)
        if not isinstance(cycle_value, list):
            self.fail("cycle should resolve to a list")
        self.assertIs(cycle_value[0], cycle_value)
        if not callable(runtime_ref):
            self.fail("runtime_ref should resolve to a callable")
        self.assertEqual(partial(runtime_ref, 2)(), 3)

    def test_resolve_saved_root_variable_raises_for_missing_name(self) -> None:
        root_payload = _saved_payload("root")

        with self.assertRaises(ResolutionError):
            resolve_saved_root_variable(
                root_payload,
                "missing",
                is_missing_main_placeholder=_is_missing_main_placeholder,
            )

    def test_resolve_saved_root_variable_raises_for_missing_runtime_binding(self) -> None:
        root_payload = _saved_payload(
            "root",
            para_value_dict={
                "selected": RuntimeCallableReference(callable_name="missing_runtime_callable")
            },
        )

        with self.assertRaises(PersistenceError):
            resolve_saved_root_variable(
                root_payload,
                "selected",
                is_missing_main_placeholder=_is_missing_main_placeholder,
            )

    def test_resolve_saved_root_variable_raises_for_runtime_placeholder_inside_selected_graph(self) -> None:
        root_payload = _saved_payload(
            "root",
            para_value_dict={
                "selected": {"bad": RuntimeValueReference("Lock", "<lock>", "not serializable")}
            },
        )

        with self.assertRaises(PersistenceError):
            resolve_saved_root_variable(
                root_payload,
                "selected",
                is_missing_main_placeholder=_is_missing_main_placeholder,
            )

    def test_resolve_saved_root_variable_raises_for_missing_main_placeholder_inside_selected_graph(self) -> None:
        root_payload = _saved_payload(
            "root",
            para_value_dict={"selected": {"bad": _missing_main_placeholder_instance()}},
        )

        with self.assertRaises(PersistenceError):
            resolve_saved_root_variable(
                root_payload,
                "selected",
                is_missing_main_placeholder=_is_missing_main_placeholder,
            )

    def test_resolve_saved_config_field_is_receiver_local(self) -> None:
        namespace: dict[str, object] = {}
        function_source = "def runtime_increment(value: int) -> int:\n    return value + 1\n"
        exec(function_source, __main__.__dict__, namespace)
        runtime_increment = namespace["runtime_increment"]
        if not callable(runtime_increment):
            self.fail("runtime_increment should be callable")

        child_payload = _saved_payload(
            "child",
            config=_serialized_config(
                {"selected": RuntimeCallableReference(callable_name="runtime_increment")}
            ),
        )
        root_payload = _saved_payload(
            "root",
            config=_serialized_config({"root_only": importable}),
            nodes=[_child_node(child_payload)],
        )

        with self.assertRaises(ResolutionError):
            resolve_saved_config_field(
                root_payload,
                "selected",
                is_missing_main_placeholder=_is_missing_main_placeholder,
            )

        with MainBindingScope("runtime_increment", runtime_increment):
            resolved = resolve_saved_config_field(
                child_payload,
                "selected",
                is_missing_main_placeholder=_is_missing_main_placeholder,
            )

        if resolved is not runtime_increment:
            self.fail("config field should resolve to the current runtime callable")
        if not callable(resolved):
            self.fail("resolved config field should be callable")
        self.assertEqual(partial(resolved, 2)(), 3)

    def test_resolve_saved_config_field_raises_for_selected_placeholder(self) -> None:
        payload = _saved_payload(
            "root",
            config=_serialized_config(
                {"selected": RuntimeValueReference("Lock", "<lock>", "not serializable")}
            ),
        )

        with self.assertRaises(PersistenceError):
            resolve_saved_config_field(
                payload,
                "selected",
                is_missing_main_placeholder=_is_missing_main_placeholder,
            )
