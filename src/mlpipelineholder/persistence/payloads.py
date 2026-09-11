"""Payload serialization and partial-callable persistence helpers."""

from __future__ import annotations

import pickle
import sys
import warnings
from dataclasses import fields, is_dataclass
from functools import partial
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from ..core.models import (
    ArtifactRecord,
    CallableValueReference,
    DataclassValueReference,
    ExpressionRegistration,
    FunctionRegistration,
    RuntimeCallableReference,
    RuntimeValueReference,
    TorchStateArtifactRecord,
)
from ..exceptions import PersistenceError, RegistrationError
from ..execution.function_registry import resolve_callable
from ..integrations.optuna.support import (
    OPTUNA_STUDIES_DB_NAME,
    is_optuna_sampler,
    is_optuna_study,
)
from ..state.output_pointers import OutputAddress, OutputPointer
from .artifacts.serializers import choose_serializer
from .artifacts.store import ArtifactStore

from ..core.base import PipelineBase


def _is_child_pipeline(node: object) -> bool:
    return isinstance(node, PipelineBase)


class PayloadMixin:
    """Serialize runtime payloads and restore registered partial callables."""

    if TYPE_CHECKING:
        registration_name: str = ""
        config: Any = None
        execution_priority: Any = None
        expression_runtime_code: Any = None
        historical_result_log_path: Any = None
        gate_block: Any = None
        _is_atom: bool = False
        torch_load_weights_only: bool = False
        producer_outputs: dict[str, dict[str, Any]] = {}
        para_value_dict: dict[str, Any] = {}
        artifact_registry: dict[str, Any] = {}
        run_history: list[Any] = []

        def qualified_node_name(self, node_name: str) -> str: ...
        def _sorted_nodes(self) -> list[Any]: ...
        @staticmethod
        def _serialize_config_for_save(config: Any) -> Any: ...
        @staticmethod
        def _callable_reference_round_trips(value: Any, import_path: str) -> bool: ...
        @staticmethod
        def _restore_callable_value(reference: CallableValueReference) -> Any: ...
        @staticmethod
        def _restore_runtime_callable(
            reference: RuntimeCallableReference, owner_label: str
        ) -> Any: ...
        @staticmethod
        def _deserialize_config_value(
            value: Any, *, verbose: bool = False, warn: Any | None = None
        ) -> Any: ...
        @classmethod
        def _serialize_dataclass_field_value(cls, value: Any) -> Any: ...
        @staticmethod
        def _is_missing_main_placeholder(value: Any) -> bool: ...

    def _serialize_node_for_save(
        self,
        node: Any,
        target_root: Path,
        cache: dict[int, Any],
    ) -> dict[str, Any]:
        if _is_child_pipeline(node):
            return {
                "kind": "pipeline",
                "registration_name": node.registration_name,
                "execution_priority": node.execution_priority,
                "overridden_outputs": {
                    output_name: (address.pipeline_name, address.node_name)
                    for output_name, address in sorted(
                        node._overridden_outputs.items()
                    )
                },
                "payload": node._serialize_payload_for_save(
                    target_root / "children" / node.registration_name,
                    cache,
                ),
            }
        return self._serialize_node(node)

    def _serialize_runtime_value_for_save(
        self,
        value: Any,
        target_root: Path,
        cache: dict[int, Any],
        node_name: str,
        output_name: str,
        sibling_outputs: dict[str, Any] | None = None,
    ) -> Any:
        if isinstance(value, ArtifactRecord):
            return value
        value_id = id(value)
        if value_id in cache:
            return cache[value_id]
        serialized = self._persist_runtime_value(
            value,
            target_root,
            cache,
            node_name,
            output_name,
            sibling_outputs=sibling_outputs,
        )
        cache[value_id] = serialized
        return serialized

    def _persist_runtime_value(
        self,
        value: Any,
        target_root: Path,
        cache: dict[int, Any],
        node_name: str,
        output_name: str,
        sibling_outputs: dict[str, Any] | None = None,
    ) -> Any:
        if is_optuna_study(value) or is_optuna_sampler(value):
            save_store = ArtifactStore(target_root)
            return save_store.save(
                variable_name=output_name,
                value=value,
                block_name=self.qualified_node_name(node_name),
                function_name="save_pipeline_runtime",
                run_id="save_pipeline",
                optuna_db_path=self._optuna_target_db_path(target_root),
            )
        try:
            torch = import_module("torch")
        except ModuleNotFoundError:
            torch = None
        if torch is not None:
            if isinstance(value, torch.nn.Module) or isinstance(value, torch.Tensor):
                save_store = ArtifactStore(target_root)
                return save_store.save(
                    variable_name=output_name,
                    value=value,
                    block_name=self.qualified_node_name(node_name),
                    function_name="save_pipeline_runtime",
                    run_id="save_pipeline",
                    torch_load_weights_only=self.torch_load_weights_only,
                )
            if isinstance(value, torch.optim.Optimizer):
                linked_model_record = self._find_linked_model_artifact(
                    cache,
                    output_name,
                    sibling_outputs or {},
                    target_root,
                    node_name,
                )
                if (
                    self._paired_model_name(output_name) is not None
                    and linked_model_record is None
                ):
                    warnings.warn(
                        f"Runtime optimizer '{node_name}.{output_name}' was saved without a linked model artifact.",
                        stacklevel=2,
                    )
                optimizer_path = self._save_torch_optimizer_state(
                    value,
                    target_root,
                    node_name,
                    output_name,
                )
                return TorchStateArtifactRecord(
                    variable_name=output_name,
                    file_path=str(optimizer_path),
                    object_kind="torch_optimizer_state",
                    metadata={
                        "linked_model_variable": None if linked_model_record is None else linked_model_record.variable_name,
                    },
                )

        if callable(value):
            return self._serialize_callable_runtime_value(
                value,
                node_name,
                output_name,
            )

        try:
            pickle.dumps(value)
            return value
        except Exception:
            if is_dataclass(value) and not isinstance(value, type):
                return DataclassValueReference(
                    class_name=type(value).__name__,
                    module=type(value).__module__,
                    data={
                        name: self._serialize_dataclass_field_value(
                            getattr(value, name)
                        )
                        for name in value.__dataclass_fields__
                    },
                    reason="not directly serializable during save_pipeline",
                )
            warnings.warn(
                f"Runtime value '{node_name}.{output_name}' could not be serialized directly; saving a reference placeholder instead.",
                stacklevel=2,
            )
            return RuntimeValueReference(
                type_name=type(value).__name__,
                repr_text=repr(value),
                reason="not directly serializable during save_pipeline",
            )

    def _optuna_target_db_path(self, target_root: Path) -> Path:
        if self._is_atom:
            return target_root.parent.parent / OPTUNA_STUDIES_DB_NAME
        return target_root / OPTUNA_STUDIES_DB_NAME

    def _serialize_callable_runtime_value(
        self,
        value: Any,
        node_name: str,
        output_name: str,
    ) -> Any:
        try:
            _, import_path, callable_name = resolve_callable(value)
        except RegistrationError:
            import_path = None
            callable_name = getattr(value, "__name__", type(value).__name__)
        if import_path is not None and import_path.startswith("__main__."):
            return RuntimeCallableReference(callable_name=callable_name)
        if import_path is None or not self._callable_reference_round_trips(value, import_path):
            warnings.warn(
                f"Callable runtime value '{node_name}.{output_name}' is not importable; saving a reference placeholder instead.",
                stacklevel=2,
            )
            return RuntimeValueReference(
                type_name=type(value).__name__,
                repr_text=repr(value),
                reason="callable is not importable during save_pipeline",
            )
        return CallableValueReference(
            callable_name=callable_name,
            import_path=import_path,
        )

    @classmethod
    def _restore_saved_runtime_mapping(
        cls,
        values: dict[str, Any],
        owner_label: str,
    ) -> dict[str, Any]:
        restored: dict[str, Any] = {}
        for value_name, value in values.items():
            if isinstance(value, CallableValueReference):
                restored[value_name] = cls._restore_callable_value(value)
            elif isinstance(value, RuntimeCallableReference):
                restored[value_name] = cls._restore_runtime_callable(
                    value,
                    f"{owner_label} '{value_name}'",
                )
            else:
                restored[value_name] = value
        return restored

    @classmethod
    def _restore_runtime_registered_callable(
        cls,
        reference: RuntimeCallableReference,
        block_name: str,
    ) -> Any:
        return cls._restore_runtime_callable(
            reference,
            f"block '{block_name}'",
        )

    @classmethod
    def _restore_partial_callable(
        cls,
        payload: dict[str, Any],
        block_name: str,
    ) -> Any:
        nested_partial = payload.get("partial")
        func_import_path = payload.get("func_import_path")
        if nested_partial is not None:
            func = cls._restore_partial_callable(nested_partial, block_name)
        elif func_import_path is not None:
            func = resolve_callable(func_import_path)[0]
        else:
            func = cls._restore_runtime_callable(
                RuntimeCallableReference(
                    callable_name=payload["func_callable_name"]
                ),
                f"block '{block_name}'",
            )
        args = tuple(
            cls._deserialize_partial_argument(item, block_name)
            for item in payload.get("args", [])
        )
        keywords = {
            key: cls._deserialize_partial_argument(item, block_name)
            for key, item in payload.get("keywords", {}).items()
        }
        return partial(func, *args, **keywords)

    @classmethod
    def _deserialize_partial_argument(cls, value: Any, block_name: str) -> Any:
        restored = cls._deserialize_config_value(value)
        if cls._is_missing_main_placeholder(restored):
            raise PersistenceError(
                f"Partial argument for block '{block_name}' references a missing "
                "__main__ callable or class and cannot be restored"
            )
        placeholder = cls._find_partial_argument_placeholder(restored)
        if placeholder is not None:
            raise PersistenceError(
                f"Partial argument for block '{block_name}' was saved as a "
                f"placeholder ({placeholder.reason}) and cannot be restored"
            )
        return restored

    @classmethod
    def _find_partial_argument_placeholder(
        cls,
        value: Any,
    ) -> RuntimeValueReference | DataclassValueReference | None:
        if isinstance(value, (RuntimeValueReference, DataclassValueReference)):
            return value
        if isinstance(value, dict):
            values = value.values()
        elif isinstance(value, (list, tuple, set, frozenset)):
            values = value
        elif is_dataclass(value) and not isinstance(value, type):
            values = (getattr(value, field.name) for field in fields(value))
        elif isinstance(value, SimpleNamespace):
            values = vars(value).values()
        else:
            return None
        for item in values:
            placeholder = cls._find_partial_argument_placeholder(item)
            if placeholder is not None:
                return placeholder
        return None

    def _find_linked_model_artifact(
        self,
        cache: dict[int, Any],
        optimizer_name: str,
        sibling_outputs: dict[str, Any],
        target_root: Path,
        node_name: str,
    ) -> ArtifactRecord | None:
        model_name = self._paired_model_name(optimizer_name)
        if model_name is None:
            return None
        model_value = sibling_outputs.get(model_name)
        if model_value is None:
            return None
        model_id = id(model_value)
        cached = cache.get(model_id)
        if isinstance(cached, ArtifactRecord):
            return cached
        serialized = self._serialize_runtime_value_for_save(
            model_value,
            target_root,
            cache,
            node_name,
            model_name,
            sibling_outputs=sibling_outputs,
        )
        if isinstance(serialized, ArtifactRecord):
            return serialized
        return None

    def _paired_model_name(self, optimizer_name: str) -> str | None:
        if "optimizer" not in optimizer_name:
            return None
        return optimizer_name.replace("optimizer", "model")

    def _save_torch_optimizer_state(
        self,
        optimizer: Any,
        target_root: Path,
        node_name: str,
        output_name: str,
    ) -> Path:
        import torch  # type: ignore

        save_store = ArtifactStore(target_root)
        artifact = save_store.save(
            variable_name=output_name,
            value=optimizer.state_dict(),
            block_name=self.qualified_node_name(node_name),
            function_name="save_pipeline_runtime_optimizer_state",
            run_id="save_pipeline",
        )
        return Path(artifact.file_path)

    def _serialize_payload(self) -> dict[str, Any]:
        return {
            "registration_name": self.registration_name,
            "config": self._serialize_config_for_save(self.config),
            "execution_priority": self.execution_priority,
            "is_atom": self._is_atom,
            "expression_runtime_code": self.expression_runtime_code,
            "historical_result_log_path": self.historical_result_log_path,
            "gate": None if self.gate_block is None else self.gate_block.serialize(),
            "nodes": [self._serialize_node(node) for node in self._sorted_nodes()],
            "producer_outputs": self.producer_outputs,
            "para_value_dict": self.para_value_dict,
            "artifact_registry": self.artifact_registry,
            "run_history": self.run_history,
        }

    def _serialize_node(self, node: Any) -> dict[str, Any]:
        if _is_child_pipeline(node):
            return {
                "kind": "pipeline",
                "registration_name": node.registration_name,
                "execution_priority": node.execution_priority,
                "overridden_outputs": {
                    output_name: (address.pipeline_name, address.node_name)
                    for output_name, address in sorted(
                        node._overridden_outputs.items()
                    )
                },
                "payload": node._serialize_payload(),
            }
        functions = []
        for registration in node.functions:
            match registration:
                case FunctionRegistration():
                    if isinstance(registration.callable_obj, partial):
                        functions.append(
                            {
                                "kind": "function",
                                "partial": self._serialize_partial_callable(
                                    registration.callable_obj
                                ),
                                "output_names": registration.output_names,
                                "save_to_disk": sorted(registration.save_to_disk),
                                "param_mapping": registration.param_mapping,
                                "var_pos_name": registration.var_pos_name,
                                "var_kw_name": registration.var_kw_name,
                                "overridden_outputs": {
                                    output_name: (
                                        address.pipeline_name,
                                        address.node_name,
                                    )
                                    for output_name, address in sorted(
                                        registration.overridden_outputs.items()
                                    )
                                },
                            }
                        )
                        continue
                    functions.append(
                        {
                            "kind": "function",
                            "import_path": registration.import_path,
                            "runtime_callable_reference": (
                                None
                                if registration.import_path is not None
                                else RuntimeCallableReference(
                                    callable_name=registration.function_name
                                )
                            ),
                            "output_names": registration.output_names,
                            "save_to_disk": sorted(registration.save_to_disk),
                            "param_mapping": registration.param_mapping,
                            "var_pos_name": registration.var_pos_name,
                            "var_kw_name": registration.var_kw_name,
                            "overridden_outputs": {
                                output_name: (
                                    address.pipeline_name,
                                    address.node_name,
                                )
                                for output_name, address in sorted(
                                    registration.overridden_outputs.items()
                                )
                            },
                        }
                    )
                case ExpressionRegistration():
                    functions.append(
                        {
                            "kind": "expression",
                            "code": registration.code,
                            "output_names": registration.output_names,
                            "save_to_disk": sorted(registration.save_to_disk),
                            "warn_on_input_mutation": registration.warn_on_input_mutation,
                            "overridden_outputs": {
                                output_name: (
                                    address.pipeline_name,
                                    address.node_name,
                                )
                                for output_name, address in sorted(
                                    registration.overridden_outputs.items()
                                )
                            },
                        }
                    )
                case _:
                    raise PersistenceError(
                        f"Unsupported registration type in block '{node.registration_name}'"
                    )
        return {
            "kind": "block",
            "registration_name": node.registration_name,
            "execution_priority": node.execution_priority,
            "functions": functions,
            "registered_args": [
                {"name": registration.name, "ordered_items": registration.ordered_items}
                for registration in node.registered_args.values()
            ],
            "registered_kwargs": [
                {"name": registration.name, "mapping_dct": registration.mapping_dct}
                for registration in node.registered_kwargs.values()
            ],
        }

    def _serialize_partial_callable(self, value: partial[Any]) -> dict[str, Any]:
        """Serialize a ``functools.partial`` as a structural, loadable payload.

        The wrapped callable is recorded by import path when it is importable
        (module function), by its runtime name when it must be looked up in
        ``__main__`` at load, or as a nested partial payload for partials of
        partials. Bound args and keywords go through the same structured
        serialization as dataclass fields; values that would become unresolved
        placeholders are rejected during saving.
        """
        func = value.func
        if isinstance(func, partial):
            func_payload = {"partial": self._serialize_partial_callable(func)}
        else:
            func_import_path = None
            func_callable_name: str | None = "callable"
            try:
                _, func_import_path, func_callable_name = resolve_callable(func)
            except RegistrationError:
                pass
            if (
                func_import_path is not None
                and not func_import_path.startswith("__main__.")
            ):
                func_payload = {"func_import_path": func_import_path}
            else:
                func_payload = {"func_callable_name": func_callable_name}
        serialized_args = [
            self._serialize_dataclass_field_value(item) for item in value.args
        ]
        serialized_keywords = {
            key: self._serialize_dataclass_field_value(item)
            for key, item in value.keywords.items()
        }
        for item in [*serialized_args, *serialized_keywords.values()]:
            placeholder = self._find_partial_argument_placeholder(item)
            if placeholder is not None:
                raise PersistenceError(
                    "Partial argument was saved as a placeholder "
                    f"({placeholder.reason}) and cannot be persisted"
                )
        return {
            **func_payload,
            "args": serialized_args,
            "keywords": serialized_keywords,
        }
