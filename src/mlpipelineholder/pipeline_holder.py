from __future__ import annotations

import ast
import builtins
import copy
import gc
import inspect
import math
import os
import pickle
import platform
import shutil
import sys
import warnings
from contextlib import ExitStack, redirect_stdout
from ctypes import CDLL
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from functools import partial
from importlib import import_module
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import uuid4
from weakref import WeakKeyDictionary

from .persistence.artifacts.store import ArtifactStore
from .persistence.cleanup import (
    _collect_obsolete_artifact_entries,
    _collect_obsolete_child_dirs,
    _delete_saved_path,
    _referenced_payload_artifact_paths,
)
from .persistence.payloads import PayloadMixin
from .persistence.pickle_io import (
    _MissingMainClassPlaceholder,
    atomic_pickle_dump,
    load_pickle_with_missing_class_fallback,
    load_pipeline_metadata,
)
from .persistence.save_load import SaveLoadMixin
from .persistence.serialization import SerializationMixin
from .persistence.validation import (
    contains_missing_main_placeholder,
    contains_missing_placeholder_outside_config,
    replace_missing_runtime_payload_values,
    require_loaded_payload_keys,
    validate_function_payload_structure,
    validate_import_path_payload,
    validate_loaded_payload_placeholders,
    validate_loaded_payload_structure,
)
from .state.output_graph import OutputGraphMixin
from .state.values import ValueAccessMixin
from .state.visibility import VisibilityMixin
from .state.invalidation import InvalidationMixin
from .state.mutation import MutationMixin
from .state.storage import StorageMixin
from .exceptions import (
    ExecutionError,
    PersistenceError,
    RegistrationError,
    ResolutionError,
)
from .execution.arguments import ArgumentMixin
from .execution.engine import EngineMixin
from .execution.function_registry import (
    _values_equal,
    callable_signature,
    default_map,
    resolve_callable,
)
from .execution.gate_block import GateBlock, _GateStatusCache
from .execution.atom_registry import atom_pipeline_class
from .execution.registration import AtomRegistrationMixin, RegistrationMixin
from .presentation.print_capture import PrintCaptureMixin
from .presentation.description import DescriptionMixin
from .presentation.logger import PipelineLogger
from .core.configuration import ConfigurationMixin
from .core.base import PipelineBase
from .core.constants import (
    _IMMUTABLE_TYPES,
    _MISSING,
    _RESERVED_BUILTIN_NAMES,
)
from .core.models import (
    ArtifactRecord,
    CallableValueReference,
    DataclassValueReference,
    ExpressionRegistration,
    FunctionRegistration,
    RunRecord,
    RuntimeCallableReference,
    RuntimeValueReference,
    TorchStateArtifactRecord,
)
from .core.naming import validate_registration_name
from .persistence.object_storage import (
    StoredObjectRecord,
    create_record,
    metadata_frame,
    record_from_payload,
    resolve_record,
    validate_object_description,
    validate_object_name,
    warn_for_pickle_fallback,
)
from .integrations.optuna.api import OptunaStudy, StudyArtifactOptions
from .integrations.optuna.support import (
    OPTUNA_STUDIES_DB_NAME,
    is_optuna_sampler,
    is_optuna_study,
)
from .state.output_pointers import (
    OutputAddress,
    OutputPointer,
    PointerDestinationMissingError,
    PointerResolutionError,
    is_strictly_upstream,
    resolve_pointer_chain,
)
from .persistence.artifacts.serializers import choose_serializer
from .integrations.dataframe import stage_dask_dataframe_for_copy

if TYPE_CHECKING:
    from .execution.block import ExecutionBlock


class PipelineHolder(
    InvalidationMixin,
    MutationMixin,
    StorageMixin,
    ValueAccessMixin,
    VisibilityMixin,
    OutputGraphMixin,
    RegistrationMixin,
    AtomRegistrationMixin,
    DescriptionMixin,
    EngineMixin,
    ArgumentMixin,
    SerializationMixin,
    SaveLoadMixin,
    PayloadMixin,
    PrintCaptureMixin,
    ConfigurationMixin,
    PipelineBase,
):
    _is_atom: bool = False

    def __init__(
        self,
        registration_name: str,
        configuration: Any | None = None,
        local_folder_path: str | Path | None = None,
        execution_priority: float | None = None,
        forced: bool = False,
        memory_saving_mode: bool = False,
        memory_profile_logging: bool = False,
        pipeline_backup_directory: str | Path | None = None,
        log_traceback_to_file: bool = True,
        show_traceback_locals: bool = False,
        use_rich_traceback_console: bool = True,
        torch_load_weights_only: bool = False,
        strict_mode: bool = False,
        _allow_existing_root: bool = False,
        _allow_legacy_config_object: bool = False,
        _preserve_existing_log: bool = False,
    ) -> None:
        self.registration_name = validate_registration_name(
            registration_name,
            owner_label="pipeline",
        )
        self._config = {} if configuration is None else configuration
        self.execution_priority = execution_priority
        self.parent_pipeline: PipelineHolder | None = None
        self._temporary_root_handle: TemporaryDirectory[str] | None = None
        generated_temp_root = local_folder_path is None
        self.project_root = self._initial_project_root(
            registration_name,
            local_folder_path,
        )
        self.pipeline_backup_root = (
            None
            if pipeline_backup_directory is None
            else Path(pipeline_backup_directory)
        )
        try:
            if not _allow_legacy_config_object:
                self._validate_config_reconstructable(self.config)
            self._validate_config_picklable(self.config)
            self._validate_builtin_name_conflicts_in_mapping(
                self._config_name_mapping(self.config),
                owner_label="configuration",
            )
            self._validate_backup_path_safety()
            if not _allow_existing_root:
                self._prepare_project_root(forced)
            self.project_root.mkdir(parents=True, exist_ok=True)
            self.metadata_root = self.project_root / "metadata"
            self.metadata_root.mkdir(parents=True, exist_ok=True)
            self.logger = PipelineLogger(
                self.metadata_root / "pipeline.log",
                log_traceback_to_file=log_traceback_to_file,
                show_traceback_locals=show_traceback_locals,
                use_rich_traceback_console=use_rich_traceback_console,
                truncate=not _preserve_existing_log,
            )
            self.logger._pipeline = self
            self.print_capture_mode = "tee"
            self.memory_saving_mode = memory_saving_mode
            self.memory_profile_logging = memory_profile_logging
            self.torch_load_weights_only = bool(torch_load_weights_only)
            self.strict_mode = bool(strict_mode)
            self._suppress_strict_validation = False
            self._invalidation_forbidden = False
            self.suppress_registration_advisories = False
            self.historical_result_log_path: str | None = None
            self._attached_result_history_override: list[str] | None = None
            self.expression_runtime_code: str | None = None
            self._expression_runtime_defined_names_cache: set[str] | None = None
            self._expression_runtime_namespace_cache: dict[str, Any] | None = None

            self.nodes: list[Any] = []
            self.nodes_by_name: dict[str, Any] = {}
            self.blocks: list[Any] = []
            self.blocks_by_name: dict[str, Any] = {}
            self.gate_block: GateBlock | None = None
            self.gate_cleanup_confirmation: bool = False
            self._gate_cleanup_predecided: bool | None = None
            self._overridden_outputs: dict[str, OutputAddress] = {}

            self.manual_values: dict[str, Any] = {}
            self.para_value_dict: dict[str, Any] = {}
            self.artifact_registry: dict[str, ArtifactRecord] = {}
            self.producer_outputs: dict[str, dict[str, Any]] = {}
            self._optuna_study_group_addresses: dict[str, set[OutputAddress]] = {}
            self._optuna_study_group_names: dict[str, tuple[str, str]] = {}
            self._optuna_study_provenance: WeakKeyDictionary[
                OptunaStudy,
                tuple[str, str],
            ] = WeakKeyDictionary()
            self.run_history: list[RunRecord] = []
            self._stored_objects: dict[str, StoredObjectRecord] = {}
            self.artifact_store = ArtifactStore(self.project_root)
        except Exception:
            if generated_temp_root and self.project_root.exists():
                shutil.rmtree(self.project_root, ignore_errors=True)
            raise

    @property
    def config(self) -> Any:
        if getattr(self, "_is_atom", False) and self.parent_pipeline is not None:
            return self.parent_pipeline.config
        return self._config

    @config.setter
    def config(self, configuration: Any) -> None:
        self._require_owned_config()
        self._config = configuration

    @property
    def optuna_studies_db_path(self) -> Path:
        if getattr(self, "_is_atom", False):
            raise AttributeError("Atom pipelines do not own Optuna study storage")
        return self.project_root / OPTUNA_STUDIES_DB_NAME

    def _optuna_studies_db_path_for_storage(self) -> Path:
        current = self
        while current._is_atom and current.parent_pipeline is not None:
            current = current.parent_pipeline
        return current.optuna_studies_db_path

    def __del__(self) -> None:
        try:
            self._cleanup_temporary_root_handle()
        except Exception:
            pass

    def _initial_project_root(
        self,
        registration_name: str,
        local_folder_path: str | Path | None,
    ) -> Path:
        if local_folder_path is not None:
            return Path(local_folder_path)
        del registration_name
        self._temporary_root_handle = TemporaryDirectory(prefix="mlpipelineholder_")
        return Path(self._temporary_root_handle.name)

    def _cleanup_temporary_root_handle(self) -> None:
        if self._temporary_root_handle is None:
            return
        self._temporary_root_handle.cleanup()
        self._temporary_root_handle = None

    def __str__(self) -> str:
        return self.describe_pipeline()

    def __repr__(self) -> str:
        return self.describe_pipeline()

    def _create_execution_block(
        self, registration_name: str, execution_priority: float
    ) -> Any:
        from .execution.block import ExecutionBlock

        return ExecutionBlock(self, registration_name, execution_priority)

    @staticmethod
    def _is_execution_block(node: Any) -> bool:
        from .execution.block import ExecutionBlock

        return isinstance(node, ExecutionBlock)

    def _seal(self) -> None:
        """Finish node construction. AtomPipeline seals to lock its structure."""
        return

    def _assert_mutable(self, action: str) -> None:
        """Guard hook for node mutation. Sealed atoms override this to raise."""
        return

    def add_gate_block(
        self, function_or_path: Any, expected_value: Any = True, forced: bool = False
    ) -> Any:
        if self.gate_block is not None and not forced:
            self.logger.warning("Skipped gate block registration: gate block already exists")
            return None
        self.gate_block = GateBlock(self, function_or_path, expected_value=expected_value)
        if not self._invalidation_forbidden:
            self._invalidate_all_outputs()
        return self.gate_block

    def set_gate_block(
        self, function_or_path: Any, expected_value: Any = True, forced: bool = False
    ) -> Any:
        return self.add_gate_block(function_or_path, expected_value=expected_value, forced=forced)

    def get_block(self, block_name: str) -> Any:
        node = self.nodes_by_name.get(block_name)
        if node is None:
            raise RegistrationError(f"Block not registered: {block_name}")
        if isinstance(node, PipelineHolder):
            raise RegistrationError(f"Registered node '{block_name}' is a child pipeline, not a block")
        return node

    def get_child_pipeline(self, pipeline_name: str) -> "PipelineHolder":
        node = self.nodes_by_name.get(pipeline_name)
        if node is None:
            raise RegistrationError(f"Child pipeline not registered: {pipeline_name}")
        if not isinstance(node, PipelineHolder):
            raise RegistrationError(f"Registered node '{pipeline_name}' is a block, not a child pipeline")
        return node

    def list_child_pipeline_names(self) -> list[str]:
        return [
            node.registration_name
            for node in self._sorted_nodes()
            if isinstance(node, PipelineHolder)
        ]

    def reset_gate_block(self) -> None:
        if self.gate_block is None:
            return
        self.gate_block = None
        if not self._invalidation_forbidden:
            self._invalidate_all_outputs()

    def define_expression_runtime(self, code: str) -> None:
        code = self._normalize_expression_runtime_code(code)
        self._validate_expression_runtime_code(code)
        self.expression_runtime_code = code
        self._expression_runtime_defined_names_cache = None
        self._expression_runtime_namespace_cache = None
        try:
            self._build_expression_runtime_namespace()
        except PersistenceError as exc:
            self.clear_expression_runtime()
            raise RegistrationError(str(exc)) from exc

    def clear_expression_runtime(self) -> None:
        self.expression_runtime_code = None
        self._expression_runtime_defined_names_cache = None
        self._expression_runtime_namespace_cache = None

    def get_expression_runtime_code(self) -> str | None:
        owner = self._effective_expression_runtime_owner()
        if owner is None:
            return None
        return owner.expression_runtime_code

    def list_declared_outputs(self) -> set[str]:
        outputs: set[str] = set()
        for node in self.nodes:
            outputs.update(self._node_declared_outputs(node))
        return outputs

    def remove_block(self, block_name: str) -> None:
        if block_name not in self.blocks_by_name:
            raise RegistrationError(f"Block not registered: {block_name}")
        block = self.blocks_by_name.pop(block_name)
        self.blocks = [candidate for candidate in self.blocks if candidate is not block]
        self.nodes = [candidate for candidate in self.nodes if candidate is not block]
        self.nodes_by_name.pop(block_name, None)
        self._erase_node_outputs(block_name)
        if not self._invalidation_forbidden:
            self._invalidate_from_priority(block.execution_priority)
        if self.parent_pipeline is not None:
            self._resync_mirror_to_parent()

    @staticmethod
    def _load_pickle_with_missing_class_fallback(raw_bytes: bytes) -> Any:
        return load_pickle_with_missing_class_fallback(raw_bytes)

    @staticmethod
    def _validate_expression_runtime_code(code: str) -> None:
        try:
            parsed = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            raise RegistrationError(f"Invalid expression runtime syntax: {exc}") from exc
        for node in parsed.body:
            if isinstance(node, ast.Import):
                continue
            if isinstance(node, ast.ImportFrom):
                if node.level != 0 or node.module is None:
                    raise RegistrationError(
                        "Expression runtime only supports absolute imports"
                    )
                if any(alias.name == "*" for alias in node.names):
                    raise RegistrationError(
                        "Expression runtime does not support wildcard imports"
                    )
                continue
            raise RegistrationError(
                "Expression runtime only supports import and from-import statements"
            )

    @staticmethod
    def _normalize_expression_runtime_code(code: str) -> str:
        normalized = dedent(code).strip()
        if not normalized:
            raise RegistrationError("Expression runtime code cannot be empty")
        return normalized

    def _effective_expression_runtime_owner(self) -> "PipelineHolder | None":
        current: PipelineHolder | None = self
        while current is not None:
            if current.expression_runtime_code is not None:
                return current
            current = current.parent_pipeline
        return None

    def _expression_runtime_defined_names(self) -> set[str]:
        owner = self._effective_expression_runtime_owner()
        if owner is None or owner.expression_runtime_code is None:
            return set()
        if owner._expression_runtime_defined_names_cache is not None:
            return set(owner._expression_runtime_defined_names_cache)
        parsed = ast.parse(owner.expression_runtime_code, mode="exec")
        names: set[str] = set()
        for node in parsed.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split(".")[0])
                continue
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names.add(alias.asname or alias.name)
        owner._expression_runtime_defined_names_cache = names
        return set(names)

    def _build_expression_runtime_namespace(self) -> dict[str, Any]:
        owner = self._effective_expression_runtime_owner()
        if owner is None or owner.expression_runtime_code is None:
            return {}
        if owner._expression_runtime_namespace_cache is not None:
            return dict(owner._expression_runtime_namespace_cache)
        globals_namespace: dict[str, Any] = {"__builtins__": builtins.__dict__}
        try:
            exec(owner.expression_runtime_code, globals_namespace, globals_namespace)
        except Exception as exc:
            raise PersistenceError(
                f"Failed to build expression runtime for pipeline '{owner.registration_name}': {type(exc).__name__}: {exc}"
            ) from exc
        runtime_namespace = {
            key: value
            for key, value in globals_namespace.items()
            if key != "__builtins__"
        }
        owner._expression_runtime_namespace_cache = runtime_namespace
        return dict(runtime_namespace)

    @classmethod
    def _contains_missing_main_placeholder(cls, value: Any) -> bool:
        return contains_missing_main_placeholder(cls, value)

    @staticmethod
    def _is_missing_main_placeholder(value: Any) -> bool:
        return isinstance(value, _MissingMainClassPlaceholder) or (
            isinstance(value, type)
            and issubclass(value, _MissingMainClassPlaceholder)
        )

    @classmethod
    def _validate_loaded_payload_placeholders(cls, payload: dict[str, Any]) -> None:
        validate_loaded_payload_placeholders(cls, payload)

    @classmethod
    def _replace_missing_runtime_payload_values(
        cls,
        payload: Any,
        pipeline_path: tuple[str, ...] = (),
    ) -> list[str]:
        return replace_missing_runtime_payload_values(cls, payload, pipeline_path)

    @classmethod
    def _validate_loaded_payload_structure(cls, payload: Any) -> None:
        validate_loaded_payload_structure(cls, payload)

    @staticmethod
    def _require_loaded_payload_keys(
        payload: dict[str, Any],
        required_keys: tuple[str, ...],
        *,
        owner_label: str,
    ) -> None:
        require_loaded_payload_keys(payload, required_keys, owner_label=owner_label)

    @classmethod
    def _validate_function_payload_structure(
        cls,
        function_payload: Any,
        *,
        owner_label: str,
    ) -> None:
        validate_function_payload_structure(cls, function_payload, owner_label=owner_label)

    @classmethod
    def _validate_import_path_payload(cls, import_path: Any) -> None:
        validate_import_path_payload(import_path)

    @classmethod
    def _contains_missing_placeholder_outside_config(
        cls,
        value: Any,
        *,
        inside_config: bool = False,
    ) -> bool:
        return contains_missing_placeholder_outside_config(
            cls, value, inside_config=inside_config
        )

    @classmethod
    def _clear_path_with_optional_confirmation(
        cls,
        target_path: Path,
        *,
        source_path: Path,
        forced_deleting: bool,
    ) -> None:
        if not target_path.exists():
            return
        if target_path.is_dir() and not any(target_path.iterdir()):
            target_path.rmdir()
            return
        if not forced_deleting:
            user_input = input(
                f"Working pipeline directory '{target_path}' will be deleted and replaced from '{source_path}'. Type 'yes' or 'y' to continue: "
            ).strip().lower()
            if user_input not in {"yes", "y"}:
                raise PersistenceError(
                    f"Aborted restoring pipeline directory '{target_path}' from '{source_path}'"
                )
        if target_path.is_dir():
            shutil.rmtree(target_path)
            return
        target_path.unlink()

    @staticmethod
    def _normalized_path(path: Path) -> Path:
        return path.expanduser().resolve(strict=False)

    @classmethod
    def _paths_overlap(cls, first_path: Path, second_path: Path) -> bool:
        first = cls._normalized_path(first_path)
        second = cls._normalized_path(second_path)
        return first == second or first in second.parents or second in first.parents

    def _validate_backup_path_safety(self) -> None:
        if self.pipeline_backup_root is None:
            return
        if self._paths_overlap(self.project_root, self.pipeline_backup_root):
            raise RegistrationError(
                f"Pipeline backup directory '{self.pipeline_backup_root}' must not overlap with pipeline directory '{self.project_root}'"
            )


    @classmethod
    def _from_payload(
        cls,
        payload: dict[str, Any],
        project_root: Path,
        parent: "PipelineHolder | None" = None,
        *,
        verbose: bool = False,
        auto_resolve_placeholders: bool = True,
    ) -> "PipelineHolder":
        reconstruction_warnings: list[str] = []
        config = cls._deserialize_saved_config(
            payload["config"],
            verbose=verbose,
            warn=reconstruction_warnings.append,
        )
        pipeline_class: type[PipelineHolder]
        if payload.get("is_atom", False):
            pipeline_class = atom_pipeline_class()
        else:
            pipeline_class = cls
        pipeline = pipeline_class(
            registration_name=payload["registration_name"],
            configuration=config,
            local_folder_path=project_root,
            execution_priority=payload.get("execution_priority"),
            memory_saving_mode=payload.get("memory_saving_mode", False),
            memory_profile_logging=payload.get("memory_profile_logging", False),
            log_traceback_to_file=payload.get("log_traceback_to_file", True),
            show_traceback_locals=payload.get("show_traceback_locals", False),
            use_rich_traceback_console=payload.get("use_rich_traceback_console", True),
            torch_load_weights_only=payload.get("torch_load_weights_only", False),
            strict_mode=payload.get("strict_mode", False),
            pipeline_backup_directory=payload.get("pipeline_backup_directory"),
            _allow_existing_root=True,
            _allow_legacy_config_object=True,
            _preserve_existing_log=parent is not None,
        )
        for message in reconstruction_warnings:
            pipeline.logger.warning(message)
        if parent is not None:
            pipeline.parent_pipeline = parent
            pipeline.logger = parent.logger
        if payload.get("expression_runtime_code") is not None:
            pipeline.define_expression_runtime(payload["expression_runtime_code"])
        pipeline.historical_result_log_path = payload.get("historical_result_log_path")
        pipeline.suppress_registration_advisories = True
        pipeline._suppress_strict_validation = True
        if payload.get("gate") is not None:
            gate_payload = payload["gate"]
            if gate_payload.get("kind") == "config_field":
                pipeline.set_gate_block(
                    gate_payload["field_name"],
                    expected_value=gate_payload.get("expected_value", True),
                )
            else:
                pipeline.set_gate_block(
                    gate_payload["import_path"],
                    expected_value=gate_payload.get("expected_value", True),
                )
        for node_payload in payload["nodes"]:
            if node_payload["kind"] == "block":
                block = pipeline.add_block(
                    node_payload["registration_name"],
                    node_payload["execution_priority"],
                )
                if block is None:
                    block = pipeline._add_block_strict(
                        node_payload["registration_name"],
                        node_payload["execution_priority"],
                    )
                for args_payload in node_payload.get("registered_args", []):
                    block.register_args(
                        args_payload["name"],
                        args_payload["ordered_items"],
                        forced=True,
                    )
                for kwargs_payload in node_payload.get("registered_kwargs", []):
                    block.register_kwargs(
                        kwargs_payload["name"],
                        kwargs_payload["mapping_dct"],
                        forced=True,
                    )
                for function_payload in node_payload["functions"]:
                    if function_payload.get("kind") == "expression":
                        registration = block._register_expression_strict(
                            function_payload["code"],
                            output_variable_name=(
                                function_payload["output_names"][0]
                                if function_payload["output_names"]
                                else None
                            ),
                            save_to_disk=bool(function_payload["save_to_disk"]),
                            forced=False,
                            warn_on_input_mutation=function_payload.get(
                                "warn_on_input_mutation", False
                            ),
                            overridden_outputs=function_payload.get(
                                "overridden_outputs"
                            ),
                        )
                    else:
                        partial_payload = function_payload.get("partial")
                        if partial_payload is not None:
                            function_source = cls._restore_partial_callable(
                                partial_payload,
                                block.registration_name,
                            )
                        else:
                            function_source = function_payload.get("import_path")
                            if function_source is None:
                                runtime_reference = function_payload.get(
                                    "runtime_callable_reference"
                                )
                                if not isinstance(
                                    runtime_reference,
                                    RuntimeCallableReference,
                                ):
                                    raise PersistenceError(
                                        f"Saved function in block '{block.registration_name}' has no callable reference"
                                    )
                                function_source = cls._restore_runtime_registered_callable(
                                    runtime_reference,
                                    block.registration_name,
                                )
                        registration = block._register_function_strict(
                            function_source,
                            function_payload["output_names"],
                            function_payload["save_to_disk"],
                            param_mapping=function_payload.get("param_mapping"),
                            var_pos_name=function_payload.get("var_pos_name"),
                            var_kw_name=function_payload.get("var_kw_name"),
                            overridden_outputs=function_payload.get(
                                "overridden_outputs"
                            ),
                        )
                    if registration is None:
                        raise PersistenceError(
                            f"Failed to restore function in block '{block.registration_name}'"
                        )
            else:
                child_root = project_root / "children" / node_payload["registration_name"]
                child = cls._from_payload(
                    node_payload["payload"],
                    child_root,
                    parent=pipeline,
                    verbose=verbose,
                    auto_resolve_placeholders=auto_resolve_placeholders,
                )
                child.execution_priority = node_payload["execution_priority"]
                child.parent_pipeline = pipeline
                child.logger = pipeline.logger
                child._overridden_outputs = pipeline._normalize_overridden_outputs(
                    sorted(child.list_declared_outputs()),
                    node_payload.get("overridden_outputs"),
                    current_node_name=child.registration_name,
                    current_priority=child.execution_priority,
                )
                pipeline._register_node(child)
        pipeline.producer_outputs = {
            node_name: cls._restore_saved_runtime_mapping(
                outputs,
                owner_label=f"outputs from '{node_name}'",
            )
            for node_name, outputs in payload.get("producer_outputs", {}).items()
        }
        pipeline.manual_values = cls._restore_saved_runtime_mapping(
            payload.get("manual_values", {}),
            owner_label="pipeline value",
        )
        for constant_name, constant_value in pipeline.manual_values.items():
            if isinstance(constant_value, RuntimeValueReference) and verbose:
                pipeline.logger.warning(
                    f"Constant '{constant_name}' was saved as a placeholder ({constant_value.reason}) "
                    "and could not be restored; reset it with set_constant_value before running, "
                    "otherwise functions consuming it will fail"
                )
        pipeline.para_value_dict = cls._restore_saved_runtime_mapping(
            payload.get("para_value_dict", {}),
            owner_label="pipeline state value",
        )
        pipeline.artifact_registry = cls._restore_saved_runtime_mapping(
            payload.get("artifact_registry", {}),
            owner_label="artifact registry value",
        )
        pipeline._stored_objects = {
            record.hash_id: record
            for record in (
                record_from_payload(record_payload)
                for record_payload in payload.get("object_storage", {}).values()
            )
        }
        pipeline.run_history = payload.get("run_history", [])
        saved_project_root = payload.get("saved_project_root")
        if saved_project_root is not None:
            pipeline._rewrite_artifact_paths(Path(saved_project_root), project_root)
            pipeline._rewrite_run_history_paths(Path(saved_project_root), project_root)
        pipeline.suppress_registration_advisories = False
        pipeline._suppress_strict_validation = False
        if pipeline._is_atom:
            pipeline._config = {}
            pipeline._seal()
        if parent is not None:
            pipeline.parent_pipeline = parent
            pipeline.logger = parent.logger
        pending_dataclass_fallbacks: list[
            tuple[PipelineHolder, str, str | None, str, DataclassValueReference]
        ] = []
        if parent is None:
            pipeline._recover_loaded_output_pointers()
            pipeline._restore_dataclass_value_references(
                verbose=verbose,
                _pending=pending_dataclass_fallbacks,
            )
        if auto_resolve_placeholders and parent is None:
            pipeline._auto_resolve_placeholder_outputs(verbose=verbose)
            pipeline._reconnect_dataclass_fallbacks(
                pending_dataclass_fallbacks,
                verbose=verbose,
            )
        elif not auto_resolve_placeholders and parent is None:
            pipeline._warn_unresolved_placeholders_at_load(verbose=verbose)
        if parent is None:
            pipeline._validate_runtime_output_pointers()
            pipeline._backfill_legacy_study_ownership()
            pipeline._replace_unloadable_persisted_values()
        return pipeline

    def _restore_dataclass_value_references(
        self,
        *,
        verbose: bool,
        _memo: dict[int, Any] | None = None,
        _pending: list[
            tuple[PipelineHolder, str, str | None, str, DataclassValueReference]
        ]
        | None = None,
    ) -> None:
        """Replace structured dataclass references with reconstructed values.

        Runs once at the root after the tree is rebuilt: dataclass values saved
        as structured references are rebuilt (a real dataclass when the class is
        importable and constructible from the saved fields, a ``SimpleNamespace``
        fallback otherwise) regardless of the ``auto_resolve_placeholders``
        flag. A shared identity memo guarantees that one logical saved reference
        is reconstructed once, so every mirror slot (producer outputs, visible
        state, parent mirrors) keeps the same object. Legacy pre-0.2.14
        placeholders that carry only a ``type_name`` are best-effort
        reconstructed with default fields when a matching importable dataclass
        exists.

        Every slot that falls back to a ``SimpleNamespace`` is recorded in the
        ``_pending`` registry (when given) as a slot descriptor rather than a
        dict reference, because placeholder recovery later rebuilds the visible
        state with fresh dict objects; a later reconnect pass can then upgrade
        the current slot once placeholder recovery has produced the real class
        as a pipeline value.
        """
        memo = {} if _memo is None else _memo
        pending = [] if _pending is None else _pending
        for slot_kind, mapping in (
            ("manual_values", self.manual_values),
            ("para_value_dict", self.para_value_dict),
        ):
            for value_name, value in list(mapping.items()):
                if isinstance(value, DataclassValueReference):
                    restored = self._restore_dataclass_value(
                        value,
                        verbose=verbose,
                        memo=memo,
                    )
                    mapping[value_name] = restored
                    if isinstance(restored, SimpleNamespace):
                        pending.append(
                            (self, slot_kind, None, value_name, value)
                        )
                elif isinstance(value, RuntimeValueReference):
                    restored = self._restore_legacy_dataclass_reference(
                        value,
                        verbose=verbose,
                        memo=memo,
                    )
                    if restored is not value:
                        mapping[value_name] = restored
        for node_name, outputs in self.producer_outputs.items():
            for value_name, value in list(outputs.items()):
                if isinstance(value, DataclassValueReference):
                    restored = self._restore_dataclass_value(
                        value,
                        verbose=verbose,
                        memo=memo,
                    )
                    outputs[value_name] = restored
                    if isinstance(restored, SimpleNamespace):
                        pending.append(
                            (
                                self,
                                "producer_outputs",
                                node_name,
                                value_name,
                                value,
                            )
                        )
                elif isinstance(value, RuntimeValueReference):
                    restored = self._restore_legacy_dataclass_reference(
                        value,
                        verbose=verbose,
                        memo=memo,
                    )
                    if restored is not value:
                        outputs[value_name] = restored
        for node in self._sorted_nodes():
            if isinstance(node, PipelineHolder):
                node._restore_dataclass_value_references(
                    verbose=verbose,
                    _memo=memo,
                    _pending=pending,
                )

    def _reconnect_dataclass_fallbacks(
        self,
        pending: list[
            tuple[PipelineHolder, str, str | None, str, DataclassValueReference]
        ],
        *,
        verbose: bool,
    ) -> None:
        """Upgrade SimpleNamespace fallbacks once placeholder recovery has run.

        Re-running blocks during ``_auto_resolve_placeholder_outputs`` can
        produce dynamically generated classes (for example a factory function
        returning a new dataclass), so saved dataclass instances that could not
        be reconnected during the first restore pass get a second attempt.
        Each slot is re-resolved against the current pipeline state (placeholder
        recovery rebuilds the visible state with fresh dict objects) and
        replaced with a real dataclass instance when its class is now
        available; every other slot keeps its SimpleNamespace fallback. Slots
        mirroring the same saved reference share one reconstructed object.
        """
        if not pending:
            return
        rebuilt: dict[int, Any] = {}
        for owner, slot_kind, slot_key, value_name, reference in pending:
            mapping = self._dataclass_fallback_mapping(
                owner,
                slot_kind,
                slot_key,
            )
            if mapping is None or not isinstance(
                mapping.get(value_name), SimpleNamespace
            ):
                continue
            reference_id = id(reference)
            result = rebuilt.get(reference_id)
            if result is None:
                data = {
                    key: self._deserialize_config_value(
                        item,
                        verbose=verbose,
                        warn=self.logger.warning,
                    )
                    for key, item in reference.data.items()
                }
                result = self._reconstruct_dataclass(
                    reference.class_name,
                    data,
                    verbose=False,
                    module_name=reference.module,
                    warn=None,
                    pipeline=owner,
                )
                rebuilt[reference_id] = result
            if not isinstance(result, SimpleNamespace):
                mapping[value_name] = result

    @staticmethod
    def _dataclass_fallback_mapping(
        owner: PipelineHolder,
        slot_kind: str,
        slot_key: str | None,
    ) -> dict[str, Any] | None:
        """Resolve the current mapping a recorded fallback slot lives in."""
        if slot_kind == "manual_values":
            return owner.manual_values
        if slot_kind == "para_value_dict":
            return owner.para_value_dict
        if slot_kind == "producer_outputs" and slot_key is not None:
            return owner.producer_outputs.get(slot_key)
        return None

    def _restore_dataclass_value(
        self,
        reference: DataclassValueReference,
        *,
        verbose: bool,
        memo: dict[int, Any],
    ) -> Any:
        reference_id = id(reference)
        cached = memo.get(reference_id)
        if cached is not None:
            return cached
        data = {
            key: self._deserialize_config_value(
                item,
                verbose=verbose,
                warn=self.logger.warning,
            )
            for key, item in reference.data.items()
        }
        reconstructed = self._reconstruct_dataclass(
            reference.class_name,
            data,
            verbose=verbose,
            module_name=reference.module,
            warn=self.logger.warning,
            pipeline=self,
        )
        memo[reference_id] = reconstructed
        return reconstructed

    def _restore_legacy_dataclass_reference(
        self,
        reference: RuntimeValueReference,
        *,
        verbose: bool,
        memo: dict[int, Any],
    ) -> Any:
        """Best-effort reconstruction of pre-0.2.14 dataclass placeholders.

        Old saves stored unpicklable dataclasses as plain
        ``RuntimeValueReference`` objects carrying only ``type_name``,
        ``repr_text``, and ``reason``, so no field data survives. When a
        dataclass with that name is importable and constructible with defaults,
        rebuild it; otherwise keep the placeholder (with a verbose-gated warning
        when a matching dataclass class was found but could not be built).
        """
        reference_id = id(reference)
        cached = memo.get(reference_id)
        if cached is not None:
            return cached
        candidate = self._find_dataclass_class(reference.type_name)
        if candidate is None:
            return reference
        try:
            reconstructed = candidate()
        except Exception:
            if verbose:
                self.logger.warning(
                    f"Saved value placeholder of dataclass '{reference.type_name}' could not be "
                    "reconstructed (the class is importable but not constructible without fields); "
                    "it remains a placeholder and raises ResolutionError when read"
                )
            memo[reference_id] = reference
            return reference
        memo[reference_id] = reconstructed
        return reconstructed

    def _has_placeholder_outputs_in_subtree(self) -> bool:
        placeholder_types = (RuntimeValueReference, DataclassValueReference)
        for outputs in self.producer_outputs.values():
            if any(isinstance(value, placeholder_types) for value in outputs.values()):
                return True
        for name, value in self.para_value_dict.items():
            if name not in self.manual_values and isinstance(value, placeholder_types):
                return True
        return any(
            node._has_placeholder_outputs_in_subtree()
            for node in self._sorted_nodes()
            if isinstance(node, PipelineHolder)
        )

    def _auto_resolve_placeholder_outputs(
        self,
        *,
        verbose: bool,
        _gate_cache: _GateStatusCache | None = None,
    ) -> None:
        """Recover produced values saved as placeholders by re-running their blocks.

        Walks nodes in upstream-to-downstream order and runs at most one block per
        placeholder recovery, injecting fresh values without invalidating any
        downstream outputs. Gate-off pipelines are skipped silently; recovery
        failures emit warnings (unconditional for execution exceptions,
        verbose-gated otherwise).
        """
        if _gate_cache is None:
            _gate_cache = _GateStatusCache()
        status, gate_error = self._pipeline_gate_status(_gate_cache=_gate_cache)
        if status == "block":
            return
        if status == "error":
            if verbose and self._has_placeholder_outputs_in_subtree():
                self.logger.warning(
                    f"Placeholder output(s) in pipeline '{self.full_path()}' are not recoverable: "
                    f"the gate could not be evaluated ({gate_error}); they remain placeholders"
                )
            return
        for node in self._sorted_nodes():
            if isinstance(node, PipelineHolder):
                node._auto_resolve_placeholder_outputs(
                    verbose=verbose,
                    _gate_cache=_gate_cache,
                )
                continue
            node_outputs = self.producer_outputs.get(node.registration_name, {})
            placeholder_names = [
                output_name
                for output_name, value in node_outputs.items()
                if isinstance(value, RuntimeValueReference)
            ]
            if not placeholder_names:
                continue
            self._recover_block_placeholder_outputs(
                node,
                placeholder_names,
                verbose=verbose,
                _gate_cache=_gate_cache,
            )
        for value_name, value in self.para_value_dict.items():
            if (
                isinstance(value, RuntimeValueReference)
                and value_name not in self.manual_values
                and self._tree_find_declaring_node(value_name) is None
                and verbose
            ):
                self.logger.warning(
                    f"Placeholder value '{value_name}' is not recoverable: its producing "
                    "block is not registered in the loaded pipeline; it remains a placeholder "
                    "and raises ResolutionError when read"
                )

    def _warn_unresolved_placeholders_at_load(
        self,
        *,
        verbose: bool,
        _seen: set[str] | None = None,
        _gate_cache: _GateStatusCache | None = None,
    ) -> None:
        """Verbose-gated load warning for produced values saved as placeholders.

        Used when ``auto_resolve_placeholders=False`` so users still learn that
        a produced value was saved as a placeholder rather than a real value.
        Gate-off pipelines are skipped silently; a gate that fails to evaluate
        logs a verbose warning instead.
        """
        if not verbose:
            return
        if _gate_cache is None:
            _gate_cache = _GateStatusCache()
        seen = set() if _seen is None else _seen
        status, gate_error = self._pipeline_gate_status(_gate_cache=_gate_cache)
        if status == "block":
            return
        if status == "error":
            self.logger.warning(
                f"Placeholder value(s) in pipeline '{self.full_path()}' could not be inspected: "
                f"the gate could not be evaluated ({gate_error})"
            )
            return
        for node in self._sorted_nodes():
            if isinstance(node, PipelineHolder):
                node._warn_unresolved_placeholders_at_load(
                    verbose=verbose,
                    _seen=seen,
                    _gate_cache=_gate_cache,
                )
                continue
            for output_name, value in self.producer_outputs.get(
                node.registration_name, {}
            ).items():
                if isinstance(value, RuntimeValueReference) and output_name not in seen:
                    seen.add(output_name)
                    self.logger.warning(
                        f"Pipeline value '{output_name}' was saved as a placeholder "
                        f"({value.reason}) rather than a real value; it raises "
                        "ResolutionError when read"
                    )
        for value_name, value in self.para_value_dict.items():
            if (
                isinstance(value, RuntimeValueReference)
                and value_name not in self.manual_values
                and self._tree_find_declaring_node(value_name) is None
                and value_name not in seen
            ):
                seen.add(value_name)
                self.logger.warning(
                    f"Pipeline value '{value_name}' was saved as a placeholder "
                    f"({value.reason}) rather than a real value; it raises "
                    "ResolutionError when read"
                )

    def _recover_block_placeholder_outputs(
        self,
        node: Any,
        placeholder_names: list[str],
        *,
        verbose: bool,
        _gate_cache: _GateStatusCache | None = None,
    ) -> None:
        upstream_outputs = self._recovery_upstream_outputs(_gate_cache=_gate_cache)
        parent_config = self._ancestor_config_values()
        visible_outputs = self._recovery_visible_outputs_before_priority(
            node.execution_priority,
            upstream_outputs=upstream_outputs,
            _gate_cache=_gate_cache,
        )
        for registration in node.functions:
            defaults = (
                {}
                if isinstance(registration, ExpressionRegistration)
                else default_map(registration.callable_obj)
            )
            for input_name in self._recovery_input_names(node, registration):
                status = self._recovery_input_status(
                    input_name,
                    visible_outputs,
                    parent_config,
                    defaults,
                )
                if status == "placeholder":
                    self._warn_placeholder_unrecoverable(
                        placeholder_names,
                        f"required input '{input_name}' is a placeholder that could not be restored",
                        verbose=verbose,
                    )
                    return
                if status == "unresolvable":
                    gate_off_reason = self._unresolvable_input_gate_off_reason(
                        input_name,
                        _gate_cache=_gate_cache,
                    )
                    self._warn_placeholder_unrecoverable(
                        placeholder_names,
                        f"required input '{input_name}' cannot be resolved{gate_off_reason}",
                        verbose=verbose,
                    )
                    return
        run_id = uuid4().hex
        run_record = RunRecord(
            run_id=run_id,
            mode=f"auto_resolve_placeholder:{node.registration_name}",
            executed_blocks=[node.registration_name],
            started_at=datetime.now(UTC).isoformat(),
        )
        self.run_history.append(run_record)
        try:
            produced_outputs = node.execute(
                run_id,
                visible_outputs,
                overrides={},
                parent_config=parent_config,
            )
        except (KeyboardInterrupt, SystemExit):
            run_record.status = "failed"
            run_record.finished_at = datetime.now(UTC).isoformat()
            raise
        except BaseException as exc:
            run_record.status = "failed"
            run_record.error_message = str(exc)
            run_record.finished_at = datetime.now(UTC).isoformat()
            self.logger.warning(
                f"Placeholder output(s) '{', '.join(sorted(placeholder_names))}' are not "
                f"recoverable: re-running block '{node.registration_name}' failed "
                f"({type(exc).__name__}: {exc})"
            )
            return
        self.producer_outputs[node.registration_name] = produced_outputs
        self._rebuild_visible_state(upstream_outputs)
        run_record.status = "success"
        run_record.produced_outputs.extend(sorted(placeholder_names))
        run_record.finished_at = datetime.now(UTC).isoformat()
        if verbose:
            self.logger.info(
                f"Recovered placeholder output(s) '{', '.join(sorted(placeholder_names))}' "
                f"by re-running block '{node.registration_name}'"
            )
        for output_name in produced_outputs:
            self._sync_value_to_ancestors_without_invalidation(output_name)

    def _recovery_input_names(self, node: Any, registration: Any) -> list[str]:
        """Effective pipeline-facing input names one registration resolves."""
        if isinstance(registration, ExpressionRegistration):
            return list(node._effective_expression_input_names(registration))
        input_names: list[str] = []
        for parameter in callable_signature(registration.callable_obj).parameters.values():
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                var_pos_name = registration.var_pos_name or parameter.name
                args_registration = node.registered_args.get(var_pos_name)
                if args_registration is not None:
                    input_names.extend(args_registration.ordered_items)
                continue
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                var_kw_name = registration.var_kw_name or parameter.name
                kwargs_registration = node.registered_kwargs.get(var_kw_name)
                if kwargs_registration is not None:
                    input_names.extend(kwargs_registration.mapping_dct.values())
                continue
            mapped_name = registration.param_mapping.get(parameter.name, parameter.name)
            if mapped_name is not None:
                input_names.append(mapped_name)
        return input_names

    def _recovery_input_status(
        self,
        input_name: str,
        visible_outputs: dict[str, Any],
        parent_config: dict[str, Any] | None,
        defaults: dict[str, Any],
    ) -> str:
        """Classify how one required input resolves during placeholder recovery.

        Returns ``"ok"``, ``"placeholder"`` (a placeholder reference would reach
        the function), or ``"unresolvable"`` (no source supplies the input).
        """
        if input_name == "logger":
            return "ok"
        if input_name in visible_outputs:
            value = visible_outputs[input_name]
        elif input_name in self.manual_values:
            value = self.manual_values[input_name]
        elif input_name in self._ancestor_manual_values():
            value = self._ancestor_manual_values()[input_name]
        elif self._config_has_field(self.config, input_name):
            value = self._config_value(self.config, input_name)
        elif parent_config and input_name in parent_config:
            value = parent_config[input_name]
        elif input_name in defaults:
            return "ok"
        else:
            return "unresolvable"
        if isinstance(value, (RuntimeValueReference, DataclassValueReference)):
            return "placeholder"
        return "ok"

    def _unresolvable_input_gate_off_reason(
        self,
        input_name: str,
        *,
        _gate_cache: _GateStatusCache | None = None,
    ) -> str:
        """Describe when an unresolvable input is only produced by gate-off blocks."""
        declaring_blocks = self._tree_declaring_blocks(input_name)
        if not declaring_blocks:
            return ""
        gated_labels = [
            f"'{node.registration_name}'"
            for pipeline, node in declaring_blocks
            if pipeline._pipeline_effectively_gated(_gate_cache=_gate_cache)
        ]
        if len(gated_labels) == len(declaring_blocks):
            return (
                f"; its only producer(s) {', '.join(gated_labels)} are gated off by config"
            )
        return ""

    def _tree_declaring_blocks(
        self,
        variable_name: str,
    ) -> list[tuple["PipelineHolder", Any]]:
        """Return ``(owning pipeline, block)`` pairs for every block declaring the name."""
        declaring: list[tuple["PipelineHolder", Any]] = []
        for node in self._sorted_nodes():
            if isinstance(node, PipelineHolder):
                if variable_name in node.list_declared_outputs():
                    declaring.extend(node._tree_declaring_blocks(variable_name))
            elif variable_name in node.declared_outputs():
                declaring.append((self, node))
        return declaring

    def _tree_find_declaring_node(self, variable_name: str) -> Any | None:
        """Find the first block declaring the name anywhere in the subtree."""
        for node in self._sorted_nodes():
            if isinstance(node, PipelineHolder):
                if variable_name in node.list_declared_outputs():
                    found = node._tree_find_declaring_node(variable_name)
                    if found is not None:
                        return found
            elif variable_name in node.declared_outputs():
                return node
        return None

    def _pipeline_gate_status(
        self,
        *,
        _gate_cache: _GateStatusCache | None = None,
    ) -> tuple[str, str | None]:
        """Classify this pipeline's gate chain as ``("pass" | "block" | "error", message)``.

        A successfully evaluated false gate blocks; a config-field gate whose
        value is ``None`` is treated as no blocking; any exception while
        evaluating a gate yields ``"error"`` with the exception text.

        When ``_gate_cache`` is given, each level's gate result is memoized on
        the level pipeline's identity plus a digest of the inputs its gate
        reads, so a gate runs at most once per unchanged input state during
        one recovery pass.
        """
        current: PipelineHolder | None = self
        while current is not None:
            gate = current.gate_block
            if gate is not None:
                status, message = self._gate_level_status(
                    current,
                    gate,
                    _gate_cache=_gate_cache,
                )
                if status in ("block", "error"):
                    return status, message
            current = current.parent_pipeline
        return "pass", None

    def _gate_level_status(
        self,
        pipeline: "PipelineHolder",
        gate: GateBlock,
        *,
        _gate_cache: _GateStatusCache | None,
    ) -> tuple[str, str | None]:
        """Evaluate (or reuse from the cache) one pipeline level's own gate."""
        if _gate_cache is None:
            return self._evaluate_gate_level(pipeline, gate)
        input_snapshot = pipeline._gate_level_input_digest()
        if input_snapshot is None:
            return self._evaluate_gate_level(pipeline, gate)
        cache_key = id(pipeline)
        cached = _gate_cache._levels.get(cache_key)
        if cached is not None and _values_equal(cached[0], input_snapshot):
            return cached[1]
        result = self._evaluate_gate_level(pipeline, gate)
        _gate_cache._levels[cache_key] = (input_snapshot, result)
        return result

    def _evaluate_gate_level(
        self,
        pipeline: "PipelineHolder",
        gate: GateBlock,
    ) -> tuple[str, str | None]:
        if gate.config_field_name is not None:
            try:
                value = pipeline._resolve_named_input(
                    gate.config_field_name,
                    gate.registration.function_name,
                    {},
                    pipeline._incoming_parent_outputs(),
                    pipeline._ancestor_config_values(),
                    {},
                    [],
                    set(pipeline._incoming_parent_outputs()).union(
                        pipeline.list_declared_outputs()
                    ),
                )
            except Exception as exc:
                return "error", f"{type(exc).__name__}: {exc}"
            if value is None or value == gate.expected_value:
                return "pass", None
            return "block", None
        try:
            gate_passes = gate.evaluate(
                {},
                pipeline._incoming_parent_outputs(),
                pipeline._ancestor_config_values(),
            )
        except Exception as exc:
            return "error", f"{type(exc).__name__}: {exc}"
        if gate_passes:
            return "pass", None
        return "block", None

    def _gate_level_input_digest(self) -> tuple[tuple[str, Any], ...] | None:
        """Snapshot every state this pipeline's own gate can read.

        Captures the incoming parent outputs, own and ancestor config fields,
        and own and ancestor manual values. Mutable values are copied so an
        in-place change invalidates the cached gate result. If a value cannot
        be copied safely, caching is disabled for that gate level.
        """
        entries: list[tuple[str, Any]] = []

        def append_snapshot(name: str, value: Any) -> bool:
            if not self._is_mutable_value(value):
                entries.append((name, value))
                return True
            try:
                snapshot = self._copy_value(value)
            except Exception:
                return False
            entries.append((name, snapshot))
            return True

        for name, value in self._incoming_parent_outputs().items():
            if not append_snapshot(name, value):
                return None
        for name, value in self._config_name_mapping(self.config).items():
            if not append_snapshot(f"config:{name}", value):
                return None
        ancestor = self.parent_pipeline
        while ancestor is not None:
            for name, value in self._config_name_mapping(ancestor.config).items():
                if not append_snapshot(
                    f"ancestor_config:{ancestor.registration_name}:{name}",
                    value,
                ):
                    return None
            ancestor = ancestor.parent_pipeline
        for name, value in self.manual_values.items():
            if not append_snapshot(f"manual:{name}", value):
                return None
        for name, value in self._ancestor_manual_values().items():
            if not append_snapshot(f"ancestor_manual:{name}", value):
                return None
        return tuple(sorted(entries, key=lambda item: item[0]))

    def _pipeline_effectively_gated(
        self,
        *,
        _gate_cache: _GateStatusCache | None = None,
    ) -> bool:
        """True when this pipeline or any ancestor gate blocks or fails to evaluate."""
        status, _ = self._pipeline_gate_status(_gate_cache=_gate_cache)
        return status in ("block", "error")

    def _recovery_upstream_outputs(
        self,
        *,
        _gate_cache: _GateStatusCache | None = None,
    ) -> dict[str, Any]:
        """Parent outputs visible to this pipeline, excluding gate-off producers."""
        if self.parent_pipeline is None or self.execution_priority is None:
            return {}
        return self.parent_pipeline._recovery_visible_outputs_before_priority(
            self.execution_priority,
            _gate_cache=_gate_cache,
        )

    def _recovery_visible_outputs_before_priority(
        self,
        priority: float | None,
        upstream_outputs: dict[str, Any] | None = None,
        *,
        _gate_cache: _GateStatusCache | None = None,
    ) -> dict[str, Any]:
        """Visible outputs for placeholder recovery, ignoring gate-off producers.

        A same-name output produced by a non-gated block is used in place of an
        identical output declared by a gate-off block, regardless of whether the
        alternative sits upstream or downstream of the gate-off block (point 4).
        """
        visible = dict(
            upstream_outputs
            if upstream_outputs is not None
            else self._recovery_upstream_outputs(_gate_cache=_gate_cache)
        )
        visible.update(self.manual_values)
        if priority is None:
            return visible
        for node in self._sorted_nodes():
            if node.execution_priority >= priority:
                break
            if isinstance(node, PipelineHolder):
                if node._pipeline_effectively_gated(_gate_cache=_gate_cache):
                    continue
                visible.update(node.para_value_dict)
            else:
                visible.update(self.producer_outputs.get(node.registration_name, {}))
        return visible

    def _warn_placeholder_unrecoverable(
        self,
        placeholder_names: list[str],
        reason: str,
        *,
        verbose: bool,
    ) -> None:
        if not verbose:
            return
        self.logger.warning(
            f"Placeholder output(s) '{', '.join(sorted(placeholder_names))}' are not "
            f"recoverable at load: {reason}; they remain placeholders and raise "
            "ResolutionError when read"
        )

    def _serialize_payload_for_save(
        self,
        target_root: Path,
        cache: dict[int, Any] | None = None,
    ) -> dict[str, Any]:
        cache = {} if cache is None else cache
        traceback_settings = self.logger.get_traceback_settings()
        return {
            "registration_name": self.registration_name,
            "config": (
                {}
                if self._is_atom
                else self._serialize_config_for_save(self.config)
            ),
            "execution_priority": self.execution_priority,
            "is_atom": self._is_atom,
            "saved_project_root": str(self.project_root),
            "pipeline_backup_directory": (
                None
                if self.pipeline_backup_root is None
                else str(self.pipeline_backup_root)
            ),
            "expression_runtime_code": self.expression_runtime_code,
            "memory_saving_mode": self.memory_saving_mode,
            "memory_profile_logging": self.memory_profile_logging,
            "log_traceback_to_file": traceback_settings["log_traceback_to_file"],
            "show_traceback_locals": traceback_settings["show_traceback_locals"],
            "use_rich_traceback_console": traceback_settings["use_rich_traceback_console"],
            "torch_load_weights_only": self.torch_load_weights_only,
            "strict_mode": self.strict_mode,
            "historical_result_log_path": self.historical_result_log_path,
            "gate": None if self.gate_block is None else self.gate_block.serialize(),
            "nodes": [self._serialize_node_for_save(node, target_root, cache) for node in self._sorted_nodes()],
            "manual_values": {
                output_name: self._serialize_runtime_value_for_save(
                    value,
                    target_root,
                    cache,
                    "manual_values",
                    output_name,
                    sibling_outputs=self.manual_values,
                )
                for output_name, value in self.manual_values.items()
            },
            "producer_outputs": {
                node_name: {
                    output_name: self._serialize_runtime_value_for_save(
                        value,
                        target_root,
                        cache,
                        node_name,
                        output_name,
                        sibling_outputs=outputs,
                    )
                    for output_name, value in outputs.items()
                }
                for node_name, outputs in self.producer_outputs.items()
            },
            "para_value_dict": {
                output_name: self._serialize_runtime_value_for_save(
                    value,
                    target_root,
                    cache,
                    "pipeline_state",
                    output_name,
                    sibling_outputs=self.para_value_dict,
                )
                for output_name, value in self.para_value_dict.items()
            },
            "artifact_registry": {
                output_name: self._serialize_runtime_value_for_save(
                    value,
                    target_root,
                    cache,
                    "artifact_registry",
                    output_name,
                    sibling_outputs=self.artifact_registry,
                )
                for output_name, value in self.artifact_registry.items()
            },
            "object_storage": self._serialize_stored_objects_for_save(target_root),
            "run_history": self.run_history,
        }

    def _node_overridden_outputs(self, node: Any) -> dict[str, OutputAddress]:
        if isinstance(node, PipelineHolder):
            return dict(node._overridden_outputs)
        overrides: dict[str, OutputAddress] = {}
        for registration in node.functions:
            overrides.update(registration.overridden_outputs)
        return overrides

    def _confirm_gate_cleanup(self, mode: str) -> bool:
        gate_label = (
            self.gate_block.config_field_name
            if self.gate_block is not None and self.gate_block.config_field_name is not None
            else (
                self.gate_block.registration.function_name
                if self.gate_block is not None
                else "?"
            )
        )
        affected = sorted(
            {
                output_name
                for outputs in self.producer_outputs.values()
                for output_name in outputs
            }
            | set(self.artifact_registry)
        )
        reason = (
            f"Gate '{gate_label}' did not pass for {mode}, so the run is skipped. "
            f"Cleaning up would invalidate {len(affected)} produced value(s) "
            f"({', '.join(affected) if affected else 'none'}) and delete their disk artifacts; "
            "downstream blocks and child pipelines consuming them would then receive None. "
            "Type 'yes' or 'y' to clean up, anything else keeps the current values (non-destructive): "
        )
        answer = input(reason).strip().lower()
        return answer in {"yes", "y"}

    def _gate_skip_without_cleanup(
        self,
        mode: str,
        overrides: dict[str, Any] | None,
        base_visible: dict[str, Any],
        parent_config: Any | None,
    ) -> bool:
        """Return True when the gate fails and cleanup was declined.

        Runs before any output invalidation so a declined confirmation leaves the
        current values and artifacts untouched.
        """
        if not self.gate_cleanup_confirmation:
            return False
        if self.gate_block is None:
            return False
        if not (self.producer_outputs or self.artifact_registry):
            return False
        if self.gate_block.evaluate(overrides or {}, base_visible, parent_config):
            return False
        self._gate_cleanup_predecided = self._confirm_gate_cleanup(mode)
        return not self._gate_cleanup_predecided

    def _build_skipped_run_record(self, mode: str) -> RunRecord:
        run_id = uuid4().hex
        run_record = RunRecord(
            run_id=run_id,
            mode=mode,
            executed_blocks=[],
            started_at=datetime.now(UTC).isoformat(),
        )
        run_record.status = "skipped"
        run_record.finished_at = datetime.now(UTC).isoformat()
        self.run_history.append(run_record)
        self.logger.warning(
            f"Skipped {mode} with run_id={run_id} without cleanup (cleanup declined)"
        )
        return run_record

    def _register_node(self, node: Any) -> None:
        if self.nodes_by_name.get(node.registration_name) is node:
            return
        self._validate_node_registration(node, node.execution_priority)
        self.nodes.append(node)
        self.nodes_by_name[node.registration_name] = node
        if not isinstance(node, PipelineHolder):
            self.blocks.append(node)
            self.blocks_by_name[node.registration_name] = node

    def _validate_output_names_against_config(self, output_names: list[str]) -> None:
        if not output_names:
            return
        self._validate_builtin_name_conflicts_in_mapping(
            {output_name: None for output_name in output_names},
            owner_label="pipeline value",
        )
        conflicts = set(output_names).intersection(self._visible_config_names())
        if conflicts:
            raise RegistrationError(
                f"Output names conflict with visible configuration fields: {sorted(conflicts)}"
            )
        constant_conflicts = set(output_names).intersection(self._tree_constant_names())
        if constant_conflicts:
            raise RegistrationError(
                f"Output names conflict with pipeline constants: {sorted(constant_conflicts)}"
            )

    def _registration_conflicts(self, node: Any, execution_priority: float | None) -> list[Any]:
        conflicts: list[Any] = []
        existing = self.nodes_by_name.get(node.registration_name)
        if existing is not None and existing is not node:
            conflicts.append(existing)
        for existing_node in self.nodes:
            if existing_node is node or existing_node in conflicts:
                continue
            if existing_node.execution_priority == execution_priority:
                conflicts.append(existing_node)
        return conflicts

    def _priority_group(self, execution_priority: float | None) -> int:
        if execution_priority is None:
            return -1
        return int(execution_priority)

    def _select_executable_node_in_group(self, nodes: list[Any]) -> Any:
        for node in sorted(nodes, key=lambda item: (item.execution_priority, item.registration_name)):
            if isinstance(node, PipelineHolder):
                if node.gate_block is None:
                    return node
                try:
                    should_run = node.gate_block.evaluate(
                        {},
                        self._visible_outputs_before_priority(node.execution_priority),
                        self.config_as_dict(),
                    )
                except Exception:
                    return node
                if should_run:
                    return node
                continue
            return node
        return None

    def _raise_on_priority_conflict_with_different_name(
        self,
        registration_name: str,
        execution_priority: float | None,
        conflicts: list[Any],
    ) -> None:
        for node in conflicts:
            if (
                node.execution_priority == execution_priority
                and node.registration_name != registration_name
            ):
                raise RegistrationError(
                    f"Execution priority {execution_priority} is already used by '{node.registration_name}'"
                )

    def _replace_conflicting_nodes(self, nodes: list[Any]) -> None:
        if not nodes:
            return
        earliest_priority = min(
            node.execution_priority for node in nodes if node.execution_priority is not None
        )
        for node in nodes:
            self._erase_node_outputs(node.registration_name)
            if isinstance(node, PipelineHolder):
                node._invalidate_all_outputs()
            self._remove_registered_node(node)
        if not self._invalidation_forbidden:
            self._invalidate_from_priority(earliest_priority)
        if self.parent_pipeline is not None:
            self._resync_mirror_to_parent()

    def _remove_registered_node(self, node: Any) -> None:
        self.nodes = [candidate for candidate in self.nodes if candidate is not node]
        self.nodes_by_name.pop(node.registration_name, None)
        if not isinstance(node, PipelineHolder):
            self.blocks = [candidate for candidate in self.blocks if candidate is not node]
            self.blocks_by_name.pop(node.registration_name, None)

    def _validate_node_registration(self, node: Any, execution_priority: float | None) -> None:
        existing = self.nodes_by_name.get(node.registration_name)
        if existing is not None and existing is not node:
            raise RegistrationError(f"Node already registered: {node.registration_name}")
        for existing_node in self.nodes:
            if existing_node is node:
                return
            if existing_node.execution_priority == execution_priority:
                raise RegistrationError(
                    f"Execution priority already registered: {execution_priority}"
                )

    def _validate_related_pipeline_names(self, candidate: "PipelineHolder") -> None:
        candidate_pipelines = list(candidate._iter_attached_pipelines())
        candidate_names = [pipeline.registration_name for pipeline in candidate_pipelines]
        duplicates = sorted(
            {
                name
                for name in candidate_names
                if candidate_names.count(name) > 1
            }
        )
        if duplicates:
            raise RegistrationError(
                f"Pipeline names must be unique inside the attached subtree: {duplicates}"
            )

        candidate_ids = {id(pipeline) for pipeline in candidate_pipelines}
        related_names = {
            pipeline.registration_name
            for pipeline in self._root_pipeline()._iter_attached_pipelines()
            if id(pipeline) not in candidate_ids
        }
        overlap = sorted(set(candidate_names).intersection(related_names))
        if overlap:
            raise RegistrationError(
                f"Pipeline names must be unique across the related pipeline tree: {overlap}"
            )

    def _root_pipeline(self) -> "PipelineHolder":
        current = self
        while current.parent_pipeline is not None:
            current = current.parent_pipeline
        return current

    def _pipeline_by_name(self, pipeline_name: str) -> "PipelineHolder":
        matches = [
            pipeline
            for pipeline in self._root_pipeline()._iter_attached_pipelines()
            if pipeline.registration_name == pipeline_name
        ]
        if len(matches) != 1:
            raise RegistrationError(
                f"Output pointer pipeline must identify one attached pipeline: {pipeline_name!r}"
            )
        return matches[0]

    def _priority_vector(
        self,
        node_name: str,
        node_priority: float | None = None,
    ) -> tuple[float, ...]:
        priorities: list[float] = []
        chain: list[PipelineHolder] = []
        current: PipelineHolder | None = self
        while current is not None and current.parent_pipeline is not None:
            chain.append(current)
            current = current.parent_pipeline
        for pipeline in reversed(chain):
            if pipeline.execution_priority is None or not math.isfinite(
                pipeline.execution_priority
            ):
                raise RegistrationError(
                    f"Pipeline '{pipeline.registration_name}' has an invalid output-pointer priority"
                )
            priorities.append(pipeline.execution_priority)
        priority = node_priority
        if priority is None:
            node = self.nodes_by_name.get(node_name)
            priority = None if node is None else node.execution_priority
        if priority is None or not math.isfinite(priority):
            raise RegistrationError(
                f"Node '{node_name}' has an invalid output-pointer priority"
            )
        priorities.append(priority)
        return tuple(priorities)

    def _iter_attached_pipelines(self) -> list["PipelineHolder"]:
        pipelines: list[PipelineHolder] = [self]
        for node in self._sorted_nodes():
            if isinstance(node, PipelineHolder):
                for pipeline in node._iter_attached_pipelines():
                    pipelines.append(pipeline)
        return pipelines

    def _tree_constant_names(self) -> set[str]:
        names: set[str] = set()
        for pipeline in self._root_pipeline()._iter_attached_pipelines():
            names.update(pipeline.manual_values)
        return names

    def _tree_declared_output_names(self) -> set[str]:
        names: set[str] = set()
        for pipeline in self._root_pipeline()._iter_attached_pipelines():
            names.update(pipeline.list_declared_outputs())
        return names

    def _tree_produced_value_names(self) -> set[str]:
        names: set[str] = set()
        constants = self._tree_constant_names()
        for pipeline in self._root_pipeline()._iter_attached_pipelines():
            for outputs in pipeline.producer_outputs.values():
                for name in outputs:
                    if name not in constants:
                        names.add(name)
        return names

    def _get_node_or_raise(self, block_name: str) -> Any:
        node = self.nodes_by_name.get(block_name)
        if node is None:
            raise RegistrationError(f"Node not registered: {block_name}")
        return node

    def _resolve_target_path(self, path_parts: tuple[str, ...]) -> tuple["PipelineHolder", Any]:
        if not path_parts:
            raise RegistrationError("At least one target name must be provided")
        current: PipelineHolder = self
        for pipeline_name in path_parts[:-1]:
            current = current.get_child_pipeline(pipeline_name)
        return current, current._get_node_or_raise(path_parts[-1])

    def _run_nested_until_path(
        self,
        path_parts: tuple[str, ...],
        overrides: dict[str, Any] | None = None,
    ) -> RunRecord:
        child_name = path_parts[0]
        child_pipeline = self.get_child_pipeline(child_name)
        child_priority = child_pipeline.execution_priority
        if child_priority is None:
            raise RegistrationError(f"Child pipeline '{child_name}' has no priority")
        (
            self._invalidate_from_priority(self._sorted_nodes()[0].execution_priority)
            if self.nodes
            else None
        )
        selected = [
            candidate
            for candidate in self._sorted_nodes()
            if candidate.execution_priority < child_priority
        ]
        if selected:
            self._execute_nodes(
                selected,
                mode=f"run_until_parent:{child_name}",
                overrides=overrides,
                upstream_outputs=self._incoming_parent_outputs(),
                parent_config=self._ancestor_config_values(),
                sync_parent_on_completion=False,
            )
        return child_pipeline.run_until(*path_parts[1:], overrides=overrides)

    def _run_nested_from_path(
        self,
        path_parts: tuple[str, ...],
        overrides: dict[str, Any] | None = None,
    ) -> RunRecord:
        child_name = path_parts[0]
        child_pipeline = self.get_child_pipeline(child_name)
        child_priority = child_pipeline.execution_priority
        if child_priority is None:
            raise RegistrationError(f"Child pipeline '{child_name}' has no priority")
        self._invalidate_from_priority(
            child_priority,
            preserve_pipeline_state=child_pipeline,
        )
        child_run = child_pipeline.run_from(*path_parts[1:], overrides=overrides)
        self.producer_outputs[child_pipeline.registration_name] = {
            name: value
            for name, value in child_pipeline.para_value_dict.items()
            if name not in child_pipeline.manual_values
        }
        self._rebuild_visible_state(self._incoming_parent_outputs())
        downstream_nodes = [
            candidate
            for candidate in self._sorted_nodes()
            if candidate.execution_priority > child_priority
        ]
        if downstream_nodes:
            self._execute_nodes(
                downstream_nodes,
                mode=f"run_from_parent_tail:{child_name}",
                overrides=overrides,
                upstream_outputs=self._visible_outputs_before_priority(downstream_nodes[0].execution_priority),
                parent_config=self._ancestor_config_values(),
            )
        return child_run

    def _run_nested_block_path(
        self,
        path_parts: tuple[str, ...],
        overrides: dict[str, Any] | None = None,
    ) -> RunRecord:
        child_name = path_parts[0]
        child_pipeline = self.get_child_pipeline(child_name)
        child_priority = child_pipeline.execution_priority
        if child_priority is None:
            raise RegistrationError(f"Child pipeline '{child_name}' has no priority")
        selected = [
            candidate
            for candidate in self._sorted_nodes()
            if candidate.execution_priority < child_priority
        ]
        if selected:
            self._execute_nodes(
                selected,
                mode=f"run_block_parent:{child_name}",
                overrides=overrides,
                upstream_outputs=self._incoming_parent_outputs(),
                parent_config=self._ancestor_config_values(),
                sync_parent_on_completion=False,
            )
        return child_pipeline.run_block(*path_parts[1:], overrides=overrides)

    def _run_nested_until(
        self,
        child_pipeline: "PipelineHolder",
        target_name: str,
        overrides: dict[str, Any] | None = None,
    ) -> RunRecord:
        child_priority = child_pipeline.execution_priority
        if child_priority is None:
            raise RegistrationError(f"Child pipeline '{child_pipeline.registration_name}' has no priority")
        selected = [
            candidate
            for candidate in self._sorted_nodes()
            if candidate.execution_priority < child_priority
        ]
        if selected:
            self._execute_nodes(
                selected,
                mode=f"run_until_parent:{child_pipeline.registration_name}",
                overrides=overrides,
                upstream_outputs=self._incoming_parent_outputs(),
                parent_config=self._ancestor_config_values(),
                sync_parent_on_completion=False,
            )
        return child_pipeline.run_until(target_name, overrides=overrides)

    def _run_nested_from(
        self,
        child_pipeline: "PipelineHolder",
        target_name: str,
        overrides: dict[str, Any] | None = None,
    ) -> RunRecord:
        child_priority = child_pipeline.execution_priority
        if child_priority is None:
            raise RegistrationError(f"Child pipeline '{child_pipeline.registration_name}' has no priority")
        snapshot = self._snapshot_runtime_state()
        previous_outputs = snapshot[0].get(child_pipeline.registration_name, {})
        self._invalidate_from_priority(child_priority)
        return child_pipeline.run_from(target_name, overrides=overrides)

    def _run_nested_block(
        self,
        child_pipeline: "PipelineHolder",
        target_name: str,
        overrides: dict[str, Any] | None = None,
    ) -> RunRecord:
        child_priority = child_pipeline.execution_priority
        if child_priority is None:
            raise RegistrationError(f"Child pipeline '{child_pipeline.registration_name}' has no priority")
        selected = [
            candidate
            for candidate in self._sorted_nodes()
            if candidate.execution_priority < child_priority
        ]
        if selected:
            self._execute_nodes(
                selected,
                mode=f"run_block_parent:{child_pipeline.registration_name}",
                overrides=overrides,
                upstream_outputs=self._incoming_parent_outputs(),
                parent_config=self._ancestor_config_values(),
                sync_parent_on_completion=False,
            )
        return child_pipeline.run_block(target_name, overrides=overrides)

    def _materialize_previous_node_inputs(
        self,
        node: Any,
        previous_outputs: dict[str, Any],
        overrides: dict[str, Any] | None,
    ) -> dict[str, Any]:
        materialized = dict(previous_outputs)
        required_outputs = self._required_input_names(node).intersection(
            self._node_declared_outputs(node)
        )
        required_outputs.difference_update(overrides or {})
        for output_name in required_outputs:
            value = materialized.get(output_name)
            if isinstance(value, (ArtifactRecord, OutputPointer)):
                materialized[output_name] = self._materialize_stored_value(value, "")
        return materialized

    def _cleanup_block_memory(self, node_name: str) -> None:
        gc.collect()
        self._attempt_allocator_trim()
        if self.memory_profile_logging:
            self._log_memory_profile(node_name, phase="after_cleanup")

    def _log_memory_profile(self, node_name: str, phase: str = "after_cleanup") -> None:
        try:
            import psutil  # type: ignore

            process = psutil.Process()
            rss_mb = process.memory_info().rss / (1024 * 1024)
            self.logger.info(f"memory {phase} {node_name}: rss={rss_mb:.2f}MB")
        except Exception:
            return

    def _attempt_allocator_trim(self) -> None:
        if platform.system() != "Linux":
            return
        try:
            libc = CDLL("libc.so.6")
            malloc_trim = getattr(libc, "malloc_trim", None)
            if malloc_trim is None:
                return
            malloc_trim(0)
        except Exception:
            return

    @staticmethod
    def _config_name_mapping(config_obj: Any) -> dict[str, Any]:
        if is_dataclass(config_obj) and not isinstance(config_obj, type):
            return SerializationMixin._config_object_as_dict(config_obj)
        if isinstance(config_obj, dict):
            return dict(config_obj)
        if hasattr(config_obj, "__dict__"):
            return dict(vars(config_obj))
        return {}

    @staticmethod
    def _validate_builtin_name_conflict(name: str, owner_label: str) -> None:
        if name in _RESERVED_BUILTIN_NAMES:
            raise RegistrationError(
                f"{owner_label.capitalize()} name '{name}' conflicts with a reserved Python builtin"
            )

    @classmethod
    def _validate_builtin_name_conflicts_in_mapping(
        cls,
        values: dict[str, Any],
        owner_label: str,
    ) -> None:
        for name in values:
            cls._validate_builtin_name_conflict(name, owner_label)

    @staticmethod
    def _validate_config_value_picklable(field_name: str, value: Any) -> None:
        if isinstance(value, (CallableValueReference, RuntimeValueReference, RuntimeCallableReference)):
            return
        try:
            pickle.dumps(value)
        except Exception as exc:
            raise RegistrationError(
                f"Config field '{field_name}' is not picklable ({type(value).__name__}: {exc}); "
                "use set_constant_value instead for values that cannot be persisted"
            ) from exc

    @staticmethod
    def _validate_config_reconstructable(config: Any) -> None:
        if isinstance(config, dict):
            return
        if is_dataclass(config) and not isinstance(config, type):
            field_names = set(config.__dataclass_fields__)
            offending: list[str] = []
            for cls in type(config).__mro__:
                if cls is object:
                    continue
                for name in vars(cls):
                    if name.startswith("__") or name in field_names:
                        continue
                    offending.append(name)
            if offending:
                raise RegistrationError(
                    f"Pipeline configuration dataclass '{type(config).__name__}' "
                    f"must be pure (fields only) so it can be reconstructed after "
                    f"save/load; found non-field member(s): {sorted(offending)}"
                )
            return
        raise RegistrationError(
            f"Pipeline configuration must be a dict or a pure dataclass instance "
            f"(fields only) so it can be reconstructed after save/load; got "
            f"{type(config).__name__}. Use a dict or define a @dataclass config."
        )

    @classmethod
    def _validate_config_picklable(cls, config: Any) -> None:
        if config is None:
            return
        if is_dataclass(config) and not isinstance(config, type):
            for name in config.__dataclass_fields__:
                cls._validate_config_value_picklable(name, getattr(config, name))
            if hasattr(config, "__dict__"):
                extra_names = set(vars(config)).difference(config.__dataclass_fields__)
                for name in extra_names:
                    cls._validate_config_value_picklable(name, getattr(config, name))
            return
        if isinstance(config, dict):
            for name, value in config.items():
                cls._validate_config_value_picklable(name, value)
            return
        if hasattr(config, "__dict__"):
            for name, value in vars(config).items():
                cls._validate_config_value_picklable(name, value)
            return
        cls._validate_config_value_picklable("configuration", config)

    def _set_config_value(self, field_name: str, value: Any) -> None:
        if is_dataclass(self.config) and not isinstance(self.config, type):
            setattr(self.config, field_name, value)
            return
        if isinstance(self.config, dict):
            self.config[field_name] = value
            return
        setattr(self.config, field_name, value)

    def _require_owned_config(self) -> None:
        if not getattr(self, "_is_atom", False):
            return
        parent_name = (
            "the parent pipeline"
            if self.parent_pipeline is None
            else f"parent pipeline '{self.parent_pipeline.registration_name}'"
        )
        raise RegistrationError(
            f"Atom pipeline '{self.registration_name}' does not own configuration; "
            f"call set_config(), set_configs(), update_config(), or update_configs() "
            f"on {parent_name}"
        )

    def _invalidate_from_priority(
        self,
        priority: float,
        include_target: bool = True,
        preserve_pipeline_state: PipelineHolder | None = None,
    ) -> None:
        if priority is None or self._invalidation_forbidden:
            return
        self._root_pipeline()._remember_optuna_study_groups()
        selected_nodes = [
            node
            for node in self._sorted_nodes()
            if node.execution_priority >= priority
            and (node.execution_priority != priority or include_target)
        ]
        slots = self._public_output_slots()
        invalidated = {
            OutputAddress(self.registration_name, node.registration_name, output_name)
            for node in selected_nodes
            for output_name in self.producer_outputs.get(
                node.registration_name,
                {},
            )
        }
        self._prepare_pointer_removal(invalidated, slots)
        removed_outputs: list[dict[str, Any]] = []
        for node in selected_nodes:
            removed_outputs.append(
                self.producer_outputs.pop(node.registration_name, {})
            )
            if (
                isinstance(node, PipelineHolder)
                and node is not preserve_pipeline_state
            ):
                node._invalidate_all_outputs()
        self._refresh_pointer_visible_state()
        for outputs in removed_outputs:
            self._delete_artifacts_from_outputs(outputs)

    def _invalidate_all_outputs(self) -> None:
        removed_outputs = list(self.producer_outputs.values())
        self.producer_outputs.clear()
        self.para_value_dict.clear()
        self.artifact_registry.clear()
        for node in self._sorted_nodes():
            if isinstance(node, PipelineHolder):
                node._invalidate_all_outputs()
        self._rebuild_visible_state(self._incoming_parent_outputs())
        for outputs in removed_outputs:
            self._delete_artifacts_from_outputs(outputs)

    def _delete_artifacts_from_outputs(self, outputs: dict[str, Any]) -> None:
        active_artifact_paths = self._collect_referenced_artifact_paths()
        for value in outputs.values():
            if isinstance(value, ArtifactRecord):
                if (
                    str(Path(value.file_path).resolve())
                    in active_artifact_paths
                ):
                    continue
                try:
                    self.artifact_store.delete(value)
                except (OSError, PersistenceError) as exc:
                    self.logger.warning(
                        f"Could not delete obsolete artifact '{value.file_path}': "
                        f"{type(exc).__name__}: {exc}"
                    )

    def _cleanup_replaced_artifact(self, previous_value: Any) -> None:
        """Retire a replaced artifact file only after its replacement committed.

        Runs after the new value has been saved and every mapping slot has
        been updated, so a serialization failure can never leave a deleted
        file behind. The previous file is removed only when no remaining slot
        (or mirror in this subtree) still references it, so passing the same
        record as its own replacement keeps the file alive.
        """
        if not isinstance(previous_value, ArtifactRecord):
            return
        self._delete_artifacts_from_outputs({"_replaced": previous_value})

    def _collect_referenced_artifact_paths(self) -> set[str]:
        paths: set[str] = set()

        def collect_from_mapping(mapping: dict[str, Any]) -> None:
            for value in mapping.values():
                if isinstance(value, ArtifactRecord):
                    paths.add(str(Path(value.file_path).resolve()))

        def collect_from_pipeline(pipeline: "PipelineHolder") -> None:
            collect_from_mapping(pipeline.manual_values)
            for record in pipeline._stored_objects.values():
                if record.artifact is not None:
                    paths.add(str(Path(record.artifact.file_path).resolve()))
            for node in pipeline._sorted_nodes():
                if isinstance(node, PipelineHolder):
                    collect_from_pipeline(node)
                else:
                    collect_from_mapping(
                        pipeline.producer_outputs.get(
                            node.registration_name,
                            {},
                        )
                    )

        collect_from_pipeline(self._root_pipeline())
        return paths

    def _persist_config_snapshot(self, path: Path) -> None:
        atomic_pickle_dump(self._serialize_config_for_save(self.config), path)

    def _attach_to_parent(self, parent: "PipelineHolder", execution_priority: float) -> None:
        # Registration moves the child's working tree underneath the parent project root.
        # Future execution uses the parent logger, but historical child RESULT display still
        # reads from the child-side historical log path captured here.
        if self.parent_pipeline is not None and self.parent_pipeline is not parent:
            raise RegistrationError(
                f"Pipeline '{self.registration_name}' is already attached to another parent"
            )
        original_root = self.project_root
        target_root = parent.project_root / "children" / self.registration_name
        target_root.mkdir(parents=True, exist_ok=True)
        if original_root != target_root and original_root.exists():
            for entry in original_root.iterdir():
                destination = target_root / entry.name
                if destination.exists():
                    if destination.is_dir():
                        shutil.rmtree(destination)
                    else:
                        destination.unlink()
                shutil.move(str(entry), str(destination))
            if original_root.exists() and not any(original_root.iterdir()):
                original_root.rmdir()
        moved_log_path = target_root / "metadata" / "pipeline.log"
        self.historical_result_log_path = str(moved_log_path)
        self.project_root = target_root
        self.metadata_root = target_root / "metadata"
        self.metadata_root.mkdir(parents=True, exist_ok=True)
        self.artifact_store = ArtifactStore(target_root)
        self.parent_pipeline = parent
        self.execution_priority = execution_priority
        self.logger = parent.logger
        self.memory_saving_mode = parent.memory_saving_mode
        self.memory_profile_logging = parent.memory_profile_logging
        inherited_strict_mode = parent._root_pipeline().strict_mode
        for pipeline in self._iter_attached_pipelines():
            pipeline.strict_mode = inherited_strict_mode
        self._rewrite_artifact_paths(original_root, target_root)
        self._rewrite_run_history_paths(original_root, target_root)
        self._refresh_descendant_roots(original_root, target_root)
        self._merge_storage_into_root()
        self._cleanup_temporary_root_handle()

    def _merge_storage_into_root(self) -> None:
        if self.parent_pipeline is None or not self._stored_objects:
            return
        root = self.parent_pipeline._root_pipeline()
        for hash_id, source_record in self._stored_objects.items():
            if hash_id in root._stored_objects:
                if source_record.artifact is not None:
                    self.artifact_store.delete(source_record.artifact)
                continue
            value = self._load_stored_object_value(source_record)
            adopted_record = StoredObjectRecord(
                hash_id=source_record.hash_id,
                object_name=source_record.object_name,
                object_type=source_record.object_type,
                object_description=source_record.object_description,
                created_at_utc=source_record.created_at_utc,
                last_modified_at_utc=source_record.last_modified_at_utc,
                value=value,
            )
            if source_record.artifact is not None:
                root._persist_stored_object(adopted_record, root.project_root)
                self.artifact_store.delete(source_record.artifact)
            root._stored_objects[hash_id] = adopted_record
        self._stored_objects.clear()

    def _sync_attached_outputs_to_parent(self) -> None:
        if self.parent_pipeline is None or self.execution_priority is None:
            return
        if self._invalidation_forbidden:
            self._resync_mirror_to_parent()
            return
        self.parent_pipeline.producer_outputs[self.registration_name] = (
            self._locally_produced_outputs()
        )
        self.parent_pipeline._invalidate_from_priority(self.execution_priority, include_target=False)

    def _rewrite_artifact_paths(self, old_root: Path, new_root: Path) -> None:
        old_prefix = str(old_root)
        new_prefix = str(new_root)

        def rewrite_value(value: Any) -> Any:
            if isinstance(value, ArtifactRecord) and value.file_path.startswith(old_prefix):
                value.file_path = value.file_path.replace(old_prefix, new_prefix, 1)
            if isinstance(value, ArtifactRecord):
                metadata = getattr(value, "metadata", None)
                if not isinstance(metadata, dict):
                    return value
                db_path = metadata.get("db_path")
                if isinstance(db_path, str) and db_path.startswith(old_prefix):
                    metadata["db_path"] = db_path.replace(
                        old_prefix,
                        new_prefix,
                        1,
                    )
            return value

        for outputs in self.producer_outputs.values():
            for key, value in list(outputs.items()):
                outputs[key] = rewrite_value(value)
        for key, value in list(self.para_value_dict.items()):
            self.para_value_dict[key] = rewrite_value(value)
        for key, value in list(self.artifact_registry.items()):
            self.artifact_registry[key] = rewrite_value(value)
        for record in self._stored_objects.values():
            if record.artifact is not None:
                record.artifact = rewrite_value(record.artifact)
        if self.historical_result_log_path and self.historical_result_log_path.startswith(
            old_prefix
        ):
            self.historical_result_log_path = self.historical_result_log_path.replace(
                old_prefix,
                new_prefix,
                1,
            )
        for node in self._sorted_nodes():
            if isinstance(node, PipelineHolder):
                node._rewrite_artifact_paths(old_root, new_root)

    def _rewrite_run_history_paths(self, old_root: Path, new_root: Path) -> None:
        old_prefix = str(old_root)
        new_prefix = str(new_root)
        for run_record in self.run_history:
            if run_record.config_snapshot_path and run_record.config_snapshot_path.startswith(old_prefix):
                run_record.config_snapshot_path = run_record.config_snapshot_path.replace(
                    old_prefix,
                    new_prefix,
                    1,
                )
        for node in self._sorted_nodes():
            if isinstance(node, PipelineHolder):
                node._rewrite_run_history_paths(old_root, new_root)

    def _refresh_descendant_roots(self, old_root: Path, new_root: Path) -> None:
        for node in self._sorted_nodes():
            if not isinstance(node, PipelineHolder):
                continue
            old_child_root = node.project_root
            try:
                relative = old_child_root.relative_to(old_root)
            except ValueError:
                relative = Path("children") / node.registration_name
            new_child_root = new_root / relative
            node.project_root = new_child_root
            node.metadata_root = new_child_root / "metadata"
            node.metadata_root.mkdir(parents=True, exist_ok=True)
            node.artifact_store = ArtifactStore(new_child_root)
            node.logger = self.logger
            node.memory_saving_mode = self.memory_saving_mode
            node.memory_profile_logging = self.memory_profile_logging
            node._rewrite_artifact_paths(old_child_root, new_child_root)
            node._rewrite_run_history_paths(old_child_root, new_child_root)
            node._refresh_descendant_roots(old_child_root, new_child_root)

    def qualified_node_name(self, node_name: str) -> str:
        return f"{self.full_path()}/{node_name}"

    def full_path(self) -> str:
        if self.parent_pipeline is None:
            return self.registration_name
        return f"{self.parent_pipeline.full_path()}/{self.registration_name}"

    def _prepare_project_root(self, forced: bool) -> None:
        if not self.project_root.exists():
            return
        if not any(self.project_root.iterdir()):
            return
        if not forced:
            raise RegistrationError(
                f"Pipeline root folder is not empty: {self.project_root}"
            )
        user_input = input(
            f"Pipeline root folder '{self.project_root}' is not empty. Type 'yes' to clear it: "
        ).strip()
        if user_input != "yes":
            raise RegistrationError(
                f"Pipeline root folder is not empty: {self.project_root}"
            )
        for entry in self.project_root.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()


PipelineHandler = PipelineHolder
