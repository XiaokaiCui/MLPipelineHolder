from __future__ import annotations

import ast
import builtins
import copy
import gc
import inspect
import os
import pickle
import platform
import re
import shutil
import sys
import warnings
from ctypes import CDLL
from contextlib import redirect_stdout
from dataclasses import asdict, fields, is_dataclass
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

from .artifact_store import ArtifactStore
from .exceptions import ExecutionError, PersistenceError, RegistrationError, ResolutionError
from .function_registry import (
    _values_equal,
    callable_signature,
    default_map,
    resolve_callable,
)
from .gate_block import GateBlock
from .logger import PipelineLogger
from .models import (
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

if TYPE_CHECKING:
    from .execution_block import ExecutionBlock


class _MissingMainClassPlaceholder:
    pass


_RESERVED_BUILTIN_NAMES = {
    name
    for name in dir(builtins)
    if not name.startswith("_")
}


_SAVE_WARNING_PATTERNS = (
    r"Saved pipelines preserve callable references",
    r".*was saved without a linked model artifact",
    r".*could not be serialized directly; saving a reference placeholder instead",
    r".*is not importable; saving a reference placeholder instead",
)

_IMMUTABLE_TYPES = (int, float, complex, bool, str, bytes, type(None))
_ACTIVE_PACKAGE_ROOT = __name__.rsplit(".", maxsplit=1)[0]
_PERSISTED_PACKAGE_ROOTS = ("mlpipelineholder", "src.mlpipelineholder")

_CLEANUP_MODES = ("none", "confirm", "auto")
_SAVED_GENERATION_RE = re.compile(
    r"^.+__.+__.+__[0-9a-f]{32}\.(?:json|npy|pkl|pt|feather|parquet|bin)$"
)
_MANAGED_CHILD_DIR_NAMES = ("artifacts", "children", "metadata", "history_logs")
_MANAGED_CHILD_FILE_NAMES = ("pipeline_state.pkl", "config.pkl", "pipeline_meta.pkl")


class _MissingClassUnpickler(pickle.Unpickler):
    def __init__(self, file_obj: BytesIO) -> None:
        super().__init__(file_obj)
        self._placeholder_cache: dict[tuple[str, str], type[Any]] = {}

    def find_class(self, module: str, name: str) -> Any:
        resolved_module = module
        for persisted_root in _PERSISTED_PACKAGE_ROOTS:
            if module == persisted_root or module.startswith(f"{persisted_root}."):
                resolved_module = f"{_ACTIVE_PACKAGE_ROOT}{module[len(persisted_root):]}"
                break
        try:
            return super().find_class(resolved_module, name)
        except (AttributeError, ImportError, ModuleNotFoundError):
            if module != "__main__":
                raise
            cache_key = (module, name)
            placeholder = self._placeholder_cache.get(cache_key)
            if placeholder is not None:
                return placeholder
            placeholder = type(name, (_MissingMainClassPlaceholder,), {})
            placeholder.__module__ = module
            self._placeholder_cache[cache_key] = placeholder
            return placeholder


class _GateStatusCache:
    """Memo of per-level gate evaluations for one placeholder-recovery pass.

    Each gate-owning pipeline retains a snapshot of every value its gate can
    read (incoming parent outputs, config fields, and manual values), so a
    cached answer is reused only while those inputs remain deeply equal. The
    cache is short-lived: it is created at the entry point of one recovery
    pass and discarded when the pass completes.
    """

    __slots__ = ("_levels",)

    def __init__(self) -> None:
        self._levels: dict[
            int,
            tuple[tuple[tuple[str, Any], ...], tuple[str, str | None]],
        ] = {}


class PipelineHandler:
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
    ) -> None:
        self.registration_name = registration_name
        self.config = {} if configuration is None else configuration
        self.execution_priority = execution_priority
        self.parent_pipeline: PipelineHandler | None = None
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
            self._is_atom: bool = False

            self.manual_values: dict[str, Any] = {}
            self.para_value_dict: dict[str, Any] = {}
            self.artifact_registry: dict[str, ArtifactRecord] = {}
            self.producer_outputs: dict[str, dict[str, Any]] = {}
            self.run_history: list[RunRecord] = []
            self.artifact_store = ArtifactStore(self.project_root)
        except Exception:
            if generated_temp_root and self.project_root.exists():
                shutil.rmtree(self.project_root, ignore_errors=True)
            raise

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

    def add_block(
        self, registration_name: str, execution_priority: float, forced: bool = False
    ) -> ExecutionBlock | None:
        if self._is_atom:
            raise RegistrationError(
                f"Atom pipeline '{self.registration_name}' is immutable "
                "and cannot accept new blocks"
            )
        from .execution_block import ExecutionBlock

        block = ExecutionBlock(self, registration_name, execution_priority)
        conflicts = self._registration_conflicts(block, execution_priority)
        self._raise_on_priority_conflict_with_different_name(
            registration_name,
            execution_priority,
            conflicts,
        )
        existing_block = self.blocks_by_name.get(registration_name)
        if (
            forced
            and existing_block is not None
            and isinstance(existing_block, ExecutionBlock)
            and existing_block.execution_priority == execution_priority
            and len(existing_block.functions) == 1
            and isinstance(existing_block.functions[0], ExpressionRegistration)
        ):
            return existing_block
        if conflicts and not forced:
            self.logger.warning(
                f"Skipped block registration '{registration_name}' at priority {execution_priority}: already exists"
            )
            return None
        if conflicts and forced:
            self._replace_conflicting_nodes(conflicts)
        try:
            self._register_node(block)
        except RegistrationError as exc:
            self.logger.warning(
                f"Skipped block registration '{registration_name}' at priority {execution_priority}: {exc}"
            )
            return None
        return block

    def _add_block_strict(self, registration_name: str, execution_priority: float):
        if self._is_atom:
            raise RegistrationError(
                f"Atom pipeline '{self.registration_name}' is immutable "
                "and cannot accept new blocks"
            )
        from .execution_block import ExecutionBlock

        block = ExecutionBlock(self, registration_name, execution_priority)
        self._register_node(block)
        return block

    def add_child_pipeline(
        self,
        child_pipeline: "PipelineHandler",
        execution_priority: float,
        registration_name: str | None = None,
        forced: bool = False,
    ) -> Any:
        if self._is_atom:
            raise RegistrationError(
                f"Atom pipeline '{self.registration_name}' is immutable "
                "and cannot accept child pipelines"
            )
        if child_pipeline is self:
            raise RegistrationError("A pipeline cannot register itself as a child pipeline")
        was_root = child_pipeline.parent_pipeline is None
        if registration_name is not None:
            child_pipeline.registration_name = registration_name
        conflicts = self._registration_conflicts(child_pipeline, execution_priority)
        self._raise_on_priority_conflict_with_different_name(
            child_pipeline.registration_name,
            execution_priority,
            conflicts,
        )
        if conflicts and not forced:
            self.logger.warning(
                f"Skipped child pipeline registration '{child_pipeline.registration_name}' at priority {execution_priority}: already exists"
            )
            return None
        if conflicts and forced:
            self._replace_conflicting_nodes(conflicts)
        if (
            forced
            and child_pipeline.parent_pipeline is not None
            and child_pipeline.parent_pipeline is not self
        ):
            child_pipeline.parent_pipeline._remove_registered_node(child_pipeline)
            child_pipeline.parent_pipeline = None
        self._validate_node_registration(child_pipeline, execution_priority)
        self._validate_related_pipeline_names(child_pipeline)
        self._validate_output_names_against_config(sorted(child_pipeline.list_declared_outputs()))
        self._validate_strict_attach(child_pipeline, execution_priority)
        child_pipeline._attach_to_parent(self, execution_priority)
        if was_root and child_pipeline._invalidation_forbidden:
            top = self._root_pipeline()
            changed = not top._invalidation_forbidden
            top._invalidation_forbidden = True
            top._sync_invalidation_flag()
            if changed:
                top.logger.warning(
                    "Attaching a former root pipeline with object invalidation "
                    "forbidden transferred FORBIDDEN state to the whole pipeline tree: "
                    "forced re-registrations and structural changes will still erase "
                    "each changed node's own outputs but will not invalidate other "
                    "upstream or downstream outputs, so stale or inconsistent values "
                    "may survive silently; call allow_invalidate_objects() to restore "
                    "normal cascade invalidation"
                )
        else:
            child_pipeline._invalidation_forbidden = self._invalidation_forbidden
            child_pipeline._sync_invalidation_flag()
        self._register_node(child_pipeline)
        if child_pipeline.para_value_dict:
            self.producer_outputs[child_pipeline.registration_name] = (
                child_pipeline._locally_produced_outputs()
            )
            self._rebuild_visible_state(self._incoming_parent_outputs())
        return child_pipeline

    def create_atom_child_pipeline(
        self,
        child_name: str,
        execution_priority: float,
        target_function: Any,
        gate_config: str | None = None,
        expected_value: Any = True,
        default_config_value: Any = True,
        output_variable_names: str | list[str] | tuple[str, ...] | None = None,
        save_to_disk_lst: list[str] | tuple[str, ...] | set[str] | None = None,
        param_mapping_dct: dict[str, str | None] | None = None,
        kwargs_dct: dict[str, str] | None = None,
        args_lst: tuple[str, ...] | list[str] | None = None,
        child_configuration: Any | None = None,
        allow_existing_root: bool = True,
        forced: bool = True,
        block_priority: float = 10.0,
        *,
        param_mapping: dict[str, str | None] | None = None,
    ) -> None:
        if self._is_atom:
            raise RegistrationError(
                f"Atom pipeline '{self.registration_name}' is immutable "
                "and cannot accept child pipelines"
            )
        output_names = (
            []
            if output_variable_names is None
            else [output_variable_names]
            if isinstance(output_variable_names, str)
            else list(output_variable_names)
        )
        save_names = set(save_to_disk_lst or [])
        if not save_names.issubset(set(output_names)):
            raise RegistrationError(
                "save_to_disk_lst must be a subset of output_variable_names in create_atom_child_pipeline"
            )
        if param_mapping is not None and param_mapping_dct is not None:
            raise RegistrationError(
                "Use either param_mapping or param_mapping_dct in create_atom_child_pipeline, not both"
            )
        effective_param_mapping = (
            param_mapping if param_mapping is not None else param_mapping_dct
        )
        child_root = self.project_root / "children" / child_name
        temp_pipeline = PipelineHandler(
            registration_name=child_name,
            configuration=self.get_full_config() if child_configuration is None else child_configuration,
            local_folder_path=child_root,
            execution_priority=execution_priority,
            forced=forced,
            strict_mode=self._root_pipeline().strict_mode,
            _allow_existing_root=allow_existing_root,
        )
        temp_pipeline.logger = self.logger
        temp_pipeline.parent_pipeline = self
        if gate_config is not None:
            gate_visible = (
                set(temp_pipeline.get_full_config())
                | set(temp_pipeline._incoming_parent_outputs())
                | set(temp_pipeline.manual_values)
                | set(temp_pipeline._ancestor_manual_values())
                | set(self._declared_output_names_before_priority(execution_priority))
            )
            if gate_config not in gate_visible:
                if temp_pipeline.strict_mode and not temp_pipeline._suppress_strict_validation:
                    raise RegistrationError(
                        f"Gate config '{gate_config}' is not found in config, visible output values, or visible manual values"
                    )
                if not temp_pipeline._suppress_strict_validation:
                    temp_pipeline.logger.warning(
                        f"Gate config '{gate_config}' is not found in config, visible output values, or visible manual values; auto-creating config field with default value"
                    )
                temp_pipeline.set_config({gate_config: default_config_value})
            temp_pipeline.set_gate_block(
                gate_config,
                expected_value=expected_value,
                forced=forced,
            )
        temp_block = temp_pipeline.add_block(f"{child_name}_block", block_priority, forced=forced)
        if temp_block is None:
            return None
        if args_lst is not None:
            temp_block.register_args(
                "default_args",
                args_lst,
                forced=forced,
            )
        if kwargs_dct is not None:
            temp_block.register_kwargs(
                "default_kwargs",
                kwargs_dct,
                forced=forced,
            )
        temp_block.register_function(
            target_function,
            output_variable_names=output_variable_names,
            save_to_disk=save_to_disk_lst,
            var_pos_name="default_args" if args_lst is not None else None,
            var_kw_name="default_kwargs" if kwargs_dct is not None else None,
            param_mapping=effective_param_mapping,
            forced=forced,
        )
        child_priority = temp_pipeline.execution_priority
        if child_priority is None:
            raise RegistrationError(f"Child pipeline '{child_name}' has no priority")
        existing = self.nodes_by_name.get(child_name)
        if (
            forced
            and isinstance(existing, PipelineHandler)
            and self._atom_matches(existing, temp_pipeline)
        ):
            return None
        if forced and existing is not None:
            self._validate_atom_replacement(
                temp_pipeline,
                existing,
                child_priority,
            )
            old_outputs = list(
                self.producer_outputs.get(existing.registration_name, {}).keys()
            )
            new_outputs = sorted(temp_pipeline.list_declared_outputs())
            old_priority = existing.execution_priority
            self._remove_registered_node(existing)
            if isinstance(existing, PipelineHandler):
                existing._invalidate_all_outputs()
            self._erase_overridden_node_outputs(
                existing.registration_name,
                old_priority,
                child_priority,
                old_outputs,
                new_outputs,
            )
            attached = self.add_child_pipeline(
                temp_pipeline,
                execution_priority=child_priority,
                forced=False,
            )
        else:
            attached = self.add_child_pipeline(
                temp_pipeline,
                execution_priority=child_priority,
                forced=forced,
            )
        if attached is not None:
            temp_pipeline._is_atom = True

    def _validate_atom_replacement(
        self,
        candidate: "PipelineHandler",
        existing: Any,
        execution_priority: float,
    ) -> None:
        nodes = self.nodes
        nodes_by_name = self.nodes_by_name
        blocks = self.blocks
        blocks_by_name = self.blocks_by_name
        self.nodes = [node for node in self.nodes if node is not existing]
        self.nodes_by_name = {
            name: node
            for name, node in self.nodes_by_name.items()
            if node is not existing
        }
        self.blocks = [block for block in self.blocks if block is not existing]
        self.blocks_by_name = {
            name: block
            for name, block in self.blocks_by_name.items()
            if block is not existing
        }
        try:
            self._validate_node_registration(candidate, execution_priority)
            self._validate_related_pipeline_names(candidate)
            self._validate_output_names_against_config(
                sorted(candidate.list_declared_outputs())
            )
            self._validate_strict_attach(candidate, execution_priority)
        finally:
            self.nodes = nodes
            self.nodes_by_name = nodes_by_name
            self.blocks = blocks
            self.blocks_by_name = blocks_by_name

    def _atom_matches(
        self,
        old: "PipelineHandler",
        new: "PipelineHandler",
    ) -> bool:
        """Whether a re-created atom pipeline is structurally identical to the old one.

        Compares registration identity, gate, inner blocks (functions, args and
        kwargs helpers) and the effective values of the config fields the atom
        actually consumes. Unused config fields and their counts are ignored, so
        parent-config drift never blocks a no-op.
        """
        if not old._is_atom:
            return False
        if old.registration_name != new.registration_name:
            return False
        if old.execution_priority != new.execution_priority:
            return False
        if not self._atom_gates_equal(old, new):
            return False
        if not self._atom_blocks_equal(old, new):
            return False
        if not self._atom_used_config_fields_equal(old, new):
            return False
        return True

    @staticmethod
    def _atom_gates_equal(
        old: "PipelineHandler",
        new: "PipelineHandler",
    ) -> bool:
        old_gate = old.gate_block
        new_gate = new.gate_block
        if old_gate is None or new_gate is None:
            return old_gate is None and new_gate is None
        if old_gate.config_field_name != new_gate.config_field_name:
            return False
        return _values_equal(
            old_gate.expected_value,
            new_gate.expected_value,
        )

    def _atom_blocks_equal(
        self,
        old: "PipelineHandler",
        new: "PipelineHandler",
    ) -> bool:
        if any(
            isinstance(node, PipelineHandler)
            for pipeline in (old, new)
            for node in pipeline._sorted_nodes()
        ):
            return False
        old_blocks = {
            node.registration_name: node
            for node in old._sorted_nodes()
            if not isinstance(node, PipelineHandler)
        }
        new_blocks = {
            node.registration_name: node
            for node in new._sorted_nodes()
            if not isinstance(node, PipelineHandler)
        }
        if set(old_blocks) != set(new_blocks):
            return False
        return all(
            self._atom_block_equal(old_blocks[name], new_blocks[name])
            for name in old_blocks
        )

    def _atom_block_equal(self, old_block: Any, new_block: Any) -> bool:
        if old_block.execution_priority != new_block.execution_priority:
            return False
        if len(old_block.functions) != len(new_block.functions):
            return False
        if not all(
            self._atom_registration_equal(old_reg, new_reg)
            for old_reg, new_reg in zip(old_block.functions, new_block.functions)
        ):
            return False
        if set(old_block.registered_args) != set(new_block.registered_args):
            return False
        if not all(
            old_block.registered_args[name].ordered_items
            == new_block.registered_args[name].ordered_items
            for name in old_block.registered_args
        ):
            return False
        if set(old_block.registered_kwargs) != set(new_block.registered_kwargs):
            return False
        return all(
            old_block.registered_kwargs[name].mapping_dct
            == new_block.registered_kwargs[name].mapping_dct
            for name in old_block.registered_kwargs
        )

    @staticmethod
    def _atom_registration_equal(old_reg: Any, new_reg: Any) -> bool:
        from .models import ExpressionRegistration
        from .function_registry import callable_identity_matches

        if isinstance(old_reg, ExpressionRegistration) != isinstance(
            new_reg, ExpressionRegistration
        ):
            return False
        if isinstance(old_reg, ExpressionRegistration):
            return (
                old_reg.code == new_reg.code
                and old_reg.output_names == new_reg.output_names
                and old_reg.save_to_disk == new_reg.save_to_disk
                and old_reg.warn_on_input_mutation == new_reg.warn_on_input_mutation
            )
        if old_reg.function_name != new_reg.function_name:
            return False
        if not callable_identity_matches(
            old_reg.import_path,
            old_reg.callable_obj,
            new_reg.import_path,
            new_reg.callable_obj,
        ):
            return False
        return (
            old_reg.output_names == new_reg.output_names
            and old_reg.save_to_disk == new_reg.save_to_disk
            and old_reg.param_mapping == new_reg.param_mapping
            and old_reg.var_pos_name == new_reg.var_pos_name
            and old_reg.var_kw_name == new_reg.var_kw_name
        )

    def _atom_used_config_fields_equal(
        self,
        old: "PipelineHandler",
        new: "PipelineHandler",
    ) -> bool:
        from .models import ExpressionRegistration

        used_names: set[str] = set()
        for pipeline in (old, new):
            if (
                pipeline.gate_block is not None
                and pipeline.gate_block.config_field_name is not None
            ):
                used_names.add(pipeline.gate_block.config_field_name)
            for node in pipeline._sorted_nodes():
                if isinstance(node, PipelineHandler):
                    continue
                for registration in node.functions:
                    if isinstance(registration, ExpressionRegistration):
                        continue
                    helper_names = {
                        registration.var_pos_name,
                        registration.var_kw_name,
                    }
                    used_names.update(
                        name
                        for name in registration.input_names
                        if name != "logger" and name not in helper_names
                    )
                    for mapped_name in registration.param_mapping.values():
                        if mapped_name is not None:
                            used_names.add(mapped_name)
                for args_registration in node.registered_args.values():
                    used_names.update(args_registration.ordered_items)
                for kwargs_registration in node.registered_kwargs.values():
                    used_names.update(kwargs_registration.mapping_dct.values())
        old_config = old.get_full_config()
        new_config = new.get_full_config()
        missing = object()
        for name in used_names:
            old_value = old_config.get(name, missing)
            new_value = new_config.get(name, missing)
            if old_value is missing or new_value is missing:
                if old_value is not new_value:
                    return False
                continue
            if not _values_equal(old_value, new_value):
                return False
        return True

    def _erase_node_outputs(self, node_name: str) -> None:
        removed = self.producer_outputs.pop(node_name, {})
        self._rebuild_visible_state(self._incoming_parent_outputs())
        self._delete_artifacts_from_outputs(removed)

    def _erase_overridden_node_outputs(
        self,
        node_name: str,
        old_priority: float | None,
        new_priority: float | None,
        old_output_names: list[str],
        new_output_names: list[str] | None = None,
    ) -> None:
        """Unified erasure for a forced override of an expression, function or atom.

        Always erases the overridden node's own produced outputs. Unless cascade
        invalidation is forbidden, it then erases from the earliest downstream
        block that consumes any affected output name: the old outputs (whose
        values change or disappear) and the new outputs (which may collide with
        downstream inputs). Downstream blocks consuming none of the affected
        names are left untouched. Old outputs are walked from the old priority
        and new outputs from the new priority, so a priority change still catches
        consumers between the two positions.
        """
        self._erase_node_outputs(node_name)
        if self.parent_pipeline is not None:
            self._resync_mirror_to_parent()
        if self._invalidation_forbidden:
            return
        users: list[tuple[PipelineHandler, Any]] = []
        affected_names: set[str] = set()
        for output_name in dict.fromkeys(old_output_names):
            name_users = self._downstream_input_users(old_priority, output_name)
            if name_users:
                users.extend(name_users)
                affected_names.add(output_name)
        for output_name in dict.fromkeys(new_output_names or []):
            name_users = self._downstream_input_users(new_priority, output_name)
            if name_users:
                users.extend(name_users)
                affected_names.add(output_name)
        if not users:
            return
        labels = sorted(
            {
                f"'{pipeline.full_path()}.{node.registration_name}'"
                for pipeline, node in users
            }
        )
        self.logger.warning(
            f"Output(s) '{', '.join(sorted(affected_names))}' of block '{node_name}' "
            f"are used as inputs by downstream block(s) {', '.join(labels)}; "
            "invalidating those blocks and everything downstream of them"
        )
        by_pipeline: dict[int, tuple[PipelineHandler, float]] = {}
        for owning_pipeline, node in users:
            if node.execution_priority is None:
                continue
            current = by_pipeline.get(id(owning_pipeline))
            if current is None or node.execution_priority < current[1]:
                by_pipeline[id(owning_pipeline)] = (
                    owning_pipeline,
                    node.execution_priority,
                )
        for owning_pipeline, priority in by_pipeline.values():
            owning_pipeline._invalidate_with_ancestor_consumers(priority)
        resynced: set[int] = set()
        if self.parent_pipeline is not None:
            self._resync_mirror_to_parent()
            resynced.add(id(self))
        for owning_pipeline, _ in by_pipeline.values():
            if id(owning_pipeline) in resynced:
                continue
            if owning_pipeline.parent_pipeline is not None:
                owning_pipeline._resync_mirror_to_parent()
                resynced.add(id(owning_pipeline))

    def _downstream_input_users(
        self,
        block_priority: float | None,
        input_name: str,
    ) -> list[tuple["PipelineHandler", Any]]:
        """Blocks anywhere downstream that consume ``input_name`` as an input.

        A consumer is only impacted when the expression is its effective source:
        the first downstream node that also produces ``input_name`` shields every
        consumer after it (later producers win in the visibility model), so the
        walk stops there. Covers blocks after the expression in its own pipeline,
        blocks in descendant pipelines, and blocks in ancestor pipelines after
        the child node on the path.
        """
        users: list[tuple[PipelineHandler, Any]] = []
        self._walk_input_users_stopping_at_producer(
            self,
            block_priority,
            input_name,
            users,
        )
        if self._has_shielding_producer(block_priority, input_name):
            return users
        current: PipelineHandler | None = self
        while current is not None and current.parent_pipeline is not None:
            parent = current.parent_pipeline
            child_node = next(
                (node for node in parent._sorted_nodes() if node is current),
                None,
            )
            if child_node is None or child_node.execution_priority is None:
                break
            self._walk_input_users_stopping_at_producer(
                parent,
                child_node.execution_priority,
                input_name,
                users,
            )
            if parent._has_shielding_producer(
                child_node.execution_priority,
                input_name,
            ):
                break
            current = parent
        return users

    def _invalidate_with_ancestor_consumers(self, priority: float) -> None:
        """Brutally invalidate from a consumer through every ancestor tail.

        The target pipeline loses the consumer and every later node. Each
        ancestor keeps the child node containing that consumer, but loses every
        node registered after that child, preserving one positional cutoff
        across nested pipeline boundaries.
        """
        self._invalidate_from_priority(priority)
        current = self
        while current.parent_pipeline is not None:
            parent = current.parent_pipeline
            current._resync_mirror_to_parent()
            if current.execution_priority is None:
                return
            parent._invalidate_from_priority(
                current.execution_priority,
                include_target=False,
            )
            current = parent

    def _has_shielding_producer(
        self,
        block_priority: float | None,
        input_name: str,
    ) -> bool:
        """Whether a later node keeps ``input_name``'s effective value stable.

        A same-name producer after the overridden node wins in the visibility
        model, so removing or changing the node's output does not alter this
        pipeline's effective value — ancestor consumers of this pipeline's
        mirror are therefore not impacted. A producing node that itself consumes
        the name does not shield (its stored output was computed from the old
        value and will change).
        """
        for node in self._sorted_nodes():
            if (
                block_priority is None
                or node.execution_priority is None
                or node.execution_priority <= block_priority
            ):
                continue
            produced_outputs = self.producer_outputs.get(node.registration_name, {})
            if input_name in produced_outputs:
                if isinstance(node, PipelineHandler):
                    return True
                return input_name not in self._block_consumed_input_names(node)
        return False

    def _walk_input_users_stopping_at_producer(
        self,
        pipeline: "PipelineHandler",
        start_priority: float | None,
        input_name: str,
        users: list[tuple["PipelineHandler", Any]],
    ) -> None:
        """Walk one pipeline's nodes in priority order, stopping at the first producer.

        Nodes at or below ``start_priority`` are skipped (the expression itself and
        everything before it). The walk stops at the first node that produces
        ``input_name`` — that node's output overrides the expression for every
        later consumer, so later consumers are not impacted. A producing node that
        also consumes the name is still flagged (its stored output was computed
        from the old value). Nested pipeline nodes are recursed into: their blocks
        see the expression's output as upstream, so they can contain impacted
        consumers before their own first producer.
        """
        for node in pipeline._sorted_nodes():
            if (
                start_priority is not None
                and (
                    node.execution_priority is None
                    or node.execution_priority <= start_priority
                )
            ):
                continue
            produced_outputs = pipeline.producer_outputs.get(
                node.registration_name,
                {},
            )
            if input_name in produced_outputs:
                if (
                    not isinstance(node, PipelineHandler)
                    and input_name in pipeline._block_consumed_input_names(node)
                ):
                    users.append((pipeline, node))
                if isinstance(node, PipelineHandler):
                    self._walk_input_users_stopping_at_producer(
                        node,
                        None,
                        input_name,
                        users,
                    )
                break
            if isinstance(node, PipelineHandler):
                self._walk_input_users_stopping_at_producer(
                    node,
                    None,
                    input_name,
                    users,
                )
                continue
            if input_name in pipeline._block_consumed_input_names(node):
                users.append((pipeline, node))

    def _block_consumed_input_names(self, block: Any) -> set[str]:
        names = set(self._required_input_names(block))
        for args_registration in block.registered_args.values():
            names.update(args_registration.ordered_items)
        for kwargs_registration in block.registered_kwargs.values():
            names.update(kwargs_registration.mapping_dct.values())
        return names

    def _resync_mirror_to_parent(self) -> None:
        current: PipelineHandler | None = self
        while current is not None and current.parent_pipeline is not None:
            parent = current.parent_pipeline
            current_outputs = current._locally_produced_outputs()
            if current_outputs:
                parent.producer_outputs[current.registration_name] = current_outputs
            else:
                parent.producer_outputs.pop(current.registration_name, None)
            parent._rebuild_visible_state(parent._incoming_parent_outputs())
            current = parent

    def _locally_produced_outputs(self) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        for node in self._sorted_nodes():
            outputs.update(self.producer_outputs.get(node.registration_name, {}))
        return outputs

    def add_gate_block(
        self, function_or_path: Any, expected_value: Any = True, forced: bool = False
    ) -> Any:
        if self._is_atom:
            raise RegistrationError(
                f"Atom pipeline '{self.registration_name}' is immutable "
                "and cannot change its gate"
            )
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

    def set_config(self, overrides: dict[str, Any]) -> None:
        declared_outputs = self.list_declared_outputs()
        manual_names = set(self.manual_values) | set(self._ancestor_manual_values())
        for field_name, value in overrides.items():
            self._validate_builtin_name_conflict(field_name, owner_label="configuration")
            self._validate_config_value_picklable(field_name, value)
            if field_name in declared_outputs:
                self.logger.warning(
                    f"Skipped config update for '{field_name}' because it conflicts with a declared output"
                )
                continue
            if field_name in manual_names:
                self.logger.warning(
                    f"Skipped config update for '{field_name}' because it conflicts with a manual value"
                )
                continue
            self._set_config_value(field_name, value)

    def update_config(self, overrides: dict[str, Any]) -> None:
        config_names = self.get_full_config()
        declared_outputs = self.list_declared_outputs()
        manual_names = set(self.manual_values) | set(self._ancestor_manual_values())
        for field_name, value in overrides.items():
            self._validate_builtin_name_conflict(field_name, owner_label="configuration")
            if field_name not in config_names:
                raise ResolutionError(f"Unknown config field: {field_name}")
            self._validate_config_value_picklable(field_name, value)
            if field_name in declared_outputs:
                self.logger.warning(
                    f"Skipped config update for '{field_name}' because it conflicts with a declared output"
                )
                continue
            if field_name in manual_names:
                self.logger.warning(
                    f"Skipped config update for '{field_name}' because it conflicts with a manual value"
                )
                continue
            self._set_config_value(field_name, value)

    def get_value(self, variable_name: str) -> Any:
        if variable_name in self._tree_constant_names():
            raise ResolutionError(
                f"Cannot get value '{variable_name}': name is a pipeline constant; use get_constant_value instead"
            )
        if variable_name in self.para_value_dict:
            value = self.para_value_dict[variable_name]
        else:
            upstream_outputs = self._incoming_parent_outputs()
            if variable_name in upstream_outputs:
                value = upstream_outputs[variable_name]
            else:
                value = self._descendant_visible_value(variable_name)
                if value is None:
                    raise ResolutionError(f"Unknown pipeline value: {variable_name}")
        return self._materialize_stored_value(
            value,
            f"Cannot get value '{variable_name}': it was saved as a placeholder "
            f"({value.reason}) and cannot be restored; recreate or reset the value"
            if isinstance(value, (RuntimeValueReference, DataclassValueReference))
            else "",
        )

    def get_node_output(self, node_name: str, output_name: str) -> Any:
        """Return one materialized output produced by an immediate block or atom."""
        node = self.nodes_by_name.get(node_name)
        if node is None:
            raise ResolutionError(
                f"Unknown immediate child node '{node_name}' in pipeline '{self.registration_name}'"
            )
        if isinstance(node, PipelineHandler):
            if not node._is_atom:
                raise ResolutionError(
                    f"Node '{node_name}' is not a block or atom pipeline"
                )
            values_by_priority: dict[float, Any] = {}
            for internal_node in node._sorted_nodes():
                outputs = node.producer_outputs.get(
                    internal_node.registration_name,
                    {},
                )
                if output_name not in outputs:
                    continue
                priority = internal_node.execution_priority
                if priority is None:
                    raise ResolutionError(
                        f"Node '{internal_node.registration_name}' has no execution priority"
                    )
                value = outputs[output_name]
                values_by_priority[priority] = node._materialize_stored_value(
                    value,
                    f"Cannot get node output '{node_name}.{output_name}': it was saved "
                    f"as a placeholder ({value.reason}) and cannot be restored"
                    if isinstance(value, (RuntimeValueReference, DataclassValueReference))
                    else "",
                )
            if not values_by_priority:
                raise ResolutionError(
                    f"Node '{node_name}' has no produced output named '{output_name}'"
                )
            if len(values_by_priority) == 1:
                return next(iter(values_by_priority.values()))
            return values_by_priority
        outputs = self.producer_outputs.get(node_name, {})
        if output_name not in outputs:
            raise ResolutionError(
                f"Node '{node_name}' has no produced output named '{output_name}'"
            )
        value = outputs[output_name]
        return self._materialize_stored_value(
            value,
            f"Cannot get node output '{node_name}.{output_name}': it was saved as a "
            f"placeholder ({value.reason}) and cannot be restored"
            if isinstance(value, (RuntimeValueReference, DataclassValueReference))
            else "",
        )

    def _materialize_stored_value(
        self,
        value: Any,
        placeholder_error: str,
    ) -> Any:
        if isinstance(value, TorchStateArtifactRecord):
            return value
        if isinstance(value, CallableValueReference):
            return self._restore_callable_value(value)
        if isinstance(value, ArtifactRecord):
            return self.artifact_store.load(value)
        if isinstance(value, (RuntimeValueReference, DataclassValueReference)):
            raise ResolutionError(placeholder_error)
        return value

    def get_constant_value(self, variable_name: str) -> Any:
        visible = dict(self._incoming_parent_manual_values())
        visible.update(self.manual_values)
        if variable_name not in visible:
            raise ResolutionError(f"Unknown pipeline constant: {variable_name}")
        value = visible[variable_name]
        return self._materialize_stored_value(
            value,
            f"Cannot get constant '{variable_name}': it was saved as a placeholder "
            f"({value.reason}) and cannot be restored; reset it with set_constant_value"
            if isinstance(value, (RuntimeValueReference, DataclassValueReference))
            else "",
        )

    @staticmethod
    def _is_mutable_value(value: Any) -> bool:
        if isinstance(value, _IMMUTABLE_TYPES):
            return False
        if isinstance(value, (tuple, frozenset)):
            return any(
                PipelineHandler._is_mutable_value(item) for item in value
            )
        return True

    @staticmethod
    def _copy_value(value: Any) -> Any:
        try:
            import pandas as pd  # type: ignore

            if isinstance(value, (pd.DataFrame, pd.Series)):
                return value.copy(deep=True)
        except Exception:
            pass
        try:
            import numpy as np  # type: ignore

            if isinstance(value, np.ndarray):
                return value.copy()
        except Exception:
            pass
        try:
            import dask.dataframe as dd  # type: ignore

            if isinstance(value, dd.DataFrame):
                return value.copy()
        except Exception:
            pass
        try:
            import torch  # type: ignore

            if isinstance(value, torch.Tensor):
                return value.detach().clone()
        except Exception:
            pass
        return copy.deepcopy(value)

    def _snapshot_value(self, variable_name: str, value: Any, *, verbose: bool) -> Any:
        """Return a deep copy of a mutable value so later in-place mutation
        outside the pipeline cannot affect the stored value. Values that
        cannot be deep-copied are stored by reference with a warning;
        metadata records and callables always pass through unchanged.
        """
        if isinstance(
            value,
            (
                ArtifactRecord,
                TorchStateArtifactRecord,
                CallableValueReference,
                RuntimeValueReference,
                RuntimeCallableReference,
            ),
        ) or callable(value):
            return value
        if not self._is_mutable_value(value):
            return value
        try:
            snapshot = self._copy_value(value)
        except Exception:
            self.logger.warning(
                f"Value '{variable_name}' is not deep-copyable; stored by reference"
            )
            return value
        if verbose:
            self.logger.info(f"Value '{variable_name}' deeply copied")
        return snapshot

    def _save_value_to_disk(
        self,
        variable_name: str,
        value: Any,
        *,
        function_name: str,
        verbose: bool,
    ) -> ArtifactRecord:
        try:
            record = self.artifact_store.save(
                variable_name=variable_name,
                value=value,
                block_name=self.registration_name,
                function_name=function_name,
                run_id=uuid4().hex,
                torch_load_weights_only=self.torch_load_weights_only,
            )
        except Exception as exc:
            raise PersistenceError(
                f"Failed to save value '{variable_name}' to disk: {type(exc).__name__}: {exc}"
            ) from exc
        if verbose:
            self.logger.info(
                f"Value '{variable_name}' saved to disk; protected from in-place changes"
            )
        return record

    def set_constant_value(
        self,
        variable_name: str,
        value: Any,
        *,
        copy: bool = True,
        verbose: bool = False,
        to_disk: bool = False,
    ) -> None:
        self._validate_builtin_name_conflict(variable_name, owner_label="pipeline constant")
        if variable_name in self._visible_config_names():
            raise RegistrationError(
                f"Constant name '{variable_name}' conflicts with a visible configuration field"
            )
        if variable_name in self._tree_declared_output_names():
            raise RegistrationError(
                f"Constant name '{variable_name}' conflicts with a declared output name in the pipeline tree"
            )
        if variable_name in self._tree_produced_value_names():
            raise RegistrationError(
                f"Constant name '{variable_name}' conflicts with an existing produced value name in the pipeline tree"
            )
        previous_value = self.manual_values.get(variable_name)
        if isinstance(previous_value, ArtifactRecord):
            self.artifact_store.delete(previous_value)
        if isinstance(
            value,
            (ArtifactRecord, TorchStateArtifactRecord),
        ):
            pass
        elif to_disk:
            value = self._save_value_to_disk(
                variable_name,
                value,
                function_name="set_constant_value",
                verbose=verbose,
            )
        elif copy:
            value = self._snapshot_value(variable_name, value, verbose=verbose)
        self.manual_values[variable_name] = value
        self.para_value_dict[variable_name] = value
        if isinstance(value, ArtifactRecord):
            self.artifact_registry[variable_name] = value
        else:
            self.artifact_registry.pop(variable_name, None)
        if self.parent_pipeline is not None:
            self._sync_attached_outputs_to_parent()

    def update_value(
        self,
        variable_name: str,
        value: Any,
        *,
        copy: bool = True,
        verbose: bool = False,
    ) -> None:
        self._validate_builtin_name_conflict(variable_name, owner_label="pipeline value")
        if variable_name in self._tree_constant_names():
            raise ResolutionError(
                f"Cannot update value '{variable_name}': name is a pipeline constant; use set_constant_value instead"
            )
        if variable_name not in self.para_value_dict:
            raise ResolutionError(f"Unknown pipeline value: {variable_name}")

        previous_value = self.para_value_dict.get(variable_name)
        if isinstance(previous_value, ArtifactRecord):
            self.artifact_store.delete(previous_value)

        if isinstance(
            value,
            (ArtifactRecord, TorchStateArtifactRecord),
        ):
            pass
        elif isinstance(previous_value, ArtifactRecord):
            value = self._save_value_to_disk(
                variable_name,
                value,
                function_name="update_value",
                verbose=verbose,
            )
        elif copy:
            value = self._snapshot_value(variable_name, value, verbose=verbose)

        self.para_value_dict[variable_name] = value
        if isinstance(value, ArtifactRecord):
            self.artifact_registry[variable_name] = value
        else:
            self.artifact_registry.pop(variable_name, None)

        for node in reversed(self._sorted_nodes()):
            outputs = self.producer_outputs.get(node.registration_name)
            if outputs is None or variable_name not in outputs:
                continue
            outputs[variable_name] = value
            break
        self._sync_value_update_to_parent(variable_name, value)

    def set_value(
        self,
        variable_name: str,
        value: Any,
        *,
        copy: bool = True,
        verbose: bool = False,
    ) -> None:
        self._validate_builtin_name_conflict(variable_name, owner_label="pipeline value")
        if variable_name in self._tree_constant_names():
            raise ResolutionError(
                f"Cannot set value '{variable_name}': name is a pipeline constant; use set_constant_value instead"
            )
        if variable_name in self.para_value_dict:
            self.update_value(variable_name, value, copy=copy, verbose=verbose)
            return
        if variable_name in self._incoming_parent_outputs():
            self._nearest_upstream_produced_owner(variable_name).update_value(
                variable_name, value, copy=copy, verbose=verbose
            )
            return
        owner = self._descendant_produced_owner(variable_name)
        if owner is not None:
            owner.update_value(variable_name, value, copy=copy, verbose=verbose)
            return
        if variable_name in self._tree_declared_output_names():
            self._inject_produced_value(variable_name, value, copy=copy, verbose=verbose)
            return
        raise ResolutionError(f"Unknown pipeline value: {variable_name}")

    def _nearest_upstream_produced_owner(self, variable_name: str) -> "PipelineHandler":
        current = self.parent_pipeline
        while current is not None:
            winning_child: PipelineHandler | None = None
            for node in current._sorted_nodes():
                if node.execution_priority >= self.execution_priority:
                    break
                if (
                    isinstance(node, PipelineHandler)
                    and variable_name in node.para_value_dict
                    and variable_name not in node.manual_values
                ):
                    winning_child = node
            if winning_child is not None:
                deeper = winning_child._descendant_produced_owner(variable_name)
                return deeper if deeper is not None else winning_child
            if (
                variable_name in current.para_value_dict
                and variable_name not in current.manual_values
            ):
                deeper = current._descendant_produced_owner(variable_name)
                return deeper if deeper is not None else current
            current = current.parent_pipeline
        raise ResolutionError(f"Unknown produced value owner for: {variable_name}")

    def _descendant_produced_owner(self, variable_name: str) -> "PipelineHandler | None":
        for node in reversed(self._sorted_nodes()):
            if not isinstance(node, PipelineHandler):
                continue
            if variable_name in node.para_value_dict and variable_name not in node.manual_values:
                deeper = node._descendant_produced_owner(variable_name)
                return deeper if deeper is not None else node
        return None

    def _find_declaring_node(self, variable_name: str) -> Any | None:
        for node in self._sorted_nodes():
            if variable_name in self._node_declared_outputs(node):
                return node
        return None

    def _find_declaring_pipeline(self, variable_name: str) -> "PipelineHandler | None":
        if self._find_declaring_node(variable_name) is not None:
            return self
        current = self.parent_pipeline
        while current is not None:
            if current._find_declaring_node(variable_name) is not None:
                return current
            current = current.parent_pipeline
        return None

    def _inject_produced_value(
        self,
        variable_name: str,
        value: Any,
        *,
        copy: bool = True,
        verbose: bool = False,
    ) -> None:
        pipeline = self._find_declaring_pipeline(variable_name)
        if pipeline is None:
            raise ResolutionError(f"Unknown pipeline value: {variable_name}")
        if pipeline is not self:
            pipeline._inject_produced_value(
                variable_name, value, copy=copy, verbose=verbose
            )
            return
        node = self._find_declaring_node(variable_name)
        if node is None:
            raise ResolutionError(f"Unknown pipeline value: {variable_name}")
        if isinstance(node, PipelineHandler):
            node._inject_produced_value(
                variable_name, value, copy=copy, verbose=verbose
            )
            return
        if isinstance(
            value,
            (ArtifactRecord, TorchStateArtifactRecord),
        ):
            pass
        elif variable_name in node.functions_output_disk_names():
            value = self._save_value_to_disk(
                variable_name,
                value,
                function_name="set_value",
                verbose=verbose,
            )
        elif copy:
            value = self._snapshot_value(variable_name, value, verbose=verbose)
        outputs = self.producer_outputs.setdefault(node.registration_name, {})
        outputs[variable_name] = value
        self.para_value_dict[variable_name] = value
        if isinstance(value, ArtifactRecord):
            self.artifact_registry[variable_name] = value
        else:
            self.artifact_registry.pop(variable_name, None)
        if self.parent_pipeline is not None:
            self._sync_attached_outputs_to_parent()

    def _inject_recovered_value(self, variable_name: str, value: Any) -> None:
        pipeline = self._find_declaring_pipeline(variable_name)
        if pipeline is None:
            raise ResolutionError(f"Unknown pipeline value: {variable_name}")
        if pipeline is not self:
            pipeline._inject_recovered_value(variable_name, value)
            return
        node = self._find_declaring_node(variable_name)
        if node is None:
            raise ResolutionError(f"Unknown pipeline value: {variable_name}")
        if isinstance(node, PipelineHandler):
            node._inject_recovered_value(variable_name, value)
            return
        self.producer_outputs.setdefault(node.registration_name, {})[
            variable_name
        ] = value
        self._refresh_visible_value(variable_name)
        self._sync_value_to_ancestors_without_invalidation(variable_name)

    def _refresh_visible_value(self, variable_name: str) -> None:
        found = False
        value: Any = None
        upstream_outputs = self._incoming_parent_outputs()
        if variable_name in upstream_outputs:
            found = True
            value = upstream_outputs[variable_name]
        for node in self._sorted_nodes():
            produced_outputs = self.producer_outputs.get(node.registration_name, {})
            if variable_name in produced_outputs:
                found = True
                value = produced_outputs[variable_name]
        if variable_name in self.manual_values:
            found = True
            value = self.manual_values[variable_name]
        if found and (
            variable_name in self.list_declared_outputs()
            or variable_name in self.manual_values
        ):
            self.para_value_dict[variable_name] = value
            if isinstance(value, ArtifactRecord):
                self.artifact_registry[variable_name] = value
            else:
                self.artifact_registry.pop(variable_name, None)
            return
        self.para_value_dict.pop(variable_name, None)
        self.artifact_registry.pop(variable_name, None)

    def _sync_value_to_ancestors_without_invalidation(
        self,
        variable_name: str,
    ) -> None:
        current = self
        while current.parent_pipeline is not None:
            parent = current.parent_pipeline
            if (
                variable_name in current.para_value_dict
                and variable_name not in current.manual_values
            ):
                parent.producer_outputs.setdefault(current.registration_name, {})[
                    variable_name
                ] = current.para_value_dict[variable_name]
            else:
                child_outputs = parent.producer_outputs.get(current.registration_name)
                if child_outputs is not None:
                    child_outputs.pop(variable_name, None)
            parent._refresh_visible_value(variable_name)
            current = parent

    def _sync_value_update_to_parent(self, variable_name: str, value: Any) -> None:
        current = self
        while current.parent_pipeline is not None:
            parent = current.parent_pipeline
            parent_outputs = parent.producer_outputs.get(current.registration_name)
            if parent_outputs is None or variable_name not in parent_outputs:
                return
            cached_output = parent_outputs[variable_name]
            parent_outputs[variable_name] = value
            if (
                variable_name in parent.para_value_dict
                and variable_name not in parent.manual_values
                and parent.para_value_dict[variable_name] is cached_output
            ):
                parent.para_value_dict[variable_name] = value
                if isinstance(value, ArtifactRecord):
                    parent.artifact_registry[variable_name] = value
                else:
                    parent.artifact_registry.pop(variable_name, None)
            current = parent

    def recover_variable_from_backup(
        self,
        name: str,
        *,
        pipeline_name: str | None = None,
    ) -> None:
        from .backup_recovery_service import recover_variable_from_backup

        recover_variable_from_backup(self, name, pipeline_name=pipeline_name)

    def get_full_config(self) -> dict[str, Any]:
        return dict(self._ancestor_config_values(), **self.config_as_dict())

    def get_config_value(self, field_name: str) -> Any:
        config = self.get_full_config()
        if field_name not in config:
            raise ResolutionError(f"Unknown config field: {field_name}")
        value = config[field_name]
        if isinstance(value, (RuntimeValueReference, DataclassValueReference)):
            raise ResolutionError(
                f"Cannot get config field '{field_name}': it was saved as a placeholder "
                f"({value.reason}) and cannot be restored"
            )
        return value

    def recover_config_from_backup(self, name: str) -> None:
        from .backup_recovery_service import recover_config_from_backup

        recover_config_from_backup(self, name)

    def get_block(self, block_name: str) -> Any:
        node = self.nodes_by_name.get(block_name)
        if node is None:
            raise RegistrationError(f"Block not registered: {block_name}")
        if isinstance(node, PipelineHandler):
            raise RegistrationError(f"Registered node '{block_name}' is a child pipeline, not a block")
        return node

    def get_child_pipeline(self, pipeline_name: str) -> "PipelineHandler":
        node = self.nodes_by_name.get(pipeline_name)
        if node is None:
            raise RegistrationError(f"Child pipeline not registered: {pipeline_name}")
        if not isinstance(node, PipelineHandler):
            raise RegistrationError(f"Registered node '{pipeline_name}' is a block, not a child pipeline")
        return node

    def list_child_pipeline_names(self) -> list[str]:
        return [
            node.registration_name
            for node in self._sorted_nodes()
            if isinstance(node, PipelineHandler)
        ]

    def reset_gate_block(self) -> None:
        if self._is_atom:
            raise RegistrationError(
                f"Atom pipeline '{self.registration_name}' is immutable "
                "and cannot change its gate"
            )
        if self.gate_block is None:
            return
        self.gate_block = None
        if not self._invalidation_forbidden:
            self._invalidate_all_outputs()

    def get_result_history(self) -> list[str]:
        # Attached child pipelines intentionally keep reading historical RESULT lines from
        # their own pre-attachment log path, while new runtime logging flows through the
        # parent logger. This preserves old child history but means nested logging is not
        # fully unified after attachment.
        if self.parent_pipeline is not None and self.historical_result_log_path is not None:
            if self._attached_result_history_override is not None:
                return list(self._attached_result_history_override)
            return self._read_result_history_from_file(self.historical_result_log_path)
        return self.logger.get_result_history()

    def print_result_history(self) -> None:
        for entry in self.get_result_history():
            print(self._color(entry, "green"))

    def clear_result_history(self) -> None:
        if self.parent_pipeline is not None and self.historical_result_log_path is not None:
            self._attached_result_history_override = []
            return
        self.logger.clear_result_history()

    def set_print_capture_mode(self, mode: str) -> None:
        if mode not in {"tee", "logger_only", "off"}:
            raise RegistrationError("print capture mode must be one of: tee, logger_only, off")
        self.print_capture_mode = mode

    def set_log_level(self, level: str) -> None:
        try:
            self.logger.set_level(level)
        except ValueError as exc:
            raise RegistrationError(str(exc)) from exc

    def set_torch_load_weights_only(self, enabled: bool) -> None:
        """Set whether torch artifacts saved by this pipeline load with weights_only=True."""
        self.torch_load_weights_only = bool(enabled)

    def set_strict_mode(self, enabled: bool) -> None:
        """Enable or disable strict-mode registration validation for this pipeline and all attached descendants."""
        for pipeline in self._iter_attached_pipelines():
            pipeline.strict_mode = bool(enabled)

    def _sync_invalidation_flag(self) -> None:
        """Copy this pipeline's invalidation flag to its whole attached subtree."""
        flag = self._invalidation_forbidden
        for pipeline in self._iter_attached_pipelines():
            pipeline._invalidation_forbidden = flag

    def forbid_invalidate_objects(self) -> None:
        """Suppress cascade invalidation after changes across the pipeline tree.

        Forced re-registrations anywhere from the root down to every descendant
        still erase the changed node's own outputs, but stop invalidating other
        upstream or downstream objects. Those retained values may be stale or
        inconsistent until this mode is lifted. The state is not persisted, only
        the root pipeline may toggle it, and every pipeline added later inherits
        the current state automatically; call `allow_invalidate_objects()` to
        restore normal behaviour.
        """
        if self.parent_pipeline is not None:
            raise RegistrationError(
                "forbid_invalidate_objects() must be called on the root pipeline"
            )
        if self._invalidation_forbidden:
            return
        self._invalidation_forbidden = True
        self._sync_invalidation_flag()
        self.logger.warning(
            "Object invalidation is now FORBIDDEN on the whole pipeline tree: forced "
            "re-registrations and structural changes will still erase each changed "
            "node's own outputs but will not invalidate other upstream or downstream "
            "outputs, so stale or inconsistent values may survive silently; call "
            "allow_invalidate_objects() to restore normal cascade invalidation"
        )

    def allow_invalidate_objects(self) -> None:
        """Restore normal erasure behaviour across the whole pipeline tree.

        Forced re-registrations and structural changes will invalidate affected
        upstream or downstream outputs again, which may discard previously
        computed results and require re-running blocks. Only the root pipeline
        may toggle this state; every pipeline already in the tree syncs
        immediately and later additions inherit the current state. Call
        `forbid_invalidate_objects()` to suppress cascade invalidation again.
        """
        if self.parent_pipeline is not None:
            raise RegistrationError(
                "allow_invalidate_objects() must be called on the root pipeline"
            )
        if not self._invalidation_forbidden:
            return
        self._invalidation_forbidden = False
        self._sync_invalidation_flag()
        self.logger.warning(
            "Object invalidation is now ALLOWED on the whole pipeline tree: forced "
            "re-registrations and structural changes will invalidate affected upstream "
            "or downstream outputs again and may discard previously computed results; "
            "call forbid_invalidate_objects() to suppress cascade invalidation again"
        )

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

    def get_priority_group(self, integer_priority: int) -> tuple[list[str], str | None]:
        group_nodes = [
            node for node in self._sorted_nodes() if self._priority_group(node.execution_priority) == integer_priority
        ]
        names = [node.registration_name for node in group_nodes]
        executable = self._select_executable_node_in_group(group_nodes)
        return names, None if executable is None else executable.registration_name

    def get_output_conflicts(self) -> dict[str, dict[str, list[str] | str]]:
        conflicts: dict[str, dict[str, list[str] | str]] = {}
        seen: dict[str, str] = {}
        for node in self._sorted_nodes():
            producer_name = self.qualified_node_name(node.registration_name)
            for output_name in sorted(self._node_declared_outputs(node)):
                if output_name not in seen:
                    seen[output_name] = producer_name
                    continue
                conflict = conflicts.setdefault(
                    output_name,
                    {"created_by": seen[output_name], "overridden_by": []},
                )
                overridden_by = conflict["overridden_by"]
                if isinstance(overridden_by, list):
                    overridden_by.append(producer_name)
        return conflicts

    def describe_output_conflicts(self) -> str:
        lines = [f"Output conflicts in {self.registration_name}:"]
        conflicts = self.get_output_conflicts()
        if not conflicts:
            lines.append("- none")
        for output_name, data in sorted(conflicts.items()):
            lines.append(f"- {output_name}")
            lines.append(f"  first created by: {data['created_by']}")
            lines.append("  overridden by:")
            for producer in data["overridden_by"]:
                lines.append(f"    - {producer}")
        for node in self._sorted_nodes():
            if isinstance(node, PipelineHandler):
                lines.append(node.describe_output_conflicts())
        return "\n".join(lines)

    def describe_pipeline(self) -> str:
        return "\n".join(self._describe_lines())

    def remove_block(self, block_name: str) -> None:
        if self._is_atom:
            raise RegistrationError(
                f"Atom pipeline '{self.registration_name}' is immutable "
                "and cannot remove blocks"
            )
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

    def run_all(self, overrides: dict[str, Any] | None = None) -> RunRecord:
        if self._gate_skip_without_cleanup(
            "run_all",
            overrides,
            self._incoming_parent_outputs(),
            self._ancestor_config_values(),
        ):
            return self._build_skipped_run_record("run_all")
        (
            self._invalidate_from_priority(self._sorted_nodes()[0].execution_priority)
            if self.nodes
            else None
        )
        return self._execute_nodes(
            self._sorted_nodes(),
            mode="run_all",
            overrides=overrides,
            upstream_outputs=self._incoming_parent_outputs(),
            parent_config=self._ancestor_config_values(),
        )[0]

    def run_until(self, *path_parts: str, overrides: dict[str, Any] | None = None) -> RunRecord:
        if len(path_parts) > 1:
            return self._run_nested_until_path(path_parts, overrides=overrides)
        pipeline, node = self._resolve_target_path(path_parts)
        if self._gate_skip_without_cleanup(
            f"run_until:{node.registration_name}",
            overrides,
            self._incoming_parent_outputs(),
            self._ancestor_config_values(),
        ):
            return self._build_skipped_run_record(f"run_until:{node.registration_name}")
        (
            self._invalidate_from_priority(self._sorted_nodes()[0].execution_priority)
            if self.nodes
            else None
        )
        selected = [
            candidate
            for candidate in self._sorted_nodes()
            if candidate.execution_priority <= node.execution_priority
        ]
        return self._execute_nodes(
            selected,
            mode=f"run_until:{node.registration_name}",
            overrides=overrides,
            upstream_outputs=self._incoming_parent_outputs(),
            parent_config=self._ancestor_config_values(),
        )[0]

    def run_from(self, *path_parts: str, overrides: dict[str, Any] | None = None) -> RunRecord:
        if len(path_parts) > 1:
            return self._run_nested_from_path(path_parts, overrides=overrides)
        pipeline, node = self._resolve_target_path(path_parts)
        if self._gate_skip_without_cleanup(
            f"run_from:{node.registration_name}",
            overrides,
            self._visible_outputs_before_priority(node.execution_priority),
            self._ancestor_config_values(),
        ):
            return self._build_skipped_run_record(f"run_from:{node.registration_name}")
        snapshot = self._snapshot_runtime_state()
        previous_outputs = snapshot[0].get(node.registration_name, {})
        previous_outputs = self._materialize_previous_node_inputs(
            node,
            previous_outputs,
            overrides,
        )
        self._invalidate_from_priority(node.execution_priority)
        return self._execute_nodes(
            [
                candidate
                for candidate in self._sorted_nodes()
                if candidate.execution_priority >= node.execution_priority
            ],
            mode=f"run_from:{node.registration_name}",
            overrides=overrides,
            upstream_outputs=self._visible_outputs_before_priority(node.execution_priority),
            parent_config=self._ancestor_config_values(),
            previous_node_outputs={node.registration_name: previous_outputs},
        )[0]

    def run_block(self, *path_parts: str, overrides: dict[str, Any] | None = None) -> RunRecord:
        if len(path_parts) > 1:
            return self._run_nested_block_path(path_parts, overrides=overrides)
        pipeline, node = self._resolve_target_path(path_parts)
        if self._gate_skip_without_cleanup(
            f"run_block:{node.registration_name}",
            overrides,
            self._visible_outputs_before_priority(node.execution_priority),
            self._ancestor_config_values(),
        ):
            return self._build_skipped_run_record(f"run_block:{node.registration_name}")
        snapshot = self._snapshot_runtime_state()
        previous_outputs = snapshot[0].get(node.registration_name, {})
        previous_outputs = self._materialize_previous_node_inputs(
            node,
            previous_outputs,
            overrides,
        )
        self._invalidate_from_priority(node.execution_priority)
        return self._execute_nodes(
            [node],
            mode=f"run_block:{node.registration_name}",
            overrides=overrides,
            upstream_outputs=self._visible_outputs_before_priority(node.execution_priority),
            parent_config=self._ancestor_config_values(),
            previous_node_outputs={node.registration_name: previous_outputs},
        )[0]

    def save_pipeline(
        self,
        path: str | Path | None = None,
        save_log_to_file: str | Path | None = None,
        *,
        verbose: bool = False,
        cleanup: str = "auto",
    ) -> Path:
        if cleanup not in _CLEANUP_MODES:
            raise ValueError(
                f"Invalid cleanup mode {cleanup!r}: expected 'none', 'confirm', or 'auto'"
            )
        with warnings.catch_warnings():
            if not verbose:
                for pattern in _SAVE_WARNING_PATTERNS:
                    warnings.filterwarnings("ignore", message=pattern)
            warnings.warn(
                "Saved pipelines preserve callable references, not historical function behavior; later source changes may affect reloaded pipelines.",
                stacklevel=2,
            )
            return self._save_pipeline_impl(path, save_log_to_file, cleanup)

    def _save_pipeline_impl(
        self,
        path: str | Path | None,
        save_log_to_file: str | Path | None,
        cleanup_mode: str,
    ) -> Path:
        target = self.project_root if path is None else Path(path)
        if self._temporary_root_handle is not None and self._normalized_path(target) != self._normalized_path(self.project_root):
            self._relocate_project_root(target)
            target = self.project_root
        if self._normalized_path(target) != self._normalized_path(self.project_root):
            self._materialize_project_tree_for_save(target)
        else:
            target.mkdir(parents=True, exist_ok=True)
        try:
            payload = self._serialize_payload_for_save(target)
        except RegistrationError as exc:
            raise PersistenceError(str(exc)) from exc
        self._atomic_pickle_dump(payload, target / "pipeline_state.pkl")
        self._atomic_pickle_dump(self._serialize_config_for_save(self.config), target / "config.pkl")
        self._write_pipeline_metadata(target)
        self.logger.info(f"Pipeline has been saved to project root: {target}")
        if self._normalized_path(target) == self._normalized_path(self.project_root):
            try:
                self._cleanup_obsolete_saved_objects(payload, target, cleanup_mode)
            except OSError as exc:
                self.logger.warning(
                    f"Skipped cleanup because saved paths could not be inspected: "
                    f"{type(exc).__name__}: {exc}"
                )
        if save_log_to_file is not None:
            self.logger.flush()
            log_target = Path(save_log_to_file)
            log_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.logger.log_file_path, log_target)
        refresh_backup = self._should_refresh_backup_on_save(target)
        if refresh_backup:
            backup_root = self.pipeline_backup_root
            if backup_root is not None:
                self._refresh_backup_copy(target, backup_root)
                self.logger.info(
                    f"Pipeline has been saved to project backup path: {backup_root}"
                )
        self._archive_current_log_to_history()
        if refresh_backup:
            backup_root = self.pipeline_backup_root
            if backup_root is not None:
                self._sync_history_logs_to_backup(backup_root)
        return target

    def _cleanup_obsolete_saved_objects(
        self,
        payload: dict[str, Any],
        target_root: Path,
        cleanup_mode: str,
    ) -> None:
        """Delete obsolete saved artifacts and orphaned child pipelines.

        The committed payload is the only liveness authority: every artifact
        path it references is kept, and every other framework-shaped
        generation entry under ``artifacts/`` plus every child pipeline
        directory absent from the serialized topology (with a fully
        framework-managed subtree) is deleted. ``none`` keeps everything,
        ``confirm`` asks first, and ``auto`` deletes without prompting.
        """
        if cleanup_mode == "none":
            return
        live_paths = PipelineHandler._referenced_payload_artifact_paths(payload)
        obsolete_children: list[Path] = []
        PipelineHandler._collect_obsolete_child_dirs(
            payload,
            target_root,
            live_paths,
            obsolete_children,
        )
        obsolete: list[Path] = []
        PipelineHandler._collect_obsolete_artifact_entries(
            payload,
            target_root,
            live_paths,
            obsolete,
        )
        obsolete.extend(obsolete_children)
        if not obsolete:
            return
        if cleanup_mode == "confirm":
            try:
                answer = input(
                    f"Delete {len(obsolete)} obsolete saved pipeline object(s)? [y/N]: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer not in ("y", "yes"):
                self.logger.info(
                    f"Skipped cleanup: {len(obsolete)} obsolete saved object(s) retained"
                )
                return
        deleted = 0
        for path in sorted(obsolete, key=lambda item: len(item.parts), reverse=True):
            try:
                if PipelineHandler._delete_saved_path(path, target_root):
                    deleted += 1
            except OSError as exc:
                self.logger.warning(
                    f"Could not delete obsolete saved path '{path}': "
                    f"{type(exc).__name__}: {exc}"
                )
        if deleted:
            self.logger.info(f"Deleted {deleted} obsolete saved pipeline object(s)")

    @staticmethod
    def _referenced_payload_artifact_paths(payload: Any) -> set[Path]:
        """Return resolved absolute paths of every artifact the payload references."""
        paths: set[Path] = set()
        seen: set[int] = set()

        def walk(value: Any) -> None:
            if isinstance(value, (ArtifactRecord, TorchStateArtifactRecord)):
                paths.add(Path(value.file_path).resolve())
                return
            if isinstance(value, dict):
                if id(value) in seen:
                    return
                seen.add(id(value))
                for key, item in value.items():
                    walk(key)
                    walk(item)
            elif isinstance(value, (list, tuple, set, frozenset)):
                if id(value) in seen:
                    return
                seen.add(id(value))
                for item in value:
                    walk(item)

        walk(payload)
        return paths

    @staticmethod
    def _collect_obsolete_artifact_entries(
        payload: dict[str, Any],
        target_root: Path,
        live_paths: set[Path],
        obsolete: list[Path],
    ) -> None:
        """Append unreferenced framework artifact generations to ``obsolete``."""
        artifacts_dir = target_root / "artifacts"
        if not artifacts_dir.is_symlink() and artifacts_dir.is_dir():
            for block_dir in artifacts_dir.iterdir():
                if block_dir.is_symlink() or not block_dir.is_dir():
                    continue
                for entry in block_dir.iterdir():
                    if entry.is_symlink():
                        continue
                    if _SAVED_GENERATION_RE.fullmatch(entry.name) is None:
                        continue
                    if entry.resolve() in live_paths:
                        continue
                    obsolete.append(entry)
        nodes = payload.get("nodes")
        if not isinstance(nodes, list):
            return
        for node in nodes:
            if not isinstance(node, dict) or node.get("kind") != "pipeline":
                continue
            name = node.get("registration_name")
            child_payload = node.get("payload")
            if not isinstance(name, str) or not isinstance(child_payload, dict):
                continue
            PipelineHandler._collect_obsolete_artifact_entries(
                child_payload,
                target_root / "children" / name,
                live_paths,
                obsolete,
            )

    @staticmethod
    def _collect_obsolete_child_dirs(
        payload: dict[str, Any],
        pipeline_root: Path,
        live_paths: set[Path],
        obsolete_children: list[Path],
    ) -> None:
        """Classify orphaned child pipeline dirs as obsolete or protected."""
        nodes = payload.get("nodes")
        if not isinstance(nodes, list):
            return
        active_names = {
            node["registration_name"]
            for node in nodes
            if isinstance(node, dict)
            and node.get("kind") == "pipeline"
            and isinstance(node.get("registration_name"), str)
        }
        children_dir = pipeline_root / "children"
        if children_dir.is_dir():
            for child in children_dir.iterdir():
                if child.is_symlink() or not child.is_dir():
                    continue
                if child.name in active_names:
                    continue
                child_root = child.resolve()
                if any(
                    live_path == child_root or live_path.is_relative_to(child_root)
                    for live_path in live_paths
                ):
                    continue
                if PipelineHandler._is_fully_managed_child_tree(child):
                    obsolete_children.append(child)
        for node in nodes:
            if not isinstance(node, dict) or node.get("kind") != "pipeline":
                continue
            name = node.get("registration_name")
            child_payload = node.get("payload")
            if not isinstance(name, str) or not isinstance(child_payload, dict):
                continue
            PipelineHandler._collect_obsolete_child_dirs(
                child_payload,
                pipeline_root / "children" / name,
                live_paths,
                obsolete_children,
            )

    @staticmethod
    def _is_fully_managed_child_tree(root: Path) -> bool:
        """True when a child directory holds only framework-managed entries."""
        for path in root.rglob("*"):
            if path.is_symlink():
                return False
            parts = path.relative_to(root).parts
            if parts[0] in _MANAGED_CHILD_DIR_NAMES:
                continue
            if len(parts) == 1 and path.is_file() and parts[0] in _MANAGED_CHILD_FILE_NAMES:
                continue
            return False
        return True

    @staticmethod
    def _delete_saved_path(path: Path, target_root: Path) -> bool:
        """Safely delete one obsolete path inside the project root."""
        if path.is_symlink() or not path.exists():
            return False
        if not path.resolve().is_relative_to(target_root.resolve()):
            return False
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True

    def save_project(
        self,
        path: str | Path | None = None,
        save_log_to_file: str | Path | None = None,
        *,
        verbose: bool = False,
    ) -> Path:
        return self.save_pipeline(path, save_log_to_file=save_log_to_file, verbose=verbose)

    def _archive_current_log_to_history(self) -> None:
        """Copy the current pipeline.log into history_logs/ with a timestamped name."""
        self.logger.flush()
        history_root = self.project_root / "history_logs"
        history_root.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC)
        stamp = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.{now.microsecond // 1000:03d}"
        target = history_root / f"{stamp}.log"
        counter = 1
        while target.exists():
            target = history_root / f"{stamp}_{counter}.log"
            counter += 1
        shutil.copy2(self.logger.log_file_path, target)

    @classmethod
    def load_pipeline(
        cls,
        path: str | Path,
        *,
        forced_deleting: bool = False,
        verbose: bool = False,
        auto_resolve_placeholders: bool = True,
    ) -> "PipelineHandler":
        warnings.warn(
            "Loaded pipelines restore current callable references, not historical function snapshots; changed source code may alter behavior.",
            stacklevel=2,
        )
        source = Path(path)
        try:
            with (source / "pipeline_state.pkl").open("rb") as handle:
                state_bytes = handle.read()
            payload = cls._load_pickle_with_missing_class_fallback(state_bytes)
            cls._validate_loaded_payload_placeholders(payload)
            cls._validate_loaded_payload_structure(payload)
            target, restore_message = cls._restore_working_tree_if_needed(
                source,
                forced_deleting=forced_deleting,
            )
            pipeline = cls._from_payload(
                payload,
                target,
                verbose=verbose,
                auto_resolve_placeholders=auto_resolve_placeholders,
            )
        except PersistenceError:
            raise
        except Exception as exc:
            # A failed build leaves the saved pipeline_state.pkl in the working
            # tree untouched, so the load can be retried; only log and artifact
            # files written mid-build remain, and save-time cleanup removes them.
            raise PersistenceError(f"Failed to load pipeline project: {exc}") from exc
        if restore_message is not None:
            pipeline.logger.info(restore_message)
        pipeline.logger.info(f"Pipeline has been loaded from the project root: {target}")
        return pipeline

    @classmethod
    def load_project(
        cls,
        path: str | Path,
        *,
        forced_deleting: bool = False,
        verbose: bool = False,
        auto_resolve_placeholders: bool = True,
    ) -> "PipelineHandler":
        return cls.load_pipeline(
            path,
            forced_deleting=forced_deleting,
            verbose=verbose,
            auto_resolve_placeholders=auto_resolve_placeholders,
        )

    def _write_pipeline_metadata(self, target: Path) -> None:
        with (target / "pipeline_meta.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "pipeline_directory": str(self.project_root),
                    "pipeline_backup_directory": (
                        None
                        if self.pipeline_backup_root is None
                        else str(self.pipeline_backup_root)
                    ),
                },
                handle,
            )

    def _should_refresh_backup_on_save(self, target: Path) -> bool:
        backup_root = self.pipeline_backup_root
        if backup_root is None:
            return False
        return self._normalized_path(target) == self._normalized_path(self.project_root)

    def _refresh_backup_copy(self, source: Path, backup_root: Path) -> None:
        if self._normalized_path(source) == self._normalized_path(backup_root):
            return
        if backup_root.exists():
            if backup_root.is_dir():
                shutil.rmtree(backup_root)
            else:
                backup_root.unlink()
        shutil.copytree(source, backup_root)

    def _sync_history_logs_to_backup(self, backup_root: Path) -> None:
        """Copy the project history_logs folder into the backup after a save snapshot."""
        history_root = self.project_root / "history_logs"
        if not history_root.is_dir():
            return
        backup_history = backup_root / "history_logs"
        if backup_history.exists():
            if backup_history.is_dir():
                shutil.rmtree(backup_history)
            else:
                backup_history.unlink()
        shutil.copytree(history_root, backup_history)

    def _materialize_project_tree_for_save(self, target: Path) -> None:
        if self._paths_overlap(self.project_root, target):
            raise PersistenceError(
                f"Cannot save pipeline from '{self.project_root}' into overlapping directory '{target}'"
            )
        self.logger.flush()
        shutil.copytree(self.project_root, target, dirs_exist_ok=True)

    @classmethod
    def _load_pipeline_metadata(cls, path: Path) -> dict[str, Any] | None:
        metadata_path = path / "pipeline_meta.pkl"
        if not metadata_path.exists():
            return None
        with metadata_path.open("rb") as handle:
            return cls._load_pickle_with_missing_class_fallback(handle.read())

    @staticmethod
    def _load_pickle_with_missing_class_fallback(raw_bytes: bytes) -> Any:
        return _MissingClassUnpickler(BytesIO(raw_bytes)).load()

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

    def _effective_expression_runtime_owner(self) -> "PipelineHandler | None":
        current: PipelineHandler | None = self
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
        if cls._is_missing_main_placeholder(value):
            return True
        if isinstance(value, dict):
            return any(cls._contains_missing_main_placeholder(item) for item in value.values())
        if isinstance(value, (list, tuple, set)):
            return any(cls._contains_missing_main_placeholder(item) for item in value)
        return False

    @staticmethod
    def _is_missing_main_placeholder(value: Any) -> bool:
        return isinstance(value, _MissingMainClassPlaceholder) or (
            isinstance(value, type)
            and issubclass(value, _MissingMainClassPlaceholder)
        )

    @classmethod
    def _validate_loaded_payload_placeholders(cls, payload: dict[str, Any]) -> None:
        if cls._contains_missing_placeholder_outside_config(payload):
            raise PersistenceError(
                "Failed to load pipeline project because a missing __main__ class was found outside the saved pipeline config"
            )

    @classmethod
    def _validate_loaded_payload_structure(cls, payload: Any) -> None:
        """Reject obviously unbuildable payloads before the working tree is touched.

        Full validation happens while ``_from_payload`` rebuilds the tree; this
        pass only performs cheap, side-effect-free checks (payload shape, node
        kinds, callable references, expression syntax) so a corrupted save is
        caught before it could replace a working tree that then fails to load.
        """
        if not isinstance(payload, dict):
            raise PersistenceError("Saved pipeline payload is not a mapping")
        cls._require_loaded_payload_keys(
            payload,
            ("registration_name", "config", "nodes"),
            owner_label="pipeline",
        )
        if not isinstance(payload["nodes"], list):
            raise PersistenceError("Saved pipeline payload has a non-list 'nodes' entry")
        for mapping_key in (
            "manual_values",
            "producer_outputs",
            "para_value_dict",
            "artifact_registry",
        ):
            if mapping_key in payload and not isinstance(payload[mapping_key], dict):
                raise PersistenceError(
                    f"Saved pipeline payload has a non-mapping '{mapping_key}' entry"
                )
        for node_payload in payload["nodes"]:
            if not isinstance(node_payload, dict):
                raise PersistenceError("Saved pipeline payload has an invalid node entry")
            cls._require_loaded_payload_keys(
                node_payload,
                ("kind", "registration_name", "execution_priority"),
                owner_label="node",
            )
            kind = node_payload["kind"]
            if kind == "pipeline":
                cls._require_loaded_payload_keys(
                    node_payload,
                    ("payload",),
                    owner_label=f"pipeline node '{node_payload['registration_name']}'",
                )
                cls._validate_loaded_payload_structure(node_payload["payload"])
                continue
            if kind == "block":
                cls._require_loaded_payload_keys(
                    node_payload,
                    ("functions",),
                    owner_label=f"block '{node_payload['registration_name']}'",
                )
                if not isinstance(node_payload["functions"], list):
                    raise PersistenceError(
                        f"Saved block '{node_payload['registration_name']}' has a non-list 'functions' entry"
                    )
                for args_payload in node_payload.get("registered_args", []):
                    if not isinstance(args_payload, dict):
                        raise PersistenceError("Saved pipeline payload has an invalid args entry")
                    cls._require_loaded_payload_keys(
                        args_payload,
                        ("name", "ordered_items"),
                        owner_label=f"args registration in block '{node_payload['registration_name']}'",
                    )
                for kwargs_payload in node_payload.get("registered_kwargs", []):
                    if not isinstance(kwargs_payload, dict):
                        raise PersistenceError("Saved pipeline payload has an invalid kwargs entry")
                    cls._require_loaded_payload_keys(
                        kwargs_payload,
                        ("name", "mapping_dct"),
                        owner_label=f"kwargs registration in block '{node_payload['registration_name']}'",
                    )
                for function_payload in node_payload["functions"]:
                    cls._validate_function_payload_structure(
                        function_payload,
                        owner_label=f"block '{node_payload['registration_name']}'",
                    )
                continue
            raise PersistenceError(
                f"Saved pipeline payload has an unknown node kind: {kind!r}"
            )
        gate_payload = payload.get("gate")
        if gate_payload is None:
            return
        if not isinstance(gate_payload, dict):
            raise PersistenceError("Saved pipeline payload has an invalid gate entry")
        cls._require_loaded_payload_keys(
            gate_payload,
            ("kind",),
            owner_label="gate",
        )
        gate_kind = gate_payload["kind"]
        if gate_kind == "callable":
            cls._require_loaded_payload_keys(
                gate_payload,
                ("import_path",),
                owner_label="callable gate",
            )
            cls._validate_import_path_payload(gate_payload["import_path"])
        elif gate_kind == "config_field":
            cls._require_loaded_payload_keys(
                gate_payload,
                ("field_name",),
                owner_label="config-field gate",
            )
        else:
            raise PersistenceError(
                f"Saved pipeline payload has an unknown gate kind: {gate_kind!r}"
            )

    @staticmethod
    def _require_loaded_payload_keys(
        payload: dict[str, Any],
        required_keys: tuple[str, ...],
        *,
        owner_label: str,
    ) -> None:
        for required_key in required_keys:
            if required_key not in payload:
                raise PersistenceError(
                    f"Saved {owner_label} payload is missing required key '{required_key}'"
                )

    @classmethod
    def _validate_function_payload_structure(
        cls,
        function_payload: Any,
        *,
        owner_label: str,
    ) -> None:
        if not isinstance(function_payload, dict):
            raise PersistenceError("Saved pipeline payload has an invalid function entry")
        cls._require_loaded_payload_keys(
            function_payload,
            ("kind", "output_names", "save_to_disk"),
            owner_label=f"function in {owner_label}",
        )
        function_kind = function_payload["kind"]
        if function_kind == "expression":
            cls._require_loaded_payload_keys(
                function_payload,
                ("code",),
                owner_label=f"expression in {owner_label}",
            )
            code = function_payload["code"]
            if not isinstance(code, str) or not code.strip():
                raise PersistenceError("Saved pipeline payload has an empty expression")
            try:
                compile(code, "<pipeline_expression>", "exec")
            except SyntaxError as exc:
                raise PersistenceError(
                    f"Saved pipeline payload has invalid expression code: {exc}"
                ) from exc
            return
        if function_kind != "function":
            raise PersistenceError(
                f"Saved function in {owner_label} has an unknown kind: {function_kind!r}"
            )
        import_path = function_payload.get("import_path")
        partial_payload = function_payload.get("partial")
        runtime_reference = function_payload.get("runtime_callable_reference")
        if import_path is not None:
            cls._validate_import_path_payload(import_path)
        elif partial_payload is not None:
            if not isinstance(partial_payload, dict):
                raise PersistenceError(
                    "Saved pipeline payload has an invalid partial callable entry"
                )
            cls._restore_partial_callable(partial_payload, owner_label)
        elif runtime_reference is not None:
            if not isinstance(runtime_reference, RuntimeCallableReference):
                raise PersistenceError(
                    "Saved pipeline payload has an invalid runtime callable reference"
                )
            cls._restore_runtime_registered_callable(runtime_reference, owner_label)
        else:
            raise PersistenceError("Saved pipeline function has no callable reference")

    @classmethod
    def _validate_import_path_payload(cls, import_path: Any) -> None:
        if not isinstance(import_path, str):
            raise PersistenceError(
                f"Saved pipeline payload has an invalid callable import path: {import_path!r}"
            )
        try:
            resolve_callable(import_path)
        except Exception as exc:
            raise PersistenceError(
                f"Saved pipeline callable '{import_path}' could not be imported: {exc}"
            ) from exc

    @classmethod
    def _contains_missing_placeholder_outside_config(
        cls,
        value: Any,
        *,
        inside_config: bool = False,
    ) -> bool:
        if cls._is_missing_main_placeholder(value):
            return not inside_config
        if isinstance(value, dict):
            for key, item in value.items():
                if cls._contains_missing_placeholder_outside_config(
                    item,
                    inside_config=inside_config or key == "config",
                ):
                    return True
            return False
        if isinstance(value, (list, tuple, set)):
            return any(
                cls._contains_missing_placeholder_outside_config(
                    item,
                    inside_config=inside_config,
                )
                for item in value
            )
        return False

    @classmethod
    def _restore_working_tree_if_needed(
        cls,
        source_path: Path,
        *,
        forced_deleting: bool,
    ) -> tuple[Path, str | None]:
        metadata = cls._load_pipeline_metadata(source_path)
        if metadata is None:
            return source_path, None
        pipeline_directory = metadata.get("pipeline_directory")
        if pipeline_directory is None:
            return source_path, None
        work_root = Path(pipeline_directory)
        if cls._normalized_path(source_path) == cls._normalized_path(work_root):
            return source_path, None
        if cls._paths_overlap(source_path, work_root):
            raise PersistenceError(
                f"Cannot restore pipeline from '{source_path}' into overlapping working directory '{work_root}'"
            )
        cls._clear_path_with_optional_confirmation(
            work_root,
            source_path=source_path,
            forced_deleting=forced_deleting,
        )
        shutil.copytree(source_path, work_root)
        return (
            work_root,
            f"Pipeline project directory has been copied from backup path: {source_path} -> {work_root}",
        )

    def _relocate_project_root(self, new_root: Path) -> None:
        old_root = self.project_root
        normalized_old_root = self._normalized_path(old_root)
        normalized_new_root = self._normalized_path(new_root)
        if normalized_old_root == normalized_new_root:
            self._cleanup_temporary_root_handle()
            return
        self.logger.flush()
        self.logger.disable_file_logging()
        new_root.mkdir(parents=True, exist_ok=True)
        for entry in old_root.iterdir():
            destination = new_root / entry.name
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(entry), str(destination))
        self.project_root = new_root
        self.metadata_root = new_root / "metadata"
        self.metadata_root.mkdir(parents=True, exist_ok=True)
        self.logger.rebind_path(self.metadata_root / "pipeline.log")
        self.artifact_store = ArtifactStore(new_root)
        self._rewrite_artifact_paths(old_root, new_root)
        self._rewrite_run_history_paths(old_root, new_root)
        self._refresh_descendant_roots(old_root, new_root)
        self._cleanup_temporary_root_handle()

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
        parent: "PipelineHandler | None" = None,
        *,
        verbose: bool = False,
        auto_resolve_placeholders: bool = True,
    ) -> "PipelineHandler":
        reconstruction_warnings: list[str] = []
        config = cls._deserialize_saved_config(
            payload["config"],
            verbose=verbose,
            warn=reconstruction_warnings.append,
        )
        pipeline = cls(
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
        )
        for message in reconstruction_warnings:
            pipeline.logger.warning(message)
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
        pipeline.run_history = payload.get("run_history", [])
        saved_project_root = payload.get("saved_project_root")
        if saved_project_root is not None:
            pipeline._rewrite_artifact_paths(Path(saved_project_root), project_root)
            pipeline._rewrite_run_history_paths(Path(saved_project_root), project_root)
        pipeline.suppress_registration_advisories = False
        pipeline._suppress_strict_validation = False
        pipeline._is_atom = bool(payload.get("is_atom", False))
        if parent is not None:
            pipeline.parent_pipeline = parent
            pipeline.logger = parent.logger
        pending_dataclass_fallbacks: list[
            tuple[PipelineHandler, str, str | None, str, DataclassValueReference]
        ] = []
        if parent is None:
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
        return pipeline

    def _restore_dataclass_value_references(
        self,
        *,
        verbose: bool,
        _memo: dict[int, Any] | None = None,
        _pending: list[
            tuple[PipelineHandler, str, str | None, str, DataclassValueReference]
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
            if isinstance(node, PipelineHandler):
                node._restore_dataclass_value_references(
                    verbose=verbose,
                    _memo=memo,
                    _pending=pending,
                )

    def _reconnect_dataclass_fallbacks(
        self,
        pending: list[
            tuple[PipelineHandler, str, str | None, str, DataclassValueReference]
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
            mapping = PipelineHandler._dataclass_fallback_mapping(
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
                result = PipelineHandler._reconstruct_dataclass(
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
        owner: PipelineHandler,
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
        candidate = PipelineHandler._find_dataclass_class(reference.type_name)
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
            if isinstance(node, PipelineHandler)
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
            if isinstance(node, PipelineHandler):
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
            if isinstance(node, PipelineHandler):
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
    ) -> list[tuple["PipelineHandler", Any]]:
        """Return ``(owning pipeline, block)`` pairs for every block declaring the name."""
        declaring: list[tuple["PipelineHandler", Any]] = []
        for node in self._sorted_nodes():
            if isinstance(node, PipelineHandler):
                if variable_name in node.list_declared_outputs():
                    declaring.extend(node._tree_declaring_blocks(variable_name))
            elif variable_name in node.declared_outputs():
                declaring.append((self, node))
        return declaring

    def _tree_find_declaring_node(self, variable_name: str) -> Any | None:
        """Find the first block declaring the name anywhere in the subtree."""
        for node in self._sorted_nodes():
            if isinstance(node, PipelineHandler):
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
        current: PipelineHandler | None = self
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
        pipeline: "PipelineHandler",
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
        pipeline: "PipelineHandler",
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
            if isinstance(node, PipelineHandler):
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
            "config": self._serialize_config_for_save(self.config),
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
            "run_history": self.run_history,
        }

    def _serialize_node_for_save(
        self,
        node: Any,
        target_root: Path,
        cache: dict[int, Any],
    ) -> dict[str, Any]:
        if isinstance(node, PipelineHandler):
            return {
                "kind": "pipeline",
                "registration_name": node.registration_name,
                "execution_priority": node.execution_priority,
                "payload": node._serialize_payload_for_save(target_root, cache),
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

    @staticmethod
    def _callable_reference_round_trips(value: Any, import_path: str) -> bool:
        try:
            resolved_callable, _, _ = resolve_callable(import_path)
        except Exception:
            return False
        return resolved_callable is value

    @staticmethod
    def _restore_callable_value(reference: CallableValueReference) -> Any:
        callable_obj, _, _ = resolve_callable(reference.import_path)
        return callable_obj

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

    @staticmethod
    def _restore_runtime_callable(
        reference: RuntimeCallableReference,
        owner_label: str,
    ) -> Any:
        main_module = sys.modules.get("__main__")
        callable_obj = (
            None
            if main_module is None
            else getattr(main_module, reference.callable_name, None)
        )
        if not callable(callable_obj):
            raise PersistenceError(
                f"Required runtime callable '{reference.callable_name}' for {owner_label} "
                "is unavailable in __main__; import or define it before loading the pipeline"
            )
        return callable_obj

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

    @staticmethod
    def _serialize_config_for_save(config: Any) -> Any:
        return PipelineHandler._serialize_config_value(config)

    @staticmethod
    def _serialize_config_value(value: Any) -> Any:
        if callable(value):
            try:
                _, import_path, callable_name = resolve_callable(value)
            except RegistrationError:
                import_path = None
                callable_name = getattr(value, "__name__", type(value).__name__)
            if import_path is not None and import_path.startswith("__main__."):
                return RuntimeCallableReference(callable_name=callable_name)
            if (
                import_path is not None
                and PipelineHandler._callable_reference_round_trips(value, import_path)
            ):
                return CallableValueReference(
                    callable_name=callable_name,
                    import_path=import_path,
                )
            warnings.warn(
                f"Callable config value '{callable_name}' is not importable; saving a reference placeholder instead.",
                stacklevel=2,
            )
            return RuntimeValueReference(
                type_name=type(value).__name__,
                repr_text=repr(value),
                reason="callable config value is not importable during save_pipeline",
            )
        if isinstance(value, dict):
            return {
                "__pipeline_serialized_config__": True,
                "kind": "dict",
                "data": {
                    key: PipelineHandler._serialize_config_value(item)
                    for key, item in value.items()
                },
            }
        if is_dataclass(value) and not isinstance(value, type):
            return {
                "__pipeline_serialized_config__": True,
                "kind": "namespace",
                "class_name": type(value).__name__,
                "module": type(value).__module__,
                "data": {
                    key: PipelineHandler._serialize_config_value(item)
                    for key, item in PipelineHandler._config_object_as_dict(value).items()
                },
            }
        if hasattr(value, "__dict__"):
            return {
                "__pipeline_serialized_config__": True,
                "kind": "namespace",
                "class_name": type(value).__name__,
                "module": type(value).__module__,
                "data": {
                    key: PipelineHandler._serialize_config_value(item)
                    for key, item in vars(value).items()
                },
            }
        if isinstance(value, list):
            return [PipelineHandler._serialize_config_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(PipelineHandler._serialize_config_value(item) for item in value)
        return value

    @staticmethod
    def _deserialize_saved_config(
        saved_config: Any,
        *,
        verbose: bool = False,
        warn: Any | None = None,
    ) -> Any:
        return PipelineHandler._deserialize_config_value(
            saved_config,
            verbose=verbose,
            warn=warn,
        )

    @staticmethod
    def _deserialize_config_value(
        value: Any,
        *,
        verbose: bool = False,
        warn: Any | None = None,
    ) -> Any:
        if isinstance(value, CallableValueReference):
            return PipelineHandler._restore_callable_value(value)
        if isinstance(value, RuntimeCallableReference):
            return PipelineHandler._restore_runtime_callable(
                value,
                "configuration value",
            )
        if not (
            isinstance(value, dict)
            and value.get("__pipeline_serialized_config__") is True
        ):
            if isinstance(value, dict):
                return {
                    key: PipelineHandler._deserialize_config_value(
                        item,
                        verbose=verbose,
                        warn=warn,
                    )
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [
                    PipelineHandler._deserialize_config_value(
                        item,
                        verbose=verbose,
                        warn=warn,
                    )
                    for item in value
                ]
            if isinstance(value, tuple):
                return tuple(
                    PipelineHandler._deserialize_config_value(
                        item,
                        verbose=verbose,
                        warn=warn,
                    )
                    for item in value
                )
            return value
        kind = value.get("kind")
        if kind == "dict":
            return {
                key: PipelineHandler._deserialize_config_value(
                    item,
                    verbose=verbose,
                    warn=warn,
                )
                for key, item in dict(value.get("data", {})).items()
            }
        if kind == "namespace":
            data = {
                key: PipelineHandler._deserialize_config_value(
                    item,
                    verbose=verbose,
                    warn=warn,
                )
                for key, item in dict(value.get("data", {})).items()
            }
            return PipelineHandler._reconstruct_dataclass(
                value.get("class_name"),
                data,
                verbose=verbose,
                module_name=value.get("module"),
                warn=warn,
            )
        return value

    @classmethod
    def _serialize_dataclass_field_value(cls, value: Any) -> Any:
        """Return a picklable structured representation of one dataclass field value.

        Picklable values are kept as-is; unpicklable values are converted into
        reconstructable references (callables, nested dataclasses, dict-like
        objects) or a last-resort placeholder.
        """
        if callable(value):
            return cls._serialize_config_value(value)
        try:
            pickle.dumps(value)
            return value
        except Exception:
            pass
        if isinstance(value, dict):
            return {
                key: cls._serialize_dataclass_field_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._serialize_dataclass_field_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._serialize_dataclass_field_value(item) for item in value)
        serialized = cls._serialize_config_value(value)
        if isinstance(
            serialized,
            (CallableValueReference, RuntimeCallableReference, RuntimeValueReference),
        ):
            return serialized
        try:
            pickle.dumps(serialized)
            return serialized
        except Exception:
            return RuntimeValueReference(
                type_name=type(value).__name__,
                repr_text=repr(value),
                reason="dataclass field not directly serializable during save_pipeline",
            )

    @staticmethod
    def _find_dataclass_class(
        class_name: str,
        module_name: str | None = None,
    ) -> type | None:
        """Locate an importable pure dataclass by name.

        When the saved module is known, the class is looked up there first;
        otherwise every loaded module's attributes are scanned. ``__main__``
        definitions (notebook-local classes) are preferred over ambiguous
        same-name matches.
        """
        if module_name is not None:
            module = sys.modules.get(module_name)
            if module is not None:
                candidate = getattr(module, class_name, None)
                if (
                    isinstance(candidate, type)
                    and is_dataclass(candidate)
                    and candidate.__name__ == class_name
                ):
                    return candidate
        candidates: list[type] = []
        for module in sys.modules.values():
            candidate = getattr(module, class_name, None)
            if (
                isinstance(candidate, type)
                and is_dataclass(candidate)
                and candidate.__name__ == class_name
            ):
                candidates.append(candidate)
        main_candidates = [
            candidate
            for candidate in candidates
            if getattr(candidate, "__module__", None) == "__main__"
        ]
        if main_candidates:
            return main_candidates[0]
        if candidates:
            return candidates[0]
        return None

    def _find_pipeline_dataclass_class(
        self,
        class_name: str,
        module_name: str | None = None,
    ) -> type | None:
        """Locate a dataclass class among pipeline-visible runtime values.

        Placeholder recovery re-runs blocks, which can produce dynamically
        generated classes (for example a factory function returning a new
        dataclass) that exist only as pipeline values after load. Checks this
        pipeline's visible state and producer outputs, walking up to ancestor
        pipelines, for a dataclass class matching ``class_name`` (and
        ``module_name`` when known).
        """
        current: PipelineHandler | None = self
        while current is not None:
            for mapping in (current.para_value_dict, current.manual_values):
                for value in mapping.values():
                    candidate = PipelineHandler._match_dataclass_class_value(
                        value,
                        class_name,
                        module_name,
                    )
                    if candidate is not None:
                        return candidate
            for outputs in current.producer_outputs.values():
                for value in outputs.values():
                    candidate = PipelineHandler._match_dataclass_class_value(
                        value,
                        class_name,
                        module_name,
                    )
                    if candidate is not None:
                        return candidate
            current = current.parent_pipeline
        return None

    @staticmethod
    def _match_dataclass_class_value(
        value: Any,
        class_name: str,
        module_name: str | None,
    ) -> type | None:
        """Return ``value`` when it is a dataclass class matching the criteria."""
        if not isinstance(value, type) or not is_dataclass(value):
            return None
        if value.__name__ != class_name:
            return None
        if module_name is not None and getattr(value, "__module__", None) != module_name:
            return None
        return value

    @staticmethod
    def _reconstruct_dataclass(
        class_name: str | None,
        data: dict[str, Any],
        *,
        verbose: bool,
        module_name: str | None = None,
        warn: Any | None = None,
        pipeline: Any | None = None,
    ) -> Any:
        """Rebuild a saved pure dataclass from its structured fields.

        Returns a real dataclass instance when the class is importable and can be
        constructed from the saved fields; otherwise falls back to a
        ``SimpleNamespace`` (with a verbose-gated warning when ``warn`` is given).
        When ``pipeline`` is given, classes visible as pipeline runtime values
        (for example dynamically generated classes produced by a block that was
        re-run during placeholder recovery) are also considered.
        """
        if class_name is not None:
            candidate_class = PipelineHandler._find_dataclass_class(
                class_name,
                module_name=module_name,
            )
            if candidate_class is None and pipeline is not None:
                candidate_class = pipeline._find_pipeline_dataclass_class(
                    class_name,
                    module_name=module_name,
                )
            if candidate_class is not None:
                init_field_names = {
                    field.name for field in fields(candidate_class) if field.init
                }
                try:
                    return candidate_class(
                        **{
                            key: value
                            for key, value in data.items()
                            if key in init_field_names
                        }
                    )
                except TypeError:
                    pass
        if verbose and warn is not None:
            warn(
                f"Saved dataclass '{class_name}' could not be reconstructed at load "
                "(class is not importable as a pure dataclass, its fields changed, or the "
                "saved fields cannot be passed to its constructor); "
                "using a SimpleNamespace fallback instead."
            )
        return SimpleNamespace(**data)

    @staticmethod
    def _config_object_as_dict(config_obj: Any) -> dict[str, Any]:
        config_dict = asdict(config_obj)
        extra_attrs = {
            key: value
            for key, value in vars(config_obj).items()
            if key not in config_dict
        }
        config_dict.update(extra_attrs)
        return config_dict

    def _serialize_node(self, node: Any) -> dict[str, Any]:
        if isinstance(node, PipelineHandler):
            return {
                "kind": "pipeline",
                "registration_name": node.registration_name,
                "execution_priority": node.execution_priority,
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

    def _execute_nodes(
        self,
        nodes: list[Any],
        mode: str,
        overrides: dict[str, Any] | None = None,
        upstream_outputs: dict[str, Any] | None = None,
        parent_config: dict[str, Any] | None = None,
        sync_parent_on_completion: bool = True,
        previous_node_outputs: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[RunRecord, dict[str, Any]]:
        run_id = uuid4().hex
        run_record = RunRecord(
            run_id=run_id,
            mode=mode,
            executed_blocks=[],
            started_at=datetime.now(UTC).isoformat(),
        )
        config_snapshot_path = self.metadata_root / f"config__{run_id}.pkl"
        run_record.config_snapshot_path = str(config_snapshot_path)
        self._persist_config_snapshot(config_snapshot_path)
        self.run_history.append(run_record)
        self.logger.info(f"Starting {mode} with run_id={run_id}")

        base_visible = dict(upstream_outputs or {})
        executed_priority_groups: set[int] = set()
        try:
            if self.gate_block is not None and not self.gate_block.evaluate(
                overrides or {},
                base_visible,
                parent_config or {},
            ):
                skipped_outputs = {
                    output_name: None
                    for output_name in self.list_declared_outputs()
                    if output_name not in base_visible
                }
                if self._gate_cleanup_predecided is not None:
                    cleanup = self._gate_cleanup_predecided
                    self._gate_cleanup_predecided = None
                elif self.gate_cleanup_confirmation and (
                    self.producer_outputs or self.artifact_registry
                ):
                    cleanup = self._confirm_gate_cleanup(mode)
                else:
                    cleanup = True
                if cleanup:
                    removed_outputs: list[dict[str, Any]] = []
                    for node in nodes:
                        removed_outputs.append(
                            self.producer_outputs.pop(node.registration_name, {})
                        )
                        if isinstance(node, PipelineHandler):
                            node._invalidate_all_outputs()
                    self.para_value_dict = skipped_outputs
                    self.artifact_registry = {}
                    for outputs in removed_outputs:
                        self._delete_artifacts_from_outputs(outputs)
                    self.logger.warning(f"Skipped {mode} with run_id={run_id}")
                else:
                    self.logger.warning(
                        f"Skipped {mode} with run_id={run_id} without cleanup (cleanup declined)"
                    )
                run_record.status = "skipped"
                run_record.produced_outputs.extend(sorted(skipped_outputs))
                if sync_parent_on_completion:
                    self._sync_attached_outputs_to_parent()
                return run_record, skipped_outputs

            for node in nodes:
                priority_group = self._priority_group(node.execution_priority)
                if priority_group in executed_priority_groups:
                    removed_node_outputs: dict[str, Any] = self.producer_outputs.pop(
                        node.registration_name,
                        {},
                    )
                    if isinstance(node, PipelineHandler):
                        node._invalidate_all_outputs()
                    self._rebuild_visible_state(upstream_outputs)
                    self._delete_artifacts_from_outputs(removed_node_outputs)
                    continue
                visible_outputs = self._visible_outputs_before_priority(
                    node.execution_priority,
                    upstream_outputs=upstream_outputs,
                )
                visible_outputs.update(self.manual_values)
                prior_outputs = (previous_node_outputs or {}).get(node.registration_name)
                if prior_outputs:
                    visible_outputs = dict(visible_outputs) | prior_outputs
                node_executed = True
                if isinstance(node, PipelineHandler):
                    child_run_record, produced_outputs = node._execute_nodes(
                        node._sorted_nodes(),
                        mode=f"run_child:{node.registration_name}",
                        overrides=overrides,
                        upstream_outputs=visible_outputs,
                        parent_config=self.get_full_config(),
                        sync_parent_on_completion=False,
                    )
                    node_executed = child_run_record.status != "skipped"
                else:
                    produced_outputs = node.execute(
                        run_id,
                        visible_outputs,
                        overrides=overrides,
                        parent_config=parent_config or {},
                    )
                self.producer_outputs[node.registration_name] = produced_outputs
                self._rebuild_visible_state(upstream_outputs)
                if self.memory_profile_logging and not isinstance(node, PipelineHandler):
                    self._log_memory_profile(node.registration_name, phase="after_compute")
                run_record.executed_blocks.append(node.registration_name)
                run_record.produced_outputs.extend(produced_outputs.keys())
                if node_executed:
                    executed_priority_groups.add(priority_group)
                if self.memory_saving_mode:
                    if not isinstance(node, PipelineHandler):
                        self._cleanup_block_memory(node.registration_name)
                elif self.memory_profile_logging and not isinstance(node, PipelineHandler):
                    self._log_memory_profile(node.registration_name, phase="after_cleanup")

            run_record.status = "success"
            run_record.produced_outputs = list(dict.fromkeys(run_record.produced_outputs))
            self._rebuild_visible_state(upstream_outputs)
            if sync_parent_on_completion:
                self._sync_attached_outputs_to_parent()
            self.logger.info(f"Completed {mode} with run_id={run_id}")
            return run_record, dict(self.para_value_dict)
        except BaseException as exc:
            run_record.status = "failed"
            run_record.error_message = str(exc)
            if not mode.startswith("run_child:"):
                self.logger.log_exception(exc, f"Failed {mode} with run_id={run_id}: {exc}")
            if isinstance(
                exc,
                (ExecutionError, ResolutionError, RegistrationError, KeyboardInterrupt, SystemExit),
            ):
                raise
            raise ExecutionError("Pipeline execution failed") from exc
        finally:
            run_record.finished_at = datetime.now(UTC).isoformat()

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
        if not isinstance(node, PipelineHandler):
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
            if isinstance(node, PipelineHandler):
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
            if isinstance(node, PipelineHandler):
                node._invalidate_all_outputs()
            self._remove_registered_node(node)
        if not self._invalidation_forbidden:
            self._invalidate_from_priority(earliest_priority)
        if self.parent_pipeline is not None:
            self._resync_mirror_to_parent()

    def _validate_strict_attach(
        self,
        child: "PipelineHandler",
        execution_priority: float,
    ) -> None:
        """Validate attaching `child` when this pipeline is in strict mode.

        Runs before any mutation so a failed check leaves the child unattached
        and this pipeline untouched. Raises RegistrationError on the first
        cross-boundary name conflict or the first failing per-registration
        strict check.
        """
        if not self.strict_mode or self._suppress_strict_validation:
            return

        child_pipelines = child._iter_attached_pipelines()
        child_config_names: set[str] = set()
        child_manual_names: set[str] = set()
        child_output_names: set[str] = set()
        for pipeline in child_pipelines:
            child_config_names.update(pipeline.config_as_dict())
            child_manual_names.update(pipeline.manual_values)
            child_output_names.update(pipeline.list_declared_outputs())

        parent_config_names = set(self._visible_config_names())
        parent_manual_names = set(self.manual_values) | set(self._ancestor_manual_values())
        parent_output_names = set(
            self._visible_outputs_before_priority(execution_priority)
        )
        # Outputs declared by earlier-priority nodes are guaranteed to exist
        # before the child runs, so they count as visible at attach time even
        # before the upstream blocks have executed.
        parent_output_names.update(
            self._declared_output_names_before_priority(execution_priority)
        )

        cross_conflicts: list[tuple[str, set[str]]] = [
            ("child config field collides with parent manual value", child_config_names & parent_manual_names),
            ("child config field collides with parent visible output", child_config_names & parent_output_names),
            ("child manual value collides with parent config field", child_manual_names & parent_config_names),
            ("child manual value collides with parent visible output", child_manual_names & parent_output_names),
            ("child output collides with parent config field", child_output_names & parent_config_names),
            ("child output collides with parent manual value", child_output_names & parent_manual_names),
        ]
        for message, names in cross_conflicts:
            if names:
                raise RegistrationError(
                    f"Attach conflict while attaching '{child.registration_name}': {message}(s) {sorted(names)}"
                )

        for pipeline in child_pipelines:
            pipeline_effective_priority = (
                pipeline.execution_priority
                if pipeline.execution_priority is not None
                else execution_priority
            )
            pipeline_upstream_declared = set(
                pipeline._declared_output_names_before_priority(
                    pipeline_effective_priority
                )
            )
            if pipeline.gate_block is not None and pipeline.gate_block.config_field_name is not None:
                gate_visible = (
                    set(pipeline.get_full_config())
                    | set(pipeline._incoming_parent_outputs())
                    | set(pipeline.manual_values)
                    | set(pipeline._ancestor_manual_values())
                    | pipeline_upstream_declared
                    | parent_config_names
                    | parent_manual_names
                    | parent_output_names
                )
                if pipeline.gate_block.config_field_name not in gate_visible:
                    raise RegistrationError(
                        f"Gate config '{pipeline.gate_block.config_field_name}' in child pipeline "
                        f"'{pipeline.registration_name}' is not found in config, visible output values, or visible manual values"
                    )
            for block in pipeline.blocks:
                block_visible_names = (
                    set(pipeline._visible_config_names())
                    | set(
                        pipeline._visible_outputs_before_priority(
                            block.execution_priority
                        )
                    )
                    | set(
                        pipeline._declared_output_names_before_priority(
                            block.execution_priority
                        )
                    )
                    | set(pipeline.manual_values)
                    | set(pipeline._ancestor_manual_values())
                    | parent_config_names
                    | parent_manual_names
                    | parent_output_names
                )
                for registration in block.functions:
                    if isinstance(registration, FunctionRegistration):
                        block._strict_validate_registration(
                            registration,
                            force_strict=True,
                            visible_names=block_visible_names,
                        )

    def _remove_registered_node(self, node: Any) -> None:
        self.nodes = [candidate for candidate in self.nodes if candidate is not node]
        self.nodes_by_name.pop(node.registration_name, None)
        if not isinstance(node, PipelineHandler):
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

    def _validate_related_pipeline_names(self, candidate: "PipelineHandler") -> None:
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

    def _root_pipeline(self) -> "PipelineHandler":
        current = self
        while current.parent_pipeline is not None:
            current = current.parent_pipeline
        return current

    def _iter_attached_pipelines(self) -> list["PipelineHandler"]:
        pipelines: list[PipelineHandler] = [self]
        for node in self._sorted_nodes():
            if isinstance(node, PipelineHandler):
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

    def _resolve_target_path(self, path_parts: tuple[str, ...]) -> tuple["PipelineHandler", Any]:
        if not path_parts:
            raise RegistrationError("At least one target name must be provided")
        current: PipelineHandler = self
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
        snapshot = self._snapshot_runtime_state()
        previous_outputs = snapshot[0].get(child_pipeline.registration_name, {})
        self._invalidate_from_priority(child_priority)
        child_target_name = path_parts[1]
        child_target = child_pipeline._get_node_or_raise(child_target_name)
        child_previous_outputs = snapshot[0].get(child_pipeline.registration_name, {})
        child_pipeline.producer_outputs[child_target.registration_name] = dict(child_previous_outputs)
        child_pipeline._rebuild_visible_state(child_pipeline._incoming_parent_outputs())
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
        child_pipeline: "PipelineHandler",
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
        child_pipeline: "PipelineHandler",
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
        child_pipeline: "PipelineHandler",
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

    def _sorted_nodes(self) -> list[Any]:
        return sorted(
            self.nodes, key=lambda node: (node.execution_priority, node.registration_name)
        )

    def _node_declared_outputs(self, node: Any) -> set[str]:
        if isinstance(node, PipelineHandler):
            return node.list_declared_outputs()
        return node.declared_outputs()

    def _rebuild_visible_state(self, upstream_outputs: dict[str, Any] | None = None) -> None:
        """Rebuild this pipeline's visible value mirrors in memory only.

        Artifact files are never deleted here: deletion happens exclusively at
        the explicit invalidation points (``_erase_node_outputs``,
        ``_invalidate_from_priority``, ``_invalidate_all_outputs``, gate-skip
        cleanup) via ``_delete_artifacts_from_outputs``, and stale generations
        are removed by save-time cleanup. Keeping this rebuild side-effect-free
        lets the run loop call it after every node without rescanning the tree.
        """
        visible = dict(upstream_outputs or {})
        for node in self._sorted_nodes():
            visible.update(self.producer_outputs.get(node.registration_name, {}))
        visible.update(self.manual_values)
        declared_outputs = self.list_declared_outputs()
        self.para_value_dict = {
            output_name: visible[output_name]
            for output_name in declared_outputs
            if output_name in visible
        }
        self.para_value_dict.update(self.manual_values)
        self.artifact_registry = {
            output_name: value
            for output_name, value in self.para_value_dict.items()
            if isinstance(value, ArtifactRecord)
        }

    def _visible_outputs_before_priority(
        self,
        priority: float | None,
        upstream_outputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        visible = dict(upstream_outputs or self._incoming_parent_outputs())
        visible.update(self.manual_values)
        if priority is None:
            return visible
        for node in self._sorted_nodes():
            if node.execution_priority >= priority:
                break
            visible.update(self._node_visible_outputs(node))
        return visible

    def _node_visible_outputs(self, node: Any) -> dict[str, Any]:
        outputs = dict(self.producer_outputs.get(node.registration_name, {}))
        if isinstance(node, PipelineHandler):
            outputs.update(node.para_value_dict)
        return outputs

    def _incoming_parent_output_names(self) -> set[str]:
        if self.parent_pipeline is None or self.execution_priority is None:
            return set()
        return self.parent_pipeline._declared_output_names_before_priority(self.execution_priority)

    def _declared_output_names_before_priority(self, priority: float | None) -> set[str]:
        output_names = set(self._incoming_parent_output_names())
        if priority is None:
            return output_names
        for node in self._sorted_nodes():
            if node.execution_priority >= priority:
                break
            output_names.update(self._node_declared_outputs(node))
        return output_names

    def _incoming_parent_outputs(self) -> dict[str, Any]:
        if self.parent_pipeline is None or self.execution_priority is None:
            return {}
        return self.parent_pipeline._visible_outputs_before_priority(self.execution_priority)

    def _incoming_parent_manual_values(self) -> dict[str, Any]:
        if self.parent_pipeline is None or self.execution_priority is None:
            return {}
        return self.parent_pipeline._visible_manual_values_before_priority(self.execution_priority)

    def _visible_manual_values_before_priority(self, priority: float | None) -> dict[str, Any]:
        visible = dict(self._incoming_parent_manual_values())
        visible.update(self.manual_values)
        if priority is None:
            return visible
        for node in self._sorted_nodes():
            if node.execution_priority >= priority:
                break
            if isinstance(node, PipelineHandler):
                visible.update(node.manual_values)
        return visible

    def _descendant_visible_value(self, variable_name: str) -> Any | None:
        for node in self._sorted_nodes():
            if not isinstance(node, PipelineHandler):
                continue
            if variable_name in node.para_value_dict:
                return node.para_value_dict[variable_name]
            descendant_value = node._descendant_visible_value(variable_name)
            if descendant_value is not None:
                return descendant_value
        return None

    def _ancestor_descendant_visible_value(self, variable_name: str) -> Any | None:
        current = self.parent_pipeline
        while current is not None:
            descendant_value = current._descendant_visible_value(variable_name)
            if descendant_value is not None:
                return descendant_value
            current = current.parent_pipeline
        return None

    def _ancestor_config_values(self) -> dict[str, Any]:
        config: dict[str, Any] = {}
        current = self.parent_pipeline
        chain: list[PipelineHandler] = []
        while current is not None:
            chain.append(current)
            current = current.parent_pipeline
        for pipeline in reversed(chain):
            config.update(pipeline.config_as_dict())
        return config

    def _ancestor_manual_values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        current = self.parent_pipeline
        chain: list[PipelineHandler] = []
        while current is not None:
            chain.append(current)
            current = current.parent_pipeline
        for pipeline in reversed(chain):
            values.update(pipeline.manual_values)
        return values

    def _visible_config_names(self) -> set[str]:
        return set(self.config_as_dict()).union(self._ancestor_config_values())

    def _registration_visible_names(self) -> set[str]:
        names = set(self._visible_config_names())
        names.update(self.list_declared_outputs())
        names.update(self._incoming_parent_output_names())
        names.update(self._tree_constant_names())
        return names

    def _registration_disk_backed_names(self) -> set[str]:
        names = self._known_disk_backed_output_names()
        try:
            for key, value in self._incoming_parent_outputs().items():
                if isinstance(value, ArtifactRecord):
                    names.add(key)
        except Exception:
            pass
        for key, value in self.manual_values.items():
            if isinstance(value, ArtifactRecord):
                names.add(key)
        for key, value in self._ancestor_manual_values().items():
            if isinstance(value, ArtifactRecord):
                names.add(key)
        return names

    def _known_disk_backed_output_names(self) -> set[str]:
        names = set(self.artifact_registry)
        for outputs in self.producer_outputs.values():
            for key, value in outputs.items():
                if isinstance(value, ArtifactRecord):
                    names.add(key)
        return names

    def _required_input_names(self, node: Any) -> set[str]:
        if isinstance(node, PipelineHandler):
            return node._required_input_names_for_pipeline()
        required = set()
        for registration in node.functions:
            if isinstance(registration, ExpressionRegistration):
                required.update(node._effective_expression_input_names(registration))
            else:
                required.update(registration.input_names)
            var_pos_name = getattr(registration, "var_pos_name", None)
            if var_pos_name is not None:
                required.add(var_pos_name)
            var_kw_name = getattr(registration, "var_kw_name", None)
            if var_kw_name is not None:
                required.add(var_kw_name)
        return required

    def _required_input_names_for_pipeline(self) -> set[str]:
        required: set[str] = set()
        if self.gate_block is not None:
            required.update(self.gate_block.registration.input_names)
            if self.gate_block.config_field_name is not None:
                required.add(self.gate_block.config_field_name)
        for node in self._sorted_nodes():
            required.update(self._required_input_names(node))
        return required

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
            if isinstance(value, ArtifactRecord):
                materialized[output_name] = self.artifact_store.load(value)
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

    def _prepare_call_arguments(
        self,
        registration: FunctionRegistration,
        overrides: dict[str, Any],
        visible_outputs: dict[str, Any],
        parent_config: dict[str, Any] | None = None,
        block: Any | None = None,
    ) -> tuple[list[Any], dict[str, Any], list[str]]:
        defaults = default_map(registration.callable_obj)
        signature = callable_signature(registration.callable_obj)
        parameters = list(signature.parameters.values())
        declared_output_names = set(visible_outputs).union(self.list_declared_outputs())
        if block is not None:
            declared_output_names.update(block.declared_outputs())
        var_pos_index = next(
            (
                index
                for index, parameter in enumerate(parameters)
                if parameter.kind == inspect.Parameter.VAR_POSITIONAL
            ),
            None,
        )
        positional_args: list[Any] = []
        keyword_args: dict[str, Any] = {}
        loaded_artifacts: list[str] = []

        for index, parameter in enumerate(parameters):
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                input_name = registration.var_pos_name or parameter.name
                if block is not None and input_name in block.registered_args:
                    value = [
                        self._resolve_named_input(
                            item_name,
                            registration.function_name,
                            overrides,
                            visible_outputs,
                            parent_config,
                            defaults,
                            loaded_artifacts,
                            declared_output_names,
                        )
                        for item_name in block.registered_args[input_name].ordered_items
                    ]
                else:
                    value = self._resolve_named_input(
                        input_name,
                        registration.function_name,
                        overrides,
                        visible_outputs,
                        parent_config,
                        defaults,
                        loaded_artifacts,
                        declared_output_names,
                        allow_missing=True,
                        missing_value=[],
                    )
                if not isinstance(value, (list, tuple)):
                    raise ResolutionError(
                        f"Variadic positional argument '{input_name}' for function '{registration.function_name}' must resolve to a list or tuple"
                    )
                positional_args.extend(value)
                continue

            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                input_name = registration.var_kw_name or parameter.name
                if block is not None and input_name in block.registered_kwargs:
                    value = {
                        key: self._resolve_named_input(
                            item_name,
                            registration.function_name,
                            overrides,
                            visible_outputs,
                            parent_config,
                            defaults,
                            loaded_artifacts,
                            declared_output_names,
                        )
                        for key, item_name in block.registered_kwargs[input_name].mapping_dct.items()
                    }
                else:
                    value = self._resolve_named_input(
                        input_name,
                        registration.function_name,
                        overrides,
                        visible_outputs,
                        parent_config,
                        defaults,
                        loaded_artifacts,
                        declared_output_names,
                        allow_missing=True,
                        missing_value={},
                    )
                if not isinstance(value, dict):
                    raise ResolutionError(
                        f"Variadic keyword argument '{input_name}' for function '{registration.function_name}' must resolve to a dict"
                    )
                overlap = set(value).intersection(keyword_args)
                if overlap:
                    raise ResolutionError(
                        f"Variadic keyword argument '{input_name}' conflicts with explicit arguments: {sorted(overlap)}"
                    )
                keyword_args.update(value)
                continue

            input_name = registration.param_mapping.get(parameter.name, parameter.name)
            if input_name is None:
                value = None
            else:
                value = self._resolve_named_input(
                    input_name,
                    registration.function_name,
                    overrides,
                    visible_outputs,
                    parent_config,
                    defaults,
                    loaded_artifacts,
                    declared_output_names,
                )

            if parameter.kind == inspect.Parameter.POSITIONAL_ONLY or (
                parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                and var_pos_index is not None
                and index < var_pos_index
            ):
                positional_args.append(value)
            else:
                keyword_args[parameter.name] = value
        return positional_args, keyword_args, loaded_artifacts

    def _resolve_named_input(
        self,
        input_name: str,
        function_name: str,
        overrides: dict[str, Any],
        visible_outputs: dict[str, Any],
        parent_config: dict[str, Any] | None,
        defaults: dict[str, Any],
        loaded_artifacts: list[str],
        declared_output_names: set[str],
        *,
        allow_missing: bool = False,
        missing_value: Any = None,
    ) -> Any:
        if input_name == "logger":
            value = self.logger
        elif input_name in overrides:
            value = overrides[input_name]
        elif input_name in visible_outputs:
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
            value = defaults[input_name]
        elif allow_missing:
            value = missing_value
        else:
            raise ResolutionError(
                f"Cannot resolve argument '{input_name}' for function '{function_name}'"
            )

        if isinstance(value, ArtifactRecord):
            value = self.artifact_store.load(value)
            loaded_artifacts.append(input_name)
        if isinstance(value, CallableValueReference):
            value = self._restore_callable_value(value)
        if isinstance(value, (RuntimeValueReference, DataclassValueReference)):
            raise ResolutionError(
                f"Cannot resolve argument '{input_name}' for function '{function_name}': "
                f"the value was saved as a placeholder ({value.reason}) and cannot be restored; "
                "recreate or reset the value before running"
            )
        return value

    def _config_has_field(self, config_obj: Any, field_name: str) -> bool:
        if is_dataclass(config_obj) and not isinstance(config_obj, type):
            return any(
                field.name == field_name for field in config_obj.__dataclass_fields__.values()
            )
        if isinstance(config_obj, dict):
            return field_name in config_obj
        return hasattr(config_obj, field_name)

    def _config_value(self, config_obj: Any, field_name: str) -> Any:
        if is_dataclass(config_obj) and not isinstance(config_obj, type):
            return getattr(config_obj, field_name)
        if isinstance(config_obj, dict):
            return config_obj[field_name]
        return getattr(config_obj, field_name)

    @staticmethod
    def _config_name_mapping(config_obj: Any) -> dict[str, Any]:
        if is_dataclass(config_obj) and not isinstance(config_obj, type):
            return PipelineHandler._config_object_as_dict(config_obj)
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

    def _invalidate_from_priority(self, priority: float, include_target: bool = True) -> None:
        if priority is None:
            return
        removed_outputs: list[dict[str, Any]] = []
        for node in list(self._sorted_nodes()):
            if node.execution_priority < priority:
                continue
            if node.execution_priority == priority and not include_target:
                continue
            removed_outputs.append(
                self.producer_outputs.pop(node.registration_name, {})
            )
            if isinstance(node, PipelineHandler):
                node._invalidate_all_outputs()
        self._rebuild_visible_state(self._incoming_parent_outputs())
        for outputs in removed_outputs:
            self._delete_artifacts_from_outputs(outputs)

    def _invalidate_all_outputs(self) -> None:
        removed_outputs = list(self.producer_outputs.values())
        self.producer_outputs.clear()
        self.para_value_dict.clear()
        self.artifact_registry.clear()
        for node in self._sorted_nodes():
            if isinstance(node, PipelineHandler):
                node._invalidate_all_outputs()
        self._rebuild_visible_state(self._incoming_parent_outputs())
        for outputs in removed_outputs:
            self._delete_artifacts_from_outputs(outputs)

    def _delete_artifacts_from_outputs(self, outputs: dict[str, Any]) -> None:
        active_artifact_paths = self._collect_referenced_artifact_paths()
        for value in outputs.values():
            if isinstance(value, ArtifactRecord):
                if value.file_path in active_artifact_paths:
                    continue
                try:
                    self.artifact_store.delete(value)
                except (OSError, PersistenceError) as exc:
                    self.logger.warning(
                        f"Could not delete obsolete artifact '{value.file_path}': "
                        f"{type(exc).__name__}: {exc}"
                    )

    def _collect_referenced_artifact_paths(self) -> set[str]:
        paths: set[str] = set()

        def collect_from_mapping(mapping: dict[str, Any]) -> None:
            for value in mapping.values():
                if isinstance(value, ArtifactRecord):
                    paths.add(value.file_path)

        collect_from_mapping(self.para_value_dict)
        collect_from_mapping(self.artifact_registry)
        collect_from_mapping(self.manual_values)
        for outputs in self.producer_outputs.values():
            collect_from_mapping(outputs)
        for node in self._sorted_nodes():
            if isinstance(node, PipelineHandler):
                paths.update(node._collect_referenced_artifact_paths())
        return paths

    def _persist_config_snapshot(self, path: Path) -> None:
        self._atomic_pickle_dump(self._serialize_config_for_save(self.config), path)

    @staticmethod
    def _atomic_pickle_dump(obj: Any, path: Path) -> None:
        """Write a pickle payload to `path` atomically.

        The payload is written to a sibling ``*.tmp`` file, fsynced, then renamed
        over `path`, so a crash mid-write never corrupts the previous file. The
        temporary file is removed on any failure.
        """
        tmp_path = path.with_name(path.name + ".tmp")
        try:
            with tmp_path.open("wb") as handle:
                pickle.dump(obj, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _attach_to_parent(self, parent: "PipelineHandler", execution_priority: float) -> None:
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
        self._cleanup_temporary_root_handle()

    def _sync_attached_outputs_to_parent(self) -> None:
        if self.parent_pipeline is None or self.execution_priority is None:
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
            return value

        for outputs in self.producer_outputs.values():
            for key, value in list(outputs.items()):
                outputs[key] = rewrite_value(value)
        for key, value in list(self.para_value_dict.items()):
            self.para_value_dict[key] = rewrite_value(value)
        for key, value in list(self.artifact_registry.items()):
            self.artifact_registry[key] = rewrite_value(value)
        if self.historical_result_log_path and self.historical_result_log_path.startswith(
            old_prefix
        ):
            self.historical_result_log_path = self.historical_result_log_path.replace(
                old_prefix,
                new_prefix,
                1,
            )
        for node in self._sorted_nodes():
            if isinstance(node, PipelineHandler):
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
            if isinstance(node, PipelineHandler):
                node._rewrite_run_history_paths(old_root, new_root)

    def _refresh_descendant_roots(self, old_root: Path, new_root: Path) -> None:
        for node in self._sorted_nodes():
            if not isinstance(node, PipelineHandler):
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

    def _describe_lines(
        self, indent: str = "", as_child: bool = False, muted: bool = False
    ) -> list[str]:
        lines: list[str] = []
        symbol_color = "blue"
        # Arguments use pure black normally and dark grey when skipped.
        # termcolor's light_grey is SGR 37 (white), which is nearly invisible
        # on the white notebook background in Colab; dark_grey (SGR 90)
        # renders as a readable grey in both Jupyter and Colab.
        arg_color = "grey" if not muted else "dark_grey"
        muted = muted or (as_child and self._should_grey_in_chart())
        if as_child:
            line = f"{self._line_style(indent, muted)}{self._chart_color('pipeline', 'magenta', muted)} {self._chart_color(f'[{self.execution_priority}]', 'cyan', muted)} {self._chart_color(self.registration_name, 'blue', muted)}"
            lines.append(line)
        else:
            lines.append(
                f"{indent}{self._chart_color('PipelineHandler', 'green', muted)}{self._chart_color('(', symbol_color, muted)}{self._chart_color(self.registration_name, 'blue', muted)}{self._chart_color(')', symbol_color, muted)}"
            )
        if self.gate_block is not None:
            gate_args = self._chart_color(
                ", ".join(self._displayed_argument_names(self.gate_block.registration, None, None)),
                arg_color,
                muted,
            )
            gate_line = (
                f"{indent}{self._line_style('├── ', muted)}{self._chart_color('[gate]', 'magenta', muted)} {self._chart_color(self.gate_block.registration.function_name, 'green', muted)}"
                f"{self._chart_color('(', symbol_color, muted)}{gate_args}{self._chart_color(')', symbol_color, muted)}"
            )
            lines.append(gate_line)
        sorted_nodes = self._sorted_nodes()
        node_muted_states = [
            muted or (isinstance(node, PipelineHandler) and node._should_grey_in_chart())
            for node in sorted_nodes
        ]
        for index, node in enumerate(sorted_nodes):
            is_last = index == len(sorted_nodes) - 1
            node_muted = node_muted_states[index]
            prefix = "└──" if is_last else "├──"
            # The spine segment owned by this level stays active while any
            # later sibling at this level will still run, so the outermost
            # vertical line never looks broken across a skipped subtree; it
            # only greys when this node and every later sibling is skipped.
            spine_muted = node_muted and all(node_muted_states[index + 1 :])
            child_indent = indent + self._spine_style(
                "    " if is_last else "│   ", spine_muted
            )
            if isinstance(node, PipelineHandler):
                child_line = (
                    f"{indent}{self._line_style(f'{prefix} ', node_muted)}{self._chart_color('child-pipeline', 'magenta', node_muted)} {self._chart_color(f'[{node.execution_priority}]', 'cyan', node_muted)} {self._chart_color(node.registration_name, 'blue', node_muted)}"
                )
                lines.append(child_line)
                lines.extend(node._describe_lines(child_indent, as_child=True, muted=node_muted)[1:])
                continue
            block_line = (
                f"{indent}{self._line_style(f'{prefix} ', node_muted)}{self._chart_color(f'[{node.execution_priority}]', 'cyan', node_muted)} {self._chart_color(node.registration_name, 'blue', node_muted)}"
            )
            lines.append(block_line)
            for function_index, registration in enumerate(node.functions):
                function_prefix = "└──" if function_index == len(node.functions) - 1 else "├──"
                outputs = [
                    (
                        self._chart_color(f"{output_name}*", "red", node_muted)
                        if output_name in registration.save_to_disk
                        else self._chart_color(output_name, "green", node_muted)
                    )
                    for output_name in registration.output_names
                ]
                args = self._chart_color(
                    ", ".join(
                        self._displayed_argument_names(
                            registration,
                            node.execution_priority,
                            node,
                        )
                    ),
                    arg_color,
                    node_muted,
                )
                function_line = (
                    f"{child_indent}{self._line_style(f'{function_prefix} ', node_muted)}{self._chart_color(registration.function_name, 'green', node_muted)}"
                    f"{self._chart_color('(', symbol_color, node_muted)}{args}{self._chart_color(')', symbol_color, node_muted)}"
                    + (
                        f" {self._chart_color('->', symbol_color, node_muted)} {', '.join(outputs)}"
                        if outputs
                        else ""
                    )
                )
                lines.append(function_line)
        return lines

    def _should_grey_in_chart(self) -> bool:
        if self.parent_pipeline is None or self.gate_block is None:
            return False
        config_field_name = self.gate_block.config_field_name
        if config_field_name is None:
            return False
        try:
            return self.get_config_value(config_field_name) != self.gate_block.expected_value
        except ResolutionError:
            return False

    def _chart_color(self, text: str, color: str, muted: bool) -> str:
        swapped_normal_map = {
            "magenta": "light_magenta",
            "cyan": "light_cyan",
            "blue": "light_blue",
            "green": "light_green",
            "yellow": "light_yellow",
            "red": "light_red",
        }
        if not muted:
            return import_module("termcolor").colored(
                text,
                swapped_normal_map.get(color, color),
                force_color=True,
            )
        return import_module("termcolor").colored(
            text,
            color,
            force_color=True,
        )

    def _line_style(self, text: str, muted: bool) -> str:
        if not muted:
            return text
        return import_module("termcolor").colored(text, "light_grey", force_color=True)

    def _spine_style(self, text: str, muted: bool) -> str:
        """Color a spine segment (``│   `` or spaces) for the chart."""
        if not muted:
            return text
        return import_module("termcolor").colored(text, "light_grey", force_color=True)

    def _displayed_argument_names(
        self,
        registration: FunctionRegistration | ExpressionRegistration,
        priority: float | None,
        block: Any | None = None,
    ) -> list[str]:
        visible_output_names = self._declared_output_names_before_priority(priority)
        visible_config_names = self._visible_config_names()
        input_names = (
            block._effective_expression_input_names(registration)
            if block is not None and isinstance(registration, ExpressionRegistration)
            else registration.input_names
        )
        displayed = [
            name
            for name in input_names
            if name in visible_output_names or name in visible_config_names
        ]
        var_pos_name = getattr(registration, "var_pos_name", None)
        if (
            block is not None
            and var_pos_name is not None
            and var_pos_name in block.registered_args
        ):
            displayed.append(var_pos_name)
        var_kw_name = getattr(registration, "var_kw_name", None)
        if (
            block is not None
            and var_kw_name is not None
            and var_kw_name in block.registered_kwargs
        ):
            displayed.append(var_kw_name)
        return displayed

    def _read_result_history_from_file(self, file_path: str) -> list[str]:
        path = Path(file_path)
        if not path.exists():
            return []
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if " RESULT " in line
        ]

    def _color(self, text: str, color: str) -> str:
        return import_module("termcolor").colored(text, color, force_color=True)

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

    def _capture_prints(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        if self.print_capture_mode == "off":
            return func(*args, **kwargs)

        # Convenience feature only: redirect_stdout is process-level state, so heavily
        # parallel print-heavy functions can still interleave output. Explicit logger usage
        # remains the safer option for important structured messages.
        buffer = StringIO()
        stdout_target: Any = (
            _TeeStdout(sys.stdout, buffer) if self.print_capture_mode == "tee" else buffer
        )
        try:
            with redirect_stdout(stdout_target):
                result = func(*args, **kwargs)
        finally:
            # Flush even when the function raises, so prints made before the
            # exception are still recorded in the pipeline log.
            self._flush_captured_prints(buffer)
        return result

    def _flush_captured_prints(self, buffer: StringIO) -> None:
        captured = buffer.getvalue()
        if not captured:
            return
        for line in captured.splitlines():
            if line:
                self.logger._write("PRINT", line, emit_console=False)

    def config_as_dict(self) -> dict[str, Any]:
        if is_dataclass(self.config) and not isinstance(self.config, type):
            config_dict = asdict(self.config)
            extra_attrs = {
                key: value
                for key, value in vars(self.config).items()
                if key not in config_dict
            }
            config_dict.update(extra_attrs)
            return config_dict
        if isinstance(self.config, dict):
            return dict(self.config)
        if hasattr(self.config, "__dict__"):
            return dict(vars(self.config))
        raise PersistenceError("Configuration object is not serializable to dict")

    def _snapshot_runtime_state(
        self,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, ArtifactRecord], dict[str, Any]]:
        return (
            {name: dict(outputs) for name, outputs in self.producer_outputs.items()},
            dict(self.para_value_dict),
            dict(self.artifact_registry),
            dict(self.manual_values),
        )

    def _restore_runtime_state(
        self,
        snapshot: tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, ArtifactRecord], dict[str, Any]],
    ) -> None:
        producer_outputs, para_values, artifacts, manual_values = snapshot
        self.producer_outputs = {name: dict(outputs) for name, outputs in producer_outputs.items()}
        self.para_value_dict = dict(para_values)
        self.artifact_registry = dict(artifacts)
        self.manual_values = dict(manual_values)


class _TeeStdout:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            flush = getattr(stream, "flush", None)
            if callable(flush):
                flush()
