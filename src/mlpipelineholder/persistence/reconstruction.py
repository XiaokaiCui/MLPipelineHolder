"""Pipeline reconstruction from saved payloads and dataclass value recovery."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from ..core.models import (
    ArtifactRecord,
    CallableValueReference,
    DataclassValueReference,
    RuntimeCallableReference,
    RuntimeValueReference,
)
from ..core.base import PipelineBase
from ..exceptions import PersistenceError, RegistrationError
from .object_storage import record_from_payload
from ..execution.atom_registry import atom_pipeline_class


def _is_child_pipeline(node: object) -> bool:
    return isinstance(node, PipelineBase)


class ReconstructionMixin:
    """Rebuilding a pipeline tree from a serialized payload."""

    if TYPE_CHECKING:
        logger: Any = None
        manual_values: dict[str, Any] = {}
        para_value_dict: dict[str, Any] = {}
        producer_outputs: dict[str, dict[str, Any]] = {}

        def _sorted_nodes(self) -> list[Any]: ...
        @staticmethod
        def _deserialize_saved_config(
            saved_config: Any, *, verbose: bool = False, warn: Any | None = None
        ) -> Any: ...
        @staticmethod
        def _deserialize_config_value(
            value: Any, *, verbose: bool = False, warn: Any | None = None
        ) -> Any: ...
        @staticmethod
        def _reconstruct_dataclass(
            class_name: str | None,
            data: dict[str, Any],
            *,
            verbose: bool,
            module_name: str | None = None,
            warn: Any | None = None,
            pipeline: Any | None = None,
        ) -> Any: ...
        @staticmethod
        def _find_dataclass_class(class_name: str, module_name: str | None = None) -> Any: ...
        @classmethod
        def _restore_saved_runtime_mapping(
            cls, values: dict[str, Any], owner_label: str
        ) -> dict[str, Any]: ...
        @classmethod
        def _restore_runtime_registered_callable(
            cls, reference: RuntimeCallableReference, block_name: str
        ) -> Any: ...
        @classmethod
        def _restore_partial_callable(cls, payload: dict[str, Any], block_name: str) -> Any: ...

    @classmethod
    def _from_payload(
        cls,
        payload: dict[str, Any],
        project_root: Path,
        parent: Any = None,
        *,
        verbose: bool = False,
        auto_resolve_placeholders: bool = True,
    ) -> Any:
        reconstruction_warnings: list[str] = []
        config = cls._deserialize_saved_config(
            payload["config"],
            verbose=verbose,
            warn=reconstruction_warnings.append,
        )
        pipeline_class: Any
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
            colourful_logs=(
                parent.colourful_logs
                if parent is not None
                else payload.get("colourful_logs", False)
            ),
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
            tuple[Any, str, str | None, str, DataclassValueReference]
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
            tuple[Any, str, str | None, str, DataclassValueReference]
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
            if _is_child_pipeline(node):
                node._restore_dataclass_value_references(
                    verbose=verbose,
                    _memo=memo,
                    _pending=pending,
                )

    def _reconnect_dataclass_fallbacks(
        self,
        pending: list[
            tuple[Any, str, str | None, str, DataclassValueReference]
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
        owner: Any,
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
