from __future__ import annotations

import inspect
import pickle
import shutil
import sys
import warnings
from contextlib import redirect_stdout
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from importlib import import_module
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from .artifact_store import ArtifactStore
from .exceptions import ExecutionError, PersistenceError, RegistrationError, ResolutionError
from .function_registry import default_map
from .gate_block import GateBlock
from .logger import PipelineLogger
from .models import ArtifactRecord, FunctionRegistration, RunRecord


class PipelineHandler:
    def __init__(
        self,
        registration_name: str,
        configuration: Any,
        local_folder_path: str | Path,
        execution_priority: int | None = None,
    ) -> None:
        self.registration_name = registration_name
        self.config = configuration
        self.execution_priority = execution_priority
        self.parent_pipeline: PipelineHandler | None = None
        self.project_root = Path(local_folder_path)
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.metadata_root = self.project_root / "metadata"
        self.metadata_root.mkdir(parents=True, exist_ok=True)
        self.logger = PipelineLogger(self.metadata_root / "pipeline.log")
        self.print_capture_mode = "tee"
        self.historical_result_log_path: str | None = None
        self._attached_result_history_override: list[str] | None = None

        self.nodes: list[Any] = []
        self.nodes_by_name: dict[str, Any] = {}
        self.blocks: list[Any] = []
        self.blocks_by_name: dict[str, Any] = {}
        self.gate_block: GateBlock | None = None

        self.para_value_dict: dict[str, Any] = {}
        self.artifact_registry: dict[str, ArtifactRecord] = {}
        self.producer_outputs: dict[str, dict[str, Any]] = {}
        self.run_history: list[RunRecord] = []
        self.artifact_store = ArtifactStore(self.project_root)

    def __str__(self) -> str:
        return self.describe_pipeline()

    def __repr__(self) -> str:
        return self.describe_pipeline()

    def add_block(self, registration_name: str, execution_priority: int) -> Any:
        from .execution_block import ExecutionBlock

        block = ExecutionBlock(self, registration_name, execution_priority)
        try:
            self._register_node(block)
        except RegistrationError as exc:
            self.logger.warning(
                f"Skipped block registration '{registration_name}' at priority {execution_priority}: {exc}"
            )
            return None
        return block

    def _add_block_strict(self, registration_name: str, execution_priority: int):
        from .execution_block import ExecutionBlock

        block = ExecutionBlock(self, registration_name, execution_priority)
        self._register_node(block)
        return block

    def add_child_pipeline(
        self,
        child_pipeline: "PipelineHandler",
        execution_priority: int,
        registration_name: str | None = None,
    ) -> "PipelineHandler":
        if child_pipeline is self:
            raise RegistrationError("A pipeline cannot register itself as a child pipeline")
        if registration_name is not None:
            child_pipeline.registration_name = registration_name
        self._validate_node_registration(child_pipeline, execution_priority)
        self._validate_output_names_against_config(sorted(child_pipeline.list_declared_outputs()))
        child_pipeline._attach_to_parent(self, execution_priority)
        self._register_node(child_pipeline)
        return child_pipeline

    def set_gate_block(self, function_or_path: Any) -> None:
        if self.gate_block is not None:
            raise RegistrationError("A pipeline can only have one gate block")
        self.gate_block = GateBlock(self, function_or_path)

    def update_config(self, overrides: dict[str, Any]) -> None:
        declared_outputs = self.list_declared_outputs()
        for field_name, value in overrides.items():
            if field_name in declared_outputs:
                self.logger.warning(
                    f"Skipped config update for '{field_name}' because it conflicts with a declared output"
                )
                continue
            self._set_config_value(field_name, value)

    def get_value(self, variable_name: str) -> Any:
        if variable_name not in self.para_value_dict:
            raise ResolutionError(f"Unknown pipeline value: {variable_name}")
        value = self.para_value_dict[variable_name]
        if isinstance(value, ArtifactRecord):
            return self.artifact_store.load(value)
        return value

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

    def list_declared_outputs(self) -> set[str]:
        outputs: set[str] = set()
        for node in self.nodes:
            outputs.update(self._node_declared_outputs(node))
        return outputs

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
        if block_name not in self.blocks_by_name:
            raise RegistrationError(f"Block not registered: {block_name}")
        block = self.blocks_by_name.pop(block_name)
        self.blocks = [candidate for candidate in self.blocks if candidate is not block]
        self.nodes = [candidate for candidate in self.nodes if candidate is not block]
        self.nodes_by_name.pop(block_name, None)
        self._invalidate_from_priority(block.execution_priority)

    def run_all(self, overrides: dict[str, Any] | None = None) -> RunRecord:
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

    def run_until(self, block_name: str, overrides: dict[str, Any] | None = None) -> RunRecord:
        node = self._get_node_or_raise(block_name)
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
            mode=f"run_until:{block_name}",
            overrides=overrides,
            upstream_outputs=self._incoming_parent_outputs(),
            parent_config=self._ancestor_config_values(),
        )[0]

    def run_from(self, block_name: str, overrides: dict[str, Any] | None = None) -> RunRecord:
        node = self._get_node_or_raise(block_name)
        self._invalidate_from_priority(node.execution_priority)
        return self._execute_nodes(
            [
                candidate
                for candidate in self._sorted_nodes()
                if candidate.execution_priority >= node.execution_priority
            ],
            mode=f"run_from:{block_name}",
            overrides=overrides,
            upstream_outputs=self._visible_outputs_before_priority(node.execution_priority),
            parent_config=self._ancestor_config_values(),
        )[0]

    def run_block(self, block_name: str, overrides: dict[str, Any] | None = None) -> RunRecord:
        node = self._get_node_or_raise(block_name)
        self._invalidate_from_priority(node.execution_priority)
        return self._execute_nodes(
            [node],
            mode=f"run_block:{block_name}",
            overrides=overrides,
            upstream_outputs=self._visible_outputs_before_priority(node.execution_priority),
            parent_config=self._ancestor_config_values(),
        )[0]

    def save_project(self, path: str | Path | None = None) -> Path:
        warnings.warn(
            "Saved pipelines preserve import paths, not historical function behavior; later source changes may affect reloaded pipelines.",
            stacklevel=2,
        )
        target = self.project_root if path is None else Path(path)
        target.mkdir(parents=True, exist_ok=True)
        payload = self._serialize_payload()
        with (target / "pipeline_state.pkl").open("wb") as handle:
            pickle.dump(payload, handle)
        with (target / "config.pkl").open("wb") as handle:
            pickle.dump(self.config, handle)
        return target

    @classmethod
    def load_project(cls, path: str | Path) -> "PipelineHandler":
        warnings.warn(
            "Loaded pipelines restore current importable functions, not historical function snapshots; changed source code may alter behavior.",
            stacklevel=2,
        )
        target = Path(path)
        try:
            with (target / "pipeline_state.pkl").open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            raise PersistenceError("Failed to load pipeline project") from exc
        return cls._from_payload(payload, target)

    @classmethod
    def _from_payload(
        cls,
        payload: dict[str, Any],
        project_root: Path,
        parent: "PipelineHandler | None" = None,
    ) -> "PipelineHandler":
        pipeline = cls(
            registration_name=payload["registration_name"],
            configuration=payload["config"],
            local_folder_path=project_root,
            execution_priority=payload.get("execution_priority"),
        )
        pipeline.historical_result_log_path = payload.get("historical_result_log_path")
        if payload.get("gate") is not None:
            pipeline.set_gate_block(payload["gate"]["import_path"])
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
                for function_payload in node_payload["functions"]:
                    registration = block._register_function_strict(
                        function_payload["import_path"],
                        function_payload["output_names"],
                        function_payload["save_to_disk"],
                        kw_mapping=function_payload.get("kw_mapping"),
                        var_pos_name=function_payload.get("var_pos_name"),
                        var_kw_name=function_payload.get("var_kw_name"),
                    )
                    if registration is None:
                        raise PersistenceError(
                            f"Failed to restore function in block '{block.registration_name}'"
                        )
            else:
                child = cls._from_payload(node_payload["payload"], project_root, parent=pipeline)
                child.execution_priority = node_payload["execution_priority"]
                child.parent_pipeline = pipeline
                child.logger = pipeline.logger
                pipeline._register_node(child)
        pipeline.producer_outputs = payload.get("producer_outputs", {})
        pipeline.para_value_dict = payload.get("para_value_dict", {})
        pipeline.artifact_registry = payload.get("artifact_registry", {})
        pipeline.run_history = payload.get("run_history", [])
        if parent is not None:
            pipeline.parent_pipeline = parent
            pipeline.logger = parent.logger
        return pipeline

    def _serialize_payload(self) -> dict[str, Any]:
        return {
            "registration_name": self.registration_name,
            "config": self.config,
            "execution_priority": self.execution_priority,
            "historical_result_log_path": self.historical_result_log_path,
            "gate": None if self.gate_block is None else self.gate_block.serialize(),
            "nodes": [self._serialize_node(node) for node in self._sorted_nodes()],
            "producer_outputs": self.producer_outputs,
            "para_value_dict": self.para_value_dict,
            "artifact_registry": self.artifact_registry,
            "run_history": self.run_history,
        }

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
            if registration.import_path is None:
                raise PersistenceError(
                    f"Function '{registration.function_name}' is not importable; save/load requires importable callables"
                )
            functions.append(
                {
                    "import_path": registration.import_path,
                    "output_names": registration.output_names,
                    "save_to_disk": sorted(registration.save_to_disk),
                    "kw_mapping": registration.kw_mapping,
                    "var_pos_name": registration.var_pos_name,
                    "var_kw_name": registration.var_kw_name,
                }
            )
        return {
            "kind": "block",
            "registration_name": node.registration_name,
            "execution_priority": node.execution_priority,
            "functions": functions,
        }

    def _execute_nodes(
        self,
        nodes: list[Any],
        mode: str,
        overrides: dict[str, Any] | None = None,
        upstream_outputs: dict[str, Any] | None = None,
        parent_config: dict[str, Any] | None = None,
        sync_parent_on_completion: bool = True,
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
                for node in nodes:
                    self._delete_artifacts_from_outputs(
                        self.producer_outputs.pop(node.registration_name, {})
                    )
                    if isinstance(node, PipelineHandler):
                        node._invalidate_all_outputs()
                self.para_value_dict = skipped_outputs
                self.artifact_registry = {}
                run_record.status = "skipped"
                run_record.produced_outputs.extend(sorted(skipped_outputs))
                if sync_parent_on_completion:
                    self._sync_attached_outputs_to_parent()
                self.logger.warning(f"Skipped {mode} with run_id={run_id}")
                return run_record, skipped_outputs

            for node in nodes:
                visible_outputs = self._visible_outputs_before_priority(
                    node.execution_priority,
                    upstream_outputs=upstream_outputs,
                )
                if isinstance(node, PipelineHandler):
                    _, produced_outputs = node._execute_nodes(
                        node._sorted_nodes(),
                        mode=f"run_child:{node.registration_name}",
                        overrides=overrides,
                        upstream_outputs=visible_outputs,
                        parent_config=self.config_as_dict(),
                        sync_parent_on_completion=False,
                    )
                else:
                    produced_outputs = node.execute(
                        run_id,
                        visible_outputs,
                        overrides=overrides,
                        parent_config=parent_config or {},
                    )
                self.producer_outputs[node.registration_name] = produced_outputs
                run_record.executed_blocks.append(node.registration_name)
                run_record.produced_outputs.extend(produced_outputs.keys())

            run_record.status = "success"
            self._rebuild_visible_state(upstream_outputs)
            if sync_parent_on_completion:
                self._sync_attached_outputs_to_parent()
            self.logger.info(f"Completed {mode} with run_id={run_id}")
            return run_record, dict(self.para_value_dict)
        except Exception as exc:
            run_record.status = "failed"
            run_record.error_message = str(exc)
            self.logger.error(f"Failed {mode} with run_id={run_id}: {exc}")
            if isinstance(exc, (ExecutionError, ResolutionError, RegistrationError)):
                raise
            raise ExecutionError("Pipeline execution failed") from exc
        finally:
            run_record.finished_at = datetime.now(UTC).isoformat()

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
        conflicts = set(output_names).intersection(self._visible_config_names())
        if conflicts:
            raise RegistrationError(
                f"Output names conflict with visible configuration fields: {sorted(conflicts)}"
            )

    def _validate_node_registration(self, node: Any, execution_priority: int | None) -> None:
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

    def _get_node_or_raise(self, block_name: str) -> Any:
        node = self.nodes_by_name.get(block_name)
        if node is None:
            raise RegistrationError(f"Node not registered: {block_name}")
        return node

    def _sorted_nodes(self) -> list[Any]:
        return sorted(
            self.nodes, key=lambda node: (node.execution_priority, node.registration_name)
        )

    def _node_declared_outputs(self, node: Any) -> set[str]:
        if isinstance(node, PipelineHandler):
            return node.list_declared_outputs()
        return node.declared_outputs()

    def _rebuild_visible_state(self, upstream_outputs: dict[str, Any] | None = None) -> None:
        all_artifacts: list[ArtifactRecord] = []
        visible = dict(upstream_outputs or {})
        for node in self._sorted_nodes():
            produced_outputs = self.producer_outputs.get(node.registration_name, {})
            for value in produced_outputs.values():
                if isinstance(value, ArtifactRecord):
                    all_artifacts.append(value)
            visible.update(produced_outputs)
        declared_outputs = self.list_declared_outputs()
        self.para_value_dict = {
            output_name: visible[output_name]
            for output_name in declared_outputs
            if output_name in visible
        }
        self.artifact_registry = {
            output_name: value
            for output_name, value in self.para_value_dict.items()
            if isinstance(value, ArtifactRecord)
        }
        active_artifacts = {
            id(value)
            for value in self.para_value_dict.values()
            if isinstance(value, ArtifactRecord)
        }
        for artifact in all_artifacts:
            if id(artifact) in active_artifacts:
                continue
            self.artifact_store.delete(artifact)

    def _visible_outputs_before_priority(
        self,
        priority: int | None,
        upstream_outputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        visible = dict(upstream_outputs or self._incoming_parent_outputs())
        if priority is None:
            return visible
        for node in self._sorted_nodes():
            if node.execution_priority >= priority:
                break
            visible.update(self.producer_outputs.get(node.registration_name, {}))
        return visible

    def _incoming_parent_output_names(self) -> set[str]:
        if self.parent_pipeline is None or self.execution_priority is None:
            return set()
        return self.parent_pipeline._declared_output_names_before_priority(self.execution_priority)

    def _declared_output_names_before_priority(self, priority: int | None) -> set[str]:
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

    def _visible_config_names(self) -> set[str]:
        return set(self.config_as_dict()).union(self._ancestor_config_values())

    def _prepare_call_arguments(
        self,
        registration: FunctionRegistration,
        overrides: dict[str, Any],
        visible_outputs: dict[str, Any],
        parent_config: dict[str, Any] | None = None,
    ) -> tuple[list[Any], dict[str, Any], list[str]]:
        defaults = default_map(registration.callable_obj)
        signature = inspect.signature(registration.callable_obj)
        parameters = list(signature.parameters.values())
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
                input_name = parameter.name
                value = self._resolve_named_input(
                    input_name,
                    registration.function_name,
                    overrides,
                    visible_outputs,
                    parent_config,
                    defaults,
                    loaded_artifacts,
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
                input_name = parameter.name
                value = self._resolve_named_input(
                    input_name,
                    registration.function_name,
                    overrides,
                    visible_outputs,
                    parent_config,
                    defaults,
                    loaded_artifacts,
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

            input_name = parameter.name
            value = self._resolve_named_input(
                input_name,
                registration.function_name,
                overrides,
                visible_outputs,
                parent_config,
                defaults,
                loaded_artifacts,
            )

            if parameter.kind == inspect.Parameter.POSITIONAL_ONLY or (
                parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                and var_pos_index is not None
                and index < var_pos_index
            ):
                positional_args.append(value)
            else:
                keyword_args[input_name] = value
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

    def _set_config_value(self, field_name: str, value: Any) -> None:
        if is_dataclass(self.config) and not isinstance(self.config, type):
            setattr(self.config, field_name, value)
            return
        if isinstance(self.config, dict):
            self.config[field_name] = value
            return
        setattr(self.config, field_name, value)

    def _invalidate_from_priority(self, priority: int, include_target: bool = True) -> None:
        if priority is None:
            return
        for node in list(self._sorted_nodes()):
            if node.execution_priority < priority:
                continue
            if node.execution_priority == priority and not include_target:
                continue
            self._delete_artifacts_from_outputs(
                self.producer_outputs.pop(node.registration_name, {})
            )
            if isinstance(node, PipelineHandler):
                node._invalidate_all_outputs()
        self._rebuild_visible_state(self._incoming_parent_outputs())

    def _invalidate_all_outputs(self) -> None:
        for outputs in self.producer_outputs.values():
            self._delete_artifacts_from_outputs(outputs)
        self.producer_outputs.clear()
        self.para_value_dict.clear()
        self.artifact_registry.clear()
        for node in self._sorted_nodes():
            if isinstance(node, PipelineHandler):
                node._invalidate_all_outputs()

    def _delete_artifacts_from_outputs(self, outputs: dict[str, Any]) -> None:
        for value in outputs.values():
            if isinstance(value, ArtifactRecord):
                self.artifact_store.delete(value)

    def _persist_config_snapshot(self, path: Path) -> None:
        with path.open("wb") as handle:
            pickle.dump(self.config, handle)

    def _attach_to_parent(self, parent: "PipelineHandler", execution_priority: int) -> None:
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
        moved_log_path = target_root / "metadata" / "pipeline.log"
        self.historical_result_log_path = str(moved_log_path)
        self.project_root = target_root
        self.metadata_root = target_root / "metadata"
        self.metadata_root.mkdir(parents=True, exist_ok=True)
        self.artifact_store = ArtifactStore(target_root)
        self.parent_pipeline = parent
        self.execution_priority = execution_priority
        self.logger = parent.logger
        self._rewrite_artifact_paths(original_root, target_root)

    def _sync_attached_outputs_to_parent(self) -> None:
        if self.parent_pipeline is None or self.execution_priority is None:
            return
        self.parent_pipeline.producer_outputs[self.registration_name] = dict(self.para_value_dict)
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

    def qualified_node_name(self, node_name: str) -> str:
        return f"{self.full_path()}/{node_name}"

    def full_path(self) -> str:
        if self.parent_pipeline is None:
            return self.registration_name
        return f"{self.parent_pipeline.full_path()}/{self.registration_name}"

    def _describe_lines(self, indent: str = "", as_child: bool = False) -> list[str]:
        lines: list[str] = []
        if as_child:
            lines.append(
                f"{indent}{self._color('pipeline', 'magenta')} {self._color(f'[{self.execution_priority}]', 'cyan')} {self._color(self.registration_name, 'blue')}"
            )
        else:
            lines.append(
                f"{indent}{self._color('PipelineHandler', 'green')}({self._color(self.registration_name, 'blue')})"
            )
        if self.gate_block is not None:
            gate_args = self._color(
                ", ".join(self._displayed_argument_names(self.gate_block.registration, None)),
                "yellow",
            )
            lines.append(
                f"{indent}├── {self._color('[gate]', 'magenta')} {self._color(self.gate_block.registration.function_name, 'green')}({gate_args}) -> bool"
            )
        sorted_nodes = self._sorted_nodes()
        for index, node in enumerate(sorted_nodes):
            is_last = index == len(sorted_nodes) - 1
            prefix = "└──" if is_last else "├──"
            child_indent = indent + ("    " if is_last else "│   ")
            if isinstance(node, PipelineHandler):
                lines.append(
                    f"{indent}{prefix} {self._color('child-pipeline', 'magenta')} {self._color(f'[{node.execution_priority}]', 'cyan')} {self._color(node.registration_name, 'blue')}"
                )
                lines.extend(node._describe_lines(child_indent, as_child=True)[1:])
                continue
            lines.append(
                f"{indent}{prefix} {self._color(f'[{node.execution_priority}]', 'cyan')} {self._color(node.registration_name, 'blue')}"
            )
            for function_index, registration in enumerate(node.functions):
                function_prefix = "└──" if function_index == len(node.functions) - 1 else "├──"
                outputs = [
                    (
                        self._color(f"{output_name}*", "red")
                        if output_name in registration.save_to_disk
                        else self._color(output_name, "green")
                    )
                    for output_name in registration.output_names
                ]
                args = self._color(
                    ", ".join(
                        self._displayed_argument_names(
                            registration,
                            node.execution_priority,
                        )
                    ),
                    "yellow",
                )
                lines.append(
                    f"{child_indent}{function_prefix} {self._color(registration.function_name, 'green')}({args})"
                    + (f" -> {', '.join(outputs)}" if outputs else "")
                )
        return lines

    def _displayed_argument_names(
        self,
        registration: FunctionRegistration,
        priority: int | None,
    ) -> list[str]:
        visible_output_names = self._declared_output_names_before_priority(priority)
        visible_config_names = self._visible_config_names()
        return [
            name
            for name in registration.input_names
            if name in visible_output_names or name in visible_config_names
        ]

    def _read_result_history_from_file(self, file_path: str) -> list[str]:
        path = Path(file_path)
        if not path.exists():
            return []
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if "[RESULT]" in line
        ]

    def _color(self, text: str, color: str) -> str:
        return import_module("termcolor").colored(text, color, force_color=True)

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
        with redirect_stdout(stdout_target):
            result = func(*args, **kwargs)
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
            return asdict(self.config)
        if isinstance(self.config, dict):
            return dict(self.config)
        if hasattr(self.config, "__dict__"):
            return dict(vars(self.config))
        raise PersistenceError("Configuration object is not serializable to dict")


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
