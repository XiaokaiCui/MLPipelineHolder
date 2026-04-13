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
from .models import ArtifactRecord, FunctionRegistration, RunRecord, RuntimeValueReference, TorchStateArtifactRecord


class PipelineHandler:
    def __init__(
        self,
        registration_name: str,
        configuration: Any | None,
        local_folder_path: str | Path,
        execution_priority: float | None = None,
    ) -> None:
        self.registration_name = registration_name
        self.config = {} if configuration is None else configuration
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

    def add_block(
        self, registration_name: str, execution_priority: float, forced: bool = False
    ) -> Any:
        from .execution_block import ExecutionBlock

        block = ExecutionBlock(self, registration_name, execution_priority)
        conflicts = self._registration_conflicts(block, execution_priority)
        self._raise_on_priority_conflict_with_different_name(
            registration_name,
            execution_priority,
            conflicts,
        )
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
        if child_pipeline is self:
            raise RegistrationError("A pipeline cannot register itself as a child pipeline")
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
        self._validate_node_registration(child_pipeline, execution_priority)
        self._validate_output_names_against_config(sorted(child_pipeline.list_declared_outputs()))
        child_pipeline._attach_to_parent(self, execution_priority)
        self._register_node(child_pipeline)
        return child_pipeline

    def add_gate_block(
        self, function_or_path: Any, expected_value: Any = True, forced: bool = False
    ) -> Any:
        if self.gate_block is not None and not forced:
            self.logger.warning("Skipped gate block registration: gate block already exists")
            return None
        self.gate_block = GateBlock(self, function_or_path, expected_value=expected_value)
        self._invalidate_all_outputs()
        return self.gate_block

    def set_gate_block(
        self, function_or_path: Any, expected_value: Any = True, forced: bool = False
    ) -> Any:
        return self.add_gate_block(function_or_path, expected_value=expected_value, forced=forced)

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
        if isinstance(value, TorchStateArtifactRecord):
            return value
        if isinstance(value, ArtifactRecord):
            return self.artifact_store.load(value)
        return value

    def get_full_config(self) -> dict[str, Any]:
        return dict(self._ancestor_config_values(), **self.config_as_dict())

    def get_config_value(self, field_name: str) -> Any:
        config = self.get_full_config()
        if field_name not in config:
            raise ResolutionError(f"Unknown config field: {field_name}")
        return config[field_name]

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

    def reset_gate_block(self) -> None:
        if self.gate_block is None:
            return
        self.gate_block = None
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

    def run_until(self, *path_parts: str, overrides: dict[str, Any] | None = None) -> RunRecord:
        if len(path_parts) > 1:
            return self._run_nested_until_path(path_parts, overrides=overrides)
        pipeline, node = self._resolve_target_path(path_parts)
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
        snapshot = self._snapshot_runtime_state()
        previous_outputs = snapshot[0].get(node.registration_name, {})
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
        snapshot = self._snapshot_runtime_state()
        previous_outputs = snapshot[0].get(node.registration_name, {})
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
    ) -> Path:
        warnings.warn(
            "Saved pipelines preserve import paths, not historical function behavior; later source changes may affect reloaded pipelines.",
            stacklevel=2,
        )
        target = self.project_root if path is None else Path(path)
        target.mkdir(parents=True, exist_ok=True)
        try:
            payload = self._serialize_payload_for_save(target)
        except RegistrationError as exc:
            raise PersistenceError(str(exc)) from exc
        with (target / "pipeline_state.pkl").open("wb") as handle:
            pickle.dump(payload, handle)
        with (target / "config.pkl").open("wb") as handle:
            pickle.dump(self.config, handle)
        if save_log_to_file is not None:
            self.logger.flush()
            log_target = Path(save_log_to_file)
            log_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.logger.log_file_path, log_target)
        return target

    def save_project(
        self,
        path: str | Path | None = None,
        save_log_to_file: str | Path | None = None,
    ) -> Path:
        return self.save_pipeline(path, save_log_to_file=save_log_to_file)

    @classmethod
    def load_pipeline(cls, path: str | Path) -> "PipelineHandler":
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
    def load_project(cls, path: str | Path) -> "PipelineHandler":
        return cls.load_pipeline(path)

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
                for function_payload in node_payload["functions"]:
                    registration = block._register_function_strict(
                        function_payload["import_path"],
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
            else:
                child_root = project_root / "children" / node_payload["registration_name"]
                child = cls._from_payload(node_payload["payload"], child_root, parent=pipeline)
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

    def _serialize_payload_for_save(
        self,
        target_root: Path,
        cache: dict[int, Any] | None = None,
    ) -> dict[str, Any]:
        cache = {} if cache is None else cache
        return {
            "registration_name": self.registration_name,
            "config": self.config,
            "execution_priority": self.execution_priority,
            "historical_result_log_path": self.historical_result_log_path,
            "gate": None if self.gate_block is None else self.gate_block.serialize(),
            "nodes": [self._serialize_node_for_save(node, target_root, cache) for node in self._sorted_nodes()],
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
            import torch  # type: ignore

            if isinstance(value, torch.nn.Module) or isinstance(value, torch.Tensor):
                save_store = ArtifactStore(target_root)
                return save_store.save(
                    variable_name=output_name,
                    value=value,
                    block_name=self.qualified_node_name(node_name),
                    function_name="save_pipeline_runtime",
                    run_id="save_pipeline",
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
        except Exception:
            pass

        try:
            pickle.dumps(value)
            return value
        except Exception:
            warnings.warn(
                f"Runtime value '{node_name}.{output_name}' could not be serialized directly; saving a reference placeholder instead.",
                stacklevel=2,
            )
            return RuntimeValueReference(
                type_name=type(value).__name__,
                repr_text=repr(value),
                reason="not directly serializable during save_pipeline",
            )

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
                    "param_mapping": registration.param_mapping,
                    "var_pos_name": registration.var_pos_name,
                    "var_kw_name": registration.var_kw_name,
                }
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
                priority_group = self._priority_group(node.execution_priority)
                if priority_group in executed_priority_groups:
                    self._delete_artifacts_from_outputs(
                        self.producer_outputs.pop(node.registration_name, {})
                    )
                    if isinstance(node, PipelineHandler):
                        node._invalidate_all_outputs()
                    continue
                visible_outputs = self._visible_outputs_before_priority(
                    node.execution_priority,
                    upstream_outputs=upstream_outputs,
                )
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
                run_record.executed_blocks.append(node.registration_name)
                run_record.produced_outputs.extend(produced_outputs.keys())
                if node_executed:
                    executed_priority_groups.add(priority_group)

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
            self._remove_registered_node(node)
        self._invalidate_from_priority(earliest_priority)

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
        child_run = child_pipeline.run_from(*path_parts[1:], overrides=overrides)
        self.producer_outputs[child_pipeline.registration_name] = dict(child_pipeline.para_value_dict)
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
        priority: float | None,
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
        block: Any | None = None,
    ) -> tuple[list[Any], dict[str, Any], list[str]]:
        defaults = default_map(registration.callable_obj)
        signature = inspect.signature(registration.callable_obj)
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
        elif self._config_has_field(self.config, input_name):
            value = self._config_value(self.config, input_name)
        elif parent_config and input_name in parent_config:
            value = parent_config[input_name]
        elif input_name in defaults:
            value = defaults[input_name]
        elif input_name in declared_output_names:
            value = None
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

    def _invalidate_from_priority(self, priority: float, include_target: bool = True) -> None:
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
        self._rewrite_artifact_paths(original_root, target_root)
        self._rewrite_run_history_paths(original_root, target_root)
        self._refresh_descendant_roots(original_root, target_root)

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
                "yellow",
                muted,
            )
            gate_line = (
                f"{self._line_style(f'{indent}├── ', muted)}{self._chart_color('[gate]', 'magenta', muted)} {self._chart_color(self.gate_block.registration.function_name, 'green', muted)}"
                f"{self._chart_color('(', symbol_color, muted)}{gate_args}{self._chart_color(')', symbol_color, muted)}"
            )
            lines.append(gate_line)
        sorted_nodes = self._sorted_nodes()
        for index, node in enumerate(sorted_nodes):
            is_last = index == len(sorted_nodes) - 1
            prefix = "└──" if is_last else "├──"
            child_indent = indent + ("    " if is_last else "│   ")
            if isinstance(node, PipelineHandler):
                child_muted = muted or node._should_grey_in_chart()
                child_line = (
                    f"{self._line_style(f'{indent}{prefix} ', child_muted)}{self._chart_color('child-pipeline', 'magenta', child_muted)} {self._chart_color(f'[{node.execution_priority}]', 'cyan', child_muted)} {self._chart_color(node.registration_name, 'blue', child_muted)}"
                )
                lines.append(child_line)
                lines.extend(node._describe_lines(child_indent, as_child=True, muted=child_muted)[1:])
                continue
            block_line = (
                f"{self._line_style(f'{indent}{prefix} ', muted)}{self._chart_color(f'[{node.execution_priority}]', 'cyan', muted)} {self._chart_color(node.registration_name, 'blue', muted)}"
            )
            lines.append(block_line)
            for function_index, registration in enumerate(node.functions):
                function_prefix = "└──" if function_index == len(node.functions) - 1 else "├──"
                outputs = [
                    (
                        self._chart_color(f"{output_name}*", "red", muted)
                        if output_name in registration.save_to_disk
                        else self._chart_color(output_name, "green", muted)
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
                    "yellow",
                    muted,
                )
                function_line = (
                    f"{self._line_style(f'{child_indent}{function_prefix} ', muted)}{self._chart_color(registration.function_name, 'green', muted)}"
                    f"{self._chart_color('(', symbol_color, muted)}{args}{self._chart_color(')', symbol_color, muted)}"
                    + (
                        f" {self._chart_color('->', symbol_color, muted)} {', '.join(outputs)}"
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

    def _displayed_argument_names(
        self,
        registration: FunctionRegistration,
        priority: float | None,
        block: Any | None = None,
    ) -> list[str]:
        visible_output_names = self._declared_output_names_before_priority(priority)
        visible_config_names = self._visible_config_names()
        displayed = [
            name
            for name in registration.input_names
            if name in visible_output_names or name in visible_config_names
        ]
        if (
            block is not None
            and registration.var_pos_name is not None
            and registration.var_pos_name in block.registered_args
        ):
            displayed.append(registration.var_pos_name)
        if (
            block is not None
            and registration.var_kw_name is not None
            and registration.var_kw_name in block.registered_kwargs
        ):
            displayed.append(registration.var_kw_name)
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

    def _snapshot_runtime_state(
        self,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, ArtifactRecord]]:
        return (
            {name: dict(outputs) for name, outputs in self.producer_outputs.items()},
            dict(self.para_value_dict),
            dict(self.artifact_registry),
        )

    def _restore_runtime_state(
        self,
        snapshot: tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, ArtifactRecord]],
    ) -> None:
        producer_outputs, para_values, artifacts = snapshot
        self.producer_outputs = {name: dict(outputs) for name, outputs in producer_outputs.items()}
        self.para_value_dict = dict(para_values)
        self.artifact_registry = dict(artifacts)


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
