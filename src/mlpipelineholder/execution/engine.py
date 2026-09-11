"""Execution navigation entry points: run_all, run_until, run_from, run_block."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..core.models import ArtifactRecord, RunRecord
from ..exceptions import ExecutionError, RegistrationError, ResolutionError
from ..integrations.optuna.api import StudyArtifactOptions
from ..integrations.optuna.support import is_optuna_study
from ..state.output_pointers import OutputAddress, OutputPointer

_holder_base: type | None = None


def _register_engine_holder_base(holder_base: type) -> None:
    global _holder_base
    _holder_base = holder_base


def _is_child_pipeline(node: object) -> bool:
    return _holder_base is not None and isinstance(node, _holder_base)


class EngineMixin:
    """Public run entry points shared by the PipelineHolder facade."""

    if TYPE_CHECKING:
        nodes: list[Any] = []

        def _gate_skip_without_cleanup(
            self,
            mode: str,
            overrides: dict[str, Any] | None,
            base_visible: dict[str, Any],
            parent_config: Any | None,
        ) -> bool: ...
        def _build_skipped_run_record(self, mode: str) -> RunRecord: ...
        def _invalidate_from_priority(
            self,
            priority: float,
            include_target: bool = True,
            preserve_pipeline_state: Any = None,
        ) -> None: ...
        def _sorted_nodes(self) -> list[Any]: ...
        def _incoming_parent_outputs(self) -> dict[str, Any]: ...
        def _visible_outputs_before_priority(
            self,
            priority: float | None,
            upstream_outputs: dict[str, Any] | None = None,
        ) -> dict[str, Any]: ...
        def _ancestor_config_values(self) -> dict[str, Any]: ...
        def _execute_nodes(
            self,
            nodes: list[Any],
            mode: str,
            overrides: dict[str, Any] | None = None,
            upstream_outputs: dict[str, Any] | None = None,
            parent_config: dict[str, Any] | None = None,
            sync_parent_on_completion: bool = True,
            previous_node_outputs: dict[str, dict[str, Any]] | None = None,
        ) -> tuple[RunRecord, dict[str, Any]]: ...
        def _resolve_target_path(self, path_parts: tuple[str, ...]) -> tuple[Any, Any]: ...
        def _run_nested_until_path(
            self, path_parts: tuple[str, ...], overrides: dict[str, Any] | None = None
        ) -> RunRecord: ...
        def _run_nested_from_path(
            self, path_parts: tuple[str, ...], overrides: dict[str, Any] | None = None
        ) -> RunRecord: ...
        def _run_nested_block_path(
            self, path_parts: tuple[str, ...], overrides: dict[str, Any] | None = None
        ) -> RunRecord: ...
        def _snapshot_runtime_state(self) -> Any: ...
        def _materialize_previous_node_inputs(
            self,
            node: Any,
            previous_outputs: dict[str, Any],
            overrides: dict[str, Any] | None,
        ) -> dict[str, Any]: ...

        metadata_root: Any = None
        run_history: list[Any] = []
        logger: Any = None
        gate_block: Any = None
        _gate_cleanup_predecided: bool | None = None
        gate_cleanup_confirmation: bool = False
        producer_outputs: dict[str, dict[str, Any]] = {}
        artifact_registry: dict[str, Any] = {}
        para_value_dict: dict[str, Any] = {}
        manual_values: dict[str, Any] = {}
        _is_atom: bool = False
        parent_pipeline: Any = None
        registration_name: str = ""
        artifact_store: Any = None
        _optuna_study_group_names: dict[str, Any] = {}
        _optuna_study_group_addresses: dict[str, Any] = {}
        _invalidation_forbidden: bool = False
        memory_profile_logging: bool = False
        memory_saving_mode: bool = False

        def _persist_config_snapshot(self, path: Path) -> None: ...
        def list_declared_outputs(self) -> set[str]: ...
        def _confirm_gate_cleanup(self, mode: str) -> bool: ...
        def _delete_artifacts_from_outputs(self, outputs: dict[str, Any]) -> None: ...
        def _sync_attached_outputs_to_parent(self) -> None: ...
        def _priority_group(self, execution_priority: float | None) -> int: ...
        def _rebuild_visible_state(
            self, upstream_outputs: dict[str, Any] | None = None
        ) -> None: ...
        def get_full_config(self) -> dict[str, Any]: ...
        def _commit_node_outputs(
            self,
            node: Any,
            produced_outputs: dict[str, Any],
            upstream_outputs: dict[str, Any] | None,
        ) -> None: ...
        def _cleanup_block_memory(self, node_name: str) -> None: ...
        def _log_memory_profile(self, node_name: str, phase: str = "after_cleanup") -> None: ...
        @staticmethod
        def _is_optuna_study_output(value: Any) -> bool: ...
        def _node_overridden_outputs(self, node: Any) -> Any: ...
        def _pipeline_by_name(self, pipeline_name: str) -> Any: ...
        def _node_declared_outputs(self, node: Any) -> set[str]: ...
        def _root_pipeline(self) -> Any: ...
        def _remember_optuna_study_groups(self) -> None: ...
        def qualified_node_name(self, node_name: str) -> str: ...
        def _same_priority_group_address(
            self, first: OutputAddress, second: OutputAddress
        ) -> bool: ...
        def _resync_mirror_to_parent(self) -> None: ...
        def _refresh_pointer_visible_state(self) -> None: ...
        def _cleanup_replaced_artifact(self, previous_value: Any) -> None: ...

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
                        if _is_child_pipeline(node):
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
                    if _is_child_pipeline(node):
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
                if _is_child_pipeline(node):
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
                self._commit_node_outputs(node, produced_outputs, upstream_outputs)
                if self.memory_profile_logging and not _is_child_pipeline(node):
                    self._log_memory_profile(node.registration_name, phase="after_compute")
                run_record.executed_blocks.append(node.registration_name)
                run_record.produced_outputs.extend(produced_outputs.keys())
                if node_executed:
                    executed_priority_groups.add(priority_group)
                if self.memory_saving_mode:
                    if not _is_child_pipeline(node):
                        self._cleanup_block_memory(node.registration_name)
                elif self.memory_profile_logging and not _is_child_pipeline(node):
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

    def _commit_node_outputs(
        self,
        node: Any,
        produced_outputs: dict[str, Any],
        upstream_outputs: dict[str, Any] | None,
    ) -> None:
        study_output_names = {
            output_name
            for output_name, value in produced_outputs.items()
            if self._is_optuna_study_output(value)
        }
        if _is_child_pipeline(node):
            terminal_pipeline: Any = None
            terminal_node_name: str | None = None
        elif self._is_atom and self.parent_pipeline is not None:
            terminal_pipeline = self.parent_pipeline
            terminal_node_name = self.registration_name
        else:
            terminal_pipeline = self
            terminal_node_name = node.registration_name
        current_address = (
            {
                output_name: OutputAddress(
                    terminal_pipeline.registration_name,
                    terminal_node_name,
                    output_name,
                )
                for output_name in produced_outputs
                if output_name in study_output_names
            }
            if terminal_pipeline is not None and terminal_node_name is not None
            else {}
        )
        node_overrides = self._node_overridden_outputs(node)
        overrides = {
            output_name: address
            for output_name, address in node_overrides.items()
            if output_name not in study_output_names
        }
        targets: list[tuple[str, Any, OutputAddress]] = []
        try:
            for output_name, address in overrides.items():
                if output_name not in produced_outputs:
                    continue
                target_pipeline = self._pipeline_by_name(address.pipeline_name)
                if (
                    address.node_name not in target_pipeline.nodes_by_name
                    or output_name
                    not in target_pipeline._node_declared_outputs(
                        target_pipeline.nodes_by_name[address.node_name]
                    )
                ):
                    raise ExecutionError(
                        f"Output pointer target is no longer valid: "
                        f"{address.pipeline_name}.{address.node_name}.{output_name}"
                    )
                targets.append((output_name, target_pipeline, address))
        except BaseException:
            self._delete_artifacts_from_outputs(produced_outputs)
            raise

        prepared_outputs = dict(produced_outputs)
        root = self._root_pipeline()
        root._remember_optuna_study_groups()
        try:
            for output_name, address in current_address.items():
                value = prepared_outputs[output_name]
                group_name = root._optuna_study_group_names.get(output_name)
                if is_optuna_study(value):
                    prepared_outputs[output_name] = self.artifact_store.save(
                        variable_name=output_name,
                        value=value,
                        block_name=self.qualified_node_name(node.registration_name),
                        function_name="study_output",
                        run_id=uuid4().hex,
                        optuna_db_path=root.optuna_studies_db_path,
                        optuna_options=StudyArtifactOptions(
                            allocate_if_occupied=group_name is None,
                            study_name=None if group_name is None else group_name[0],
                            original_study_name=(
                                None if group_name is None else group_name[1]
                            ),
                            owner_kind="output",
                            owner_key=(
                                f"{address.pipeline_name}.{address.node_name}."
                                f"{address.output_name}"
                            ),
                        ),
                    )
                artifact = prepared_outputs[output_name]
                if isinstance(artifact, ArtifactRecord):
                    study_name = artifact.metadata.get("study_name")
                    original_name = artifact.metadata.get(
                        "original_study_name",
                        study_name,
                    )
                    if isinstance(study_name, str) and isinstance(original_name, str):
                        root._optuna_study_group_names[output_name] = (
                            study_name,
                            original_name,
                        )
                root._optuna_study_group_addresses.setdefault(
                    output_name,
                    set(),
                ).add(address)
        except BaseException:
            self._delete_artifacts_from_outputs(prepared_outputs)
            raise

        affected: dict[int, Any] = {id(self): self}
        for _, pipeline, _ in targets:
            affected[id(pipeline)] = pipeline
        for output_name in current_address:
            for address in root._optuna_study_group_addresses.get(output_name, set()):
                pipeline = root._pipeline_by_name(address.pipeline_name)
                affected[id(pipeline)] = pipeline
        snapshots = {
            identity: {
                name: dict(outputs)
                for name, outputs in pipeline.producer_outputs.items()
            }
            for identity, pipeline in affected.items()
        }
        replaced: list[Any] = list(
            self.producer_outputs.get(node.registration_name, {}).values()
        )
        try:
            self.producer_outputs[node.registration_name] = dict(prepared_outputs)
            for output_name, target_pipeline, address in targets:
                target_outputs = target_pipeline.producer_outputs.setdefault(
                    address.node_name,
                    {},
                )
                if output_name in target_outputs:
                    replaced.append(target_outputs[output_name])
                target_outputs[output_name] = OutputPointer(
                    OutputAddress(
                        self.registration_name,
                        node.registration_name,
                        output_name,
                    )
                )
            for output_name, terminal_address in current_address.items():
                for address in root._optuna_study_group_addresses[output_name]:
                    if address == terminal_address:
                        continue
                    if self._same_priority_group_address(address, terminal_address):
                        continue
                    target_pipeline = root._pipeline_by_name(address.pipeline_name)
                    target_outputs = target_pipeline.producer_outputs.setdefault(
                        address.node_name,
                        {},
                    )
                    if output_name in target_outputs:
                        replaced.append(target_outputs[output_name])
                    target_outputs[output_name] = OutputPointer(terminal_address)
            for _, target_pipeline, _ in targets:
                target_pipeline._rebuild_visible_state(
                    target_pipeline._incoming_parent_outputs()
                )
                target_pipeline._resync_mirror_to_parent()
            self._rebuild_visible_state(upstream_outputs)
            self._resync_mirror_to_parent()
            self._refresh_pointer_visible_state()
        except BaseException:
            for identity, pipeline in affected.items():
                pipeline.producer_outputs = {
                    name: dict(outputs)
                    for name, outputs in snapshots[identity].items()
                }
            self._refresh_pointer_visible_state()
            self._delete_artifacts_from_outputs(prepared_outputs)
            raise
        for previous in replaced:
            self._cleanup_replaced_artifact(previous)
        for output_name, terminal_address in current_address.items():
            artifact = prepared_outputs[output_name]
            if not isinstance(artifact, ArtifactRecord):
                continue
            study_name = artifact.metadata.get("study_name")
            redirected_count = len(
                root._optuna_study_group_addresses[output_name] - {terminal_address}
            )
            self.logger.info(
                f"Optuna Study output group rewired: output={output_name!r}, "
                f"study={study_name!r}, terminal={terminal_address!r}, "
                f"redirected={redirected_count}, "
                f"invalidation_forbidden={root._invalidation_forbidden}, "
                f"override_ignored={output_name in node_overrides}"
            )
