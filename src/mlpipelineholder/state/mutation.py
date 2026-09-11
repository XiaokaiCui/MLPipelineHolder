"""Mutation state: setting constants and values, injection, and mirror sync."""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..core.models import ArtifactRecord, TorchStateArtifactRecord
from ..exceptions import PersistenceError, RegistrationError, ResolutionError
from ..integrations.dataframe import stage_dask_dataframe_for_copy
from ..integrations.optuna.api import StudyArtifactOptions
from ..state.output_pointers import (
    OutputPointer,
    PointerResolutionError,
    resolve_pointer_chain,
)

from ..core.base import PipelineBase


def _is_child_pipeline(node: object) -> bool:
    return isinstance(node, PipelineBase)


class MutationMixin:
    """Setting constants and produced values, and syncing downstream mirrors."""

    if TYPE_CHECKING:
        manual_values: dict[str, Any] = {}
        para_value_dict: dict[str, Any] = {}
        producer_outputs: dict[str, dict[str, Any]] = {}
        artifact_registry: dict[str, Any] = {}
        registration_name: str = ""
        parent_pipeline: Any = None
        execution_priority: float | None = None
        logger: Any = None
        artifact_store: Any = None
        torch_load_weights_only: bool = False

        def _visible_config_names(self) -> set[str]: ...
        def _tree_declared_output_names(self) -> set[str]: ...
        def _tree_produced_value_names(self) -> set[str]: ...
        def _tree_constant_names(self) -> set[str]: ...
        def _reject_managed_study_reuse(
            self,
            value: Any,
            permitted_owner: tuple[str, str] | None = None,
            permitted_owner_kinds: frozenset[str] = frozenset({"output"}),
        ) -> None: ...
        def _optuna_study_value(self, value: Any) -> Any: ...
        def _managed_study_owner(self, value: Any) -> Any: ...
        def _optuna_studies_db_path_for_storage(self) -> Any: ...
        def _snapshot_value(
            self, variable_name: str, value: Any, *, verbose: bool
        ) -> Any: ...
        def _sync_attached_outputs_to_parent(self) -> None: ...
        def _cleanup_replaced_artifact(self, previous_value: Any) -> None: ...
        def _sorted_nodes(self) -> list[Any]: ...
        def _root_pipeline(self) -> Any: ...
        def _incoming_parent_outputs(self) -> dict[str, Any]: ...
        def list_declared_outputs(self) -> set[str]: ...
        def _node_declared_outputs(self, node: Any) -> set[str]: ...
        @staticmethod
        def _validate_builtin_name_conflict(name: str, owner_label: str) -> None: ...

    def _save_value_to_disk(
        self,
        variable_name: str,
        value: Any,
        *,
        function_name: str,
        verbose: bool,
        stage_dask_source: bool = False,
    ) -> ArtifactRecord:
        try:
            with ExitStack() as stack:
                value_to_save = value
                if stage_dask_source:
                    value_to_save = stage_dask_dataframe_for_copy(
                        value,
                        self.artifact_store.artifact_root,
                        stack,
                    )
                record = self.artifact_store.save(
                    variable_name=variable_name,
                    value=value_to_save,
                    block_name=self.registration_name,
                    function_name=function_name,
                    run_id=uuid4().hex,
                    torch_load_weights_only=self.torch_load_weights_only,
                    optuna_db_path=self._optuna_studies_db_path_for_storage(),
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
        owner_key = f"{self.registration_name}.{variable_name}"
        self._reject_managed_study_reuse(
            value,
            permitted_owner=("constant", owner_key),
            permitted_owner_kinds=frozenset({"output", "storage"}),
        )
        study_value = self._optuna_study_value(value)
        if study_value is not None:
            source_owner = self._managed_study_owner(value)
            if source_owner is not None and source_owner[0] == "storage":
                self.logger.info(
                    f"Optuna Study copied from stored object '{source_owner[1]}' "
                    f"into constant '{variable_name}'"
                )
            previous_name = None
            original_name = None
            if (
                isinstance(previous_value, ArtifactRecord)
                and previous_value.serializer == "optuna-study"
            ):
                persisted_name = previous_value.metadata.get("study_name")
                lineage_name = previous_value.metadata.get("original_study_name")
                if isinstance(persisted_name, str):
                    previous_name = persisted_name
                if isinstance(lineage_name, str):
                    original_name = lineage_name
            replacement = self.artifact_store.save(
                variable_name=variable_name,
                value=study_value,
                block_name=self.registration_name,
                function_name="set_constant_value",
                run_id=uuid4().hex,
                optuna_db_path=self._optuna_studies_db_path_for_storage(),
                optuna_options=StudyArtifactOptions(
                    independent_copy=previous_name is None,
                    study_name=previous_name,
                    original_study_name=original_name,
                    owner_kind="constant",
                    owner_key=owner_key,
                ),
            )
        elif isinstance(
            value,
            (ArtifactRecord, TorchStateArtifactRecord),
        ):
            replacement = value
        elif to_disk:
            replacement = self._save_value_to_disk(
                variable_name,
                value,
                function_name="set_constant_value",
                verbose=verbose,
            )
        elif copy:
            replacement = self._snapshot_value(variable_name, value, verbose=verbose)
        else:
            replacement = value
        self.manual_values[variable_name] = replacement
        self.para_value_dict[variable_name] = replacement
        if isinstance(replacement, ArtifactRecord):
            self.artifact_registry[variable_name] = replacement
        else:
            self.artifact_registry.pop(variable_name, None)
        if self.parent_pipeline is not None:
            self._sync_attached_outputs_to_parent()
        self._cleanup_replaced_artifact(previous_value)

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

        producing_node = next(
            (
                node
                for node in reversed(self._sorted_nodes())
                if variable_name
                in self.producer_outputs.get(node.registration_name, {})
            ),
            None,
        )
        if producing_node is not None and _is_child_pipeline(producing_node):
            producing_node.update_value(
                variable_name,
                value,
                copy=copy,
                verbose=verbose,
            )
            return
        if producing_node is None:
            raise ResolutionError(f"Unknown produced value owner for: {variable_name}")
        self._replace_local_node_output(
            producing_node,
            variable_name,
            value,
            copy=copy,
            verbose=verbose,
        )

    def _replace_local_node_output(
        self,
        node: Any,
        variable_name: str,
        value: Any,
        *,
        copy: bool,
        verbose: bool,
    ) -> None:
        outputs = self.producer_outputs[node.registration_name]
        previous_value = outputs[variable_name]
        if isinstance(previous_value, OutputPointer):
            try:
                owner, _ = resolve_pointer_chain(
                    previous_value.destination,
                    self._root_pipeline()._read_output_address,
                )
            except PointerResolutionError as exc:
                raise ResolutionError(str(exc)) from exc
            raise ResolutionError(
                f"Cannot set node output '{node.registration_name}.{variable_name}' "
                f"in pipeline '{self.registration_name}': this slot is an output "
                f"pointer. The real object is stored at "
                f"'{owner.pipeline_name}.{owner.node_name}.{owner.output_name}'; "
                f"call set_node_output('{owner.node_name}', '{owner.output_name}', ...) "
                f"on pipeline '{owner.pipeline_name}' instead"
            )
        if isinstance(
            value,
            (ArtifactRecord, TorchStateArtifactRecord),
        ):
            replacement = value
        elif isinstance(previous_value, ArtifactRecord):
            replacement = self._save_value_to_disk(
                variable_name,
                value,
                function_name="update_value",
                verbose=verbose,
                stage_dask_source=True,
            )
        elif copy:
            replacement = self._snapshot_value(variable_name, value, verbose=verbose)
        else:
            replacement = value
        outputs[variable_name] = replacement
        self._refresh_visible_value(variable_name)
        self._sync_value_to_ancestors_without_invalidation(variable_name)
        self._cleanup_replaced_artifact(previous_value)

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
            current_value = self.para_value_dict[variable_name]
            if isinstance(current_value, OutputPointer):
                try:
                    owner, _ = resolve_pointer_chain(
                        current_value.destination,
                        self._root_pipeline()._read_output_address,
                    )
                except PointerResolutionError as exc:
                    raise ResolutionError(str(exc)) from exc
                raise ResolutionError(
                    f"Cannot set value '{variable_name}' in pipeline "
                    f"'{self.registration_name}': this slot is an output pointer. "
                    f"The real object is stored at "
                    f"'{owner.pipeline_name}.{owner.node_name}.{owner.output_name}'; "
                    f"call set_value('{owner.output_name}', ...) on pipeline "
                    f"'{owner.pipeline_name}' instead"
                )
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

    def _nearest_upstream_produced_owner(self, variable_name: str) -> Any:
        current = self.parent_pipeline
        while current is not None:
            winning_child: Any = None
            for node in current._sorted_nodes():
                if node.execution_priority >= self.execution_priority:
                    break
                if (
                    _is_child_pipeline(node)
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

    def _descendant_produced_owner(self, variable_name: str) -> Any:
        for node in reversed(self._sorted_nodes()):
            if not _is_child_pipeline(node):
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

    def _find_declaring_pipeline(self, variable_name: str) -> Any:
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
        if _is_child_pipeline(node):
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
        if _is_child_pipeline(node):
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
        current: Any = self
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
        current: Any = self
        while current.parent_pipeline is not None:
            parent = current.parent_pipeline
            parent_outputs = parent.producer_outputs.get(current.registration_name)
            if parent_outputs is None or variable_name not in parent_outputs:
                return
            parent_outputs[variable_name] = value
            parent._refresh_visible_value(variable_name)
            current = parent
