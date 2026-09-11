"""Value access: outputs, constants, and artifact materialization."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from ..core.constants import _IMMUTABLE_TYPES, _MISSING
from ..core.models import (
    ArtifactRecord,
    CallableValueReference,
    DataclassValueReference,
    RuntimeCallableReference,
    RuntimeValueReference,
    TorchStateArtifactRecord,
)
from ..exceptions import ResolutionError
from ..integrations.optuna.support import is_optuna_sampler, is_optuna_study
from ..state.output_pointers import (
    OutputPointer,
    PointerResolutionError,
    resolve_pointer_chain,
)

_holder_base: type | None = None


def _register_values_holder_base(holder_base: type) -> None:
    """Register the holder class so value access can recognise child pipelines."""
    global _holder_base
    _holder_base = holder_base


def _is_child_pipeline(node: object) -> bool:
    return _holder_base is not None and isinstance(node, _holder_base)


class ValueAccessMixin:
    """Reading and materializing produced outputs and pipeline constants."""

    if TYPE_CHECKING:
        para_value_dict: dict[str, Any] = {}
        manual_values: dict[str, Any] = {}
        producer_outputs: dict[str, dict[str, Any]] = {}
        artifact_registry: dict[str, Any] = {}
        nodes_by_name: dict[str, Any] = {}
        registration_name: str = ""
        artifact_store: Any = None
        logger: Any = None

        def _tree_constant_names(self) -> set[str]: ...
        def _incoming_parent_outputs(self) -> dict[str, Any]: ...
        def _descendant_visible_value(self, variable_name: str) -> Any: ...
        def _visible_output_names(self) -> set[str]: ...
        def _visible_constant_names(self) -> set[str]: ...
        def _incoming_parent_manual_values(self) -> dict[str, Any]: ...
        def _root_pipeline(self) -> Any: ...
        def _pipeline_by_name(self, pipeline_name: str) -> Any: ...
        @staticmethod
        def _restore_callable_value(reference: CallableValueReference) -> Any: ...
        @staticmethod
        def _validate_builtin_name_conflict(name: str, owner_label: str) -> None: ...
        def _replace_local_node_output(
            self,
            node: Any,
            variable_name: str,
            value: Any,
            *,
            copy: bool,
            verbose: bool,
        ) -> None: ...

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
                if value is _MISSING:
                    raise ResolutionError(f"Unknown pipeline value: {variable_name}")
        return self._materialize_stored_value(
            value,
            f"Cannot get value '{variable_name}': it was saved as a placeholder "
            f"({value.reason}) and cannot be restored; recreate or reset the value"
            if isinstance(value, (RuntimeValueReference, DataclassValueReference))
            else "",
        )

    def list_visible_output(self) -> set[str]:
        """Return a detached set of produced output names visible without side effects.

        Own and descendant outputs plus upstream ancestor and earlier-sibling
        outputs count by stored key, including ``None`` and deferred values,
        without materialization.
        """
        return self._visible_output_names()

    def has_visible_output(
        self, variable_name: str
    ) -> bool:
        """Report key-based produced-output visibility without materialization or side effects."""
        return variable_name in self._visible_output_names()

    def get_node_output(
        self, node_name: str, output_name: str
    ) -> Any:
        """Return one materialized output produced by an immediate block or atom."""
        node = self.nodes_by_name.get(node_name)
        if node is None:
            raise ResolutionError(
                f"Unknown immediate child node '{node_name}' in pipeline '{self.registration_name}'"
            )
        if _is_child_pipeline(node):
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

    def set_node_output(
        self,
        node_name: str,
        output_name: str,
        value: Any,
        *,
        copy: bool = True,
        verbose: bool = False,
    ) -> None:
        """Replace one output produced by an immediate block or atom."""
        self._validate_builtin_name_conflict(output_name, owner_label="pipeline value")
        node = self.nodes_by_name.get(node_name)
        if node is None:
            raise ResolutionError(
                f"Unknown immediate child node '{node_name}' in pipeline '{self.registration_name}'"
            )
        if _is_child_pipeline(node):
            if not node._is_atom:
                raise ResolutionError(
                    f"Node '{node_name}' is not a block or atom pipeline"
                )
            producers = [
                internal_node
                for internal_node in node._sorted_nodes()
                if output_name
                in node.producer_outputs.get(internal_node.registration_name, {})
            ]
            if not producers:
                raise ResolutionError(
                    f"Node '{node_name}' has no produced output named '{output_name}'"
                )
            if len(producers) > 1:
                priorities = [
                    internal_node.execution_priority for internal_node in producers
                ]
                raise ResolutionError(
                    f"Node '{node_name}' has multiple produced outputs named "
                    f"'{output_name}' at priorities {priorities}; cannot set it "
                    "unambiguously"
                )
            node._replace_local_node_output(
                producers[0],
                output_name,
                value,
                copy=copy,
                verbose=verbose,
            )
            return
        outputs = self.producer_outputs.get(node_name, {})
        if output_name not in outputs:
            raise ResolutionError(
                f"Node '{node_name}' has no produced output named '{output_name}'"
            )
        self._replace_local_node_output(
            node,
            output_name,
            value,
            copy=copy,
            verbose=verbose,
        )

    def _materialize_stored_value(
        self,
        value: Any,
        placeholder_error: str,
    ) -> Any:
        if isinstance(value, OutputPointer):
            try:
                owner_address, terminal = resolve_pointer_chain(
                    value.destination,
                    self._root_pipeline()._read_output_address,
                )
            except PointerResolutionError as exc:
                raise ResolutionError(str(exc)) from exc
            owner = self._pipeline_by_name(owner_address.pipeline_name)
            return owner._materialize_stored_value(terminal, placeholder_error)
        if isinstance(value, TorchStateArtifactRecord):
            return value
        if isinstance(value, CallableValueReference):
            return self._restore_callable_value(value)
        if isinstance(value, ArtifactRecord):
            materialized = self.artifact_store.load(value)
            if value.serializer == "optuna-study" and is_optuna_study(materialized):
                owner_kind = value.metadata.get("study_owner_kind")
                owner_key = value.metadata.get("study_owner_key")
                if isinstance(owner_kind, str) and isinstance(owner_key, str):
                    self._root_pipeline()._optuna_study_provenance[materialized] = (
                        owner_kind,
                        owner_key,
                    )
            return materialized
        if isinstance(value, (RuntimeValueReference, DataclassValueReference)):
            raise ResolutionError(placeholder_error)
        return value

    def get_constant_value(
        self, variable_name: str
    ) -> Any:
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

    def list_visible_constant(self) -> set[str]:
        """Return a detached set of constant names visible without side effects.

        Own constants plus ancestor and earlier-sibling constants count by
        stored key, including ``None`` and deferred values, without
        materialization. Later siblings and descendant-owned constants are
        excluded.
        """
        return self._visible_constant_names()

    def has_visible_constant(
        self, variable_name: str
    ) -> bool:
        """Report key-based constant visibility without materialization or side effects."""
        return variable_name in self._visible_constant_names()

    @classmethod
    def _is_mutable_value(cls, value: Any) -> bool:
        if isinstance(value, _IMMUTABLE_TYPES):
            return False
        if isinstance(value, (tuple, frozenset)):
            return any(cls._is_mutable_value(item) for item in value)
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

    def _snapshot_value(
        self, variable_name: str, value: Any, *, verbose: bool
    ) -> Any:
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
        ) or callable(value) or is_optuna_study(value) or is_optuna_sampler(value):
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
