"""Visibility state: scope resolution and parent/child mirror rebuilding."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.constants import _MISSING
from ..core.models import ArtifactRecord, ExpressionRegistration

from ..core.base import PipelineBase


def _is_child_pipeline(node: object) -> bool:
    return isinstance(node, PipelineBase)


class VisibilityMixin:
    """Visible-value resolution and in-memory mirror rebuilding."""

    if TYPE_CHECKING:
        nodes: list[Any] = []
        producer_outputs: dict[str, dict[str, Any]] = {}
        manual_values: dict[str, Any] = {}
        para_value_dict: dict[str, Any] = {}
        artifact_registry: dict[str, Any] = {}
        parent_pipeline: Any = None
        execution_priority: float | None = None
        gate_block: Any = None
        _is_atom: bool = False

        def list_declared_outputs(self) -> set[str]: ...
        def config_as_dict(self) -> dict[str, Any]: ...
        def _tree_constant_names(self) -> set[str]: ...

    def _sorted_nodes(self) -> list[Any]:
        return sorted(
            self.nodes, key=lambda node: (node.execution_priority, node.registration_name)
        )

    def _node_declared_outputs(self, node: Any) -> set[str]:
        if _is_child_pipeline(node):
            return node.list_declared_outputs()
        return node.declared_outputs()

    def _node_declared_disk_backed_outputs(self, node: Any) -> set[str]:
        if _is_child_pipeline(node):
            return node._declared_disk_backed_output_names()
        return {
            output_name
            for registration in node.functions
            for output_name in registration.save_to_disk
        }

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
        visible = dict(
            self._incoming_parent_outputs()
            if upstream_outputs is None
            else upstream_outputs
        )
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
        if _is_child_pipeline(node):
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

    def _declared_disk_backed_output_names(self) -> set[str]:
        names = set(self._incoming_parent_disk_backed_output_names())
        for node in self._sorted_nodes():
            names.update(self._node_declared_disk_backed_outputs(node))
        return names

    def _declared_disk_backed_output_names_before_priority(
        self,
        priority: float | None,
    ) -> set[str]:
        names = set(self._incoming_parent_disk_backed_output_names())
        if priority is None:
            return names
        for node in self._sorted_nodes():
            if node.execution_priority >= priority:
                break
            names.update(self._node_declared_disk_backed_outputs(node))
        return names

    def _incoming_parent_disk_backed_output_names(self) -> set[str]:
        if self.parent_pipeline is None or self.execution_priority is None:
            return set()
        return self.parent_pipeline._declared_disk_backed_output_names_before_priority(
            self.execution_priority
        )

    def _incoming_parent_outputs(self) -> dict[str, Any]:
        if self.parent_pipeline is None or self.execution_priority is None:
            return {}
        return self.parent_pipeline._visible_outputs_before_priority(self.execution_priority)

    def _subtree_produced_output_names(self) -> set[str]:
        """Collect own and descendant produced keys without inspecting values or side effects."""
        output_names: set[str] = set()
        for outputs in self.producer_outputs.values():
            output_names.update(outputs)
        for node in self._sorted_nodes():
            if _is_child_pipeline(node):
                output_names.update(node._subtree_produced_output_names())
        return output_names

    def _visible_output_names(self) -> set[str]:
        """Collect visible produced keys, excluding mirrored constants, without side effects.

        Visibility includes own and descendant outputs plus upstream ancestor
        and earlier-sibling outputs. Key presence counts ``None``, artifacts,
        pointers, and placeholders without materialization.
        """
        output_names = self._subtree_produced_output_names()
        output_names.update(self._incoming_parent_outputs())
        return output_names - self._tree_constant_names()

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
            if _is_child_pipeline(node):
                visible.update(node.manual_values)
        return visible

    def _visible_constant_names(self) -> set[str]:
        """Collect own, ancestor, and earlier-sibling constant keys without side effects."""
        return set(self._incoming_parent_manual_values()) | set(self.manual_values)

    def _descendant_visible_value(self, variable_name: str) -> Any:
        for node in self._sorted_nodes():
            if not _is_child_pipeline(node):
                continue
            if variable_name in node.para_value_dict:
                return node.para_value_dict[variable_name]
            descendant_value = node._descendant_visible_value(variable_name)
            if descendant_value is not _MISSING:
                return descendant_value
        return _MISSING

    def _ancestor_descendant_visible_value(self, variable_name: str) -> Any:
        current = self.parent_pipeline
        while current is not None:
            descendant_value = current._descendant_visible_value(variable_name)
            if descendant_value is not _MISSING:
                return descendant_value
            current = current.parent_pipeline
        return _MISSING

    def _ancestor_config_values(self) -> dict[str, Any]:
        config: dict[str, Any] = {}
        current = self.parent_pipeline
        chain: list[Any] = []
        while current is not None:
            chain.append(current)
            current = current.parent_pipeline
        for pipeline in reversed(chain):
            config.update(pipeline.config_as_dict())
        return config

    def _ancestor_manual_values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        current = self.parent_pipeline
        chain: list[Any] = []
        while current is not None:
            chain.append(current)
            current = current.parent_pipeline
        for pipeline in reversed(chain):
            values.update(pipeline.manual_values)
        return values

    def _visible_config_names(self) -> set[str]:
        if self._is_atom and self.parent_pipeline is not None:
            return self.parent_pipeline._visible_config_names()
        return set(self.config_as_dict()).union(self._ancestor_config_values())

    def _registration_visible_names(self) -> set[str]:
        names = set(self._visible_config_names())
        names.update(self.list_declared_outputs())
        names.update(self._incoming_parent_output_names())
        names.update(self._tree_constant_names())
        return names

    def _registration_disk_backed_names(
        self,
        priority: float | None,
    ) -> set[str]:
        names = self._known_disk_backed_output_names()
        names.update(self._declared_disk_backed_output_names_before_priority(priority))
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
        if _is_child_pipeline(node):
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
