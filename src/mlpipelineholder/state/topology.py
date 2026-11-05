"""Pipeline topology: roots, lookup, priority vectors, and attached-tree walks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from ..core.base import PipelineBase
from ..exceptions import RegistrationError


def _is_child_pipeline(node: object) -> bool:
    return isinstance(node, PipelineBase)


class TopologyMixin:
    """Traversal and identity helpers over the attached pipeline tree."""

    if TYPE_CHECKING:
        parent_pipeline: Any = None
        nodes_by_name: dict[str, Any] = {}
        _config: Any = None
        _clean_config_baseline: Any = None
        _is_atom: bool = False

        def _sorted_nodes(self) -> list[Any]: ...
        def list_declared_outputs(self) -> set[str]: ...
        @staticmethod
        def _config_name_mapping(config_obj: Any) -> dict[str, Any]: ...

    def _root_pipeline(self) -> Any:
        current = self
        while current.parent_pipeline is not None:
            current = current.parent_pipeline
        return current

    def _pipeline_by_name(self, pipeline_name: str) -> Any:
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
        chain: list[Any] = []
        current: Any = self
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

    def _iter_attached_pipelines(self) -> list[Any]:
        pipelines: list[Any] = [self]
        for node in self._sorted_nodes():
            if _is_child_pipeline(node):
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

    def _constant_conflict_names(
        self,
        configs: dict[int, Any] | None = None,
    ) -> set[str]:
        """Names that can actually conflict with a constant in this scope.

        A constant declared on this pipeline is visible to the pipeline and its
        descendants, so only locally declared outputs (including descendants,
        via ``list_declared_outputs``) and configuration visible from this
        pipeline and its ancestors can conflict with it. Sibling declarations
        are intentionally excluded; strict-mode attachment performs its own
        cross-boundary validation.

        Pass ``configs`` (keyed by ``id(pipeline)``) to validate against
        definition-time configuration during clean-start preparation. Without
        it, the current runtime configuration is used.
        """
        names: set[str] = set(self.list_declared_outputs())
        current: Any = self
        while current is not None:
            if not (current._is_atom and current.parent_pipeline is not None):
                if configs is None:
                    config = current._config
                else:
                    config = configs.get(id(current), current._clean_config_baseline)
                names.update(current._config_name_mapping(config))
            current = current.parent_pipeline
        return names
