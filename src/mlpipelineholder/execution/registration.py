"""Atom registration comparison: whether an atom re-registration is unchanged."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.models import ExpressionRegistration, FunctionRegistration
from ..core.naming import validate_registration_name
from ..exceptions import RegistrationError
from ..state.output_pointers import OutputAddress, is_strictly_upstream
from .atom_registry import atom_pipeline_class
from .function_registry import _values_equal, callable_identity_matches

from ..core.base import PipelineBase


def _is_child_pipeline(node: object) -> bool:
    return isinstance(node, PipelineBase)


class AtomRegistrationMixin:
    """Structural comparison used to decide whether an atom re-registration is a no-op."""

    if TYPE_CHECKING:
        nodes: list[Any] = []
        nodes_by_name: dict[str, Any] = {}
        blocks: list[Any] = []
        blocks_by_name: dict[str, Any] = {}
        gate_block: Any = None
        _overridden_outputs: dict[Any, Any] = {}
        registration_name: str = ""
        execution_priority: float | None = None
        _is_atom: bool = False

        def _validate_node_registration(
            self, node: Any, execution_priority: float | None
        ) -> None: ...
        def _validate_related_pipeline_names(self, candidate: Any) -> None: ...
        def _validate_output_names_against_config(self, output_names: list[str]) -> None: ...
        def _validate_strict_attach(
            self, child: Any, execution_priority: float
        ) -> None: ...
        def _sorted_nodes(self) -> list[Any]: ...

    def _validate_atom_replacement(
        self,
        candidate: Any,
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
        old: Any,
        new: Any,
    ) -> bool:
        """Whether a re-created atom pipeline is structurally identical to the old one.

        Compares registration identity, gate, and inner blocks, including their
        functions and args/kwargs helpers. Parent configuration is runtime state
        and therefore does not affect the atom definition.
        """
        if not old._is_atom:
            return False
        if old.registration_name != new.registration_name:
            return False
        if old.execution_priority != new.execution_priority:
            return False
        if old._overridden_outputs != new._overridden_outputs:
            return False
        if not self._atom_gates_equal(old, new):
            return False
        if not self._atom_blocks_equal(old, new):
            return False
        return True

    @staticmethod
    def _atom_gates_equal(
        old: Any,
        new: Any,
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
        old: Any,
        new: Any,
    ) -> bool:
        if any(
            _is_child_pipeline(node)
            for pipeline in (old, new)
            for node in pipeline._sorted_nodes()
        ):
            return False
        old_blocks = {
            node.registration_name: node
            for node in old._sorted_nodes()
            if not _is_child_pipeline(node)
        }
        new_blocks = {
            node.registration_name: node
            for node in new._sorted_nodes()
            if not _is_child_pipeline(node)
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
                and old_reg.overridden_outputs == new_reg.overridden_outputs
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
            and old_reg.ignore_underscore_outputs
            == new_reg.ignore_underscore_outputs
            and old_reg.save_to_disk == new_reg.save_to_disk
            and old_reg.param_mapping == new_reg.param_mapping
            and old_reg.var_pos_name == new_reg.var_pos_name
            and old_reg.var_kw_name == new_reg.var_kw_name
            and old_reg.overridden_outputs == new_reg.overridden_outputs
        )


class RegistrationMixin:
    """Registering blocks, child pipelines, and atom child pipelines."""

    if TYPE_CHECKING:
        registration_name: str = ""
        execution_priority: float | None = None
        parent_pipeline: Any = None
        project_root: Any = None
        logger: Any = None
        producer_outputs: dict[str, dict[str, Any]] = {}
        nodes: list[Any] = []
        nodes_by_name: dict[str, Any] = {}
        blocks: list[Any] = []
        blocks_by_name: dict[str, Any] = {}
        _is_atom: bool = False
        _invalidation_forbidden: bool = False

        def _registration_conflicts(
            self, node: Any, execution_priority: float | None
        ) -> list[Any]: ...
        def _raise_on_priority_conflict_with_different_name(
            self,
            registration_name: str,
            execution_priority: float | None,
            conflicts: list[Any],
        ) -> None: ...
        def _replace_conflicting_nodes(self, nodes: list[Any]) -> None: ...
        def _register_node(self, node: Any) -> None: ...
        def _remove_registered_node(self, node: Any) -> None: ...
        def _validate_node_registration(
            self, node: Any, execution_priority: float | None
        ) -> None: ...
        def _validate_related_pipeline_names(self, candidate: Any) -> None: ...
        def _validate_output_names_against_config(
            self, output_names: list[str]
        ) -> None: ...
        def _validate_atom_replacement(
            self, candidate: Any, existing: Any, execution_priority: float
        ) -> None: ...
        def _atom_matches(self, old: Any, new: Any) -> bool: ...
        def _root_pipeline(self) -> Any: ...
        def _declared_output_names_before_priority(
            self, priority: float | None
        ) -> set[str]: ...
        def _sorted_nodes(self) -> list[Any]: ...
        def _erase_overridden_node_outputs(
            self,
            node_name: str,
            old_priority: float | None,
            new_priority: float | None,
            old_output_names: list[str],
            new_output_names: list[str] | None = None,
            *,
            erase_output_names: list[str] | None = None,
        ) -> None: ...
        def _locally_produced_outputs(self) -> dict[str, Any]: ...
        def _rebuild_visible_state(
            self, upstream_outputs: dict[str, Any] | None = None
        ) -> None: ...
        def _incoming_parent_outputs(self) -> dict[str, Any]: ...
        def _sync_invalidation_flag(self) -> None: ...
        def _invalidate_all_outputs(self) -> None: ...
        def get_full_config(self) -> dict[str, Any]: ...
        def set_gate_block(
            self,
            function_or_path: Any,
            expected_value: Any = True,
            forced: bool = False,
        ) -> Any: ...
        def list_declared_outputs(self) -> set[str]: ...
        strict_mode: bool = False
        _suppress_strict_validation: bool = False
        manual_values: Any = None
        def _visible_config_names(self) -> set[str]: ...
        def _ancestor_manual_values(self) -> dict[str, Any]: ...
        def _visible_outputs_before_priority(
            self,
            priority: float | None,
            upstream_outputs: dict[str, Any] | None = None,
        ) -> dict[str, Any]: ...
        def _priority_vector(
            self, node_name: str, node_priority: float | None = None
        ) -> tuple[float, ...]: ...
        def _pipeline_by_name(self, pipeline_name: str) -> Any: ...
        def _node_declared_outputs(self, node: Any) -> set[str]: ...
        def _create_execution_block(
            self, registration_name: str, execution_priority: float
        ) -> Any: ...
        @staticmethod
        def _is_execution_block(node: Any) -> bool: ...

    def add_block(
        self: Any, registration_name: str, execution_priority: float, forced: bool = True
    ) -> Any:
        validate_registration_name(registration_name, owner_label="block")
        block = self._create_execution_block(registration_name, execution_priority)
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
            and self._is_execution_block(existing_block)
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

    def _add_block_strict(self: Any, registration_name: str, execution_priority: float):
        validate_registration_name(registration_name, owner_label="block")
        block = self._create_execution_block(registration_name, execution_priority)
        self._register_node(block)
        return block

    def add_child_pipeline(
        self,
        child_pipeline: Any,
        execution_priority: float,
        registration_name: str | None = None,
        forced: bool = True,
    ) -> Any:
        if child_pipeline is self:
            raise RegistrationError("A pipeline cannot register itself as a child pipeline")
        was_root = child_pipeline.parent_pipeline is None
        if registration_name is not None:
            validate_registration_name(registration_name, owner_label="pipeline")
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
        output_variable_names: str | list[str] | tuple[str, ...] | None = None,
        save_to_disk_lst: list[str] | tuple[str, ...] | set[str] | None = None,
        param_mapping_dct: dict[str, str | None] | None = None,
        kwargs_dct: dict[str, str] | None = None,
        args_lst: tuple[str, ...] | list[str] | None = None,
        forced: bool = True,
        block_priority: float = 10.0,
        *,
        param_mapping: dict[str, str | None] | None = None,
        overridden_outputs: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        atom_cls: Any = atom_pipeline_class()
        atom_cls.create(
            self,
            child_name,
            execution_priority,
            target_function,
            gate_config=gate_config,
            expected_value=expected_value,
            output_variable_names=output_variable_names,
            save_to_disk_lst=save_to_disk_lst,
            param_mapping_dct=param_mapping_dct,
            kwargs_dct=kwargs_dct,
            args_lst=args_lst,
            forced=forced,
            block_priority=block_priority,
            param_mapping=param_mapping,
            overridden_outputs=overridden_outputs,
        )

    def _validate_strict_attach(
        self,
        child: Any,
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

    def _normalize_overridden_outputs(
        self,
        output_names: list[str],
        declarations: dict[str, tuple[str, str]] | None,
        *,
        current_node_name: str,
        current_priority: float | None,
    ) -> dict[str, OutputAddress]:
        if declarations is None:
            return {}
        if not isinstance(declarations, dict):
            raise RegistrationError("overridden_outputs must be a dict or None")
        unknown_outputs = sorted(set(declarations).difference(output_names))
        if unknown_outputs:
            raise RegistrationError(
                f"overridden_outputs keys must be declared outputs: {unknown_outputs}"
            )
        current_vector = self._priority_vector(
            current_node_name,
            current_priority,
        )
        normalized: dict[str, OutputAddress] = {}
        for output_name, target in declarations.items():
            if (
                not isinstance(output_name, str)
                or not isinstance(target, tuple)
                or len(target) != 2
                or not all(isinstance(item, str) for item in target)
            ):
                raise RegistrationError(
                    "Each overridden_outputs entry must map a string output name to "
                    "a (pipeline_name, node_name) tuple of strings"
                )
            pipeline_name, node_name = target
            target_pipeline = self._pipeline_by_name(pipeline_name)
            target_node = target_pipeline.nodes_by_name.get(node_name)
            if target_node is None:
                raise RegistrationError(
                    f"Output pointer target node does not exist: {pipeline_name}.{node_name}"
                )
            if _is_child_pipeline(target_node) and not target_node._is_atom:
                raise RegistrationError(
                    f"Output pointer target '{pipeline_name}.{node_name}' must be a block or atom pipeline"
                )
            if output_name not in target_pipeline._node_declared_outputs(target_node):
                raise RegistrationError(
                    f"Output pointer target '{pipeline_name}.{node_name}' does not declare '{output_name}'"
                )
            target_vector = target_pipeline._priority_vector(node_name)
            if not is_strictly_upstream(target_vector, current_vector):
                raise RegistrationError(
                    f"Output pointer target '{pipeline_name}.{node_name}' must be globally upstream of "
                    f"'{self.registration_name}.{current_node_name}'"
                )
            normalized[output_name] = OutputAddress(
                pipeline_name,
                node_name,
                output_name,
            )
        return normalized
