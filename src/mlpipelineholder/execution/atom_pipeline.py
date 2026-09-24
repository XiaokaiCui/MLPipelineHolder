"""Atom pipeline: a sealed block-like child node inside its parent pipeline."""

from __future__ import annotations

from typing import Any

from ..exceptions import RegistrationError
from ..pipeline_holder import PipelineHolder
from .atom_registry import register_atom_pipeline_class
from .registration import _is_child_pipeline


class AtomPipeline(PipelineHolder):
    """A sealed block-like child node with an optional gate and no owned configuration.

    Instances are created exclusively through
    :meth:`PipelineHolder.create_atom_child_pipeline`, which delegates to
    :meth:`AtomPipeline.create`. While an atom is assembled it behaves like a regular
    holder; once construction completes the instance is sealed and the extension
    methods below reject every structural or configuration mutation. The parent then
    executes the atom as one indivisible node, exactly like a block: internal blocks
    run together, the optional gate controls execution, and outputs are exposed
    through the atom's public mirror slot.
    """

    _is_atom = True
    _sealed = False

    def _seal(self) -> None:
        """Lock the atom; extension methods reject mutation after this call."""
        self._sealed = True

    def _assert_mutable(self, action: str) -> None:
        if self._sealed:
            raise RegistrationError(
                f"Atom pipeline '{self.registration_name}' is immutable and cannot {action}"
            )

    def add_block(
        self, registration_name: str, execution_priority: float, forced: bool = True
    ) -> Any:
        self._assert_mutable("accept new blocks")
        return super().add_block(registration_name, execution_priority, forced=forced)

    def _add_block_strict(self, registration_name: str, execution_priority: float) -> Any:
        self._assert_mutable("accept new blocks")
        return super()._add_block_strict(registration_name, execution_priority)

    def add_child_pipeline(
        self,
        child_pipeline: Any,
        execution_priority: float,
        registration_name: str | None = None,
        forced: bool = True,
    ) -> Any:
        self._assert_mutable("accept child pipelines")
        return super().add_child_pipeline(
            child_pipeline,
            execution_priority,
            registration_name=registration_name,
            forced=forced,
        )

    def create_atom_child_pipeline(self, *args: Any, **kwargs: Any) -> None:
        raise RegistrationError(
            f"Atom pipeline '{self.registration_name}' is immutable "
            "and cannot accept child pipelines"
        )

    def remove_block(self, block_name: str) -> None:
        self._assert_mutable("remove blocks")
        super().remove_block(block_name)

    def add_gate_block(
        self, function_or_path: Any, expected_value: Any = True, forced: bool = False
    ) -> Any:
        self._assert_mutable("change its gate")
        return super().add_gate_block(
            function_or_path, expected_value=expected_value, forced=forced
        )

    def reset_gate_block(self) -> None:
        self._assert_mutable("change its gate")
        super().reset_gate_block()

    def set_configs(self, overrides: dict[str, Any]) -> None:
        self._require_owned_config()

    def update_configs(self, overrides: dict[str, Any]) -> None:
        self._require_owned_config()

    @classmethod
    def create(
        cls,
        parent: Any,
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
        """Create and attach an atom child under ``parent``; identical re-creation is a no-op."""
        if parent._is_atom:
            raise RegistrationError(
                f"Atom pipeline '{parent.registration_name}' is immutable "
                "and cannot accept child pipelines"
            )
        output_names = (
            []
            if output_variable_names is None
            else [output_variable_names]
            if isinstance(output_variable_names, str)
            else list(output_variable_names)
        )
        produced_output_names = [
            output_name for output_name in output_names if output_name != "_"
        ]
        save_names = set(save_to_disk_lst or [])
        if "_" in save_names:
            raise RegistrationError(
                "Ignored output marker '_' cannot be included in save_to_disk_lst"
            )
        if not save_names.issubset(set(produced_output_names)):
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
        normalized_overrides = parent._normalize_overridden_outputs(
            produced_output_names,
            overridden_outputs,
            current_node_name=child_name,
            current_priority=execution_priority,
        )
        child_root = parent.project_root / "children" / child_name
        atom_pipeline_cls = cls
        temp_pipeline = atom_pipeline_cls(
            registration_name=child_name,
            local_folder_path=child_root,
            execution_priority=execution_priority,
            strict_mode=parent._root_pipeline().strict_mode,
            colourful_logs=parent._root_pipeline().colourful_logs,
        )
        temp_pipeline.logger = parent.logger
        temp_pipeline.parent_pipeline = parent
        temp_pipeline._overridden_outputs = normalized_overrides
        if gate_config is not None:
            gate_visible = (
                set(temp_pipeline.get_full_config())
                | set(temp_pipeline._incoming_parent_outputs())
                | set(temp_pipeline.manual_values)
                | set(temp_pipeline._ancestor_manual_values())
                | set(parent._declared_output_names_before_priority(execution_priority))
            )
            if gate_config not in gate_visible:
                if temp_pipeline.strict_mode and not temp_pipeline._suppress_strict_validation:
                    raise RegistrationError(
                        f"Gate config '{gate_config}' is not found in config, visible output values, or visible manual values"
                    )
                if not temp_pipeline._suppress_strict_validation:
                    temp_pipeline.logger.warning(
                        f"Gate config '{gate_config}' is not found in config, visible output values, or visible manual values"
                    )
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
        registration = temp_block.register_function(
            target_function,
            output_variable_names=output_variable_names,
            save_to_disk=save_to_disk_lst,
            var_pos_name="default_args" if args_lst is not None else None,
            var_kw_name="default_kwargs" if kwargs_dct is not None else None,
            param_mapping=effective_param_mapping,
            forced=forced,
        )
        if registration is None:
            raise RegistrationError(
                f"Failed to register target function for atom pipeline '{child_name}'"
            )
        child_priority = temp_pipeline.execution_priority
        if child_priority is None:
            raise RegistrationError(f"Child pipeline '{child_name}' has no priority")
        existing = parent.nodes_by_name.get(child_name)
        if (
            forced
            and _is_child_pipeline(existing)
            and parent._atom_matches(existing, temp_pipeline)
        ):
            return None
        if forced and existing is not None:
            parent._validate_atom_replacement(
                temp_pipeline,
                existing,
                child_priority,
            )
            old_outputs = list(
                parent.producer_outputs.get(existing.registration_name, {}).keys()
            )
            new_outputs = sorted(temp_pipeline.list_declared_outputs())
            old_priority = existing.execution_priority
            parent._remove_registered_node(existing)
            if _is_child_pipeline(existing):
                existing._invalidate_all_outputs()
            parent._erase_overridden_node_outputs(
                existing.registration_name,
                old_priority,
                child_priority,
                old_outputs,
                new_outputs,
            )
            attached = parent.add_child_pipeline(
                temp_pipeline,
                execution_priority=child_priority,
                forced=False,
            )
        else:
            attached = parent.add_child_pipeline(
                temp_pipeline,
                execution_priority=child_priority,
                forced=forced,
            )
        if attached is not None:
            temp_pipeline._seal()


register_atom_pipeline_class(AtomPipeline)
