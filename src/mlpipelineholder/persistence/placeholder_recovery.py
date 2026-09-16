"""Placeholder recovery: rerunning blocks whose restored outputs are unusable."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..core.base import PipelineBase
from ..core.models import (
    DataclassValueReference,
    ExpressionRegistration,
    RunRecord,
    RuntimeValueReference,
)
from ..exceptions import ResolutionError
from ..execution.function_registry import callable_signature, default_map
from ..execution.gate_cache import _GateStatusCache


def _is_child_pipeline(node: object) -> bool:
    return isinstance(node, PipelineBase)


class PlaceholderRecoveryMixin:
    """Re-running blocks on load to replace unresolvable placeholder outputs."""

    if TYPE_CHECKING:
        logger: Any = None
        config: Any = None
        manual_values: dict[str, Any] = {}
        para_value_dict: dict[str, Any] = {}
        producer_outputs: dict[str, dict[str, Any]] = {}
        run_history: list[Any] = []
        strict_mode: bool = False

        def full_path(self) -> str: ...
        def _sorted_nodes(self) -> list[Any]: ...
        def _rebuild_visible_state(
            self, upstream_outputs: dict[str, Any] | None = None
        ) -> None: ...
        def _ancestor_config_values(self) -> dict[str, Any]: ...
        def _ancestor_manual_values(self) -> dict[str, Any]: ...
        def _config_has_field(self, config_obj: Any, field_name: str) -> bool: ...
        def _config_value(self, config_obj: Any, field_name: str) -> Any: ...
        def _sync_value_to_ancestors_without_invalidation(self, variable_name: str) -> None: ...
        def _tree_find_declaring_node(self, variable_name: str) -> Any | None: ...
        def _warn_placeholder_unrecoverable(
            self, placeholder_names: list[str], reason: str, *, verbose: bool
        ) -> None: ...
        def _pipeline_gate_status(
            self, *, _gate_cache: _GateStatusCache | None = None
        ) -> tuple[str, str | None]: ...
        def _recovery_upstream_outputs(
            self, *, _gate_cache: _GateStatusCache | None = None
        ) -> dict[str, Any]: ...
        def _recovery_visible_outputs_before_priority(
            self,
            priority: float | None,
            upstream_outputs: dict[str, Any] | None = None,
            *,
            _gate_cache: _GateStatusCache | None = None,
        ) -> dict[str, Any]: ...
        def _unresolvable_input_gate_off_reason(
            self, input_name: str, *, _gate_cache: _GateStatusCache | None = None
        ) -> str: ...

    def _has_placeholder_outputs_in_subtree(self) -> bool:
        placeholder_types = (RuntimeValueReference, DataclassValueReference)
        for outputs in self.producer_outputs.values():
            if any(isinstance(value, placeholder_types) for value in outputs.values()):
                return True
        for name, value in self.para_value_dict.items():
            if name not in self.manual_values and isinstance(value, placeholder_types):
                return True
        return any(
            node._has_placeholder_outputs_in_subtree()
            for node in self._sorted_nodes()
            if _is_child_pipeline(node)
        )

    def _auto_resolve_placeholder_outputs(
        self,
        *,
        verbose: bool,
        _gate_cache: _GateStatusCache | None = None,
    ) -> None:
        """Recover produced values saved as placeholders by re-running their blocks.

        Walks nodes in upstream-to-downstream order and runs at most one block per
        placeholder recovery, injecting fresh values without invalidating any
        downstream outputs. Gate-off pipelines are skipped silently; recovery
        failures emit warnings (unconditional for execution exceptions,
        verbose-gated otherwise).
        """
        if _gate_cache is None:
            _gate_cache = _GateStatusCache()
        status, gate_error = self._pipeline_gate_status(_gate_cache=_gate_cache)
        if status == "block":
            return
        if status == "error":
            if verbose and self._has_placeholder_outputs_in_subtree():
                self.logger.warning(
                    f"Placeholder output(s) in pipeline '{self.full_path()}' are not recoverable: "
                    f"the gate could not be evaluated ({gate_error}); they remain placeholders"
                )
            return
        for node in self._sorted_nodes():
            if _is_child_pipeline(node):
                node._auto_resolve_placeholder_outputs(
                    verbose=verbose,
                    _gate_cache=_gate_cache,
                )
                continue
            node_outputs = self.producer_outputs.get(node.registration_name, {})
            placeholder_names = [
                output_name
                for output_name, value in node_outputs.items()
                if isinstance(value, RuntimeValueReference)
            ]
            if not placeholder_names:
                continue
            self._recover_block_placeholder_outputs(
                node,
                placeholder_names,
                verbose=verbose,
                _gate_cache=_gate_cache,
            )
        for value_name, value in self.para_value_dict.items():
            if (
                isinstance(value, RuntimeValueReference)
                and value_name not in self.manual_values
                and self._tree_find_declaring_node(value_name) is None
                and verbose
            ):
                self.logger.warning(
                    f"Placeholder value '{value_name}' is not recoverable: its producing "
                    "block is not registered in the loaded pipeline; it remains a placeholder "
                    "and raises ResolutionError when read"
                )

    def _warn_unresolved_placeholders_at_load(
        self,
        *,
        verbose: bool,
        _seen: set[str] | None = None,
        _gate_cache: _GateStatusCache | None = None,
    ) -> None:
        """Verbose-gated load warning for produced values saved as placeholders.

        Used when ``auto_resolve_placeholders=False`` so users still learn that
        a produced value was saved as a placeholder rather than a real value.
        Gate-off pipelines are skipped silently; a gate that fails to evaluate
        logs a verbose warning instead.
        """
        if not verbose:
            return
        if _gate_cache is None:
            _gate_cache = _GateStatusCache()
        seen = set() if _seen is None else _seen
        status, gate_error = self._pipeline_gate_status(_gate_cache=_gate_cache)
        if status == "block":
            return
        if status == "error":
            self.logger.warning(
                f"Placeholder value(s) in pipeline '{self.full_path()}' could not be inspected: "
                f"the gate could not be evaluated ({gate_error})"
            )
            return
        for node in self._sorted_nodes():
            if _is_child_pipeline(node):
                node._warn_unresolved_placeholders_at_load(
                    verbose=verbose,
                    _seen=seen,
                    _gate_cache=_gate_cache,
                )
                continue
            for output_name, value in self.producer_outputs.get(
                node.registration_name, {}
            ).items():
                if isinstance(value, RuntimeValueReference) and output_name not in seen:
                    seen.add(output_name)
                    self.logger.warning(
                        f"Pipeline value '{output_name}' was saved as a placeholder "
                        f"({value.reason}) rather than a real value; it raises "
                        "ResolutionError when read"
                    )
        for value_name, value in self.para_value_dict.items():
            if (
                isinstance(value, RuntimeValueReference)
                and value_name not in self.manual_values
                and self._tree_find_declaring_node(value_name) is None
                and value_name not in seen
            ):
                seen.add(value_name)
                self.logger.warning(
                    f"Pipeline value '{value_name}' was saved as a placeholder "
                    f"({value.reason}) rather than a real value; it raises "
                    "ResolutionError when read"
                )

    def _recover_block_placeholder_outputs(
        self,
        node: Any,
        placeholder_names: list[str],
        *,
        verbose: bool,
        _gate_cache: _GateStatusCache | None = None,
    ) -> None:
        upstream_outputs = self._recovery_upstream_outputs(_gate_cache=_gate_cache)
        parent_config = self._ancestor_config_values()
        visible_outputs = self._recovery_visible_outputs_before_priority(
            node.execution_priority,
            upstream_outputs=upstream_outputs,
            _gate_cache=_gate_cache,
        )
        for registration in node.functions:
            defaults = (
                {}
                if isinstance(registration, ExpressionRegistration)
                else {}
                if self.strict_mode
                else default_map(registration.callable_obj)
            )
            unmapped_required = self._strict_unmapped_required_input(registration)
            if unmapped_required is not None:
                self._warn_placeholder_unrecoverable(
                    placeholder_names,
                    f"required input '{unmapped_required}' has no explicit mapping or callable default",
                    verbose=verbose,
                )
                return
            for input_name in self._recovery_input_names(node, registration):
                status = self._recovery_input_status(
                    input_name,
                    visible_outputs,
                    parent_config,
                    defaults,
                )
                if status == "placeholder":
                    self._warn_placeholder_unrecoverable(
                        placeholder_names,
                        f"required input '{input_name}' is a placeholder that could not be restored",
                        verbose=verbose,
                    )
                    return
                if status == "unresolvable":
                    gate_off_reason = self._unresolvable_input_gate_off_reason(
                        input_name,
                        _gate_cache=_gate_cache,
                    )
                    self._warn_placeholder_unrecoverable(
                        placeholder_names,
                        f"required input '{input_name}' cannot be resolved{gate_off_reason}",
                        verbose=verbose,
                    )
                    return
        run_id = uuid4().hex
        run_record = RunRecord(
            run_id=run_id,
            mode=f"auto_resolve_placeholder:{node.registration_name}",
            executed_blocks=[node.registration_name],
            started_at=datetime.now(UTC).isoformat(),
        )
        self.run_history.append(run_record)
        try:
            produced_outputs = node.execute(
                run_id,
                visible_outputs,
                overrides={},
                parent_config=parent_config,
            )
        except (KeyboardInterrupt, SystemExit):
            run_record.status = "failed"
            run_record.finished_at = datetime.now(UTC).isoformat()
            raise
        except BaseException as exc:
            run_record.status = "failed"
            run_record.error_message = str(exc)
            run_record.finished_at = datetime.now(UTC).isoformat()
            self.logger.warning(
                f"Placeholder output(s) '{', '.join(sorted(placeholder_names))}' are not "
                f"recoverable: re-running block '{node.registration_name}' failed "
                f"({type(exc).__name__}: {exc})"
            )
            return
        self.producer_outputs[node.registration_name] = produced_outputs
        self._rebuild_visible_state(upstream_outputs)
        run_record.status = "success"
        run_record.produced_outputs.extend(sorted(placeholder_names))
        run_record.finished_at = datetime.now(UTC).isoformat()
        if verbose:
            self.logger.info(
                f"Recovered placeholder output(s) '{', '.join(sorted(placeholder_names))}' "
                f"by re-running block '{node.registration_name}'"
            )
        for output_name in produced_outputs:
            self._sync_value_to_ancestors_without_invalidation(output_name)

    def _recovery_input_names(self, node: Any, registration: Any) -> list[str]:
        """Effective pipeline-facing input names one registration resolves."""
        if isinstance(registration, ExpressionRegistration):
            return list(node._effective_expression_input_names(registration))
        input_names: list[str] = []
        for parameter in callable_signature(registration.callable_obj).parameters.values():
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                var_pos_name = registration.var_pos_name or parameter.name
                args_registration = node.registered_args.get(var_pos_name)
                if args_registration is not None:
                    input_names.extend(args_registration.ordered_items)
                continue
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                var_kw_name = registration.var_kw_name or parameter.name
                kwargs_registration = node.registered_kwargs.get(var_kw_name)
                if kwargs_registration is not None:
                    input_names.extend(kwargs_registration.mapping_dct.values())
                continue
            if parameter.name in registration.param_mapping:
                mapped_name = registration.param_mapping[parameter.name]
            elif self.strict_mode:
                mapped_name = "logger" if parameter.name == "logger" else None
            else:
                mapped_name = parameter.name
            if mapped_name is not None:
                input_names.append(mapped_name)
        return input_names

    def _strict_unmapped_required_input(self, registration: Any) -> str | None:
        if not self.strict_mode or isinstance(registration, ExpressionRegistration):
            return None
        for parameter in callable_signature(registration.callable_obj).parameters.values():
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if parameter.name == "logger" or parameter.name in registration.param_mapping:
                continue
            if parameter.default is inspect.Parameter.empty:
                return parameter.name
        return None

    def _recovery_input_status(
        self,
        input_name: str,
        visible_outputs: dict[str, Any],
        parent_config: dict[str, Any] | None,
        defaults: dict[str, Any],
    ) -> str:
        """Classify how one required input resolves during placeholder recovery.

        Returns ``"ok"``, ``"placeholder"`` (a placeholder reference would reach
        the function), or ``"unresolvable"`` (no source supplies the input).
        """
        if input_name == "logger":
            return "ok"
        if input_name in visible_outputs:
            value = visible_outputs[input_name]
        elif input_name in self.manual_values:
            value = self.manual_values[input_name]
        elif input_name in self._ancestor_manual_values():
            value = self._ancestor_manual_values()[input_name]
        elif self._config_has_field(self.config, input_name):
            value = self._config_value(self.config, input_name)
        elif parent_config and input_name in parent_config:
            value = parent_config[input_name]
        elif input_name in defaults:
            return "ok"
        else:
            return "unresolvable"
        if isinstance(value, (RuntimeValueReference, DataclassValueReference)):
            return "placeholder"
        return "ok"
