"""Argument resolution for registered functions and expressions."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from ..core.models import (
    ArtifactRecord,
    CallableValueReference,
    DataclassValueReference,
    FunctionRegistration,
    RuntimeValueReference,
)
from ..exceptions import ResolutionError
from .function_registry import callable_signature, default_map


class ArgumentMixin:
    """Resolve function call arguments from overrides, outputs, config, and defaults."""

    if TYPE_CHECKING:
        logger: Any = None
        manual_values: dict[str, Any] = {}
        config: Any = None
        artifact_store: Any = None

        def list_declared_outputs(self) -> set[str]: ...
        def _ancestor_manual_values(self) -> dict[str, Any]: ...
        def _config_has_field(self, config_obj: Any, field_name: str) -> bool: ...
        def _config_value(self, config_obj: Any, field_name: str) -> Any: ...
        @staticmethod
        def _restore_callable_value(reference: CallableValueReference) -> Any: ...

    def _prepare_call_arguments(
        self,
        registration: FunctionRegistration,
        overrides: dict[str, Any],
        visible_outputs: dict[str, Any],
        parent_config: dict[str, Any] | None = None,
        block: Any | None = None,
    ) -> tuple[list[Any], dict[str, Any], list[str]]:
        defaults = default_map(registration.callable_obj)
        signature = callable_signature(registration.callable_obj)
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
            if input_name is None:
                value = None
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
        elif input_name in self.manual_values:
            value = self.manual_values[input_name]
        elif input_name in self._ancestor_manual_values():
            value = self._ancestor_manual_values()[input_name]
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
        if isinstance(value, CallableValueReference):
            value = self._restore_callable_value(value)
        if isinstance(value, (RuntimeValueReference, DataclassValueReference)):
            raise ResolutionError(
                f"Cannot resolve argument '{input_name}' for function '{function_name}': "
                f"the value was saved as a placeholder ({value.reason}) and cannot be restored; "
                "recreate or reset the value before running"
            )
        return value
