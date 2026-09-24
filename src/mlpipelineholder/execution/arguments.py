"""Argument resolution for registered functions and expressions."""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ..core.models import (
    ArtifactRecord,
    CallableValueReference,
    DataclassValueReference,
    FunctionRegistration,
    ResolutionSource,
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
        strict_mode: bool = False
        parent_pipeline: Any = None

        def list_declared_outputs(self) -> set[str]: ...
        def has_visible_output(self, variable_name: str) -> bool: ...
        def get_value(self, variable_name: str) -> Any: ...
        def _materialize_stored_value(
            self,
            value: Any,
            placeholder_error: str,
        ) -> Any: ...
        def _ancestor_manual_values(self) -> dict[str, Any]: ...
        def _ancestor_config_values(self) -> dict[str, Any]: ...
        def _config_has_field(self, config_obj: Any, field_name: str) -> bool: ...
        def _config_value(self, config_obj: Any, field_name: str) -> Any: ...
        def _get_stored_object_by_name(
            self,
            object_name: str,
            *,
            cache: bool,
        ) -> Any: ...
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
        strict_function_resolution = self.strict_mode and block is not None
        resolver_defaults = {} if strict_function_resolution else defaults
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
                            resolver_defaults,
                            loaded_artifacts,
                            declared_output_names,
                        )
                        for item_name in block.registered_args[input_name].ordered_items
                    ]
                elif strict_function_resolution:
                    value = []
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
                            resolver_defaults,
                            loaded_artifacts,
                            declared_output_names,
                        )
                        for key, item_name in block.registered_kwargs[input_name].mapping_dct.items()
                    }
                elif strict_function_resolution:
                    value = {}
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

            if parameter.name in registration.param_mapping:
                input_name = registration.param_mapping[parameter.name]
                if input_name is None:
                    value = None
                else:
                    value = self._resolve_named_input(
                        input_name,
                        registration.function_name,
                        overrides,
                        visible_outputs,
                        parent_config,
                        resolver_defaults,
                        loaded_artifacts,
                        declared_output_names,
                    )
            elif strict_function_resolution:
                if parameter.name == "logger":
                    value = self.logger
                elif parameter.default is not inspect.Parameter.empty:
                    value = parameter.default
                else:
                    raise ResolutionError(
                        f"Cannot resolve argument '{parameter.name}' for function "
                        f"'{registration.function_name}': strict mode requires an "
                        "explicit mapping or a callable default"
                    )
            else:
                input_name = parameter.name
                value = self._resolve_named_input(
                    input_name,
                    registration.function_name,
                    overrides,
                    visible_outputs,
                    parent_config,
                    resolver_defaults,
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
        include_root_storage: bool = False,
        cache_root_storage: bool = True,
    ) -> Any:
        value, _ = self._resolve_named_input_with_source(
            input_name,
            function_name,
            overrides,
            visible_outputs,
            parent_config,
            defaults,
            loaded_artifacts,
            declared_output_names,
            allow_missing=allow_missing,
            missing_value=missing_value,
            include_root_storage=include_root_storage,
            cache_root_storage=cache_root_storage,
        )
        return value

    def _resolve_named_input_with_source(
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
        include_root_storage: bool = False,
        cache_root_storage: bool = True,
        visible_constants: dict[str, Any] | None = None,
        same_node_previous_outputs: set[str] | None = None,
    ) -> tuple[Any, ResolutionSource]:
        del declared_output_names
        if input_name == "logger":
            value = self.logger
            source = ResolutionSource(kind="logger", name="logger")
        elif input_name in overrides:
            value = overrides[input_name]
            source = ResolutionSource(kind="runtime_override", name=input_name)
        elif input_name in visible_outputs:
            value = visible_outputs[input_name]
            source = ResolutionSource(
                kind="pipeline_output",
                name=input_name,
                same_node_previous_output=input_name
                in (same_node_previous_outputs or set()),
            )
        elif visible_constants is not None and input_name in visible_constants:
            value = visible_constants[input_name]
            source = ResolutionSource(kind="constant", name=input_name)
        elif input_name in self.manual_values:
            value = self.manual_values[input_name]
            source = ResolutionSource(kind="constant", name=input_name)
        elif input_name in self._ancestor_manual_values():
            value = self._ancestor_manual_values()[input_name]
            source = ResolutionSource(kind="constant", name=input_name)
        elif self._config_has_field(self.config, input_name):
            value = self._config_value(self.config, input_name)
            source = ResolutionSource(kind="config", name=input_name)
        elif parent_config and input_name in parent_config:
            value = parent_config[input_name]
            source = ResolutionSource(kind="config", name=input_name)
        elif input_name in defaults:
            value = defaults[input_name]
            source = ResolutionSource(kind="function_default", name=input_name)
        elif include_root_storage and self.parent_pipeline is None:
            value = self._get_stored_object_by_name(
                input_name,
                cache=cache_root_storage,
            )
            source = ResolutionSource(kind="storage", name=input_name, materialized=True)
        elif allow_missing:
            value = missing_value
            source = ResolutionSource(kind="missing", name=input_name)
        else:
            raise ResolutionError(
                f"Cannot resolve argument '{input_name}' for function '{function_name}'"
            )

        if isinstance(value, ArtifactRecord):
            value = self.artifact_store.load(value)
            loaded_artifacts.append(input_name)
            source = replace(source, materialized=True)
        if isinstance(value, CallableValueReference):
            value = self._restore_callable_value(value)
            source = replace(source, materialized=True)
        if isinstance(value, (RuntimeValueReference, DataclassValueReference)):
            raise ResolutionError(
                f"Cannot resolve argument '{input_name}' for function '{function_name}': "
                f"the value was saved as a placeholder ({value.reason}) and cannot be restored; "
                "recreate or reset the value before running"
            )
        return value, source

    def _resolve_investigation_input(
        self,
        input_name: str,
        function_name: str,
        *,
        visible_outputs: dict[str, Any] | None = None,
    ) -> Any:
        if visible_outputs is None:
            resolved_outputs: dict[str, Any] = {}
            if self.has_visible_output(input_name):
                resolved_outputs[input_name] = self.get_value(input_name)
        else:
            resolved_outputs = {}
            if input_name in visible_outputs:
                value = visible_outputs[input_name]
                resolved_outputs[input_name] = self._materialize_stored_value(
                    value,
                    f"Cannot inspect value '{input_name}': it was saved as a placeholder "
                    f"({value.reason}) and cannot be restored"
                    if isinstance(value, (RuntimeValueReference, DataclassValueReference))
                    else "",
                )
        return self._resolve_named_input(
            input_name,
            function_name,
            {},
            resolved_outputs,
            self._ancestor_config_values(),
            {},
            [],
            self.list_declared_outputs(),
            include_root_storage=True,
            cache_root_storage=False,
        )
