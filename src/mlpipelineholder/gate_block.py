from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .exceptions import ExecutionError, RegistrationError
from .function_registry import inspect_input_names, resolve_callable
from .models import FunctionRegistration

if TYPE_CHECKING:
    from .pipeline_handler import PipelineHandler


class GateBlock:
    """Runs a single boolean function before the rest of a pipeline."""

    def __init__(self, parent: PipelineHandler, function_or_path: Any, expected_value: Any = True) -> None:
        self.parent = parent
        self.config_field_name: str | None = None
        self.expected_value = expected_value

        if isinstance(function_or_path, str) and "." not in function_or_path:
            self.config_field_name = function_or_path
            callable_obj = None
            import_path = None
            function_name = function_or_path
            input_names: list[str] = []
        else:
            callable_obj, import_path, function_name = resolve_callable(function_or_path)
            input_names = inspect_input_names(callable_obj)
        self.registration = FunctionRegistration(
            function_name=function_name,
            import_path=import_path,
            callable_obj=callable_obj,
            input_names=input_names,
            output_names=["__gate__"],
            save_to_disk=set(),
        )

    def evaluate(
        self,
        overrides: dict[str, Any],
        visible_outputs: dict[str, Any],
        parent_config: Any | None = None,
    ) -> bool:
        if self.config_field_name is not None:
            value = self.parent._resolve_named_input(
                self.config_field_name,
                self.registration.function_name,
                overrides,
                visible_outputs,
                parent_config,
                {},
                [],
                set(visible_outputs).union(self.parent.list_declared_outputs()),
            )
            return value == self.expected_value

        positional_args, keyword_args, _ = self.parent._prepare_call_arguments(
            self.registration,
            overrides,
            visible_outputs,
            parent_config,
        )
        result = self.registration.callable_obj(*positional_args, **keyword_args)
        if not isinstance(result, bool):
            raise ExecutionError("Gate block must return a boolean value")
        return result

    def serialize(self) -> dict[str, str]:
        if self.config_field_name is not None:
            return {
                "kind": "config_field",
                "field_name": self.config_field_name,
                "expected_value": self.expected_value,
            }
        if self.registration.import_path is None:
            raise RegistrationError(
                f"Gate function '{self.registration.function_name}' is not importable"
            )
        return {"kind": "callable", "import_path": self.registration.import_path}
