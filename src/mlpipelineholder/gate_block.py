from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .exceptions import ExecutionError, RegistrationError
from .function_registry import inspect_input_names, resolve_callable
from .models import FunctionRegistration

if TYPE_CHECKING:
    from .pipeline_handler import PipelineHandler


class GateBlock:
    """Runs a single boolean function before the rest of a pipeline."""

    def __init__(self, parent: PipelineHandler, function_or_path: Any) -> None:
        callable_obj, import_path, function_name = resolve_callable(function_or_path)
        input_names = inspect_input_names(callable_obj)
        self.parent = parent
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
        if self.registration.import_path is None:
            raise RegistrationError(
                f"Gate function '{self.registration.function_name}' is not importable"
            )
        return {"import_path": self.registration.import_path}
