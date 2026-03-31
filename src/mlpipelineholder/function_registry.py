from __future__ import annotations

import importlib
import inspect
from typing import Any

from .exceptions import RegistrationError


def resolve_callable(function_or_path: Any) -> tuple[Any, str | None, str]:
    if isinstance(function_or_path, str):
        module_path, _, attr_name = function_or_path.rpartition(".")
        if not module_path or not attr_name:
            raise RegistrationError(f"Invalid import path: {function_or_path}")
        module = importlib.import_module(module_path)
        try:
            callable_obj = getattr(module, attr_name)
        except AttributeError as exc:
            raise RegistrationError(f"Cannot import callable: {function_or_path}") from exc
        if not callable(callable_obj):
            raise RegistrationError(f"Imported object is not callable: {function_or_path}")
        return callable_obj, function_or_path, attr_name

    if not callable(function_or_path):
        raise RegistrationError("Registered object must be callable or import path string")

    module_name = getattr(function_or_path, "__module__", None)
    qualname = getattr(
        function_or_path, "__qualname__", getattr(function_or_path, "__name__", "callable")
    )
    function_name = getattr(function_or_path, "__name__", qualname)
    import_path = None
    if (
        module_name
        and inspect.isfunction(function_or_path)
        and "<locals>" not in qualname
        and function_name != "<lambda>"
    ):
        import_path = f"{module_name}.{qualname}"
    return function_or_path, import_path, function_name


def inspect_input_names(callable_obj: Any) -> list[str]:
    signature = inspect.signature(callable_obj)
    input_names: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise RegistrationError("Variadic parameters (*args/**kwargs) are not supported")
        input_names.append(parameter.name)
    return input_names


def default_map(callable_obj: Any) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    defaults: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.default is not inspect._empty:
            defaults[parameter.name] = parameter.default
    return defaults
