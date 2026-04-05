from __future__ import annotations

import importlib
import inspect
from functools import wraps
from typing import Any

from .exceptions import RegistrationError


def resolve_callable(function_or_path: Any) -> tuple[Any, str | None, str]:
    """Resolve a callable object or import path into a callable plus persistence metadata."""

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
        input_names.append(parameter.name)
    return input_names


def inspect_exposed_input_names(
    callable_obj: Any,
    param_mapping: dict[str, str] | None = None,
    var_pos_name: str | None = None,
    var_kw_name: str | None = None,
) -> list[str]:
    signature = inspect.signature(callable_obj)
    param_mapping = param_mapping or {}
    input_names: list[str] = []
    seen: set[str] = set()
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            exposed_name = var_pos_name or parameter.name
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            exposed_name = var_kw_name or parameter.name
        else:
            exposed_name = param_mapping.get(parameter.name, parameter.name)
        if exposed_name in seen:
            continue
        seen.add(exposed_name)
        input_names.append(exposed_name)
    return input_names


def rename_args(
    func: Any,
    param_mapping: dict[str, str] | None = None,
    var_pos_name: str | None = None,
    var_kw_name: str | None = None,
) -> Any:
    """Expose safer pipeline-facing parameter names without changing the original callable."""

    param_mapping = param_mapping or {}
    if len(set(param_mapping.values())) != len(param_mapping.values()):
        raise RegistrationError("param_mapping contains duplicate target names")

    signature = inspect.signature(func)
    renamed_parameters = []
    seen_names: set[str] = set()
    reverse_param_mapping = {new_name: old_name for old_name, new_name in param_mapping.items()}

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            new_name = var_pos_name or parameter.name
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            new_name = var_kw_name or parameter.name
        else:
            new_name = param_mapping.get(parameter.name, parameter.name)
        if new_name in seen_names:
            raise RegistrationError(f"Duplicate exposed argument name: {new_name}")
        seen_names.add(new_name)
        renamed_parameters.append(parameter.replace(name=new_name))

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        remapped_kwargs = {
            reverse_param_mapping.get(name, name): value for name, value in kwargs.items()
        }
        return func(*args, **remapped_kwargs)

    setattr(wrapper, "__signature__", signature.replace(parameters=renamed_parameters))
    setattr(wrapper, "__mlpipeline_original__", func)
    setattr(wrapper, "__mlpipeline_param_mapping__", dict(param_mapping))
    setattr(wrapper, "__mlpipeline_var_pos_name__", var_pos_name)
    setattr(wrapper, "__mlpipeline_var_kw_name__", var_kw_name)
    return wrapper


def default_map(callable_obj: Any) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    defaults: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.default is not inspect._empty:
            defaults[parameter.name] = parameter.default
    return defaults
