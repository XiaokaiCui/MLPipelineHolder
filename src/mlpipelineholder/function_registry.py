from __future__ import annotations

import importlib
import inspect
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from functools import partial, wraps
from typing import Any, get_type_hints

from .code_comparison import code_objects_equal
from .exceptions import RegistrationError


def callable_signature(callable_obj: Any) -> inspect.Signature:
    try:
        return inspect.signature(callable_obj)
    except ValueError:
        if callable_obj is not partial:
            raise
        return inspect.Signature(
            parameters=[
                inspect.Parameter("func", inspect.Parameter.POSITIONAL_ONLY),
                inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
                inspect.Parameter("keywords", inspect.Parameter.VAR_KEYWORD),
            ]
        )


def resolve_callable(function_or_path: Any) -> tuple[Any, str | None, str]:
    """Resolve a callable object or import path into a callable plus persistence metadata."""

    if isinstance(function_or_path, str):
        callable_obj = _resolve_import_path_object(function_or_path)
        if not callable(callable_obj):
            raise RegistrationError(f"Imported object is not callable: {function_or_path}")
        _, _, attr_name = function_or_path.rpartition(".")
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
    elif (
        inspect.ismethod(function_or_path)
        and isinstance(function_or_path.__self__, type)
        and "<locals>" not in qualname
    ):
        owner_class = function_or_path.__self__
        import_path = (
            f"{owner_class.__module__}."
            f"{owner_class.__qualname__}.{function_or_path.__func__.__name__}"
        )
    return function_or_path, import_path, function_name


def _resolve_import_path_object(function_or_path: str) -> Any:
    """Import the longest module prefix of a dotted path, then traverse attributes.

    Import paths may be qualified by classes (e.g. ``mypkg.module.MyClass.method``
    for a static method), so the resolver imports the longest prefix that is a real
    module and walks the remaining components with ``getattr``. A prefix that fails
    to import because it is simply not a module is skipped; a module that fails to
    import because of a missing dependency propagates its original error.
    """
    parts = function_or_path.split(".")
    if len(parts) < 2:
        raise RegistrationError(f"Invalid import path: {function_or_path}")
    for index in range(len(parts) - 1, 0, -1):
        candidate_module = ".".join(parts[:index])
        try:
            module = importlib.import_module(candidate_module)
        except ModuleNotFoundError as exc:
            # exc.name is the first non-importable prefix, which may be an
            # ancestor of the candidate when its parent is a plain module
            # rather than a package; only a genuinely broken module
            # (a missing dependency) should propagate.
            if exc.name != candidate_module and not candidate_module.startswith(f"{exc.name}."):
                raise
            continue
        obj: Any = module
        try:
            for attribute in parts[index:]:
                obj = getattr(obj, attribute)
        except AttributeError as exc:
            raise RegistrationError(f"Cannot import callable: {function_or_path}") from exc
        return obj
    raise RegistrationError(f"Invalid import path: {function_or_path}")


def callable_identity_matches(
    existing_import_path: str | None,
    existing_callable: Any,
    import_path: str | None,
    callable_obj: Any,
) -> bool:
    """Whether a stored registration and a new callable refer to the same function.

    ``functools.partial`` objects are compared structurally: the wrapped callable
    is matched recursively (import path, or identity for runtime callables) and
    the bound ``args``/``keywords`` must be equal — so rebuilding an identical
    partial inline still no-ops, while a changed binding or a redefined wrapped
    function correctly replaces. Importable module functions are matched by
    import path (both the loaded registration and a freshly resolved callable
    point at the same module attribute). Redefined ``__main__`` functions are
    matched by executable definition so rerunning an unchanged notebook cell
    no-ops, while runtime-only callables still require object identity.
    """
    if isinstance(existing_callable, partial) and isinstance(callable_obj, partial):
        return _partial_identity_matches(existing_callable, callable_obj)
    if existing_import_path is not None and import_path is not None:
        if (
            existing_import_path.startswith("__main__")
            or import_path.startswith("__main__")
        ):
            return existing_import_path == import_path and (
                existing_callable is callable_obj
                or _main_function_definitions_equal(existing_callable, callable_obj)
            )
        return existing_import_path == import_path
    return existing_callable is callable_obj


def _main_function_definitions_equal(existing: Any, new: Any) -> bool:
    if not inspect.isfunction(existing) or not inspect.isfunction(new):
        return False
    return (
        code_objects_equal(existing.__code__, new.__code__, _values_equal)
        and _values_equal(existing.__defaults__, new.__defaults__)
        and _values_equal(existing.__kwdefaults__, new.__kwdefaults__)
    )


def _partial_identity_matches(
    existing: partial[Any],
    new: partial[Any],
) -> bool:
    existing_func, existing_path, _ = resolve_callable(existing.func)
    new_func, new_path, _ = resolve_callable(new.func)
    if not callable_identity_matches(
        existing_path,
        existing_func,
        new_path,
        new_func,
    ):
        return False
    return _values_equal(existing.args, new.args) and _values_equal(
        existing.keywords,
        new.keywords,
    )


def _values_equal(left: Any, right: Any) -> bool:
    """Deep equality for config-like values, robust to numpy/pandas objects."""
    if type(left) is not type(right):
        return False
    if is_dataclass(left) and not isinstance(left, type):
        for field_info in fields(left):
            left_present = hasattr(left, field_info.name)
            right_present = hasattr(right, field_info.name)
            if left_present != right_present:
                return False
            if left_present and not _values_equal(
                getattr(left, field_info.name),
                getattr(right, field_info.name),
            ):
                return False
        return True
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return False
        return all(_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right)
        )
    try:
        import numpy as np

        if isinstance(left, np.ndarray):
            return left.shape == right.shape and bool((left == right).all())
    except Exception:
        pass
    equals = getattr(left, "equals", None)
    if callable(equals):
        try:
            return bool(equals(right))
        except Exception:
            return False
    try:
        return bool(left == right)
    except Exception:
        return False


def inspect_input_names(callable_obj: Any) -> list[str]:
    signature = callable_signature(callable_obj)
    input_names: list[str] = []
    for parameter in signature.parameters.values():
        input_names.append(parameter.name)
    return input_names


def infer_declared_output_count(callable_obj: Any) -> int | None:
    signature = callable_signature(callable_obj)
    try:
        hints = get_type_hints(callable_obj)
    except Exception:
        return None
    annotation = hints.get("return", signature.return_annotation)
    if annotation is inspect.Signature.empty:
        return None
    if annotation is None:
        return 0
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())
    if origin is tuple and args:
        if len(args) == 2 and args[1] is Ellipsis:
            return None
        return len(args)
    if origin in (list, Sequence):
        return None
    return 1


def inspect_exposed_input_names(
    callable_obj: Any,
    param_mapping: dict[str, str | None] | None = None,
    var_pos_name: str | None = None,
    var_kw_name: str | None = None,
) -> list[str]:
    signature = callable_signature(callable_obj)
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
        if exposed_name is None:
            continue
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

    signature = callable_signature(func)
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
    signature = callable_signature(callable_obj)
    defaults: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.default is not inspect._empty:
            defaults[parameter.name] = parameter.default
    return defaults
