from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from types import SimpleNamespace
import sys

from .exceptions import PersistenceError, RegistrationError, ResolutionError
from .function_registry import resolve_callable
from .models import (
    CallableValueReference,
    DataclassValueReference,
    RuntimeCallableReference,
    RuntimeValueReference,
)


@dataclass(frozen=True, slots=True)
class _ResolutionContext:
    selection_label: str
    is_missing_main_placeholder: Callable[[object], bool]
    dataclass_class_resolver: Callable[[str, str | None], type | None] | None


def resolve_saved_root_variable(
    saved_root_payload: Mapping[str, object],
    name: str,
    *,
    is_missing_main_placeholder: Callable[[object], bool],
    dataclass_class_resolver: Callable[[str, str | None], type | None] | None = None,
) -> object:
    selected_value = _select_root_visible_value(saved_root_payload, name)
    return _resolve_selected_value(
        selected_value,
        selection_label=f"saved pipeline value '{name}'",
        is_missing_main_placeholder=is_missing_main_placeholder,
        dataclass_class_resolver=dataclass_class_resolver,
    )


def resolve_saved_config_field(
    saved_pipeline_payload: Mapping[str, object],
    name: str,
    *,
    is_missing_main_placeholder: Callable[[object], bool],
    dataclass_class_resolver: Callable[[str, str | None], type | None] | None = None,
) -> object:
    selected_value = _select_local_config_field(saved_pipeline_payload, name)
    return _resolve_selected_value(
        selected_value,
        selection_label=f"saved config field '{name}'",
        is_missing_main_placeholder=is_missing_main_placeholder,
        dataclass_class_resolver=dataclass_class_resolver,
    )


def _resolve_selected_value(
    selected_value: object,
    *,
    selection_label: str,
    is_missing_main_placeholder: Callable[[object], bool],
    dataclass_class_resolver: Callable[[str, str | None], type | None] | None = None,
) -> object:
    context = _ResolutionContext(
        selection_label=selection_label,
        is_missing_main_placeholder=is_missing_main_placeholder,
        dataclass_class_resolver=dataclass_class_resolver,
    )
    try:
        return _resolve_selected_graph(selected_value, context, {})
    except PersistenceError as exc:
        raise PersistenceError(
            f"Failed to resolve {selection_label} from backup"
        ) from exc


def _select_root_visible_value(saved_pipeline_payload: Mapping[str, object], name: str) -> object:
    local_values = saved_pipeline_payload.get("para_value_dict")
    if isinstance(local_values, Mapping) and name in local_values:
        return local_values[name]
    for child_payload in _child_pipeline_payloads(saved_pipeline_payload):
        child_values = child_payload.get("para_value_dict")
        if isinstance(child_values, Mapping) and name in child_values:
            return child_values[name]
        descendant_value = _select_descendant_visible_value(child_payload, name)
        if descendant_value is not _MISSING:
            return descendant_value
    raise ResolutionError(f"Unknown pipeline value: {name}")


def _select_descendant_visible_value(saved_pipeline_payload: Mapping[str, object], name: str) -> object:
    for child_payload in _child_pipeline_payloads(saved_pipeline_payload):
        child_values = child_payload.get("para_value_dict")
        if isinstance(child_values, Mapping) and name in child_values:
            return child_values[name]
        descendant_value = _select_descendant_visible_value(child_payload, name)
        if descendant_value is not _MISSING:
            return descendant_value
    return _MISSING


def _child_pipeline_payloads(saved_pipeline_payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    child_payloads: list[Mapping[str, object]] = []
    nodes = saved_pipeline_payload.get("nodes")
    if not isinstance(nodes, list):
        return child_payloads
    for node in nodes:
        match node:
            case {"kind": "pipeline", "payload": dict() as payload}:
                child_payloads.append({str(key): value for key, value in payload.items()})
    return child_payloads


def _select_local_config_field(saved_pipeline_payload: Mapping[str, object], name: str) -> object:
    serialized_config = saved_pipeline_payload.get("config")
    if not isinstance(serialized_config, Mapping):
        raise ResolutionError(f"Unknown config field: {name}")
    if serialized_config.get("__pipeline_serialized_config__") is not True:
        raise ResolutionError(f"Unknown config field: {name}")
    data = serialized_config.get("data")
    if not isinstance(data, Mapping) or name not in data:
        raise ResolutionError(f"Unknown config field: {name}")
    return data[name]


def _resolve_selected_graph(
    value: object,
    context: _ResolutionContext,
    memo: dict[int, object],
) -> object:
    value_id = id(value)
    cached = memo.get(value_id)
    if cached is not None:
        return cached

    match value:
        case CallableValueReference() as reference:
            return _restore_callable_value(reference, context.selection_label)
        case RuntimeCallableReference() as reference:
            return _restore_runtime_callable(reference, context.selection_label)
        case RuntimeValueReference() as reference:
            raise PersistenceError(
                f"Unsupported runtime placeholder '{reference.type_name}' found inside {context.selection_label}: {reference.reason}"
            )
        case DataclassValueReference() as reference:
            return _restore_dataclass_reference(reference, context, memo)
        case dict() if value.get("__pipeline_serialized_config__") is True:
            return _resolve_serialized_config_wrapper(value, context, memo)
        case dict():
            resolved_dict: dict[object, object] = {}
            memo[value_id] = resolved_dict
            for key, item in value.items():
                resolved_key = _resolve_selected_graph(key, context, memo)
                resolved_item = _resolve_selected_graph(item, context, memo)
                resolved_dict[resolved_key] = resolved_item
            return resolved_dict
        case list():
            resolved_list: list[object] = []
            memo[value_id] = resolved_list
            for item in value:
                resolved_list.append(_resolve_selected_graph(item, context, memo))
            return resolved_list
        case tuple():
            resolved_tuple = tuple(
                _resolve_selected_graph(item, context, memo) for item in value
            )
            memo[value_id] = resolved_tuple
            return resolved_tuple
        case set():
            resolved_set: set[object] = set()
            memo[value_id] = resolved_set
            for item in value:
                resolved_set.add(_resolve_selected_graph(item, context, memo))
            return resolved_set
        case frozenset():
            resolved_frozenset = frozenset(
                _resolve_selected_graph(item, context, memo) for item in value
            )
            memo[value_id] = resolved_frozenset
            return resolved_frozenset
        case _ if context.is_missing_main_placeholder(value):
            raise PersistenceError(
                f"Missing __main__ class placeholder found inside {context.selection_label}"
            )
        case _ if callable(value):
            return value
        case _:
            return value


def _resolve_serialized_config_wrapper(
    wrapper: dict[object, object],
    context: _ResolutionContext,
    memo: dict[int, object],
) -> object:
    wrapper_id = id(wrapper)
    wrapper_kind = wrapper.get("kind")
    wrapper_data = wrapper.get("data")
    if not isinstance(wrapper_data, Mapping):
        return wrapper
    if wrapper_kind == "dict":
        resolved_dict: dict[str, object] = {}
        memo[wrapper_id] = resolved_dict
        for key, item in wrapper_data.items():
            resolved_dict[str(key)] = _resolve_selected_graph(item, context, memo)
        return resolved_dict
    if wrapper_kind == "namespace":
        resolved_namespace = SimpleNamespace()
        memo[wrapper_id] = resolved_namespace
        for key, item in wrapper_data.items():
            setattr(
                resolved_namespace,
                str(key),
                _resolve_selected_graph(item, context, memo),
            )
        return resolved_namespace
    return wrapper


def _restore_callable_value(reference: CallableValueReference, selection_label: str) -> object:
    try:
        callable_obj, _, _ = resolve_callable(reference.import_path)
    except (ImportError, RegistrationError) as exc:
        raise PersistenceError(
            f"Importable callable '{reference.import_path}' required for {selection_label} could not be restored"
        ) from exc
    return callable_obj


def _restore_dataclass_reference(
    reference: DataclassValueReference,
    context: _ResolutionContext,
    memo: dict[int, object],
) -> object:
    """Rebuild a saved dataclass placeholder from its structured fields.

    Mirrors the load-time reconstruction semantics: a real dataclass instance
    when the class is importable (or resolvable through the optional
    ``dataclass_class_resolver``, for example a class produced by the live
    pipeline) and constructible from the saved fields, a ``SimpleNamespace``
    fallback otherwise. Nested field values are resolved through the graph
    resolver so callables and serialized config wrappers inside the dataclass
    are restored as well.
    """
    reference_id = id(reference)
    cached = memo.get(reference_id)
    if cached is not None:
        return cached
    data = {
        key: _resolve_selected_graph(item, context, memo)
        for key, item in reference.data.items()
    }
    candidate = _find_importable_dataclass_class(
        reference.class_name,
        reference.module,
    )
    if candidate is None and context.dataclass_class_resolver is not None:
        candidate = context.dataclass_class_resolver(
            reference.class_name,
            reference.module,
        )
    if candidate is not None:
        init_field_names = {field.name for field in fields(candidate) if field.init}
        try:
            reconstructed = candidate(
                **{
                    key: value
                    for key, value in data.items()
                    if key in init_field_names
                }
            )
            memo[reference_id] = reconstructed
            return reconstructed
        except TypeError:
            pass
    reconstructed = SimpleNamespace(**data)
    memo[reference_id] = reconstructed
    return reconstructed


def _find_importable_dataclass_class(
    class_name: str,
    module_name: str | None,
) -> type | None:
    """Locate an importable pure dataclass by name.

    When the saved module is known, the class is looked up there first;
    otherwise every loaded module's attributes are scanned. ``__main__``
    definitions (notebook-local classes) are preferred over ambiguous
    same-name matches.
    """
    if module_name is not None:
        module = sys.modules.get(module_name)
        if module is not None:
            candidate = getattr(module, class_name, None)
            if (
                isinstance(candidate, type)
                and is_dataclass(candidate)
                and candidate.__name__ == class_name
            ):
                return candidate
    candidates: list[type] = []
    for module in sys.modules.values():
        candidate = getattr(module, class_name, None)
        if (
            isinstance(candidate, type)
            and is_dataclass(candidate)
            and candidate.__name__ == class_name
        ):
            candidates.append(candidate)
    main_candidates = [
        candidate
        for candidate in candidates
        if getattr(candidate, "__module__", None) == "__main__"
    ]
    if main_candidates:
        return main_candidates[0]
    if candidates:
        return candidates[0]
    return None


def _restore_runtime_callable(
    reference: RuntimeCallableReference,
    selection_label: str,
) -> object:
    main_module = sys.modules.get("__main__")
    callable_obj = (
        None if main_module is None else getattr(main_module, reference.callable_name, None)
    )
    if not callable(callable_obj):
        raise PersistenceError(
            f"Required runtime callable '{reference.callable_name}' for {selection_label} is unavailable in __main__"
        )
    return callable_obj


_MISSING = object()
