"""Saved-payload structural validation and missing-__main__ placeholder handling."""

from __future__ import annotations

from typing import Any

from ..core.models import RuntimeCallableReference
from ..exceptions import PersistenceError
from ..execution.function_registry import resolve_callable
from .object_storage import record_from_payload


def contains_missing_main_placeholder(cls: Any, value: Any) -> bool:
    if cls._is_missing_main_placeholder(value):
        return True
    if isinstance(value, dict):
        return any(contains_missing_main_placeholder(cls, item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(contains_missing_main_placeholder(cls, item) for item in value)
    return False


def validate_loaded_payload_placeholders(cls: Any, payload: dict[str, Any]) -> None:
    if contains_missing_placeholder_outside_config(cls, payload):
        raise PersistenceError(
            "Failed to load pipeline project because a missing __main__ class was found outside the saved pipeline config"
        )


def replace_missing_runtime_payload_values(
    cls: Any,
    payload: Any,
    pipeline_path: tuple[str, ...] = (),
) -> list[str]:
    if not isinstance(payload, dict):
        return []
    registration_name = payload.get("registration_name")
    current_path = (
        (*pipeline_path, registration_name)
        if isinstance(registration_name, str)
        else pipeline_path
    )
    path_label = "/".join(current_path) or "pipeline"
    invalid: list[str] = []
    for mapping_name, owner_kind in (
        ("manual_values", "constant"),
        ("para_value_dict", "pipeline value"),
        ("artifact_registry", "artifact registry value"),
    ):
        mapping = payload.get(mapping_name)
        if not isinstance(mapping, dict):
            continue
        for value_name, value in list(mapping.items()):
            if contains_missing_main_placeholder(cls, value):
                mapping[value_name] = None
                invalid.append(f"{owner_kind} '{path_label}.{value_name}'")
    producer_outputs = payload.get("producer_outputs")
    if isinstance(producer_outputs, dict):
        for node_name, outputs in producer_outputs.items():
            if not isinstance(outputs, dict):
                continue
            for output_name, value in list(outputs.items()):
                if contains_missing_main_placeholder(cls, value):
                    outputs[output_name] = None
                    invalid.append(
                        f"output '{path_label}.{node_name}.{output_name}'"
                    )
    nodes = payload.get("nodes")
    if isinstance(nodes, list):
        for node_payload in nodes:
            if (
                isinstance(node_payload, dict)
                and node_payload.get("kind") == "pipeline"
            ):
                invalid.extend(
                    replace_missing_runtime_payload_values(
                        cls,
                        node_payload.get("payload"),
                        current_path,
                    )
                )
    return invalid


def validate_loaded_payload_structure(cls: Any, payload: Any) -> None:
    """Reject obviously unbuildable payloads before the working tree is touched.

    Full validation happens while ``_from_payload`` rebuilds the tree; this
    pass only performs cheap, side-effect-free checks (payload shape, node
    kinds, callable references, expression syntax) so a corrupted save is
    caught before it could replace a working tree that then fails to load.
    """
    if not isinstance(payload, dict):
        raise PersistenceError("Saved pipeline payload is not a mapping")
    require_loaded_payload_keys(
        payload,
        ("registration_name", "config", "nodes"),
        owner_label="pipeline",
    )
    if not isinstance(payload["nodes"], list):
        raise PersistenceError("Saved pipeline payload has a non-list 'nodes' entry")
    for mapping_key in (
        "manual_values",
        "producer_outputs",
        "para_value_dict",
        "artifact_registry",
        "object_storage",
    ):
        if mapping_key in payload and not isinstance(payload[mapping_key], dict):
            raise PersistenceError(
                f"Saved pipeline payload has a non-mapping '{mapping_key}' entry"
            )
    for hash_id, record_payload in payload.get("object_storage", {}).items():
        if not isinstance(hash_id, str):
            raise PersistenceError(
                "Saved object storage contains a non-string hash key"
            )
        record = record_from_payload(record_payload)
        if record.hash_id != hash_id:
            raise PersistenceError(
                f"Saved object storage hash key '{hash_id}' does not match "
                f"record hash_id '{record.hash_id}'"
            )
    for node_payload in payload["nodes"]:
        if not isinstance(node_payload, dict):
            raise PersistenceError("Saved pipeline payload has an invalid node entry")
        require_loaded_payload_keys(
            node_payload,
            ("kind", "registration_name", "execution_priority"),
            owner_label="node",
        )
        kind = node_payload["kind"]
        if kind == "pipeline":
            require_loaded_payload_keys(
                node_payload,
                ("payload",),
                owner_label=f"pipeline node '{node_payload['registration_name']}'",
            )
            validate_loaded_payload_structure(cls, node_payload["payload"])
            continue
        if kind == "block":
            require_loaded_payload_keys(
                node_payload,
                ("functions",),
                owner_label=f"block '{node_payload['registration_name']}'",
            )
            if not isinstance(node_payload["functions"], list):
                raise PersistenceError(
                    f"Saved block '{node_payload['registration_name']}' has a non-list 'functions' entry"
                )
            for args_payload in node_payload.get("registered_args", []):
                if not isinstance(args_payload, dict):
                    raise PersistenceError("Saved pipeline payload has an invalid args entry")
                require_loaded_payload_keys(
                    args_payload,
                    ("name", "ordered_items"),
                    owner_label=f"args registration in block '{node_payload['registration_name']}'",
                )
            for kwargs_payload in node_payload.get("registered_kwargs", []):
                if not isinstance(kwargs_payload, dict):
                    raise PersistenceError("Saved pipeline payload has an invalid kwargs entry")
                require_loaded_payload_keys(
                    kwargs_payload,
                    ("name", "mapping_dct"),
                    owner_label=f"kwargs registration in block '{node_payload['registration_name']}'",
                )
            for function_payload in node_payload["functions"]:
                validate_function_payload_structure(
                    cls,
                    function_payload,
                    owner_label=f"block '{node_payload['registration_name']}'",
                )
            continue
        raise PersistenceError(
            f"Saved pipeline payload has an unknown node kind: {kind!r}"
        )
    gate_payload = payload.get("gate")
    if gate_payload is None:
        return
    if not isinstance(gate_payload, dict):
        raise PersistenceError("Saved pipeline payload has an invalid gate entry")
    require_loaded_payload_keys(
        gate_payload,
        ("kind",),
        owner_label="gate",
    )
    gate_kind = gate_payload["kind"]
    if gate_kind == "callable":
        require_loaded_payload_keys(
            gate_payload,
            ("import_path",),
            owner_label="callable gate",
        )
        validate_import_path_payload(gate_payload["import_path"])
    elif gate_kind == "config_field":
        require_loaded_payload_keys(
            gate_payload,
            ("field_name",),
            owner_label="config-field gate",
        )
    else:
        raise PersistenceError(
            f"Saved pipeline payload has an unknown gate kind: {gate_kind!r}"
        )


def require_loaded_payload_keys(
    payload: dict[str, Any],
    required_keys: tuple[str, ...],
    *,
    owner_label: str,
) -> None:
    for required_key in required_keys:
        if required_key not in payload:
            raise PersistenceError(
                f"Saved {owner_label} payload is missing required key '{required_key}'"
            )


def validate_function_payload_structure(
    cls: Any,
    function_payload: Any,
    *,
    owner_label: str,
) -> None:
    if not isinstance(function_payload, dict):
        raise PersistenceError("Saved pipeline payload has an invalid function entry")
    require_loaded_payload_keys(
        function_payload,
        ("kind", "output_names", "save_to_disk"),
        owner_label=f"function in {owner_label}",
    )
    function_kind = function_payload["kind"]
    if function_kind == "expression":
        require_loaded_payload_keys(
            function_payload,
            ("code",),
            owner_label=f"expression in {owner_label}",
        )
        code = function_payload["code"]
        if not isinstance(code, str) or not code.strip():
            raise PersistenceError("Saved pipeline payload has an empty expression")
        try:
            compile(code, "<pipeline_expression>", "exec")
        except SyntaxError as exc:
            raise PersistenceError(
                f"Saved pipeline payload has invalid expression code: {exc}"
            ) from exc
        return
    if function_kind != "function":
        raise PersistenceError(
            f"Saved function in {owner_label} has an unknown kind: {function_kind!r}"
        )
    import_path = function_payload.get("import_path")
    partial_payload = function_payload.get("partial")
    runtime_reference = function_payload.get("runtime_callable_reference")
    if import_path is not None:
        validate_import_path_payload(import_path)
    elif partial_payload is not None:
        if not isinstance(partial_payload, dict):
            raise PersistenceError(
                "Saved pipeline payload has an invalid partial callable entry"
            )
        cls._restore_partial_callable(partial_payload, owner_label)
    elif runtime_reference is not None:
        if not isinstance(runtime_reference, RuntimeCallableReference):
            raise PersistenceError(
                "Saved pipeline payload has an invalid runtime callable reference"
            )
        cls._restore_runtime_registered_callable(runtime_reference, owner_label)
    else:
        raise PersistenceError("Saved pipeline function has no callable reference")


def validate_import_path_payload(import_path: Any) -> None:
    if not isinstance(import_path, str):
        raise PersistenceError(
            f"Saved pipeline payload has an invalid callable import path: {import_path!r}"
        )
    try:
        resolve_callable(import_path)
    except Exception as exc:
        raise PersistenceError(
            f"Saved pipeline callable '{import_path}' could not be imported: {exc}"
        ) from exc


def contains_missing_placeholder_outside_config(
    cls: Any,
    value: Any,
    *,
    inside_config: bool = False,
) -> bool:
    if cls._is_missing_main_placeholder(value):
        return not inside_config
    if isinstance(value, dict):
        for key, item in value.items():
            if contains_missing_placeholder_outside_config(
                cls,
                item,
                inside_config=inside_config or key == "config",
            ):
                return True
        return False
    if isinstance(value, (list, tuple, set)):
        return any(
            contains_missing_placeholder_outside_config(
                cls,
                item,
                inside_config=inside_config,
            )
            for item in value
        )
    return False
