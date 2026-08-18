from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

from .exceptions import PersistenceError, ResolutionError
from .models import ArtifactRecord, TorchStateArtifactRecord

if TYPE_CHECKING:
    from .pipeline_handler import PipelineHandler


@dataclass(frozen=True, slots=True)
class _BackupFingerprint:
    file_name: str
    size: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _BackupSnapshot:
    backup_root: Path
    root_registration_name: str
    state_payload: dict[str, object]
    fingerprints: tuple[_BackupFingerprint, ...]

    def payload_for_path(self, path_parts: tuple[str, ...]) -> dict[str, object]:
        if not path_parts:
            raise ResolutionError("Pipeline path cannot be empty")
        if path_parts[0] != self.root_registration_name:
            raise ResolutionError(
                f"Backup snapshot path must start with root '{self.root_registration_name}'"
            )
        payload = self.state_payload
        for child_name in path_parts[1:]:
            payload = _child_payload_for_name(payload, child_name)
        return payload

    def assert_unchanged(self) -> None:
        current = _fingerprint_required_files(self.backup_root)
        if current != self.fingerprints:
            raise PersistenceError(
                f"Backup snapshot at '{self.backup_root}' changed during recovery preflight"
            )


def read_backup_snapshot(root_pipeline: PipelineHandler) -> _BackupSnapshot:
    root = root_pipeline
    while root.parent_pipeline is not None:
        root = root.parent_pipeline
    backup_root = root.pipeline_backup_root
    if backup_root is None:
        raise PersistenceError("Root pipeline has no configured backup directory")
    if not backup_root.exists() or not backup_root.is_dir():
        raise PersistenceError(
            f"Configured backup directory does not exist: '{backup_root}'"
        )
    required = _required_paths(backup_root)
    loader = cast(
        Callable[[bytes], object],
        getattr(cast(object, root), "_load_pickle_with_missing_class_fallback"),
    )
    state_payload = _require_mapping(
        _load_pickle(required["pipeline_state.pkl"], loader, "pipeline_state.pkl"),
        "backup state payload",
    )
    _ = _load_pickle(required["config.pkl"], loader, "config.pkl")
    metadata = _load_pickle(required["pipeline_meta.pkl"], loader, "pipeline_meta.pkl")
    _validate_root_payload(state_payload, root, backup_root)
    _validate_metadata(metadata, root, backup_root)
    _validate_nested_payloads(state_payload)
    _validate_artifact_sources(
        state_payload,
        Path(str(state_payload["saved_project_root"])),
        backup_root,
    )
    return _BackupSnapshot(
        backup_root=backup_root,
        root_registration_name=root.registration_name,
        state_payload=state_payload,
        fingerprints=_fingerprint_required_files(backup_root),
    )


def _required_paths(backup_root: Path) -> dict[str, Path]:
    paths = {
        name: backup_root / name
        for name in ("pipeline_state.pkl", "config.pkl", "pipeline_meta.pkl")
    }
    for name, path in paths.items():
        if not path.is_file():
            raise PersistenceError(
                f"Backup directory '{backup_root}' is missing required file '{name}'"
            )
    return paths


def _load_pickle(path: Path, loader: Callable[[bytes], object], label: str) -> object:
    try:
        return loader(path.read_bytes())
    except Exception as exc:
        raise PersistenceError(f"Failed to read backup file '{label}'") from exc


def _validate_root_payload(
    payload: object, root: PipelineHandler, backup_root: Path
) -> None:
    mapping = _require_mapping(payload, "backup state payload")
    _require_root_shape(mapping, "backup state payload")
    if mapping["registration_name"] != root.registration_name:
        raise PersistenceError(
            "Backup root registration does not match the live root pipeline"
        )
    if mapping["saved_project_root"] != str(root.project_root):
        raise PersistenceError("Backup saved project root does not match the live root")
    if mapping["pipeline_backup_directory"] != str(backup_root):
        raise PersistenceError("Backup payload directory does not match the configured backup")


def _validate_metadata(
    metadata: object, root: PipelineHandler, backup_root: Path
) -> None:
    mapping = _require_mapping(metadata, "backup metadata")
    if mapping.get("pipeline_directory") != str(root.project_root):
        raise PersistenceError("Backup metadata project root does not match the live root")
    if mapping.get("pipeline_backup_directory") != str(backup_root):
        raise PersistenceError(
            "Backup metadata directory does not match the configured backup"
        )


def _validate_nested_payloads(payload: dict[str, object]) -> None:
    _require_root_shape(payload, "backup payload")
    seen_children: set[str] = set()
    for node in _require_list(payload.get("nodes"), "backup payload nodes"):
        node_mapping = _require_mapping(node, "backup node")
        kind = node_mapping.get("kind")
        if kind == "pipeline":
            registration_name = _require_string(
                node_mapping.get("registration_name"), "pipeline node registration_name"
            )
            if registration_name in seen_children:
                raise PersistenceError(
                    f"Backup payload contains duplicate child pipeline '{registration_name}'"
                )
            seen_children.add(registration_name)
            child_payload = _require_mapping(
                node_mapping.get("payload"), "child pipeline payload"
            )
            _require_root_shape(child_payload, f"child pipeline payload '{registration_name}'")
            if child_payload["registration_name"] != registration_name:
                raise PersistenceError(
                    f"Child pipeline payload '{registration_name}' does not match its node"
                )
            _validate_nested_payloads(child_payload)
            continue
        if kind != "block":
            raise PersistenceError(f"Backup node has unsupported kind '{kind}'")
        _ = _require_string(node_mapping.get("registration_name"), "block registration_name")


def _require_root_shape(payload: dict[str, object], label: str) -> None:
    _ = _require_string(payload.get("registration_name"), f"{label} registration_name")
    _ = _require_string(payload.get("saved_project_root"), f"{label} saved_project_root")
    backup_dir = payload.get("pipeline_backup_directory")
    if not isinstance(backup_dir, (str, type(None))):
        raise PersistenceError(f"{label} pipeline_backup_directory must be a string or None")
    _ = _require_list(payload.get("nodes"), f"{label} nodes")
    _ = _require_mapping(payload.get("manual_values"), f"{label} manual_values")
    producer_outputs = _require_mapping(payload.get("producer_outputs"), f"{label} producer_outputs")
    for owner_name, outputs in producer_outputs.items():
        _ = _require_mapping(outputs, f"{label} producer_outputs['{owner_name}']")
    _ = _require_mapping(payload.get("para_value_dict"), f"{label} para_value_dict")
    _ = _require_mapping(payload.get("artifact_registry"), f"{label} artifact_registry")
    _ = _require_list(payload.get("run_history"), f"{label} run_history")


def _child_payload_for_name(
    payload: dict[str, object], child_name: str
) -> dict[str, object]:
    for node in _require_list(payload.get("nodes"), "backup path nodes"):
        node_mapping = _require_mapping(node, "backup path node")
        if node_mapping.get("kind") != "pipeline":
            continue
        if node_mapping.get("registration_name") == child_name:
            return _require_mapping(
                node_mapping.get("payload"), f"child pipeline payload '{child_name}'"
            )
    raise ResolutionError(f"Backup snapshot does not contain pipeline path segment '{child_name}'")


def _fingerprint_required_files(backup_root: Path) -> tuple[_BackupFingerprint, ...]:
    paths = _required_paths(backup_root)
    return tuple(_fingerprint(path) for path in paths.values())


def _fingerprint(path: Path) -> _BackupFingerprint:
    stat_result = path.stat()
    return _BackupFingerprint(
        file_name=path.name,
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _validate_artifact_sources(
    value: object,
    saved_project_root: Path,
    backup_root: Path,
    seen: set[int] | None = None,
) -> None:
    visited = set() if seen is None else seen
    value_id = id(value)
    if value_id in visited:
        return
    visited.add(value_id)
    if isinstance(value, (ArtifactRecord, TorchStateArtifactRecord)):
        _validate_artifact_source(value.file_path, saved_project_root, backup_root)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_artifact_sources(key, saved_project_root, backup_root, visited)
            _validate_artifact_sources(item, saved_project_root, backup_root, visited)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _validate_artifact_sources(item, saved_project_root, backup_root, visited)


def _validate_artifact_source(
    recorded_path: str,
    saved_project_root: Path,
    backup_root: Path,
) -> None:
    try:
        relative_path = Path(recorded_path).relative_to(saved_project_root)
    except ValueError as exc:
        raise PersistenceError(
            "Saved artifact path is outside the saved project root"
        ) from exc
    source = backup_root / relative_path
    current = backup_root
    for part in relative_path.parts:
        current /= part
        if current.is_symlink():
            raise PersistenceError("Backup artifact source contains a symlink")
    if not source.exists() or not (source.is_file() or source.is_dir()):
        raise PersistenceError(f"Backup artifact source is missing: '{source}'")
    if source.is_dir() and any(path.is_symlink() for path in source.rglob("*")):
        raise PersistenceError("Backup artifact tree contains a symlink")


def _require_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PersistenceError(f"{label} must be a mapping")
    return cast(dict[str, object], value)


def _require_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise PersistenceError(f"{label} must be a list")
    return cast(list[object], value)


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise PersistenceError(f"{label} must be a string")
    return value
