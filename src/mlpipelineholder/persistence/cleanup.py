"""Obsolete saved-object cleanup: artifact generations and orphaned child trees."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from ..core.models import ArtifactRecord, TorchStateArtifactRecord
from ..integrations.optuna.support import OPTUNA_STUDIES_DB_NAME

_SAVED_GENERATION_RE = re.compile(
    r"^.+__.+__.+__[0-9a-f]{32}\.(?:json|npy|pkl|pt|feather|parquet|bin)$"
)
_MANAGED_CHILD_DIR_NAMES = ("artifacts", "children", "metadata", "history_logs")
_MANAGED_CHILD_FILE_NAMES = (
    "pipeline_state.pkl",
    "config.pkl",
    "pipeline_meta.pkl",
    OPTUNA_STUDIES_DB_NAME,
)


def _referenced_payload_artifact_paths(payload: Any) -> set[Path]:
    """Return resolved absolute paths of every artifact the payload references."""
    paths: set[Path] = set()
    seen: set[int] = set()

    def walk(value: Any) -> None:
        if isinstance(value, (ArtifactRecord, TorchStateArtifactRecord)):
            paths.add(Path(value.file_path).resolve())
            return
        if isinstance(value, dict):
            if id(value) in seen:
                return
            seen.add(id(value))
            for key, item in value.items():
                walk(key)
                walk(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            if id(value) in seen:
                return
            seen.add(id(value))
            for item in value:
                walk(item)

    walk(payload)
    return paths


def _collect_obsolete_artifact_entries(
    payload: dict[str, Any],
    target_root: Path,
    live_paths: set[Path],
    obsolete: list[Path],
) -> None:
    """Append unreferenced framework artifact generations to ``obsolete``."""
    artifacts_dir = target_root / "artifacts"
    if not artifacts_dir.is_symlink() and artifacts_dir.is_dir():
        for block_dir in artifacts_dir.iterdir():
            if block_dir.is_symlink() or not block_dir.is_dir():
                continue
            for entry in block_dir.iterdir():
                if entry.is_symlink():
                    continue
                if _SAVED_GENERATION_RE.fullmatch(entry.name) is None:
                    continue
                if entry.resolve() in live_paths:
                    continue
                obsolete.append(entry)
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return
    for node in nodes:
        if not isinstance(node, dict) or node.get("kind") != "pipeline":
            continue
        name = node.get("registration_name")
        child_payload = node.get("payload")
        if not isinstance(name, str) or not isinstance(child_payload, dict):
            continue
        _collect_obsolete_artifact_entries(
            child_payload,
            target_root / "children" / name,
            live_paths,
            obsolete,
        )


def _collect_obsolete_child_dirs(
    payload: dict[str, Any],
    pipeline_root: Path,
    live_paths: set[Path],
    obsolete_children: list[Path],
) -> None:
    """Classify orphaned child pipeline dirs as obsolete or protected."""
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return
    active_names = {
        node["registration_name"]
        for node in nodes
        if isinstance(node, dict)
        and node.get("kind") == "pipeline"
        and isinstance(node.get("registration_name"), str)
    }
    children_dir = pipeline_root / "children"
    if children_dir.is_dir():
        for child in children_dir.iterdir():
            if child.is_symlink() or not child.is_dir():
                continue
            if child.name in active_names:
                continue
            child_root = child.resolve()
            if any(
                live_path == child_root or live_path.is_relative_to(child_root)
                for live_path in live_paths
            ):
                continue
            if _is_fully_managed_child_tree(child):
                obsolete_children.append(child)
    for node in nodes:
        if not isinstance(node, dict) or node.get("kind") != "pipeline":
            continue
        name = node.get("registration_name")
        child_payload = node.get("payload")
        if not isinstance(name, str) or not isinstance(child_payload, dict):
            continue
        _collect_obsolete_child_dirs(
            child_payload,
            pipeline_root / "children" / name,
            live_paths,
            obsolete_children,
        )


def _is_fully_managed_child_tree(root: Path) -> bool:
    """True when a child directory holds only framework-managed entries."""
    for path in root.rglob("*"):
        if path.is_symlink():
            return False
        parts = path.relative_to(root).parts
        if parts[0] in _MANAGED_CHILD_DIR_NAMES:
            continue
        if len(parts) == 1 and path.is_file() and parts[0] in _MANAGED_CHILD_FILE_NAMES:
            continue
        return False
    return True


def _delete_saved_path(path: Path, target_root: Path) -> bool:
    """Safely delete one obsolete path inside the project root."""
    if path.is_symlink() or not path.exists():
        return False
    if not path.resolve().is_relative_to(target_root.resolve()):
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True
