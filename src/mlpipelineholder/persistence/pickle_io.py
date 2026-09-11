"""Atomic pickle I/O and tolerant unpickling for saved pipeline state."""

from __future__ import annotations

import os
import pickle
from io import BytesIO
from pathlib import Path
from typing import Any

_ACTIVE_PACKAGE_ROOT = (__package__ or __name__).rsplit(".", maxsplit=1)[0]
_PERSISTED_PACKAGE_ROOTS = ("mlpipelineholder", "src.mlpipelineholder")

# Historical flat module names (pre-subpackage layout) mapped to their canonical homes so
# pipelines saved by older releases still unpickle after the reorganization.
_LEGACY_FLAT_MODULES: dict[str, str] = {
    "artifact_recovery": "persistence.artifact_recovery",
    "artifact_store": "persistence.artifacts.store",
    "backup_recovery": "persistence.backup.recovery",
    "backup_recovery_service": "persistence.backup.service",
    "backup_snapshot": "persistence.backup.snapshot",
    "backup_value_resolver": "persistence.backup.value_resolver",
    "code_comparison": "execution.code_comparison",
    "execution_block": "execution.block",
    "function_registry": "execution.function_registry",
    "gate_block": "execution.gate_block",
    "logger": "presentation.logger",
    "models": "core.models",
    "naming": "core.naming",
    "object_storage": "persistence.object_storage",
    "optuna_api": "integrations.optuna.api",
    "optuna_sqlite": "integrations.optuna.sqlite",
    "optuna_support": "integrations.optuna.support",
    "output_pointers": "state.output_pointers",
    "pipeline_handler": "pipeline_holder",
    "serializers": "persistence.artifacts.serializers",
}


class _MissingMainClassPlaceholder:
    pass


class _MissingClassUnpickler(pickle.Unpickler):
    def __init__(self, file_obj: BytesIO) -> None:
        super().__init__(file_obj)
        self._placeholder_cache: dict[tuple[str, str], type[Any]] = {}

    def find_class(self, module: str, name: str) -> Any:
        resolved_module = module
        for persisted_root in _PERSISTED_PACKAGE_ROOTS:
            if module == persisted_root or module.startswith(f"{persisted_root}."):
                resolved_module = f"{_ACTIVE_PACKAGE_ROOT}{module[len(persisted_root):]}"
                break
        prefix = f"{_ACTIVE_PACKAGE_ROOT}."
        if resolved_module.startswith(prefix):
            mapped = _LEGACY_FLAT_MODULES.get(resolved_module[len(prefix):])
            if mapped is not None:
                resolved_module = prefix + mapped
        try:
            return super().find_class(resolved_module, name)
        except (AttributeError, ImportError, ModuleNotFoundError):
            if module != "__main__":
                raise
            cache_key = (module, name)
            placeholder = self._placeholder_cache.get(cache_key)
            if placeholder is not None:
                return placeholder
            placeholder = type(name, (_MissingMainClassPlaceholder,), {})
            placeholder.__module__ = module
            self._placeholder_cache[cache_key] = placeholder
            return placeholder


def atomic_pickle_dump(obj: Any, path: Path) -> None:
    """Write a pickle payload to `path` atomically.

    The payload is written to a sibling ``*.tmp`` file, fsynced, then renamed
    over `path`, so a crash mid-write never corrupts the previous file. The
    temporary file is removed on any failure.
    """
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("wb") as handle:
            pickle.dump(obj, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_pipeline_metadata(
    project_root: Path, target: Path, saved_backup_root: Path | None
) -> None:
    with (target / "pipeline_meta.pkl").open("wb") as handle:
        pickle.dump(
            {
                "pipeline_directory": str(project_root),
                "pipeline_backup_directory": (
                    None
                    if saved_backup_root is None
                    else str(saved_backup_root)
                ),
            },
            handle,
        )


def load_pipeline_metadata(path: Path) -> dict[str, Any] | None:
    metadata_path = path / "pipeline_meta.pkl"
    if not metadata_path.exists():
        return None
    with metadata_path.open("rb") as handle:
        return load_pickle_with_missing_class_fallback(handle.read())


def load_pickle_with_missing_class_fallback(raw_bytes: bytes) -> Any:
    return _MissingClassUnpickler(BytesIO(raw_bytes)).load()
