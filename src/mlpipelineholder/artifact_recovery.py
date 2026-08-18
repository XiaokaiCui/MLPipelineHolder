from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import shutil
from typing import cast, final
from uuid import uuid4

from .exceptions import PersistenceError
from .models import ArtifactRecord, TorchStateArtifactRecord

_ArtifactLike = ArtifactRecord | TorchStateArtifactRecord


def _rename_path(source: Path, target: Path) -> None:
    _ = source.rename(target)


def _delete_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    _ = path.unlink()


def _iter_artifact_paths(value: object, seen: set[int] | None = None) -> set[str]:
    refs: set[int] = set() if seen is None else seen
    value_id = id(value)
    if value_id in refs:
        return set()
    refs.add(value_id)
    if isinstance(value, (ArtifactRecord, TorchStateArtifactRecord)):
        return {value.file_path}
    if isinstance(value, dict):
        dict_paths: set[str] = set()
        mapping = cast(dict[object, object], value)
        for key, item in mapping.items():
            dict_paths.update(_iter_artifact_paths(key, refs))
            dict_paths.update(_iter_artifact_paths(item, refs))
        return dict_paths
    if isinstance(value, list):
        seq_paths: set[str] = set()
        list_value = cast(list[object], value)
        for item in list_value:
            seq_paths.update(_iter_artifact_paths(item, refs))
        return seq_paths
    if isinstance(value, tuple):
        tuple_paths: set[str] = set()
        tuple_value = cast(tuple[object, ...], value)
        for item in tuple_value:
            tuple_paths.update(_iter_artifact_paths(item, refs))
        return tuple_paths
    if isinstance(value, set):
        set_paths: set[str] = set()
        set_value = cast(set[object], value)
        for item in set_value:
            set_paths.update(_iter_artifact_paths(item, refs))
        return set_paths
    if isinstance(value, frozenset):
        frozen_paths: set[str] = set()
        for item in value:
            frozen_paths.update(_iter_artifact_paths(item, refs))
        return frozen_paths
    return set()


def _delete_unreferenced_artifact(record: _ArtifactLike, live_values: list[object]) -> None:
    for value in live_values:
        if record.file_path in _iter_artifact_paths(value):
            return
    _delete_path(Path(record.file_path))


@dataclass(slots=True)
class _PreparedClone:
    staged_path: Path
    final_path: Path
    cloned_record: _ArtifactLike


@final
class _ArtifactRecoveryTransaction:
    _saved_project_root: Path
    _backup_root: Path
    _artifact_root: Path
    _staging_root: Path
    _memo: dict[int, object]
    _record_memo: dict[tuple[type[object], str], _ArtifactLike]
    _prepared: list[_PreparedClone]

    def __init__(self, saved_project_root: str | Path, backup_root: str | Path, live_project_root: str | Path) -> None:
        self._saved_project_root = Path(saved_project_root)
        self._backup_root = Path(backup_root)
        self._artifact_root = Path(live_project_root) / "artifacts"
        self._staging_root = self._artifact_root / f".recovery-{uuid4().hex}"
        _ = self._artifact_root.mkdir(parents=True, exist_ok=True)
        self._memo = {}
        self._record_memo = {}
        self._prepared = []

    def clone_value(self, value: object) -> object:
        try:
            return self._clone_value(value)
        except Exception as exc:
            self.rollback()
            if isinstance(exc, PersistenceError):
                raise
            raise PersistenceError("Failed to stage backup artifacts for recovery") from exc

    def commit(self) -> None:
        finalized: list[Path] = []
        try:
            for prepared in self._prepared:
                _ = prepared.final_path.parent.mkdir(parents=True, exist_ok=True)
                _rename_path(prepared.staged_path, prepared.final_path)
                finalized.append(prepared.final_path)
        except Exception as exc:
            for path in reversed(finalized):
                _delete_path(path)
            self.rollback()
            raise PersistenceError("Failed to finalize recovered artifacts") from exc
        _delete_path(self._staging_root)
        self._cleanup_empty_recovered_dir()

    def rollback(self) -> None:
        for prepared in reversed(self._prepared):
            _delete_path(prepared.staged_path)
        _delete_path(self._staging_root)
        self._cleanup_empty_recovered_dir()

    def _clone_value(self, value: object) -> object:
        value_id = id(value)
        if value_id in self._memo:
            return self._memo[value_id]
        if isinstance(value, ArtifactRecord):
            return self._clone_record(value)
        if isinstance(value, TorchStateArtifactRecord):
            return self._clone_torch_record(value)
        if isinstance(value, list):
            list_clone: list[object] = []
            self._memo[value_id] = list_clone
            list_value = cast(list[object], value)
            list_clone.extend(self._clone_value(item) for item in list_value)
            return list_clone
        if isinstance(value, dict):
            dict_clone: dict[object, object] = {}
            self._memo[value_id] = dict_clone
            dict_value = cast(dict[object, object], value)
            for key, item in dict_value.items():
                dict_clone[self._clone_value(key)] = self._clone_value(item)
            return dict_clone
        if isinstance(value, tuple):
            tuple_value = cast(tuple[object, ...], value)
            tuple_clone = tuple(self._clone_value(item) for item in tuple_value)
            self._memo[value_id] = tuple_clone
            return tuple_clone
        if isinstance(value, set):
            set_value = cast(set[object], value)
            set_clone = {self._clone_value(item) for item in set_value}
            self._memo[value_id] = set_clone
            return set_clone
        if isinstance(value, frozenset):
            frozenset_clone = frozenset(self._clone_value(item) for item in value)
            self._memo[value_id] = frozenset_clone
            return frozenset_clone
        return value

    def _clone_record(self, record: ArtifactRecord) -> ArtifactRecord:
        memo_key = (ArtifactRecord, record.file_path)
        cached = self._record_memo.get(memo_key)
        if isinstance(cached, ArtifactRecord):
            return cached
        source = self._resolve_source_path(record.file_path, require_file=None)
        final_path = self._build_final_path(record.variable_name, source)
        cloned = replace(record, file_path=str(final_path))
        self._record_memo[memo_key] = cloned
        self._prepared.append(
            _PreparedClone(
                staged_path=self._stage_source(source),
                final_path=final_path,
                cloned_record=cloned,
            )
        )
        return cloned

    def _clone_torch_record(self, record: TorchStateArtifactRecord) -> TorchStateArtifactRecord:
        memo_key = (TorchStateArtifactRecord, record.file_path)
        cached = self._record_memo.get(memo_key)
        if isinstance(cached, TorchStateArtifactRecord):
            return cached
        source = self._resolve_source_path(record.file_path, require_file=True)
        final_path = self._build_final_path(record.variable_name, source)
        cloned = replace(record, file_path=str(final_path), metadata=dict(record.metadata))
        self._record_memo[memo_key] = cloned
        self._prepared.append(
            _PreparedClone(
                staged_path=self._stage_source(source),
                final_path=final_path,
                cloned_record=cloned,
            )
        )
        return cloned

    def _resolve_source_path(self, recorded_path: str, require_file: bool | None) -> Path:
        try:
            relative_path = Path(recorded_path).relative_to(self._saved_project_root)
        except ValueError as exc:
            raise PersistenceError("Saved artifact path is outside the saved project root") from exc
        if relative_path.is_absolute() or any(part in {"..", ""} for part in relative_path.parts):
            raise PersistenceError("Saved artifact path is not trusted")
        if self._backup_root.is_symlink():
            raise PersistenceError("Backup root cannot be a symlink")
        source = self._backup_root / relative_path
        self._assert_path_is_contained(source)
        if not source.exists():
            raise PersistenceError("Saved backup artifact source is missing")
        self._assert_no_symlinks(source, relative_path)
        if require_file is True and not source.is_file():
            raise PersistenceError("Saved torch artifact source changed kind")
        if require_file is None and not (source.is_file() or source.is_dir()):
            raise PersistenceError("Saved artifact source changed kind")
        return source

    def _assert_path_is_contained(self, source: Path) -> None:
        try:
            _ = source.relative_to(self._backup_root)
        except ValueError as exc:
            raise PersistenceError("Backup artifact source escapes the configured backup root") from exc

    def _assert_no_symlinks(self, source: Path, relative_path: Path) -> None:
        current = self._backup_root
        for part in relative_path.parts:
            current = current / part
            if current.is_symlink():
                raise PersistenceError("Backup artifact source contains a symlink")
        if source.is_dir():
            for nested_path in source.rglob("*"):
                if nested_path.is_symlink():
                    raise PersistenceError("Backup artifact tree contains a symlink")

    def _stage_source(self, source: Path) -> Path:
        _ = self._staging_root.mkdir(parents=True, exist_ok=True)
        staged_path = self._staging_root / f"{uuid4().hex}{''.join(source.suffixes)}"
        if source.is_dir():
            _ = shutil.copytree(source, staged_path)
        else:
            _ = staged_path.parent.mkdir(parents=True, exist_ok=True)
            _ = shutil.copy2(source, staged_path)
        return staged_path

    def _build_final_path(self, variable_name: str, source: Path) -> Path:
        safe_name = variable_name.replace("/", "_")
        suffix = "" if source.is_dir() else ''.join(source.suffixes)
        return self._artifact_root / "recovered" / f"{safe_name}__recovered__{uuid4().hex}{suffix}"

    def _cleanup_empty_recovered_dir(self) -> None:
        recovered_root = self._artifact_root / "recovered"
        if recovered_root.exists() and not any(recovered_root.iterdir()):
            _ = recovered_root.rmdir()
