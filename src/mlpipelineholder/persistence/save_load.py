"""Pipeline save/load entry points and saved-project tree metadata handling."""

from __future__ import annotations

from dataclasses import dataclass

import shutil
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..exceptions import PersistenceError, RegistrationError
from .artifacts.store import ArtifactStore
from .pickle_io import (
    atomic_pickle_dump,
    load_pickle_with_missing_class_fallback,
    load_pipeline_metadata,
    write_pipeline_metadata,
)
from .cleanup import (
    _collect_obsolete_artifact_entries,
    _collect_obsolete_child_dirs,
    _delete_saved_path,
    _referenced_payload_artifact_paths,
)

_SAVE_WARNING_PATTERNS = (
    r"Saved pipelines preserve callable references",
    r".*was saved without a linked model artifact",
    r".*could not be serialized directly; saving a reference placeholder instead",
    r".*is not importable; saving a reference placeholder instead",
)

_CLEANUP_MODES = ("none", "confirm", "auto")


@dataclass(frozen=True, slots=True, kw_only=True)
class _PipelineLoadLocation:
    project_root: Path
    restore_message: str | None = None
    backup_root: Path | None = None
    update_backup_root: bool = False


class SaveLoadMixin:
    """Public save/load entry points shared by the PipelineHolder facade."""

    if TYPE_CHECKING:
        project_root: Any = None
        pipeline_backup_root: Any = None
        logger: Any = None
        config: Any = None
        metadata_root: Any = None
        artifact_store: Any = None
        _temporary_root_handle: Any = None

        def _root_pipeline(self) -> Any: ...
        def _serialize_payload_for_save(
            self, target_root: Path, cache: dict[int, Any] | None = None
        ) -> dict[str, Any]: ...
        def _validate_runtime_output_pointers(self) -> None: ...
        @staticmethod
        def _serialize_config_for_save(config: Any) -> Any: ...
        @staticmethod
        def _normalized_path(path: Path) -> Path: ...
        def _cleanup_temporary_root_handle(self) -> None: ...
        def _rewrite_artifact_paths(self, old_root: Path, new_root: Path) -> None: ...
        def _rewrite_run_history_paths(self, old_root: Path, new_root: Path) -> None: ...
        def _refresh_descendant_roots(self, old_root: Path, new_root: Path) -> None: ...
        @classmethod
        def _paths_overlap(cls, first_path: Path, second_path: Path) -> bool: ...
        @classmethod
        def _clear_path_with_optional_confirmation(
            cls, target_path: Path, *, source_path: Path, forced_deleting: bool
        ) -> None: ...
        @classmethod
        def _validate_loaded_payload_placeholders(
            cls, payload: dict[str, Any]
        ) -> None: ...
        @classmethod
        def _replace_missing_runtime_payload_values(
            cls, payload: Any, pipeline_path: tuple[str, ...] = ()
        ) -> list[str]: ...
        @classmethod
        def _validate_loaded_payload_structure(cls, payload: Any) -> None: ...
        @classmethod
        def _from_payload(
            cls,
            payload: dict[str, Any],
            project_root: Path,
            parent: Any = None,
            *,
            verbose: bool = False,
            auto_resolve_placeholders: bool = True,
        ) -> Any: ...

    def save_pipeline(
        self,
        path: str | Path | None = None,
        save_log_to_file: str | Path | None = None,
        *,
        verbose: bool = False,
        cleanup: str = "auto",
    ) -> Path:
        if cleanup not in _CLEANUP_MODES:
            raise ValueError(
                f"Invalid cleanup mode {cleanup!r}: expected 'none', 'confirm', or 'auto'"
            )
        with warnings.catch_warnings():
            if not verbose:
                for pattern in _SAVE_WARNING_PATTERNS:
                    warnings.filterwarnings("ignore", message=pattern)
            warnings.warn(
                "Saved pipelines preserve callable references, not historical function behavior; later source changes may affect reloaded pipelines.",
                stacklevel=2,
            )
            return self._save_pipeline_impl(path, save_log_to_file, cleanup)

    def _save_pipeline_impl(
        self,
        path: str | Path | None,
        save_log_to_file: str | Path | None,
        cleanup_mode: str,
    ) -> Path:
        self._root_pipeline()._validate_runtime_output_pointers()
        target = self.project_root if path is None else Path(path)
        if self._temporary_root_handle is not None and self._normalized_path(target) != self._normalized_path(self.project_root):
            self._relocate_project_root(target)
            target = self.project_root
        target_is_project_root = self._normalized_path(target) == self._normalized_path(self.project_root)
        if not target_is_project_root:
            self._materialize_project_tree_for_save(target)
        else:
            target.mkdir(parents=True, exist_ok=True)
        try:
            payload = self._serialize_payload_for_save(target)
        except RegistrationError as exc:
            raise PersistenceError(str(exc)) from exc
        saved_backup_root = self.pipeline_backup_root if target_is_project_root else target
        payload["pipeline_backup_directory"] = (
            None if saved_backup_root is None else str(saved_backup_root)
        )
        atomic_pickle_dump(payload, target / "pipeline_state.pkl")
        atomic_pickle_dump(self._serialize_config_for_save(self.config), target / "config.pkl")
        write_pipeline_metadata(self.project_root, target, saved_backup_root)
        self.logger.info(f"Pipeline has been saved to project root: {target}")
        if target_is_project_root:
            try:
                self._cleanup_obsolete_saved_objects(payload, target, cleanup_mode)
            except OSError as exc:
                self.logger.warning(
                    f"Skipped cleanup because saved paths could not be inspected: "
                    f"{type(exc).__name__}: {exc}"
                )
        if save_log_to_file is not None:
            self.logger.flush()
            log_target = Path(save_log_to_file)
            log_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.logger.log_file_path, log_target)
        refresh_backup = self._should_refresh_backup_on_save(target)
        if refresh_backup:
            backup_root = self.pipeline_backup_root
            if backup_root is not None:
                self._refresh_backup_copy(target, backup_root)
                self.logger.info(
                    f"Pipeline has been saved to project backup path: {backup_root}"
                )
        self._archive_current_log_to_history()
        if refresh_backup:
            backup_root = self.pipeline_backup_root
            if backup_root is not None:
                self._sync_history_logs_to_backup(backup_root)
        return target

    def _cleanup_obsolete_saved_objects(
        self,
        payload: dict[str, Any],
        target_root: Path,
        cleanup_mode: str,
    ) -> None:
        """Delete obsolete saved artifacts and orphaned child pipelines.

        The committed payload is the only liveness authority: every artifact
        path it references is kept, and every other framework-shaped
        generation entry under ``artifacts/`` plus every child pipeline
        directory absent from the serialized topology (with a fully
        framework-managed subtree) is deleted. ``none`` keeps everything,
        ``confirm`` asks first, and ``auto`` deletes without prompting.
        """
        if cleanup_mode == "none":
            return
        live_paths = _referenced_payload_artifact_paths(payload)
        obsolete_children: list[Path] = []
        _collect_obsolete_child_dirs(
            payload,
            target_root,
            live_paths,
            obsolete_children,
        )
        obsolete: list[Path] = []
        _collect_obsolete_artifact_entries(
            payload,
            target_root,
            live_paths,
            obsolete,
        )
        obsolete.extend(obsolete_children)
        if not obsolete:
            return
        if cleanup_mode == "confirm":
            try:
                answer = input(
                    f"Delete {len(obsolete)} obsolete saved pipeline object(s)? [y/N]: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer not in ("y", "yes"):
                self.logger.info(
                    f"Skipped cleanup: {len(obsolete)} obsolete saved object(s) retained"
                )
                return
        deleted = 0
        for path in sorted(obsolete, key=lambda item: len(item.parts), reverse=True):
            try:
                if _delete_saved_path(path, target_root):
                    deleted += 1
            except OSError as exc:
                self.logger.warning(
                    f"Could not delete obsolete saved path '{path}': "
                    f"{type(exc).__name__}: {exc}"
                )
        if deleted:
            self.logger.info(f"Deleted {deleted} obsolete saved pipeline object(s)")

    def save_project(
        self,
        path: str | Path | None = None,
        save_log_to_file: str | Path | None = None,
        *,
        verbose: bool = False,
    ) -> Path:
        return self.save_pipeline(path, save_log_to_file=save_log_to_file, verbose=verbose)

    def _archive_current_log_to_history(self) -> None:
        """Copy the current pipeline.log into history_logs/ with a timestamped name."""
        self.logger.flush()
        history_root = self.project_root / "history_logs"
        history_root.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC)
        stamp = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.{now.microsecond // 1000:03d}"
        target = history_root / f"{stamp}.log"
        counter = 1
        while target.exists():
            target = history_root / f"{stamp}_{counter}.log"
            counter += 1
        shutil.copy2(self.logger.log_file_path, target)

    @classmethod
    def _restore_working_tree_if_needed(
        cls,
        source_path: Path,
        *,
        forced_deleting: bool,
    ) -> _PipelineLoadLocation:
        metadata = load_pipeline_metadata(source_path)
        if metadata is None:
            return _PipelineLoadLocation(project_root=source_path)
        pipeline_directory = metadata.get("pipeline_directory")
        if pipeline_directory is None:
            return _PipelineLoadLocation(project_root=source_path)
        work_root = Path(pipeline_directory)
        if cls._normalized_path(source_path) == cls._normalized_path(work_root):
            return _PipelineLoadLocation(project_root=source_path)
        backup_directory = metadata.get("pipeline_backup_directory")
        backup_root = (
            None if backup_directory is None else Path(backup_directory)
        )
        source_is_backup = backup_root is not None and (
            cls._normalized_path(source_path) == cls._normalized_path(backup_root)
        )
        update_backup_root = False
        selected_backup_root = backup_root
        if not source_is_backup:
            update_project_root = input(
                f"Pipeline was loaded from unrecognized location '{source_path}'. "
                f"Recorded project root is '{work_root}' and recorded backup path is "
                f"'{backup_root}'. Use the loading path as the new project root? "
                "Type 'yes' or 'y' to continue, otherwise type 'no': "
            ).strip().lower()
            if update_project_root in {"yes", "y"}:
                new_backup = input(
                    "Enter a new pipeline backup path, or press Enter for no backup: "
                ).strip()
                return _PipelineLoadLocation(
                    project_root=source_path,
                    backup_root=(
                        None if not new_backup else Path(new_backup).expanduser()
                    ),
                    update_backup_root=True,
                )
            selected_backup_root = source_path
            update_backup_root = True
        if cls._paths_overlap(source_path, work_root):
            raise PersistenceError(
                f"Cannot restore pipeline from '{source_path}' into overlapping working directory '{work_root}'"
            )
        cls._clear_path_with_optional_confirmation(
            work_root,
            source_path=source_path,
            forced_deleting=forced_deleting,
        )
        shutil.copytree(source_path, work_root)
        return _PipelineLoadLocation(
            project_root=work_root,
            restore_message=(
                f"Pipeline project directory has been copied from backup path: "
                f"{source_path} -> {work_root}"
            ),
            backup_root=selected_backup_root,
            update_backup_root=update_backup_root,
        )

    def _relocate_project_root(self, new_root: Path) -> None:
        old_root = self.project_root
        normalized_old_root = self._normalized_path(old_root)
        normalized_new_root = self._normalized_path(new_root)
        if normalized_old_root == normalized_new_root:
            self._cleanup_temporary_root_handle()
            return
        self.logger.flush()
        self.logger.disable_file_logging()
        new_root.mkdir(parents=True, exist_ok=True)
        for entry in old_root.iterdir():
            destination = new_root / entry.name
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(entry), str(destination))
        self.project_root = new_root
        self.metadata_root = new_root / "metadata"
        self.metadata_root.mkdir(parents=True, exist_ok=True)
        self.logger.rebind_path(self.metadata_root / "pipeline.log")
        self.artifact_store = ArtifactStore(new_root)
        self._rewrite_artifact_paths(old_root, new_root)
        self._rewrite_run_history_paths(old_root, new_root)
        self._refresh_descendant_roots(old_root, new_root)
        self._cleanup_temporary_root_handle()

    @classmethod
    def load_pipeline(
        cls,
        path: str | Path,
        *,
        forced_deleting: bool = False,
        verbose: bool = False,
        auto_resolve_placeholders: bool = True,
    ) -> Any:
        warnings.warn(
            "Loaded pipelines restore current callable references, not historical function snapshots; changed source code may alter behavior.",
            stacklevel=2,
        )
        source = Path(path)
        try:
            with (source / "pipeline_state.pkl").open("rb") as handle:
                state_bytes = handle.read()
            payload = load_pickle_with_missing_class_fallback(state_bytes)
            invalid_runtime_values = cls._replace_missing_runtime_payload_values(
                payload
            )
            cls._validate_loaded_payload_placeholders(payload)
            cls._validate_loaded_payload_structure(payload)
            load_location = cls._restore_working_tree_if_needed(
                source,
                forced_deleting=forced_deleting,
            )
            if load_location.update_backup_root:
                payload["pipeline_backup_directory"] = (
                    None
                    if load_location.backup_root is None
                    else str(load_location.backup_root)
                )
            pipeline = cls._from_payload(
                payload,
                load_location.project_root,
                verbose=verbose,
                auto_resolve_placeholders=auto_resolve_placeholders,
            )
        except PersistenceError:
            raise
        except Exception as exc:
            # A failed build leaves the saved pipeline_state.pkl in the working
            # tree untouched, so the load can be retried; only log and artifact
            # files written mid-build remain, and save-time cleanup removes them.
            raise PersistenceError(f"Failed to load pipeline project: {exc}") from exc
        if load_location.restore_message is not None:
            pipeline.logger.info(load_location.restore_message)
        for owner_label in invalid_runtime_values:
            pipeline.logger.warning(
                f"Loaded {owner_label} as None because its saved value depends on "
                "a missing __main__ class"
            )
        pipeline.logger.info(
            f"Pipeline has been loaded from the project root: {load_location.project_root}"
        )
        return pipeline

    @classmethod
    def load_project(
        cls,
        path: str | Path,
        *,
        forced_deleting: bool = False,
        verbose: bool = False,
        auto_resolve_placeholders: bool = True,
    ) -> Any:
        return cls.load_pipeline(
            path,
            forced_deleting=forced_deleting,
            verbose=verbose,
            auto_resolve_placeholders=auto_resolve_placeholders,
        )

    def _should_refresh_backup_on_save(self, target: Path) -> bool:
        backup_root = self.pipeline_backup_root
        if backup_root is None:
            return False
        return self._normalized_path(target) == self._normalized_path(self.project_root)

    def _refresh_backup_copy(self, source: Path, backup_root: Path) -> None:
        if self._normalized_path(source) == self._normalized_path(backup_root):
            return
        if backup_root.exists():
            if backup_root.is_dir():
                shutil.rmtree(backup_root)
            else:
                backup_root.unlink()
        shutil.copytree(source, backup_root)

    def _sync_history_logs_to_backup(self, backup_root: Path) -> None:
        """Copy the project history_logs folder into the backup after a save snapshot."""
        history_root = self.project_root / "history_logs"
        if not history_root.is_dir():
            return
        backup_history = backup_root / "history_logs"
        if backup_history.exists():
            if backup_history.is_dir():
                shutil.rmtree(backup_history)
            else:
                backup_history.unlink()
        shutil.copytree(history_root, backup_history)

    def _materialize_project_tree_for_save(self, target: Path) -> None:
        if self._paths_overlap(self.project_root, target):
            raise PersistenceError(
                f"Cannot save pipeline from '{self.project_root}' into overlapping directory '{target}'"
            )
        self.logger.flush()
        shutil.copytree(self.project_root, target, dirs_exist_ok=True)
