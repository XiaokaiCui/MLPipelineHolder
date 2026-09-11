"""Project-tree helpers: path normalization, cleanup prompts, and path rewrites."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.base import PipelineBase
from ..core.models import ArtifactRecord
from ..exceptions import PersistenceError
from .artifacts.store import ArtifactStore


def _is_child_pipeline(node: object) -> bool:
    return isinstance(node, PipelineBase)


class ProjectTreeMixin:
    """Path and tree maintenance used by save/load and relocation flows."""

    if TYPE_CHECKING:
        logger: Any = None
        project_root: Any = None
        artifact_store: Any = None
        _stored_objects: dict[str, Any] = {}
        run_history: list[Any] = []
        producer_outputs: dict[str, dict[str, Any]] = {}
        para_value_dict: dict[str, Any] = {}
        artifact_registry: dict[str, Any] = {}
        historical_result_log_path: Any = None
        memory_saving_mode: bool = False
        memory_profile_logging: bool = False

        def _sorted_nodes(self) -> list[Any]: ...

    @classmethod
    def _clear_path_with_optional_confirmation(
        cls,
        target_path: Path,
        *,
        source_path: Path,
        forced_deleting: bool,
    ) -> None:
        if not target_path.exists():
            return
        if target_path.is_dir() and not any(target_path.iterdir()):
            target_path.rmdir()
            return
        if not forced_deleting:
            user_input = input(
                f"Working pipeline directory '{target_path}' will be deleted and replaced from '{source_path}'. Type 'yes' or 'y' to continue: "
            ).strip().lower()
            if user_input not in {"yes", "y"}:
                raise PersistenceError(
                    f"Aborted restoring pipeline directory '{target_path}' from '{source_path}'"
                )
        if target_path.is_dir():
            shutil.rmtree(target_path)
            return
        target_path.unlink()

    @staticmethod
    def _normalized_path(path: Path) -> Path:
        return path.expanduser().resolve(strict=False)

    @classmethod
    def _paths_overlap(cls, first_path: Path, second_path: Path) -> bool:
        first = cls._normalized_path(first_path)
        second = cls._normalized_path(second_path)
        return first == second or first in second.parents or second in first.parents

    def _rewrite_artifact_paths(self, old_root: Path, new_root: Path) -> None:
        old_prefix = str(old_root)
        new_prefix = str(new_root)

        def rewrite_value(value: Any) -> Any:
            if isinstance(value, ArtifactRecord) and value.file_path.startswith(old_prefix):
                value.file_path = value.file_path.replace(old_prefix, new_prefix, 1)
            if isinstance(value, ArtifactRecord):
                metadata = getattr(value, "metadata", None)
                if not isinstance(metadata, dict):
                    return value
                db_path = metadata.get("db_path")
                if isinstance(db_path, str) and db_path.startswith(old_prefix):
                    metadata["db_path"] = db_path.replace(
                        old_prefix,
                        new_prefix,
                        1,
                    )
            return value

        for outputs in self.producer_outputs.values():
            for key, value in list(outputs.items()):
                outputs[key] = rewrite_value(value)
        for key, value in list(self.para_value_dict.items()):
            self.para_value_dict[key] = rewrite_value(value)
        for key, value in list(self.artifact_registry.items()):
            self.artifact_registry[key] = rewrite_value(value)
        for record in self._stored_objects.values():
            if record.artifact is not None:
                record.artifact = rewrite_value(record.artifact)
        if self.historical_result_log_path and self.historical_result_log_path.startswith(
            old_prefix
        ):
            self.historical_result_log_path = self.historical_result_log_path.replace(
                old_prefix,
                new_prefix,
                1,
            )
        for node in self._sorted_nodes():
            if _is_child_pipeline(node):
                node._rewrite_artifact_paths(old_root, new_root)

    def _rewrite_run_history_paths(self, old_root: Path, new_root: Path) -> None:
        old_prefix = str(old_root)
        new_prefix = str(new_root)
        for run_record in self.run_history:
            if run_record.config_snapshot_path and run_record.config_snapshot_path.startswith(old_prefix):
                run_record.config_snapshot_path = run_record.config_snapshot_path.replace(
                    old_prefix,
                    new_prefix,
                    1,
                )
        for node in self._sorted_nodes():
            if _is_child_pipeline(node):
                node._rewrite_run_history_paths(old_root, new_root)

    def _refresh_descendant_roots(self, old_root: Path, new_root: Path) -> None:
        for node in self._sorted_nodes():
            if not _is_child_pipeline(node):
                continue
            old_child_root = node.project_root
            try:
                relative = old_child_root.relative_to(old_root)
            except ValueError:
                relative = Path("children") / node.registration_name
            new_child_root = new_root / relative
            node.project_root = new_child_root
            node.metadata_root = new_child_root / "metadata"
            node.metadata_root.mkdir(parents=True, exist_ok=True)
            node.artifact_store = ArtifactStore(new_child_root)
            node.logger = self.logger
            node.memory_saving_mode = self.memory_saving_mode
            node.memory_profile_logging = self.memory_profile_logging
            node._rewrite_artifact_paths(old_child_root, new_child_root)
            node._rewrite_run_history_paths(old_child_root, new_child_root)
            node._refresh_descendant_roots(old_child_root, new_child_root)
