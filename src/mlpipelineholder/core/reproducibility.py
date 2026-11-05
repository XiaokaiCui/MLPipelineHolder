"""Root-owned seed and definition-time clean-start baselines."""

from __future__ import annotations

import copy
import os
import pickle
import random
import re
import shutil
import sys
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np

from .models import (
    ArtifactRecord,
    DataclassValueReference,
    RuntimeValueReference,
    TorchStateArtifactRecord,
)
from ..exceptions import RegistrationError
from ..integrations.dataframe import dump_dataframe
from ..integrations.optuna.api import StudyArtifactOptions
from ..integrations.optuna.support import (
    OPTUNA_STUDY_SERIALIZER,
    is_optuna_study,
)
from ..state.output_pointers import OutputPointer


class _BaselineCaptureError(RegistrationError):
    """A baseline publication failure attributed to one pipeline."""

    def __init__(self, pipeline: Any, message: str) -> None:
        super().__init__(message)
        self.pipeline = pipeline


@dataclass(frozen=True)
class _StudyBaseline:
    """Detached Optuna state; no reference to the mutable backing database."""

    name: str
    directions: tuple[Any, ...]
    trials: tuple[Any, ...]
    user_attrs: dict[str, Any]
    system_attrs: dict[str, Any]
    sampler: Any


@dataclass(frozen=True)
class _PendingStudyRestore:
    """Managed Optuna state that is persisted only at commit time."""

    pipeline: Any
    variable_name: str
    baseline: _StudyBaseline
    physical_name: str
    original_name: str
    owner_kind: str
    owner_key: str
    db_path: Path


@dataclass
class _PreparedCleanStart:
    states: list[tuple[Any, Any, dict[str, Any]]]
    staged_artifacts: list[tuple[Any, ArtifactRecord | TorchStateArtifactRecord]]
    rollback_dir: Path
    output_backups: list[tuple[Path, Path]]
    pending_studies: list[_PendingStudyRestore]
    sqlite_backups: list[tuple[Path, Path, bool]]


class ReproducibilityMixin:
    """Keep independent baseline files, never references into mutable run state."""

    if TYPE_CHECKING:
        registration_name: str = ""
        parent_pipeline: Any = None
        _seed: int | None = None
        _deterministic_torch: bool = False
        _clean_config_baseline: Any = None
        _clean_constant_baseline: dict[str, dict[str, Any]] | None = None
        _clean_baseline_error: str | None = None
        manual_values: dict[str, Any] = {}
        logger: Any = None
        artifact_store: Any = None
        project_root: Any = None
        metadata_root: Any = None
        _invalidation_forbidden: bool = False
        _is_atom: bool = False
        _config: Any = None
        nodes: list[Any] = []
        blocks: list[Any] = []
        _expression_runtime_namespace_cache: Any = None

        def _root_pipeline(self) -> Any: ...
        def _iter_attached_pipelines(self) -> list[Any]: ...
        def _constant_conflict_names(
            self, configs: dict[int, Any] | None = None,
        ) -> set[str]: ...
        def _invalidate_all_outputs(self) -> None: ...
        def _invalidate_from_priority(self, priority: float) -> None: ...
        def _sorted_nodes(self) -> list[Any]: ...
        def _cleanup_replaced_artifact(self, previous_value: Any) -> None: ...
        def _rebuild_visible_state(self, upstream_outputs: Any = None) -> None: ...
        def _incoming_parent_outputs(self) -> dict[str, Any]: ...
        def _resync_mirror_to_parent(self) -> None: ...
        def _unload_from_memory(self, object_names: Any = None) -> None: ...
        def _materialize_stored_value(self, value: Any, placeholder_error: str) -> Any: ...
        def _optuna_studies_db_path_for_storage(self) -> Any: ...
        def config_as_dict(self) -> dict[str, Any]: ...
        @staticmethod
        def _config_name_mapping(config_obj: Any) -> dict[str, Any]: ...

    @staticmethod
    def _copy_definition_config(config: Any) -> Any:
        try:
            return copy.deepcopy(config)
        except Exception:
            # Configuration is already required to round-trip through pickle;
            # use that as the fallback for objects without a working deepcopy.
            try:
                return pickle.loads(pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL))
            except Exception as exc:
                raise RegistrationError(
                    f"Cannot snapshot the pipeline's definition-time configuration: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

    def set_seed(self, seed: int | None, *, deterministic_torch: bool = False) -> None:
        """Set the optional global run_all seed on the root pipeline only."""
        if self.parent_pipeline is not None:
            raise RegistrationError("set_seed() must be called on the root pipeline")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32):
            raise ValueError("seed must be an integer between 0 and 2**32 - 1, or None")
        if not isinstance(deterministic_torch, bool):
            raise TypeError("deterministic_torch must be a boolean")
        self._seed = seed
        self._deterministic_torch = deterministic_torch if seed is not None else False
        if seed is not None:
            self._warn_seeded_parallel_blocks(self._iter_attached_pipelines())

    def _warn_seeded_parallel_blocks(
        self, pipelines: list[Any], *, only_block: Any | None = None,
    ) -> None:
        root = self._root_pipeline()
        if root._seed is None or root.suppress_registration_advisories:
            return
        for pipeline in pipelines:
            for block in pipeline.blocks:
                if len(block.functions) > 1 and (only_block is None or block is only_block):
                    self.logger.warning(
                        f"Seeded parallel block '{pipeline.full_path()}/{block.registration_name}' "
                        "uses shared global RNGs; results may not be reproducible"
                    )

    def _preflight_seed_runtime(self) -> None:
        if self._root_pipeline()._deterministic_torch:
            try:
                import torch  # noqa: F401
            except ImportError:
                raise RegistrationError("deterministic_torch=True requires PyTorch") from None

    @contextmanager
    def _torch_deterministic_scope(self) -> Iterator[None]:
        root = self._root_pipeline()
        if not root._deterministic_torch:
            yield
            return
        import torch
        previous_mode = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        previous_benchmark = torch.backends.cudnn.benchmark
        try:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            yield
        finally:
            torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)
            torch.backends.cudnn.benchmark = previous_benchmark

    def _seed_run_all(self) -> None:
        root = self._root_pipeline()
        if root._seed is None:
            return
        random.seed(root._seed)
        np.random.seed(root._seed)
        try:
            import torch
        except ImportError:
            if root._deterministic_torch:
                raise RegistrationError("deterministic_torch=True requires PyTorch") from None
        else:
            torch.manual_seed(root._seed)

    @property
    def _clean_baseline_root(self) -> Path:
        return Path(self.project_root) / "clean_baseline"

    def _study_baseline(self, study: Any) -> _StudyBaseline:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            system_attrs = dict(getattr(study, "system_attrs", {}))
        return _StudyBaseline(
            name=study.study_name,
            directions=tuple(study.directions),
            trials=tuple(study.get_trials(deepcopy=True)),
            user_attrs=dict(study.user_attrs),
            system_attrs=system_attrs,
            sampler=study.sampler,
        )

    def _write_pickle_baseline(
        self, directory: Path, index: int, value: Any,
    ) -> dict[str, Any]:
        target = directory / f"constant_{index}.bin"
        with target.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return {"kind": "pickle", "file": target.name}

    def _write_baseline_entry(
        self, directory: Path, index: int, name: str, value: Any,
    ) -> dict[str, Any]:
        if isinstance(value, TorchStateArtifactRecord):
            source = Path(value.file_path)
            target = directory / f"constant_{index}{source.suffix if source.is_file() else ''}"
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
            return {
                "kind": "torch_state",
                "file": target.name,
                "object_kind": value.object_kind,
                "metadata": dict(value.metadata),
            }
        if isinstance(value, ArtifactRecord):
            if value.serializer == OPTUNA_STUDY_SERIALIZER:
                study = self._materialize_stored_value(value, "")
                entry = self._write_pickle_baseline(
                    directory, index, self._study_baseline(study),
                )
                metadata = value.metadata if isinstance(value.metadata, dict) else {}
                physical_name = metadata.get("study_name")
                if not isinstance(physical_name, str) or not physical_name:
                    physical_name = study.study_name
                original_name = metadata.get("original_study_name")
                if not isinstance(original_name, str) or not original_name:
                    original_name = physical_name
                owner_kind = metadata.get("study_owner_kind")
                if not isinstance(owner_kind, str) or not owner_kind:
                    owner_kind = "constant"
                owner_key = metadata.get("study_owner_key")
                if not isinstance(owner_key, str) or not owner_key:
                    owner_key = f"{self.registration_name}.{name}"
                entry.update(
                    {
                        "kind": "optuna",
                        "study_name": physical_name,
                        "original_study_name": original_name,
                        "study_owner_kind": owner_kind,
                        "study_owner_key": owner_key,
                    }
                )
                return entry
            source = Path(value.file_path)
            target = directory / f"constant_{index}{source.suffix if source.is_file() else ''}"
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
            return {
                "kind": "artifact",
                "file": target.name,
                "serializer": value.serializer,
                "torch_load_weights_only": bool(value.torch_load_weights_only),
                "metadata": dict(value.metadata),
            }
        if is_optuna_study(value):
            value = self._study_baseline(value)
        dask = sys.modules.get("dask.dataframe")
        if dask is not None and isinstance(value, dask.DataFrame):
            target = directory / f"constant_{index}.parquet"
            dump_dataframe(value, "parquet", target)
            return {"kind": "dask", "file": target.name}
        return self._write_pickle_baseline(directory, index, value)

    def _stage_constant_baseline(
        self,
    ) -> tuple[dict[str, dict[str, Any]], Path, str]:
        """Prepare this pipeline's baseline without touching its live snapshot.

        Each capture gets an immutable generation id. The staged files become
        ``clean_baseline/generations/<id>/`` on publication, and the manifest
        records that id so restoration never depends on reusable file names.
        """
        generation = uuid4().hex
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".clean-baseline-stage-{generation}-", dir=self.metadata_root,
            )
        )
        manifest: dict[str, dict[str, Any]] = {}
        try:
            for index, (name, value) in enumerate(self.manual_values.items()):
                try:
                    entry = self._write_baseline_entry(
                        staging, index, name, value,
                    )
                except RegistrationError:
                    raise
                except Exception as exc:
                    raise RegistrationError(
                        f"Constant '{name}' cannot be independently snapshotted and "
                        f"restored: {type(exc).__name__}: {exc}"
                    ) from exc
                entry["generation"] = generation
                manifest[name] = entry
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return manifest, staging, generation

    def _recover_baseline_publication(self, pipeline: Any) -> None:
        """Reconcile leftovers from interrupted baseline publication.

        New baselines are immutable, generation-identified directories, so a
        crash leaves either a stale staging directory or an unreferenced
        generation; the recorded manifest still points at the complete previous
        generation. Legacy projects used a single mutable directory plus a
        ``.clean-baseline-previous`` swap backup; without a durable generation
        id the only safe recovery is to restore that backup.
        """
        metadata_root = Path(pipeline.metadata_root)
        backup = metadata_root / ".clean-baseline-previous"
        directory = pipeline._clean_baseline_root
        if backup.exists() or backup.is_symlink():
            if (directory / "generations").is_dir():
                # The generation layout is already active; a legacy swap
                # backup is stale and must not replace it.
                shutil.rmtree(backup, ignore_errors=True)
            else:
                if directory.exists() or directory.is_symlink():
                    shutil.rmtree(directory, ignore_errors=True)
                os.replace(backup, directory)
        for staging in metadata_root.glob(".clean-baseline-stage-*"):
            shutil.rmtree(staging, ignore_errors=True)

    def _capture_constant_baselines(self, pipelines: list[Any]) -> None:
        """Stage every requested pipeline, then publish immutable generations."""
        for pipeline in pipelines:
            self._recover_baseline_publication(pipeline)
        staged: list[tuple[Any, dict[str, dict[str, Any]], Path, str]] = []
        published: list[tuple[Any, dict[str, dict[str, Any]] | None, Path]] = []
        try:
            for pipeline in pipelines:
                try:
                    manifest, staging, generation = pipeline._stage_constant_baseline()
                except Exception as exc:
                    if isinstance(exc, _BaselineCaptureError):
                        raise
                    raise _BaselineCaptureError(
                        pipeline,
                        f"Constant baseline for '{pipeline.registration_name}' could not be "
                        f"captured: {type(exc).__name__}: {exc}",
                    ) from exc
                staged.append((pipeline, manifest, staging, generation))
            self._validate_constant_snapshot_names(
                [(pipeline, manifest) for pipeline, manifest, _, _ in staged]
            )
            for pipeline, manifest, staging, generation in staged:
                directory = pipeline._clean_baseline_root
                if directory.is_symlink():
                    raise _BaselineCaptureError(
                        pipeline,
                        f"Refusing to replace symlinked clean baseline: {directory}",
                    )
                generations = directory / "generations"
                if generations.is_symlink():
                    raise _BaselineCaptureError(
                        pipeline,
                        f"Refusing to use a symlinked baseline generation root: {generations}",
                    )
                generations.mkdir(parents=True, exist_ok=True)
                destination = generations / generation
                if destination.exists() or destination.is_symlink():
                    raise _BaselineCaptureError(
                        pipeline,
                        f"Baseline generation already exists: {destination}",
                    )
                old_manifest = pipeline._clean_constant_baseline
                os.replace(staging, destination)
                pipeline._clean_constant_baseline = manifest
                published.append((pipeline, old_manifest, destination))
        except BaseException:
            for pipeline, manifest, destination in reversed(published):
                if destination.exists():
                    shutil.rmtree(destination, ignore_errors=True)
                pipeline._clean_constant_baseline = manifest
            raise
        finally:
            for _, _, staging, _ in staged:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)

    def _validate_constant_snapshot_names(
        self, snapshots: list[tuple[Any, dict[str, dict[str, Any]]]],
    ) -> None:
        root = self._root_pipeline()
        definition_configs = {
            id(pipeline): pipeline._clean_config_baseline
            for pipeline in root._iter_attached_pipelines()
        }
        for pipeline, snapshot in snapshots:
            conflicts = set(snapshot) & pipeline._constant_conflict_names(
                definition_configs
            )
            if conflicts:
                raise _BaselineCaptureError(
                    pipeline,
                    f"Clean baseline for '{pipeline.registration_name}' conflicts with current registration: "
                    f"{sorted(conflicts)}",
                )

    def _ensure_constant_baselines(self, *, required: bool = False) -> None:
        """Capture missing baselines for the requested scope.

        Ordinary execution keeps the first-execution contract: the first
        execution entry anywhere in the attached tree attempts a root-wide
        capture. A clean start only depends on the state in the pipeline it was
        called on, so a required capture on a child is restricted to that
        subtree and never stages, validates, or marks unrelated pipelines.
        """
        root = self._root_pipeline()
        selected = self._iter_attached_pipelines()
        capture_scope = selected if required else root._iter_attached_pipelines()
        for pipeline in capture_scope:
            try:
                self._recover_baseline_publication(pipeline)
            except Exception as exc:
                message = (
                    f"Could not recover interrupted baseline publication for "
                    f"'{pipeline.registration_name}': {type(exc).__name__}: {exc}"
                )
                pipeline._clean_baseline_error = message
                if required and pipeline in selected:
                    raise RegistrationError(message) from exc
                self.logger.warning(message)
        failed = [pipeline for pipeline in selected if pipeline._clean_baseline_error]
        if failed and required:
            raise RegistrationError(
                f"Clean baseline unavailable for '{failed[0].registration_name}': "
                f"{failed[0]._clean_baseline_error}; call snapshot_constants() after repairing constants"
            )
        pending = [
            pipeline for pipeline in capture_scope
            if pipeline._clean_constant_baseline is None and pipeline._clean_baseline_error is None
        ]
        if not pending:
            return
        try:
            root._capture_constant_baselines(pending)
        except _BaselineCaptureError as exc:
            exc.pipeline._clean_baseline_error = str(exc)
            if required and exc.pipeline in selected:
                raise RegistrationError(
                    f"Clean baseline unavailable for '{exc.pipeline.registration_name}': "
                    f"{exc}; call snapshot_constants() after repairing constants"
                ) from exc
            self.logger.warning(
                "Automatic constant baseline unavailable; ordinary execution continues. "
                f"Call snapshot_constants() after repairing constants. {exc}"
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if required and self in selected:
                self._clean_baseline_error = message
                raise RegistrationError(
                    f"Clean baseline unavailable: {message}"
                ) from exc
            self.logger.warning(
                "Automatic constant baseline unavailable; ordinary execution continues. "
                f"Call snapshot_constants() after repairing constants. {message}"
            )
        else:
            for pipeline in pending:
                pipeline._clean_baseline_error = None

    def snapshot_constants(self) -> None:
        """Replace the root tree's automatic constant baseline with current values."""
        if self.parent_pipeline is not None:
            raise RegistrationError("snapshot_constants() must be called on the root pipeline")
        pipelines = self._iter_attached_pipelines()
        self._capture_constant_baselines(pipelines)
        for pipeline in pipelines:
            pipeline._clean_baseline_error = None

    def snapshot_config(self) -> None:
        """Replace the definition-time configuration baseline for the root tree.

        Only the root pipeline may call this. The current configuration of the
        root and every attached descendant becomes the configuration restored
        by ``run_all(clean_start=True)``. The snapshot uses the same in-memory
        ``definition_config`` representation as the automatic baseline and is
        persisted with the pipeline.
        """
        if self.parent_pipeline is not None:
            raise RegistrationError("snapshot_config() must be called on the root pipeline")
        snapshots = [
            (pipeline, pipeline._copy_definition_config(pipeline._config))
            for pipeline in self._iter_attached_pipelines()
        ]
        for pipeline, config in snapshots:
            pipeline._clean_config_baseline = config

    def snapshot_configs(self) -> None:
        """Alias for :meth:`snapshot_config`."""
        self.snapshot_config()

    @staticmethod
    def _rebuild_study(value: _StudyBaseline) -> Any:
        import optuna
        study = optuna.create_study(
            study_name=value.name,
            directions=list(value.directions),
            sampler=value.sampler,
        )
        study.add_trials(list(value.trials))
        for name, item in value.user_attrs.items():
            study.set_user_attr(name, item)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            for name, item in value.system_attrs.items():
                study.set_system_attr(name, item)
        return study

    def _baseline_source(self, name: str, entry: dict[str, Any]) -> Path:
        file_name = entry.get("file")
        if not isinstance(file_name, str) or not file_name:
            raise RegistrationError(f"Constant '{name}' has an invalid baseline file name")
        candidate = Path(file_name)
        if (
            candidate.is_absolute()
            or len(candidate.parts) != 1
            or candidate.name in {"", ".", ".."}
            or candidate.name != file_name
        ):
            raise RegistrationError(f"Constant '{name}' has an unsafe baseline file name")
        root = self._clean_baseline_root
        if root.is_symlink():
            raise RegistrationError(
                f"Constant '{name}' baseline root is a symlink: {root}"
            )
        generation = entry.get("generation")
        if generation is not None:
            if (
                not isinstance(generation, str)
                or re.fullmatch(r"[0-9a-f]{32}", generation) is None
            ):
                raise RegistrationError(
                    f"Constant '{name}' has an unsafe baseline generation"
                )
            generations_root = root / "generations"
            if generations_root.is_symlink():
                raise RegistrationError(
                    f"Constant '{name}' baseline generation root is a symlink: "
                    f"{generations_root}"
                )
            generation_dir = generations_root / generation
            if generation_dir.is_symlink():
                raise RegistrationError(
                    f"Constant '{name}' baseline generation is a symlink: "
                    f"{generation_dir}"
                )
            source = generation_dir / candidate
        else:
            source = root / candidate
        if source.is_symlink():
            raise RegistrationError(
                f"Constant '{name}' baseline file is a symlink: {source}"
            )
        try:
            root_resolved = root.resolve(strict=True)
            source_resolved = source.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RegistrationError(
                f"Constant '{name}' baseline file is missing or unsafe: {source}"
            ) from exc
        if root_resolved not in source_resolved.parents:
            raise RegistrationError(
                f"Constant '{name}' baseline path escapes the trusted baseline root: {source}"
            )
        if source_resolved.is_dir():
            for current, dirnames, filenames in os.walk(
                source_resolved, followlinks=False,
            ):
                for child in (*dirnames, *filenames):
                    child_path = Path(current) / child
                    if child_path.is_symlink():
                        raise RegistrationError(
                            f"Constant '{name}' baseline directory contains a symlink: {child_path}"
                        )
        return source_resolved

    def _load_baseline_entry(self, name: str, entry: dict[str, Any]) -> Any:
        kind = entry.get("kind")
        file_name = entry.get("file")
        if not isinstance(kind, str) or not isinstance(file_name, str):
            raise RegistrationError(
                f"Constant '{name}' has an unrestorable legacy baseline; "
                "replace the constant and call snapshot_constants()"
            )
        source = self._baseline_source(name, entry)
        if kind == "artifact":
            metadata = entry.get("metadata", {})
            if not isinstance(metadata, dict) or not isinstance(entry.get("serializer"), str):
                raise RegistrationError(f"Constant '{name}' has invalid artifact baseline metadata")
            destination = self.artifact_store.copy_artifact_file(source, name)
            return ArtifactRecord(
                variable_name=name,
                serializer=entry["serializer"],
                file_path=str(destination),
                produced_by_block="clean_baseline",
                produced_by_function="constant",
                run_id="clean_baseline",
                torch_load_weights_only=bool(entry.get("torch_load_weights_only", False)),
                metadata=dict(metadata),
            )
        if kind == "torch_state":
            metadata = entry.get("metadata", {})
            if not isinstance(metadata, dict) or not isinstance(entry.get("object_kind"), str):
                raise RegistrationError(f"Constant '{name}' has invalid torch baseline metadata")
            destination = self.artifact_store.copy_artifact_file(source, name)
            return TorchStateArtifactRecord(
                variable_name=name,
                file_path=str(destination),
                object_kind=entry["object_kind"],
                metadata=dict(metadata),
            )
        if kind == "optuna":
            physical_name = entry.get("study_name")
            original_name = entry.get("original_study_name")
            owner_kind = entry.get("study_owner_kind")
            owner_key = entry.get("study_owner_key")
            if not (
                isinstance(physical_name, str)
                and physical_name
                and isinstance(original_name, str)
                and original_name
                and isinstance(owner_kind, str)
                and owner_kind
                and isinstance(owner_key, str)
                and owner_key
            ):
                raise RegistrationError(
                    f"Constant '{name}' has invalid Optuna baseline metadata"
                )
            with source.open("rb") as handle:
                baseline = pickle.load(handle)
            if not isinstance(baseline, _StudyBaseline):
                raise RegistrationError(
                    f"Constant '{name}' has an invalid Optuna baseline payload"
                )
            return _PendingStudyRestore(
                pipeline=self,
                variable_name=name,
                baseline=baseline,
                physical_name=physical_name,
                original_name=original_name,
                owner_kind=owner_kind,
                owner_key=owner_key,
                db_path=Path(self._optuna_studies_db_path_for_storage()),
            )
        if kind == "dask":
            # Restore into an independent managed generation. A lazy Dask graph
            # must never reference the replaceable baseline directory itself:
            # later snapshots (and their cleanup) may replace it while the
            # restored constant is still live.
            destination = self.artifact_store.copy_artifact_file(
                source, name, function_name="constant_dask",
            )
            return ArtifactRecord(
                variable_name=name,
                serializer="parquet",
                file_path=str(destination),
                produced_by_block="clean_baseline",
                produced_by_function="constant",
                run_id="clean_baseline",
                torch_load_weights_only=False,
                metadata={
                    "python_type": "dask.dataframe.core.DataFrame",
                    "clean_baseline_generation": "dask",
                },
            )
        if kind == "pickle":
            with source.open("rb") as handle:
                value = pickle.load(handle)
            if isinstance(value, _StudyBaseline):
                return _PendingStudyRestore(
                    pipeline=self,
                    variable_name=name,
                    baseline=value,
                    physical_name=value.name,
                    original_name=value.name,
                    owner_kind="constant",
                    owner_key=f"{self.registration_name}.{name}",
                    db_path=Path(self._optuna_studies_db_path_for_storage()),
                )
            return value
        raise RegistrationError(
            f"Constant '{name}' has an unrestorable legacy baseline; "
            "replace the constant and call snapshot_constants()"
        )

    def _backup_sqlite_database(
        self, prepared: _PreparedCleanStart, db_path: Path,
    ) -> None:
        """Snapshot one shared SQLite database before a managed restore writes it."""
        resolved = Path(db_path).expanduser()
        key = resolved.resolve() if resolved.exists() else resolved.absolute()
        if any(existing == key for existing, _, _ in prepared.sqlite_backups):
            return
        backup = prepared.rollback_dir / f"sqlite_{uuid4().hex}.db"
        if not resolved.exists():
            prepared.sqlite_backups.append((resolved, backup, False))
            return
        import sqlite3

        source = sqlite3.connect(resolved, timeout=30)
        target = sqlite3.connect(backup, timeout=30)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        prepared.sqlite_backups.append((resolved.resolve(), backup, True))

    def _materialize_pending_studies(self, prepared: _PreparedCleanStart) -> None:
        """Persist managed Optuna baselines at commit time and collect their records."""
        constants_by_pipeline = {
            id(pipeline): constants for pipeline, _, constants in prepared.states
        }
        for pending in prepared.pending_studies:
            self._backup_sqlite_database(prepared, pending.db_path)
            study = self._rebuild_study(pending.baseline)
            generation_dir = (
                pending.pipeline.artifact_store.artifact_root / "clean_baseline"
            )
            before = (
                {entry.name for entry in generation_dir.iterdir()}
                if generation_dir.is_dir()
                else set()
            )
            try:
                record = pending.pipeline.artifact_store.save(
                    variable_name=pending.variable_name,
                    value=study,
                    block_name="clean_baseline",
                    function_name="constant",
                    run_id="clean_baseline",
                    optuna_db_path=pending.db_path,
                    optuna_options=StudyArtifactOptions(
                        independent_copy=False,
                        study_name=pending.physical_name,
                        original_study_name=pending.original_name,
                        owner_kind=pending.owner_kind,
                        owner_key=pending.owner_key,
                        preserve_system_attrs=True,
                    ),
                )
            except BaseException:
                if generation_dir.is_dir():
                    for entry in generation_dir.iterdir():
                        if entry.name in before:
                            continue
                        if entry.is_dir():
                            shutil.rmtree(entry, ignore_errors=True)
                        else:
                            entry.unlink(missing_ok=True)
                raise
            prepared.staged_artifacts.append((pending.pipeline, record))
            constants_by_pipeline[id(pending.pipeline)][pending.variable_name] = record

    def _restore_sqlite_backups(self, prepared: _PreparedCleanStart) -> None:
        """Revert databases modified by managed-study restoration."""
        import sqlite3

        for db_path, backup, existed in reversed(prepared.sqlite_backups):
            try:
                if not existed:
                    for suffix in ("-wal", "-shm"):
                        sidecar = Path(f"{db_path}{suffix}")
                        if sidecar.exists():
                            sidecar.unlink()
                    if db_path.exists():
                        db_path.unlink()
                    continue
                if not backup.exists():
                    raise FileNotFoundError(
                        f"SQLite rollback backup is missing: {backup}"
                    )
                for suffix in ("-wal", "-shm"):
                    sidecar = Path(f"{db_path}{suffix}")
                    if sidecar.exists():
                        sidecar.unlink()
                source = sqlite3.connect(backup, timeout=30)
                target = sqlite3.connect(db_path)
                try:
                    source.backup(target)
                finally:
                    target.close()
                    source.close()
            except Exception as exc:
                self.logger.warning(
                    f"Could not roll back Optuna database '{db_path}': "
                    f"{type(exc).__name__}: {exc}. A recovery copy is retained."
                )
                recovery = db_path.with_name(
                    f"{db_path.name}.rollback-{uuid4().hex}.bak"
                )
                try:
                    if backup.exists():
                        shutil.copy2(backup, recovery)
                except OSError:
                    pass

    def _discard_prepared_artifacts(
        self, prepared: _PreparedCleanStart, *, keep_rollback_dir: bool = False,
    ) -> None:
        self._restore_sqlite_backups(prepared)
        for pipeline, record in prepared.staged_artifacts:
            try:
                pipeline.artifact_store.delete(record)
            except OSError as exc:
                self.logger.warning(f"Could not remove staged clean artifact '{record.file_path}': {exc}")
        if not keep_rollback_dir:
            shutil.rmtree(prepared.rollback_dir, ignore_errors=True)

    def _prepare_clean_start(self) -> _PreparedCleanStart:
        if self._root_pipeline()._invalidation_forbidden:
            raise RegistrationError(
                "clean_start=True cannot run while object invalidation is forbidden"
            )
        self._ensure_constant_baselines(required=True)
        manifests: list[tuple[Any, Any, dict[str, dict[str, Any]]]] = []
        for pipeline in self._iter_attached_pipelines():
            manifest = pipeline._clean_constant_baseline or {}
            try:
                config = pipeline._copy_definition_config(pipeline._clean_config_baseline)
                for name, entry in manifest.items():
                    if not isinstance(entry, dict):
                        raise RegistrationError(
                            f"Constant '{name}' has an unrestorable legacy baseline; "
                            "replace the constant and call snapshot_constants()"
                        )
                    pipeline._baseline_source(name, entry)
            except Exception as exc:
                raise RegistrationError(
                    f"Cannot restore clean baseline for '{pipeline.registration_name}': {exc}"
                ) from exc
            manifests.append((pipeline, config, manifest))
        configs = {id(pipeline): config for pipeline, config, _ in manifests}
        for pipeline, _, constants in manifests:
            conflicts = set(constants) & pipeline._constant_conflict_names(configs)
            if conflicts:
                raise RegistrationError(
                    f"Clean baseline for '{pipeline.registration_name}' conflicts with current registration: "
                    f"{sorted(conflicts)}"
                )
        rollback_dir = Path(tempfile.mkdtemp(prefix=".clean-start-rollback-", dir=self.metadata_root))
        prepared = _PreparedCleanStart([], [], rollback_dir, [], [], [])
        try:
            for pipeline, config, manifest in manifests:
                constants: dict[str, Any] = {}
                for name, entry in manifest.items():
                    value = pipeline._load_baseline_entry(name, entry)
                    if isinstance(value, (ArtifactRecord, TorchStateArtifactRecord)):
                        prepared.staged_artifacts.append((pipeline, value))
                    elif isinstance(value, _PendingStudyRestore):
                        prepared.pending_studies.append(value)
                    if isinstance(value, (OutputPointer, RuntimeValueReference, DataclassValueReference)):
                        raise RegistrationError(
                            f"Constant '{name}' has an unrestorable legacy baseline; "
                            "replace the constant and call snapshot_constants()"
                        )
                    constants[name] = value
                prepared.states.append((pipeline, config, constants))
            # Deferred optimization: these copies guarantee rollback safety but
            # temporarily require disk space proportional to the selected
            # subtree's disk-backed outputs. A staged-deletion design would
            # avoid the copy at the cost of more complex pointer/shared-file
            # handling; the current requirement is documented in the API
            # reference.
            seen: set[Path] = set()
            for pipeline in self._iter_attached_pipelines():
                for outputs in pipeline.producer_outputs.values():
                    for value in outputs.values():
                        if not isinstance(value, ArtifactRecord):
                            continue
                        path = Path(value.file_path).resolve()
                        if path in seen or not path.exists() or not path.is_relative_to(
                            pipeline.artifact_store.artifact_root.resolve()
                        ):
                            continue
                        seen.add(path)
                        copy_path = rollback_dir / f"output_{len(seen)}{path.suffix}"
                        if path.is_dir():
                            shutil.copytree(path, copy_path)
                        else:
                            shutil.copy2(path, copy_path)
                        prepared.output_backups.append((path, copy_path))
        except BaseException:
            self._discard_prepared_artifacts(prepared)
            raise
        return prepared

    def _restore_clean_start(self, prepared: _PreparedCleanStart) -> None:
        root = self._root_pipeline()
        try:
            snapshots = [
                (pipeline, pipeline._snapshot_runtime_state(), pipeline._config,
                 pipeline._expression_runtime_namespace_cache, pipeline._gate_cleanup_predecided,
                 dict(pipeline._inspection_cache_bindings), dict(pipeline._inspection_cache_entries),
                 dict(pipeline._inspection_copy_skipped))
                for pipeline in root._iter_attached_pipelines()
            ]
            group_names = dict(root._optuna_study_group_names)
            group_addresses = {
                name: set(addresses) for name, addresses in root._optuna_study_group_addresses.items()
            }
            old_constants = [(pipeline, tuple(pipeline.manual_values.values())) for pipeline, _, _ in prepared.states]
        except BaseException:
            self._discard_prepared_artifacts(prepared)
            raise
        try:
            self.logger.warning(
                "clean_start=True restores definition-time configurations and saved constant "
                "values, discarding later configuration and constant changes"
            )
            self._materialize_pending_studies(prepared)
            if self.nodes:
                self._invalidate_from_priority(self._sorted_nodes()[0].execution_priority)
            self._invalidate_all_outputs()
            for pipeline, config, constants in prepared.states:
                if not pipeline._is_atom:
                    pipeline._config = config
                pipeline.manual_values = constants
                pipeline._unload_from_memory()
                pipeline._inspection_copy_skipped.clear()
                pipeline._expression_runtime_namespace_cache = None
                pipeline._gate_cleanup_predecided = True
            for pipeline, _, _ in reversed(prepared.states):
                pipeline._rebuild_visible_state(pipeline._incoming_parent_outputs())
            if self.parent_pipeline is not None:
                self._resync_mirror_to_parent()
        except BaseException as exc:
            recovery_errors: list[str] = []
            preserve_rollback_dir = False
            for original, backup in prepared.output_backups:
                try:
                    if original.is_dir():
                        shutil.rmtree(original)
                    elif original.exists():
                        original.unlink()
                    original.parent.mkdir(parents=True, exist_ok=True)
                    if backup.is_dir():
                        shutil.copytree(backup, original)
                    else:
                        shutil.copy2(backup, original)
                except Exception as recovery_exc:
                    recovery = original.with_name(
                        f".{original.name}.rollback-{uuid4().hex}"
                    )
                    try:
                        if backup.is_dir():
                            shutil.copytree(backup, recovery)
                        else:
                            shutil.copy2(backup, recovery)
                    except OSError:
                        try:
                            backup.replace(recovery)
                        except OSError:
                            recovery = backup
                            preserve_rollback_dir = True
                    message = (
                        f"Could not restore artifact '{original}' during clean-start "
                        f"rollback: {type(recovery_exc).__name__}: {recovery_exc}; "
                        f"recovery copy retained at '{recovery}'"
                    )
                    recovery_errors.append(message)
                    self.logger.warning(message)
            for (pipeline, state, config, namespace, gate, bindings, entries, skipped) in snapshots:
                try:
                    pipeline._config = config
                    pipeline._restore_runtime_state(state)
                    pipeline._expression_runtime_namespace_cache = namespace
                    pipeline._gate_cleanup_predecided = gate
                    pipeline._inspection_cache_bindings = bindings
                    pipeline._inspection_cache_entries = entries
                    pipeline._inspection_copy_skipped = skipped
                except Exception as recovery_exc:
                    message = (
                        f"Could not restore runtime state for pipeline "
                        f"'{pipeline.registration_name}' during clean-start rollback: "
                        f"{type(recovery_exc).__name__}: {recovery_exc}"
                    )
                    recovery_errors.append(message)
                    self.logger.warning(message)
            root._optuna_study_group_names = group_names
            root._optuna_study_group_addresses = group_addresses
            try:
                self._discard_prepared_artifacts(
                    prepared, keep_rollback_dir=preserve_rollback_dir
                )
            except Exception as cleanup_exc:
                message = (
                    f"Clean-start rollback cleanup failed: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
                recovery_errors.append(message)
                self.logger.warning(message)
            for message in recovery_errors:
                exc.add_note(message)
            raise
        shutil.rmtree(prepared.rollback_dir, ignore_errors=True)
        for pipeline, values in old_constants:
            for value in values:
                pipeline._cleanup_replaced_artifact(value)
