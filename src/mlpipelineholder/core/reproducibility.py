"""Root-owned seed and definition-time clean-start baselines."""

from __future__ import annotations

import copy
import os
import pickle
import random
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
from ..integrations.dataframe import dump_dataframe, load_dataframe
from ..integrations.optuna.support import (
    OPTUNA_STUDY_SERIALIZER,
    is_optuna_study,
)
from ..state.output_pointers import OutputPointer


@dataclass(frozen=True)
class _StudyBaseline:
    """Detached Optuna state; no reference to the mutable backing database."""

    name: str
    directions: tuple[Any, ...]
    trials: tuple[Any, ...]
    user_attrs: dict[str, Any]
    system_attrs: dict[str, Any]
    sampler: Any


@dataclass
class _PreparedCleanStart:
    states: list[tuple[Any, Any, dict[str, Any]]]
    staged_artifacts: list[tuple[Any, ArtifactRecord | TorchStateArtifactRecord]]
    rollback_dir: Path
    output_backups: list[tuple[Path, Path]]


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
        def _tree_declared_output_names(self) -> set[str]: ...
        def _visible_config_names(self) -> set[str]: ...
        def _invalidate_all_outputs(self) -> None: ...
        def _invalidate_from_priority(self, priority: float) -> None: ...
        def _sorted_nodes(self) -> list[Any]: ...
        def _cleanup_replaced_artifact(self, previous_value: Any) -> None: ...
        def _rebuild_visible_state(self, upstream_outputs: Any = None) -> None: ...
        def _incoming_parent_outputs(self) -> dict[str, Any]: ...
        def _resync_mirror_to_parent(self) -> None: ...
        def _unload_from_memory(self, object_names: Any = None) -> None: ...
        def _materialize_stored_value(self, value: Any, placeholder_error: str) -> Any: ...
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
                return self._write_pickle_baseline(
                    directory, index, self._study_baseline(study),
                )
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

    def _stage_constant_baseline(self) -> tuple[dict[str, dict[str, Any]], Path]:
        """Prepare this pipeline's baseline without touching its live snapshot."""
        staging = Path(tempfile.mkdtemp(prefix=".clean-baseline-stage-", dir=self.metadata_root))
        manifest: dict[str, dict[str, Any]] = {}
        try:
            for index, (name, value) in enumerate(self.manual_values.items()):
                try:
                    manifest[name] = self._write_baseline_entry(
                        staging, index, name, value,
                    )
                except RegistrationError:
                    raise
                except Exception as exc:
                    raise RegistrationError(
                        f"Constant '{name}' cannot be independently snapshotted and "
                        f"restored: {type(exc).__name__}: {exc}"
                    ) from exc
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return manifest, staging

    def _capture_constant_baselines(self, pipelines: list[Any]) -> None:
        """Stage the entire tree first; publish all manifests or roll them back."""
        staged: list[tuple[Any, dict[str, dict[str, Any]], Path]] = []
        published: list[tuple[Any, dict[str, dict[str, Any]] | None, Path | None]] = []
        try:
            for pipeline in pipelines:
                manifest, staging = pipeline._stage_constant_baseline()
                staged.append((pipeline, manifest, staging))
            self._validate_constant_snapshot_names(
                [(pipeline, manifest) for pipeline, manifest, _ in staged]
            )
            for pipeline, manifest, staging in staged:
                directory = pipeline._clean_baseline_root
                if directory.is_symlink():
                    raise RegistrationError(f"Refusing to replace symlinked clean baseline: {directory}")
                backup = Path(pipeline.metadata_root) / f".clean-baseline-old-{uuid4().hex}"
                old_manifest = pipeline._clean_constant_baseline
                had_baseline = directory.exists()
                if had_baseline:
                    os.replace(directory, backup)
                try:
                    os.replace(staging, directory)
                except BaseException:
                    if had_baseline:
                        os.replace(backup, directory)
                    raise
                pipeline._clean_constant_baseline = manifest
                published.append((pipeline, old_manifest, backup if had_baseline else None))
        except BaseException:
            for pipeline, manifest, backup in reversed(published):
                directory = pipeline._clean_baseline_root
                if directory.exists():
                    shutil.rmtree(directory)
                if backup is not None:
                    os.replace(backup, directory)
                pipeline._clean_constant_baseline = manifest
            raise
        finally:
            for _, _, staging in staged:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)
        for _, _, backup in published:
            if backup is not None:
                shutil.rmtree(backup, ignore_errors=True)

    def _validate_constant_snapshot_names(
        self, snapshots: list[tuple[Any, dict[str, dict[str, Any]]]],
    ) -> None:
        declared = self._root_pipeline()._tree_declared_output_names()
        for pipeline, snapshot in snapshots:
            visible_config: set[str] = set()
            ancestor = pipeline
            while ancestor is not None:
                visible_config.update(ancestor._config_name_mapping(ancestor._clean_config_baseline))
                ancestor = ancestor.parent_pipeline
            conflicts = set(snapshot) & (visible_config | declared)
            if conflicts:
                raise RegistrationError(
                    f"Clean baseline for '{pipeline.registration_name}' conflicts with current registration: "
                    f"{sorted(conflicts)}"
                )

    def _ensure_constant_baselines(self, *, required: bool = False) -> None:
        """Attempt automatic capture once; ordinary runs survive capture failures."""
        root = self._root_pipeline()
        pipelines = root._iter_attached_pipelines()
        selected = self._iter_attached_pipelines()
        failed = [pipeline for pipeline in selected if pipeline._clean_baseline_error]
        if failed and required:
            raise RegistrationError(
                f"Clean baseline unavailable for '{failed[0].registration_name}': "
                f"{failed[0]._clean_baseline_error}; call snapshot_constants() after repairing constants"
            )
        pending = [
            pipeline for pipeline in pipelines
            if pipeline._clean_constant_baseline is None and pipeline._clean_baseline_error is None
        ]
        if not pending:
            return
        try:
            root._capture_constant_baselines(pending)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            for pipeline in pending:
                pipeline._clean_baseline_error = message
            if required and any(pipeline in selected for pipeline in pending):
                raise RegistrationError(f"Clean baseline unavailable: {message}") from exc
            self.logger.warning(
                f"Automatic constant baseline unavailable; ordinary execution continues. "
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
        if not isinstance(file_name, str) or Path(file_name).name != file_name:
            raise RegistrationError(f"Constant '{name}' has an invalid baseline file name")
        source = self._clean_baseline_root / file_name
        if source.is_symlink() or not source.exists():
            raise RegistrationError(f"Constant '{name}' baseline file is missing or unsafe: {source}")
        return source

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
        if kind == "dask":
            return load_dataframe("parquet", source)
        if kind == "pickle":
            with source.open("rb") as handle:
                value = pickle.load(handle)
            if isinstance(value, _StudyBaseline):
                return self._rebuild_study(value)
            return value
        raise RegistrationError(
            f"Constant '{name}' has an unrestorable legacy baseline; "
            "replace the constant and call snapshot_constants()"
        )

    def _discard_prepared_artifacts(self, prepared: _PreparedCleanStart) -> None:
        for pipeline, record in prepared.staged_artifacts:
            try:
                pipeline.artifact_store.delete(record)
            except OSError as exc:
                self.logger.warning(f"Could not remove staged clean artifact '{record.file_path}': {exc}")
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
        declared = self._tree_declared_output_names()
        for pipeline, _, constants in manifests:
            visible_config_names: set[str] = set()
            ancestor = pipeline
            while ancestor is not None:
                config = configs.get(id(ancestor), ancestor._config)
                visible_config_names.update(ancestor._config_name_mapping(config))
                ancestor = ancestor.parent_pipeline
            conflicts = set(constants) & (visible_config_names | declared)
            if conflicts:
                raise RegistrationError(
                    f"Clean baseline for '{pipeline.registration_name}' conflicts with current registration: "
                    f"{sorted(conflicts)}"
                )
        rollback_dir = Path(tempfile.mkdtemp(prefix=".clean-start-rollback-", dir=self.metadata_root))
        prepared = _PreparedCleanStart([], [], rollback_dir, [])
        try:
            for pipeline, config, manifest in manifests:
                constants: dict[str, Any] = {}
                for name, entry in manifest.items():
                    value = pipeline._load_baseline_entry(name, entry)
                    if isinstance(value, (ArtifactRecord, TorchStateArtifactRecord)):
                        prepared.staged_artifacts.append((pipeline, value))
                    if isinstance(value, (OutputPointer, RuntimeValueReference, DataclassValueReference)):
                        raise RegistrationError(
                            f"Constant '{name}' has an unrestorable legacy baseline; "
                            "replace the constant and call snapshot_constants()"
                        )
                    constants[name] = value
                prepared.states.append((pipeline, config, constants))
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
            if self.nodes:
                self._invalidate_from_priority(self._sorted_nodes()[0].execution_priority)
            self._invalidate_all_outputs()
            for pipeline, config, constants in prepared.states:
                if not pipeline._is_atom:
                    pipeline._config = config
                pipeline.manual_values = constants
                pipeline._unload_from_memory()
                pipeline._expression_runtime_namespace_cache = None
                pipeline._gate_cleanup_predecided = True
            for pipeline, _, _ in reversed(prepared.states):
                pipeline._rebuild_visible_state(pipeline._incoming_parent_outputs())
            if self.parent_pipeline is not None:
                self._resync_mirror_to_parent()
        except BaseException:
            for original, backup in prepared.output_backups:
                if original.is_dir():
                    shutil.rmtree(original)
                elif original.exists():
                    original.unlink()
                original.parent.mkdir(parents=True, exist_ok=True)
                if backup.is_dir():
                    shutil.copytree(backup, original)
                else:
                    shutil.copy2(backup, original)
            for (pipeline, state, config, namespace, gate, bindings, entries, skipped) in snapshots:
                pipeline._config = config
                pipeline._restore_runtime_state(state)
                pipeline._expression_runtime_namespace_cache = namespace
                pipeline._gate_cleanup_predecided = gate
                pipeline._inspection_cache_bindings = bindings
                pipeline._inspection_cache_entries = entries
                pipeline._inspection_copy_skipped = skipped
            root._optuna_study_group_names = group_names
            root._optuna_study_group_addresses = group_addresses
            self._discard_prepared_artifacts(prepared)
            raise
        shutil.rmtree(prepared.rollback_dir, ignore_errors=True)
        for pipeline, values in old_constants:
            for value in values:
                pipeline._cleanup_replaced_artifact(value)
