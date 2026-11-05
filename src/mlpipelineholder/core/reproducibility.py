"""Root-owned seed and definition-time clean-start baselines."""

from __future__ import annotations

import copy
import pickle
import random
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .models import ArtifactRecord, DataclassValueReference, RuntimeValueReference, TorchStateArtifactRecord
from ..exceptions import RegistrationError
from ..integrations.optuna.support import is_optuna_study
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


@dataclass(frozen=True)
class _DaskBaseline:
    """Materialized source data, independent of a retired managed Parquet path."""

    frame: Any
    partitions: int


class ReproducibilityMixin:
    """Keep independent baseline copies, never references into mutable run state."""

    if TYPE_CHECKING:
        registration_name: str = ""
        parent_pipeline: Any = None
        _seed: int | None = None
        _deterministic_torch: bool = False
        _clean_config_baseline: Any = None
        _clean_constant_baseline: dict[str, bytes] | None = None
        _clean_constant_disk_names: set[str] = set()
        manual_values: dict[str, Any] = {}
        logger: Any = None
        artifact_store: Any = None
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
        except Exception as exc:
            raise RegistrationError(
                f"Cannot snapshot the pipeline's definition-time configuration: {type(exc).__name__}: {exc}"
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
        previous_benchmark = torch.backends.cudnn.benchmark
        try:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            yield
        finally:
            torch.use_deterministic_algorithms(previous_mode)
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

    def _validate_clean_constant_name(self, name: str) -> None:
        ancestor = self
        while ancestor is not None:
            if name in ancestor._config_name_mapping(ancestor._clean_config_baseline):
                raise RegistrationError(
                    f"Constant '{name}' conflicts with the definition-time configuration "
                    f"of pipeline '{ancestor.registration_name}'"
                )
            ancestor = ancestor.parent_pipeline

    def _constant_baseline_bytes(
        self, name: str, value: Any, *, materialize_artifacts: bool = True,
    ) -> bytes:
        if not materialize_artifacts and isinstance(
            value, (OutputPointer, RuntimeValueReference, DataclassValueReference),
        ):
            raise RegistrationError(
                f"Constant '{name}' cannot be independently snapshotted and restored: "
                "it refers to transient or unavailable state"
            )
        if materialize_artifacts and isinstance(value, (ArtifactRecord, TorchStateArtifactRecord)):
            value = self._materialize_stored_value(value, "")
        try:
            dask = sys.modules.get("dask.dataframe")
            if dask is not None and isinstance(value, dask.DataFrame):
                value = _DaskBaseline(value.compute(), value.npartitions)
            if is_optuna_study(value):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    system_attrs = dict(getattr(value, "system_attrs", {}))
                value = _StudyBaseline(
                    name=value.study_name,
                    directions=tuple(value.directions),
                    trials=tuple(value.get_trials(deepcopy=True)),
                    user_attrs=dict(value.user_attrs),
                    system_attrs=system_attrs,
                    sampler=value.sampler,
                )
            frozen = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.loads(frozen)
            return frozen
        except Exception as exc:
            raise RegistrationError(
                f"Constant '{name}' cannot be independently snapshotted and restored: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _validate_constant_snapshot_names(
        self, snapshots: list[tuple[Any, dict[str, bytes], set[str]]],
    ) -> None:
        declared = self._root_pipeline()._tree_declared_output_names()
        for pipeline, snapshot, _ in snapshots:
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

    def _ensure_constant_baselines(self) -> None:
        """Capture all attached siblings together before the first execution entry."""
        pending: list[tuple[Any, dict[str, bytes], set[str]]] = []
        for pipeline in self._root_pipeline()._iter_attached_pipelines():
            if pipeline._clean_constant_baseline is None:
                pending.append((pipeline, {
                    name: pipeline._constant_baseline_bytes(name, value)
                    for name, value in pipeline.manual_values.items()
                }, {
                    name for name, value in pipeline.manual_values.items()
                    if isinstance(value, (ArtifactRecord, TorchStateArtifactRecord))
                }))
        self._validate_constant_snapshot_names(pending)
        for pipeline, snapshot, disk_names in pending:
            pipeline._clean_constant_baseline = snapshot
            pipeline._clean_constant_disk_names = disk_names

    def snapshot_constants(self) -> None:
        """Replace the root tree's automatic constant baseline with current values."""
        if self.parent_pipeline is not None:
            raise RegistrationError("snapshot_constants() must be called on the root pipeline")
        prepared = [
            (pipeline, {
                name: pipeline._constant_baseline_bytes(name, value)
                for name, value in pipeline.manual_values.items()
            }, {
                name for name, value in pipeline.manual_values.items()
                if isinstance(value, (ArtifactRecord, TorchStateArtifactRecord))
            })
            for pipeline in self._iter_attached_pipelines()
        ]
        self._validate_constant_snapshot_names(prepared)
        for pipeline, snapshot, disk_names in prepared:
            pipeline._clean_constant_baseline = snapshot
            pipeline._clean_constant_disk_names = disk_names

    @staticmethod
    def _decode_constant_baseline(blob: bytes) -> Any:
        value = pickle.loads(blob)
        if isinstance(value, _DaskBaseline):
            import dask.dataframe as dd
            return dd.from_pandas(
                value.frame,
                npartitions=min(value.partitions, max(1, len(value.frame))),
                sort=False,
            )
        if not isinstance(value, _StudyBaseline):
            return value
        import optuna
        study = optuna.create_study(
            study_name=value.name, directions=list(value.directions), sampler=value.sampler,
        )
        study.add_trials(list(value.trials))
        for name, item in value.user_attrs.items():
            study.set_user_attr(name, item)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            for name, item in value.system_attrs.items():
                study.set_system_attr(name, item)
        return study

    def _prepare_clean_start(self) -> list[tuple[Any, Any, dict[str, Any]]]:
        if self._root_pipeline()._invalidation_forbidden:
            raise RegistrationError(
                "clean_start=True cannot run while object invalidation is forbidden"
            )
        self._ensure_constant_baselines()
        prepared: list[tuple[Any, Any, dict[str, Any]]] = []
        for pipeline in self._iter_attached_pipelines():
            try:
                config = pipeline._copy_definition_config(pipeline._clean_config_baseline)
                constants = {
                    name: pipeline._decode_constant_baseline(blob)
                    for name, blob in (pipeline._clean_constant_baseline or {}).items()
                }
                for name, value in constants.items():
                    if isinstance(value, (OutputPointer, RuntimeValueReference, DataclassValueReference)):
                        raise RegistrationError(
                            f"Constant '{name}' has an unrestorable legacy baseline; "
                            "replace the constant and call snapshot_constants()"
                        )
            except Exception as exc:
                raise RegistrationError(
                    f"Cannot restore clean baseline for '{pipeline.registration_name}': {exc}"
                ) from exc
            prepared.append((pipeline, config, constants))
        configs = {id(pipeline): config for pipeline, config, _ in prepared}
        for pipeline, _, constants in prepared:
            visible_config_names: set[str] = set()
            ancestor = pipeline
            while ancestor is not None:
                config = configs.get(id(ancestor), ancestor._config)
                visible_config_names.update(ancestor._config_name_mapping(config))
                ancestor = ancestor.parent_pipeline
            conflicts = set(constants) & (visible_config_names | self._tree_declared_output_names())
            if conflicts:
                raise RegistrationError(
                    f"Clean baseline for '{pipeline.registration_name}' conflicts with current registration: "
                    f"{sorted(conflicts)}"
                )
        return prepared

    def _restore_clean_start(self, prepared: list[tuple[Any, Any, dict[str, Any]]]) -> None:
        self.logger.warning(
            "clean_start=True restores definition-time configurations and saved constant "
            "values, discarding later configuration and constant changes"
        )
        old_constants = [(pipeline, dict(pipeline.manual_values)) for pipeline, _, _ in prepared]
        if self.nodes:
            self._invalidate_from_priority(self._sorted_nodes()[0].execution_priority)
        self._invalidate_all_outputs()
        for pipeline, config, constants in prepared:
            if not pipeline._is_atom:
                pipeline._config = config
            previous = next(old for owner, old in old_constants if owner is pipeline)
            disk_names = pipeline._clean_constant_disk_names
            pipeline.manual_values = {
                name: previous[name] if name in disk_names and name in previous else value
                for name, value in constants.items()
                if name not in disk_names or name in previous
            }
            for name, value in constants.items():
                if name in disk_names:
                    cast(Any, pipeline).set_constant_value(
                        name, value, to_disk=not is_optuna_study(value),
                    )
            pipeline._unload_from_memory()
            pipeline._expression_runtime_namespace_cache = None
            # A clean run explicitly permits discarding outputs, even if a gate skips.
            pipeline._gate_cleanup_predecided = True
        for pipeline, _, _ in reversed(prepared):
            pipeline._rebuild_visible_state(pipeline._incoming_parent_outputs())
        if self.parent_pipeline is not None:
            self._resync_mirror_to_parent()
        for pipeline, values in old_constants:
            for value in values.values():
                pipeline._cleanup_replaced_artifact(value)
