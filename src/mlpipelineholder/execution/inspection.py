"""Temporary value and non-committing execution inspection helpers."""

from __future__ import annotations

import copy
import copyreg
import gc
import math
import os
import shutil
import sys
import types
import warnings
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import date, datetime, time, timedelta
from enum import Enum
from itertools import islice
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath, PosixPath, WindowsPath
from types import BuiltinFunctionType, FunctionType
from typing import TYPE_CHECKING, Any, Final, Self, cast
from uuid import UUID, uuid4

from ..core.constants import _IMMUTABLE_TYPES, _MISSING
from ..core.models import (
    ArtifactRecord,
    CallableValueReference,
    DataclassValueReference,
    ResolutionSource,
    RuntimeValueReference,
    TorchStateArtifactRecord,
)
from ..exceptions import (
    InspectionCopyError,
    InspectionMemoryError,
    ResolutionError,
)
from ..persistence.artifacts.serializers import (
    choose_serializer,
    dump_value,
    extension_for,
    load_value,
)
from ..integrations.optuna.support import (
    OPTUNA_STUDY_SERIALIZER,
    is_optuna_sampler,
    is_optuna_study,
)
from ..state.output_pointers import OutputPointer, PointerResolutionError

if TYPE_CHECKING:
    from types import TracebackType


DEFAULT_INSPECTION_MEMORY_SAFETY_MARGIN: Final[float] = 0.25

_MATERIAL_MEMORY_BYTES: Final[int] = 1 << 20
_SHALLOW_SAMPLE_LIMIT: Final[int] = 32
_SHALLOW_DEPTH_LIMIT: Final[int] = 2
_PATH_ENTRY_LIMIT: Final[int] = 4096
_TEMPORARY_ALLOCATION_FRACTION: Final[float] = 0.1

# Concrete immutable leaf types whose subclasses may carry mutable instance
# state and therefore must not be shared by identity.
_IMMUTABLE_EXACT_TYPES: Final[tuple[type, ...]] = (
    Path,
    PurePath,
    PosixPath,
    PurePosixPath,
    WindowsPath,
    PureWindowsPath,
    date,
    datetime,
    time,
    timedelta,
    UUID,
    range,
    slice,
)

_ARTIFACT_MEMORY_FACTORS: Final[dict[str, float]] = {
    "numpy": 1.1,
    "torch": 1.1,
    "json": 2.0,
    "pickle": 2.0,
    "feather": 4.0,
    "parquet": 4.0,
}
_RETAINED_ARTIFACT_SERIALIZERS: Final[frozenset[str]] = frozenset(
    {"json", "pickle"}
)

_InspectionArtifactIdentity = tuple[str, str, bool, str, str]
_InspectionMemoryIdentity = tuple[str, int]
_InspectionCacheKey = tuple[
    _InspectionArtifactIdentity | _InspectionMemoryIdentity, bool
]


@dataclass(slots=True)
class _InspectionCacheEntry:
    value: Any
    compute: bool


@dataclass(slots=True)
class _InspectionCacheBinding:
    """One name bound to an artifact or to a copied in-memory value."""

    cache_key: _InspectionCacheKey
    kind: str
    priority: int | None
    compute: bool
    original_ref: Any | None = None
    # Strong identity guard for non-weak-referenceable originals (lists,
    # dicts, ...). Object ids can be reused after destruction, so the guard
    # keeps the exact original alive until the binding is replaced/unloaded.
    original_guard: Any | None = None
    original_id: int = 0


class _InspectionValueNotCopyable(Exception):
    """Raised when an in-memory value cannot be isolated by copying."""


_BOUND_OWNER_CALLABLE_TYPES: Final[tuple[type, ...]] = (
    types.MethodType,
    BuiltinFunctionType,
    types.MethodWrapperType,
)


def _bound_callable_owner(value: Any) -> Any | None:
    """Owner of any bound callable, or ``None`` for stateless callables.

    Covers Python methods, built-in methods, and method wrappers such as
    ``items.__iter__``. Module-level callables and unbound descriptors have
    no live instance owner and are stateless for copying purposes.
    """
    if not isinstance(value, _BOUND_OWNER_CALLABLE_TYPES):
        return None
    owner = getattr(value, "__self__", None)
    if owner is None or isinstance(owner, (types.ModuleType, type)):
        return None
    return owner


def _bound_builtin_owner(value: Any) -> Any | None:
    """Owner of a bound built-in method, or ``None`` for stateless builtins.

    ``BuiltinFunctionType`` covers both module-level builtins such as ``len``
    or ``math.sin`` (whose ``__self__`` is a module or absent) and methods
    bound to a live object such as ``items.append``. Only the latter are
    excluded from identity passthrough in :meth:`InspectionCopier.prepare`;
    Python methods and method wrappers reach the reduce protocol on their own.
    """
    if not isinstance(value, BuiltinFunctionType):
        return None
    return _bound_callable_owner(value)


class InspectionCopier:
    """Copy resolved values while preserving aliases within one inspection."""

    def __init__(self, logger: Any, *, allow_mutable_objects: bool) -> None:
        self._logger = logger
        self._allow_mutable_objects = allow_mutable_objects
        self._memo: dict[int, Any] = {}
        self._keepalive: list[Any] = []
        self._copy_started = False

    @property
    def copy_started(self) -> bool:
        return self._copy_started

    def prepare(
        self,
        value: Any,
        *,
        parameter_name: str,
        source: ResolutionSource,
        caller_owned: bool = False,
    ) -> tuple[Any, ResolutionSource]:
        if caller_owned or self._allow_mutable_objects:
            return value, source
        if self._is_known_immutable(value):
            return value, source
        if value is self._logger or isinstance(value, (FunctionType, type)):
            self._memo[id(value)] = value
            return value, replace(source, identity_passthrough=True)
        if isinstance(value, BuiltinFunctionType) and _bound_builtin_owner(value) is None:
            # Module-level builtins such as ``len`` or ``math.sin`` are
            # stateless templates; bound built-in methods carry an owner and
            # are reconstructed through the copy protocol below instead.
            self._memo[id(value)] = value
            return value, replace(source, identity_passthrough=True)
        if is_optuna_study(value) or is_optuna_sampler(value):
            raise self._copy_error(
                parameter_name,
                source,
                value,
                "the object is backed by shared Optuna state",
            )
        if id(value) in self._memo:
            copied = self._memo[id(value)]
            return copied, replace(source, copied=copied is not value)
        try:
            self._copy_started = True
            copied = self._copy_value(value, parameter_name, source)
        except (InspectionCopyError, MemoryError):
            raise
        except Exception as exc:
            raise self._copy_error(
                parameter_name,
                source,
                value,
                f"{type(exc).__name__}: {exc}",
            ) from exc
        if copied is value:
            raise self._copy_error(
                parameter_name,
                source,
                value,
                "copying returned the original object",
            )
        self._memo[id(value)] = copied
        # Keep the original alive for the batch so its object id cannot be
        # reused while this memo still maps it, mirroring copy._keep_alive.
        self._keepalive.append(value)
        return copied, replace(source, copied=True)

    def requires_copy(self, value: Any, *, caller_owned: bool = False) -> bool:
        """Whether protected mode would allocate a new object for ``value``.

        Used by the memory preflight to decide which resolved values need an
        estimate without performing any copying. Optuna values are excluded
        because copying rejects them outright rather than allocating for them.
        """
        if caller_owned or self._allow_mutable_objects:
            return False
        if self._is_known_immutable(value):
            return False
        if value is self._logger or isinstance(value, (FunctionType, type)):
            return False
        if isinstance(value, BuiltinFunctionType) and _bound_builtin_owner(value) is None:
            return False
        if is_optuna_study(value) or is_optuna_sampler(value):
            return False
        if (
            isinstance(value, ArtifactRecord)
            and (
                value.serializer == OPTUNA_STUDY_SERIALIZER
                or value.metadata.get("optuna_type") == "sampler"
            )
        ):
            return False
        return True

    @classmethod
    def _is_known_immutable(cls, value: Any) -> bool:
        if type(value) in _IMMUTABLE_TYPES:
            return True
        if type(value) in _IMMUTABLE_EXACT_TYPES:
            return True
        if isinstance(value, Enum):
            return True
        if type(value) is tuple or type(value) is frozenset:
            return all(cls._is_known_immutable(item) for item in value)
        return False

    def _copy_value(
        self,
        value: Any,
        parameter_name: str,
        source: ResolutionSource,
    ) -> Any:
        # Python's generic deepcopy delegates to pandas' shallow object-cell
        # copying for DataFrames nested inside containers or wrapper objects.
        # Exact builtin containers recurse through this copier; every other
        # object is reconstructed from its own copy/reduce protocol below with
        # prepared components, preserving the shared memo and pandas-aware cell
        # isolation. Specialized handlers only claim exact library types so
        # subclasses can describe themselves through their own protocol.
        if type(value) is dict:
            copied_dict: dict[Any, Any] = {}
            self._memo[id(value)] = copied_dict
            for key, item in value.items():
                copied_key = self.prepare(
                    key, parameter_name=parameter_name, source=source
                )[0]
                copied_item = self.prepare(
                    item, parameter_name=parameter_name, source=source
                )[0]
                copied_dict[copied_key] = copied_item
            return copied_dict
        if type(value) is list:
            copied_list: list[Any] = []
            self._memo[id(value)] = copied_list
            for item in value:
                copied_list.append(
                    self.prepare(item, parameter_name=parameter_name, source=source)[0]
                )
            return copied_list
        if type(value) is tuple:
            copied_items = [
                self.prepare(item, parameter_name=parameter_name, source=source)[0]
                for item in value
            ]
            # Recursive tuple/list graphs may have populated the tuple's memo
            # during the recursive copy. Reuse that copy to preserve the cycle.
            if id(value) in self._memo:
                return self._memo[id(value)]
            copied_tuple = tuple(copied_items)
            self._memo[id(value)] = copied_tuple
            return copied_tuple
        if type(value) is set:
            copied_set: set[Any] = set()
            self._memo[id(value)] = copied_set
            for item in value:
                copied_set.add(
                    self.prepare(item, parameter_name=parameter_name, source=source)[0]
                )
            return copied_set
        # Optional libraries are looked up in ``sys.modules`` instead of being
        # imported: an instance of one of their types cannot exist unless the
        # library was already imported, and importing here would make copying
        # an unrelated object pull in pandas, NumPy, Dask, and Torch.
        pd = cast(Any, sys.modules.get("pandas"))
        if pd is not None:
            if isinstance(value, pd.DataFrame):
                copied = value.copy(deep=True)
                self._memo[id(value)] = copied
                object_columns = {
                    column_index
                    for column_index, dtype in enumerate(value.dtypes)
                    if dtype == object
                }
                for column_index in object_columns:
                    for row_index in range(len(value.index)):
                        cell = value.iat[row_index, column_index]
                        copied.iat[row_index, column_index] = self.prepare(
                            cell,
                            parameter_name=parameter_name,
                            source=source,
                        )[0]
                return copied
            if isinstance(value, pd.Series):
                copied = value.copy(deep=True)
                self._memo[id(value)] = copied
                if value.dtype == object:
                    for index in range(len(value.index)):
                        copied.iat[index] = self.prepare(
                            value.iat[index],
                            parameter_name=parameter_name,
                            source=source,
                        )[0]
                return copied

        np = cast(Any, sys.modules.get("numpy"))
        if np is not None and type(value) is np.ndarray:
            if value.dtype.hasobject:
                return self._copy_object_ndarray(value, parameter_name, source)
            copied = value.copy()
            self._memo[id(value)] = copied
            return copied

        dd = cast(Any, sys.modules.get("dask.dataframe"))
        if dd is not None and (type(value) is dd.DataFrame or type(value) is dd.Series):
            copied = value.copy()
            self._memo[id(value)] = copied
            return copied

        torch = cast(Any, sys.modules.get("torch"))
        if torch is not None:
            nn = cast(Any, sys.modules.get("torch.nn"))
            if nn is not None and isinstance(value, nn.Parameter):
                return copy.deepcopy(value, self._memo)
            if type(value) is torch.Tensor:
                copied = value.detach().clone()
                self._memo[id(value)] = copied
                return copied
            if nn is not None and isinstance(value, nn.Module):
                # Reconstructing a module through its reduce protocol would
                # downgrade its Parameter tensors to plain tensors, so keep
                # Python's deepcopy for these compatibility boundaries.
                return copy.deepcopy(value, self._memo)
            optim = cast(Any, sys.modules.get("torch.optim"))
            if optim is not None and isinstance(value, optim.Optimizer):
                return copy.deepcopy(value, self._memo)

        # Classes and modules are atomic or shared templates; they keep
        # Python's deepcopy semantics. Bound methods carry their owner and are
        # reconstructed through the reduce protocol below.
        if isinstance(value, (type, types.ModuleType)):
            return copy.deepcopy(value, self._memo)

        # Mirror copy.deepcopy's selection order. An explicit ``__deepcopy__``
        # owns its isolation semantics and is trusted as written.
        deepcopier = getattr(value, "__deepcopy__", None)
        if deepcopier is not None:
            return deepcopier(self._memo)
        reductor = copyreg.dispatch_table.get(type(value))
        if reductor is not None:
            reduction = reductor(value)
        else:
            reductor = getattr(value, "__reduce_ex__", None)
            if reductor is not None:
                reduction = reductor(4)
            else:
                reductor = getattr(value, "__reduce__", None)
                if reductor is None:
                    raise TypeError(
                        f"un(deep)copyable object of type {type(value).__name__}"
                    )
                reduction = reductor()
        if isinstance(reduction, str):
            return value
        return self._copy_reduced(value, reduction, parameter_name, source)

    def _copy_reduced(
        self,
        value: Any,
        reduction: Any,
        parameter_name: str,
        source: ResolutionSource,
    ) -> Any:
        """Reconstruct ``value`` from its reduce tuple with prepared components.

        This intentionally mirrors :func:`copy._reconstruct`: the only
        difference is that each recursive ``deepcopy(component, memo)`` call
        becomes ``self.prepare(component, ...)[0]``.
        """
        if len(reduction) > 5:
            raise TypeError(
                "reduce tuple with more than five items is not supported"
            )
        func = reduction[0]
        args = reduction[1] if len(reduction) > 1 else ()
        state = reduction[2] if len(reduction) > 2 else None
        listitems = reduction[3] if len(reduction) > 3 else None
        dictitems = reduction[4] if len(reduction) > 4 else None

        copied_args = tuple(
            self.prepare(arg, parameter_name=parameter_name, source=source)[0]
            for arg in args
        )
        copied = func(*copied_args)
        self._memo[id(value)] = copied

        if state is not None:
            copied_state = self.prepare(
                state, parameter_name=parameter_name, source=source
            )[0]
            if hasattr(copied, "__setstate__"):
                copied.__setstate__(copied_state)
            else:
                dict_state: Any
                slot_state: Any
                if isinstance(copied_state, tuple) and len(copied_state) == 2:
                    dict_state, slot_state = copied_state
                else:
                    dict_state, slot_state = copied_state, None
                if dict_state is not None:
                    copied.__dict__.update(dict_state)
                if slot_state is not None:
                    for key, item in slot_state.items():
                        setattr(copied, key, item)

        if listitems is not None:
            for item in listitems:
                copied.append(
                    self.prepare(item, parameter_name=parameter_name, source=source)[0]
                )
        if dictitems is not None:
            for key, item in dictitems:
                copied[
                    self.prepare(key, parameter_name=parameter_name, source=source)[0]
                ] = self.prepare(item, parameter_name=parameter_name, source=source)[0]
        return copied

    def _copy_object_ndarray(
        self,
        value: Any,
        parameter_name: str,
        source: ResolutionSource,
    ) -> Any:
        """Copy an ndarray whose dtype holds Python objects at any depth.

        The shell is copied and memoized first so an object leaf that
        references the containing array still resolves to the copy, then every
        Python-object leaf, including fields of structured dtypes and subarray
        fields, is replaced through :meth:`prepare`.
        """
        copied = value.copy()
        self._memo[id(value)] = copied
        self._isolate_ndarray_objects(copied, value, parameter_name, source)
        return copied

    def _isolate_ndarray_objects(
        self,
        copied: Any,
        original: Any,
        parameter_name: str,
        source: ResolutionSource,
    ) -> None:
        import numpy as np  # type: ignore

        if original.dtype == object:
            for index in np.ndindex(original.shape):
                copied[index] = self.prepare(
                    original[index], parameter_name=parameter_name, source=source
                )[0]
            return
        for name in original.dtype.names or ():
            field_dtype = original.dtype.fields[name][0]
            if not field_dtype.hasobject:
                continue
            self._isolate_ndarray_objects(
                copied[name],
                original[name],
                parameter_name,
                source,
            )

    @staticmethod
    def _copy_error(
        parameter_name: str,
        source: ResolutionSource,
        value: Any,
        reason: str,
    ) -> InspectionCopyError:
        source_label = source.kind
        if source.name is not None:
            source_label += f" '{source.name}'"
        return InspectionCopyError(
            f"Cannot isolate parameter '{parameter_name}' resolved from "
            f"{source_label} (type={type(value).__name__}): {reason}. "
            "Use allow_mutable_objects=True to permit the original object, or "
            f"provide a temporary value through overrides={{'{parameter_name}': ...}}."
        )


@dataclass(slots=True)
class InspectionCopyCandidate:
    """One resolved value that a protected inspection call needs to copy."""

    parameter_name: str
    value: Any
    expected_container_kind: str | None = None


def _format_bytes(size: float) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def _format_signed_bytes(size: int) -> str:
    sign = "+" if size >= 0 else "-"
    return f"{sign}{_format_bytes(abs(size))}"


def _process_rss_bytes() -> int | None:
    rss = _read_smaps_rollup_rss()
    if rss is not None:
        return rss
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _release_native_allocators() -> None:
    """Best-effort release of PyArrow's pooled buffers.

    PyArrow keeps freed buffers in its own memory pool (jemalloc by default),
    which ``malloc_trim`` cannot reach, so RSS can stay flat on load and fail
    to drop on unload. Failures are ignored because this only improves
    reporting.
    """
    pyarrow = sys.modules.get("pyarrow")
    if pyarrow is None:
        return
    default_memory_pool = getattr(pyarrow, "default_memory_pool", None)
    if default_memory_pool is None:
        return
    try:
        pool = default_memory_pool()
        release = getattr(pool, "release_unused", None)
        if release is not None:
            release()
    except Exception:
        pass


def _read_smaps_rollup_rss(
    path: str | Path = "/proc/self/smaps_rollup",
) -> int | None:
    """Read synchronous Linux RSS accounting, falling back elsewhere.

    ``psutil.Process().memory_info().rss`` uses Linux ``statm``, whose RSS
    counters can lag large multithreaded allocations. ``smaps_rollup`` walks
    current mappings and provides a reliable point-in-time baseline without a
    sleep or duplicate computation.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("Rss:"):
                    continue
                fields_text = line.split()
                if len(fields_text) < 2:
                    return None
                return int(fields_text[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _path_logical_bytes(path: Path) -> tuple[int, bool]:
    try:
        if not path.is_dir():
            return path.stat().st_size, False
    except OSError:
        return 0, False

    total = 0
    visited = 0
    pending = [path]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    visited += 1
                    if visited > _PATH_ENTRY_LIMIT:
                        return total, True
                    try:
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                        elif entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                    except OSError:
                        continue
        except OSError:
            continue
    return total, False


def _parquet_directory_file_count(path: Path) -> tuple[int, bool]:
    count = 0
    visited = 0
    pending = [path]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    visited += 1
                    if visited > _PATH_ENTRY_LIMIT:
                        return count, True
                    try:
                        if entry.is_file(follow_symlinks=False):
                            if Path(entry.name).suffix == ".parquet":
                                count += 1
                        elif entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                    except OSError:
                        continue
        except OSError:
            continue
    return count, False


def _slot_names(value_type: type) -> tuple[str, ...]:
    names: list[str] = []
    for cls in value_type.__mro__:
        slots = cls.__dict__.get("__slots__")
        if slots is None:
            continue
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if not isinstance(name, str) or name in ("__dict__", "__weakref__"):
                continue
            # Private slot names are stored name-mangled on the instance, so
            # the declaring class must mangle them the same way to read them.
            if name.startswith("__") and not name.endswith("__"):
                name = f"_{cls.__name__.lstrip('_')}{name}"
            if name not in names:
                names.append(name)
    return tuple(names)


class _InspectionMemoryEstimator:
    """Best-effort bounded estimator for protected-copy allocations.

    The estimator intentionally avoids walking complete Python object graphs or
    object-dtype tabular values. It uses cheap storage metadata and bounded
    samples, recording material uncertainty for the confirmation path. The
    opt-in retained mode counts immutable storage and deduplicates exact array
    and tensor storage windows for inspection-cache reporting; protected-copy
    preflight keeps the default copy-allocation semantics.
    """

    def __init__(
        self,
        pointer_reader: Callable[[OutputPointer], Any] | None = None,
        *,
        retained: bool = False,
    ) -> None:
        self._pointer_reader = pointer_reader
        self._retained = retained
        self._copy_seen: set[int] = set()
        self._retained_storage_seen: set[tuple[str, int, int]] = set()
        self._retained_materialization_bytes = 0
        self._peak_transient_materialization_bytes = 0
        self._copy_bytes = 0
        self._temporary_bytes = 0
        self._cuda_bytes: dict[int, int] = {}
        self._uncertain: list[tuple[str, str]] = []

    @property
    def total_bytes(self) -> int:
        return (
            self.materialization_bytes
            + self._copy_bytes
            + self._temporary_bytes
        )

    @property
    def materialization_bytes(self) -> int:
        return (
            self._retained_materialization_bytes
            + self._peak_transient_materialization_bytes
        )

    @property
    def copy_bytes(self) -> int:
        return self._copy_bytes

    @property
    def temporary_bytes(self) -> int:
        return self._temporary_bytes

    @property
    def cuda_bytes(self) -> dict[int, int]:
        return dict(self._cuda_bytes)

    @property
    def uncertain(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._uncertain)

    def add(
        self,
        label: str,
        value: Any,
        *,
        materialize: bool = False,
        expected_container_kind: str | None = None,
    ) -> None:
        if materialize:
            self._add_materialized(
                label,
                value,
                expected_container_kind=expected_container_kind,
            )
            return
        self._copy_bytes += self._estimate_copy(value, label)

    def _mark_uncertain(self, label: str, reason: str) -> None:
        entry = (label, reason)
        if entry not in self._uncertain:
            self._uncertain.append(entry)

    def _add_materialized(
        self,
        label: str,
        value: Any,
        *,
        expected_container_kind: str | None,
    ) -> None:
        terminal = value
        if isinstance(value, OutputPointer):
            if self._pointer_reader is None:
                self._mark_uncertain(
                    label,
                    "output pointer cannot be resolved for estimation",
                )
                return
            try:
                terminal = self._pointer_reader(value)
            except PointerResolutionError as exc:
                self._mark_uncertain(
                    label,
                    f"output pointer is unresolvable: {exc}",
                )
                return
            except MemoryError:
                raise
            except Exception as exc:
                self._mark_uncertain(
                    label,
                    f"output pointer could not be assessed: {type(exc).__name__}: {exc}",
                )
                return

        if isinstance(terminal, ArtifactRecord):
            self._assess_artifact_container_type(
                terminal,
                label,
                expected_container_kind,
            )
            size = self._estimate_artifact_record(terminal, label)
            if terminal.serializer in _RETAINED_ARTIFACT_SERIALIZERS:
                self._retained_materialization_bytes += size
            else:
                self._peak_transient_materialization_bytes = max(
                    self._peak_transient_materialization_bytes,
                    size,
                )
            self._copy_bytes += size
            self._temporary_bytes = max(
                self._temporary_bytes,
                math.ceil(size * _TEMPORARY_ALLOCATION_FRACTION),
            )
            return

        self._copy_bytes += self._estimate_copy(terminal, label)

    def _assess_artifact_container_type(
        self,
        record: ArtifactRecord,
        label: str,
        expected_container_kind: str | None,
    ) -> None:
        if expected_container_kind is None:
            return
        python_type = record.metadata.get("python_type")
        expected_types = {
            "sequence": {"builtins.list", "builtins.tuple"},
            "mapping": {"builtins.dict"},
        }[expected_container_kind]
        if python_type not in expected_types:
            detail = (
                "the artifact has no saved Python container type"
                if python_type is None
                else f"the artifact records Python type '{python_type}'"
            )
            self._mark_uncertain(
                label,
                f"{detail}; protected variadic planning expected a {expected_container_kind}",
            )

    def _estimate_copy(self, value: Any, label: str, *, depth: int = 0) -> int:
        identity = id(value)
        if identity in self._copy_seen:
            return 0
        self._copy_seen.add(identity)
        try:
            return self._estimate_value(value, label, depth=depth)
        except InspectionMemoryError:
            raise
        except MemoryError:
            raise
        except Exception as exc:
            self._mark_uncertain(label, f"{type(exc).__name__}: {exc}")
            return 0

    def _estimate_value(self, value: Any, label: str, *, depth: int) -> int:
        if isinstance(value, ArtifactRecord):
            return self._estimate_nested_artifact_record(value, label, depth)
        if isinstance(value, OutputPointer):
            self._mark_uncertain(
                label,
                "nested output pointer cannot be assessed without materialising its container",
            )
            return sys.getsizeof(value)
        if type(value) in _IMMUTABLE_TYPES:
            return sys.getsizeof(value) if self._retained else 0
        if _bound_callable_owner(value) is not None:
            return self._estimate_bound_method(value, label, depth)
        if isinstance(
            value,
            (FunctionType, BuiltinFunctionType, type, types.ModuleType),
        ):
            return 0

        pd = cast(Any, sys.modules.get("pandas"))
        if pd is not None:
            if isinstance(value, pd.DataFrame):
                return self._estimate_pandas_frame(value, label)
            if isinstance(value, pd.Series):
                return self._estimate_pandas_series(value, label)

        np = cast(Any, sys.modules.get("numpy"))
        if np is not None and isinstance(value, np.ndarray):
            return self._estimate_numpy_array(value, label)

        torch = cast(Any, sys.modules.get("torch"))
        if torch is not None:
            nn = cast(Any, sys.modules.get("torch.nn"))
            if nn is not None and isinstance(value, nn.Module):
                return self._estimate_torch_module(value, label)
            if isinstance(value, torch.Tensor):
                return self._estimate_torch_tensor(value)

        dd = cast(Any, sys.modules.get("dask.dataframe"))
        if dd is not None and isinstance(value, (dd.DataFrame, dd.Series)):
            return self._estimate_dask_collection(value)

        return self._estimate_generic(value, label, depth)

    def _estimate_artifact_record(self, record: ArtifactRecord, label: str) -> int:
        serializer = record.serializer
        if (
            serializer == OPTUNA_STUDY_SERIALIZER
            or record.metadata.get("optuna_type") == "sampler"
        ):
            raise InspectionCopyError(
                "This inspection input contains Optuna state (a study or sampler) whose "
                "shared state cannot be isolated by protected copying. To inspect "
                "it using its original state, call "
                "inspect(..., allow_mutable_objects=True)."
            )
        factor = _ARTIFACT_MEMORY_FACTORS.get(serializer)
        if factor is None:
            self._mark_uncertain(
                label,
                f"artifact serializer '{serializer}' has no lightweight memory estimator",
            )
            return 0
        path = Path(record.file_path)
        if serializer == "parquet" and path.is_dir():
            size = self._estimate_parquet_directory(path, label)
            self._mark_uncertain(
                label,
                "a parquet directory is expected to load lazily, but its loader may fall back to eager pandas loading",
            )
        else:
            size, truncated = _path_logical_bytes(path)
            if truncated:
                self._mark_uncertain(
                    label,
                    f"artifact directory traversal exceeded {_PATH_ENTRY_LIMIT} entries",
                )
        return int(size * factor)

    def _estimate_nested_artifact_record(
        self,
        record: ArtifactRecord,
        label: str,
        depth: int,
    ) -> int:
        size = sys.getsizeof(record)
        if depth >= _SHALLOW_DEPTH_LIMIT:
            return size
        size += self._estimate_copy(record.metadata, label, depth=depth + 1)
        return size

    def _estimate_parquet_directory(self, path: Path, label: str) -> int:
        try:
            import dask.dataframe  # type: ignore  # noqa: F401
        except ImportError:
            size, truncated = _path_logical_bytes(path)
            if truncated:
                self._mark_uncertain(
                    label,
                    f"parquet directory traversal exceeded {_PATH_ENTRY_LIMIT} entries",
                )
            return size
        file_count, truncated = _parquet_directory_file_count(path)
        if truncated:
            self._mark_uncertain(
                label,
                f"parquet directory traversal exceeded {_PATH_ENTRY_LIMIT} entries",
            )
        return 256 * 1024 + file_count * 16 * 1024

    def _estimate_pandas_frame(self, frame: Any, label: str) -> int:
        total = int(frame.memory_usage(index=True, deep=False).sum())
        object_columns = [
            index for index, dtype in enumerate(frame.dtypes) if dtype == object
        ]
        if object_columns and len(frame.index):
            sample_budget = _SHALLOW_SAMPLE_LIMIT
            sampled_sizes: list[int] = []
            for column_index in object_columns:
                remaining_columns = max(1, len(object_columns) - len(sampled_sizes))
                column_budget = max(1, sample_budget // remaining_columns)
                for row_index in range(min(len(frame.index), column_budget)):
                    sampled_sizes.append(
                        self._estimate_copy(
                            frame.iat[row_index, column_index],
                            label,
                            depth=1,
                        )
                    )
                    sample_budget -= 1
                    if sample_budget <= 0:
                        break
                if sample_budget <= 0:
                    break
            if sampled_sizes:
                total += sum(sampled_sizes)
            if len(frame.index) * len(object_columns) > len(sampled_sizes):
                self._mark_uncertain(
                    label,
                    "object-dtype DataFrame memory is estimated from a bounded sample rather than every cell",
                )
        return total

    def _estimate_pandas_series(self, series: Any, label: str) -> int:
        total = int(series.memory_usage(index=True, deep=False))
        if series.dtype == object and len(series.index):
            sample_count = min(len(series.index), _SHALLOW_SAMPLE_LIMIT)
            sampled = [
                self._estimate_copy(series.iat[index], label, depth=1)
                for index in range(sample_count)
            ]
            total += sum(sampled)
            if len(series.index) > sample_count:
                self._mark_uncertain(
                    label,
                    "object-dtype Series memory is estimated from a bounded sample rather than every value",
                )
        return total

    def _estimate_numpy_array(self, array: Any, label: str) -> int:
        total = int(array.nbytes)
        if self._retained:
            try:
                data_pointer = int(array.__array_interface__["data"][0])
                storage_key = ("numpy", data_pointer, total)
                if storage_key in self._retained_storage_seen:
                    return 0
                self._retained_storage_seen.add(storage_key)
            except Exception:
                self._mark_uncertain(
                    label,
                    "NumPy backing storage identity could not be determined",
                )
        if array.dtype.hasobject and array.size:
            sample_count = min(int(array.size), _SHALLOW_SAMPLE_LIMIT)
            iterator = iter(array.flat)
            sampled = [
                self._estimate_copy(next(iterator), label, depth=1)
                for _ in range(sample_count)
            ]
            total += sum(sampled)
            if int(array.size) > sample_count:
                self._mark_uncertain(
                    label,
                    "object-dtype NumPy memory is estimated from a bounded sample rather than every element",
                )
        return total

    def _estimate_torch_tensor(self, tensor: Any) -> int:
        try:
            size = int(tensor.element_size() * tensor.nelement())
        except Exception:
            size = int(getattr(tensor, "nbytes", 0))
        if self._retained:
            try:
                storage_key = ("torch", int(tensor.data_ptr()), size)
                if storage_key in self._retained_storage_seen:
                    return 0
                self._retained_storage_seen.add(storage_key)
            except Exception:
                pass
        if bool(tensor.is_cuda):
            index = tensor.device.index
            device = 0 if index is None else int(index)
            self._cuda_bytes[device] = self._cuda_bytes.get(device, 0) + size
            return 0
        return size

    def _estimate_torch_module(self, module: Any, label: str) -> int:
        total = sys.getsizeof(module)
        for parameter in module.parameters():
            total += self._estimate_copy(parameter, label, depth=1)
        for buffer in module.buffers():
            total += self._estimate_copy(buffer, label, depth=1)
        return total

    def _estimate_dask_collection(self, collection: Any) -> int:
        try:
            partition_count = int(collection.npartitions)
        except MemoryError:
            raise
        except Exception:
            partition_count = 512
        task_count = max(1, partition_count) * 8
        return 256 * 1024 + task_count * 512

    def _estimate_bound_method(
        self,
        method: Any,
        label: str,
        depth: int,
    ) -> int:
        size = sys.getsizeof(method)
        instance = method.__self__
        if instance is None or isinstance(instance, type):
            return size
        if depth >= _SHALLOW_DEPTH_LIMIT:
            self._mark_uncertain(
                label,
                "bound method owner could not be assessed within the bounded inspection depth",
            )
            return size
        return size + self._estimate_copy(instance, label, depth=depth + 1)

    def _estimate_generic(self, value: Any, label: str, depth: int) -> int:
        size = sys.getsizeof(value)
        if isinstance(value, Enum):
            return 0
        if isinstance(value, bytearray):
            return size
        if depth >= _SHALLOW_DEPTH_LIMIT:
            self._mark_uncertain(
                label,
                f"the object graph for type '{type(value).__name__}' exceeds the bounded inspection depth",
            )
            return size
        if isinstance(value, dict):
            sampled = list(islice(value.items(), _SHALLOW_SAMPLE_LIMIT))
            for key, item in sampled:
                size += self._estimate_copy(key, label, depth=depth + 1)
                size += self._estimate_copy(item, label, depth=depth + 1)
            if len(value) > len(sampled):
                self._mark_uncertain(
                    label,
                    f"dictionary contents were sampled ({len(sampled)} of {len(value)} entries)",
                )
            if type(value) is dict:
                return size
        if isinstance(value, (list, tuple, set, frozenset)):
            sampled = list(islice(value, _SHALLOW_SAMPLE_LIMIT))
            for item in sampled:
                size += self._estimate_copy(item, label, depth=depth + 1)
            if len(value) > len(sampled):
                self._mark_uncertain(
                    label,
                    f"container contents were sampled ({len(sampled)} of {len(value)} items)",
                )
            if type(value) in (list, tuple, set, frozenset):
                return size

        attributes: list[Any] = []
        attribute_count = 0
        if is_dataclass(value) and not isinstance(value, type):
            field_infos = fields(value)
            attribute_count += len(field_infos)
            for field_info in islice(field_infos, _SHALLOW_SAMPLE_LIMIT):
                if hasattr(value, field_info.name):
                    attributes.append(getattr(value, field_info.name))
        elif hasattr(value, "__dict__"):
            try:
                value_attributes = vars(value)
                attribute_count += len(value_attributes)
                attributes.extend(
                    islice(value_attributes.values(), _SHALLOW_SAMPLE_LIMIT)
                )
            except TypeError:
                pass
        for slot_name in _slot_names(type(value)):
            attribute_count += 1
            if len(attributes) >= _SHALLOW_SAMPLE_LIMIT:
                continue
            try:
                item = getattr(value, slot_name)
            except AttributeError:
                continue
            attributes.append(item)
        sampled_attributes = attributes[:_SHALLOW_SAMPLE_LIMIT]
        for item in sampled_attributes:
            size += self._estimate_copy(item, label, depth=depth + 1)
        if attribute_count > len(sampled_attributes):
            self._mark_uncertain(
                label,
                f"attributes of type '{type(value).__name__}' were sampled "
                f"({len(sampled_attributes)} of {attribute_count})",
            )
        elif (
            not attributes
            and not isinstance(value, (dict, list, tuple, set, frozenset))
            and size >= _MATERIAL_MEMORY_BYTES
        ):
            self._mark_uncertain(
                label,
                f"opaque object of type '{type(value).__name__}' cannot be estimated cheaply",
            )
        return size

def _read_integer_metric(path: str) -> int | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read().strip()
    except OSError:
        return None
    if text in {"", "max"}:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _system_available_memory() -> tuple[int | None, str]:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available), "system RAM (psutil)"
    except Exception:
        pass
    for line in _read_meminfo_lines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1]) * 1024, "system RAM (/proc/meminfo)"
                except ValueError:
                    return None, ""
    return None, ""


def _read_meminfo_lines() -> list[str]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            return handle.read().splitlines()
    except OSError:
        return []


def _cgroup_memory_headroom() -> tuple[int | None, str]:
    maximum = _read_integer_metric("/sys/fs/cgroup/memory.max")
    current = _read_integer_metric("/sys/fs/cgroup/memory.current")
    if maximum is not None and current is not None and maximum < (1 << 62):
        return max(0, maximum - current), "container memory limit (cgroup v2)"
    maximum = _read_integer_metric(
        "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    )
    current = _read_integer_metric(
        "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    )
    if maximum is not None and current is not None and maximum < (1 << 62):
        return max(0, maximum - current), "container memory limit (cgroup v1)"
    return None, ""


def _process_address_space_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().vms)
    except Exception:
        pass
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            fields_text = handle.read().split()
        return int(fields_text[0]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None


def _rlimit_memory_headroom() -> tuple[int | None, str]:
    try:
        import resource
    except ImportError:
        return None, ""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    except (AttributeError, ValueError, OSError):
        return None, ""
    infinity = getattr(resource, "RLIM_INFINITY", -1)
    limit = soft if soft != infinity else hard
    if limit == infinity or limit < 0:
        return None, ""
    if limit == 0:
        return 0, "process address-space limit (RLIMIT_AS)"
    usage = _process_address_space_bytes()
    if usage is None:
        return None, ""
    return max(0, int(limit) - usage), "process address-space limit (RLIMIT_AS)"


def _available_memory_bytes() -> tuple[int | None, str, tuple[str, ...]]:
    candidates: list[tuple[int, str]] = []
    system_bytes, system_label = _system_available_memory()
    if system_bytes is not None:
        candidates.append((int(system_bytes), system_label))
    cgroup_bytes, cgroup_label = _cgroup_memory_headroom()
    if cgroup_bytes is not None:
        candidates.append((int(cgroup_bytes), cgroup_label))
    rlimit_bytes, rlimit_label = _rlimit_memory_headroom()
    if rlimit_bytes is not None:
        candidates.append((int(rlimit_bytes), rlimit_label))
    if not candidates:
        return None, "", ()
    limit, label = min(candidates, key=lambda item: item[0])
    details = tuple(
        f"{_format_bytes(value)} via {source}" for value, source in candidates
    )
    return limit, label, details


def _available_cuda_bytes(device_index: int) -> tuple[int | None, str]:
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return None, ""
        free, total = torch.cuda.mem_get_info(device_index)
        return (
            int(free),
            f"CUDA device {device_index} VRAM "
            f"({_format_bytes(free)} free of {_format_bytes(total)})",
        )
    except Exception:
        return None, ""


def _requires_materialization(value: Any) -> bool:
    return isinstance(value, (ArtifactRecord, OutputPointer))


def _inspection_memory_error(
    *,
    node_name: str,
    function_name: str,
    inputs: str,
    reason: str,
    extra_lines: tuple[str, ...] = (),
) -> InspectionMemoryError:
    lines = [
        f"Protected inspection of '{node_name}' was rejected before copying its "
        f"inputs for function '{function_name}': {reason}.",
        f"Inputs requiring protected copies: {inputs}",
        "No protected copy was started and pipeline state is unchanged.",
    ]
    lines.extend(extra_lines)
    lines.append(
        "You may free memory, reduce the inspection inputs, or explicitly use "
        "allow_mutable_objects=True to inspect the original objects without "
        "copying them."
    )
    lines.append(
        "This lightweight estimate and the available-memory measurements are "
        "approximate. Continuing can exhaust memory, terminate the Python or "
        "Jupyter process, and lose unsaved in-memory work."
    )
    return InspectionMemoryError(
        "\n".join(lines),
        node_name=node_name,
        function_name=function_name,
        inputs=inputs,
        reason=reason,
        copy_started=False,
    )


def inspection_memory_failure(
    *,
    node_name: str,
    function_name: str,
    parameter_name: str | None,
    value: Any = None,
    stage: str = "copying",
    copy_started: bool = True,
) -> InspectionMemoryError:
    """Build a stage-aware error for a ``MemoryError`` during inspection."""
    estimate_line = ""
    if value is not None:
        try:
            estimator = _InspectionMemoryEstimator()
            estimator.add(parameter_name or "input", value)
            estimate_line = (
                f"Estimated size of the affected input: "
                f"{_format_bytes(estimator.total_bytes)}"
            )
        except Exception:
            estimate_line = ""
    input_text = "" if parameter_name is None else f" input '{parameter_name}'"
    lines = [
        f"Inspection of '{node_name}' failed while {stage}{input_text} for "
        f"function '{function_name}': the process ran out of memory.",
    ]
    if copy_started:
        lines.append(
            "A protected copy had already started; inspection does not commit "
            "pipeline outputs, but memory pressure may still affect the process."
        )
    else:
        lines.append(
            "No protected copy had started when the allocation failed and no "
            "inspection result was committed. Shared mutable inputs may still "
            "have been changed if protected copying was disabled."
        )
    if estimate_line:
        lines.append(estimate_line)
    lines.append(
        "Free memory and retry, reduce the inspection inputs, or explicitly use "
        "allow_mutable_objects=True to inspect the original objects without "
        "copying them."
    )
    return InspectionMemoryError(
        "\n".join(lines),
        node_name=node_name,
        function_name=function_name,
        inputs=parameter_name,
        reason=f"MemoryError raised while {stage}",
        copy_started=copy_started,
    )


def _interactive_stdin_available() -> bool:
    """Whether a confirmation prompt can reasonably reach the user."""
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            return True
    except Exception:
        pass
    try:
        from importlib import import_module

        get_ipython = getattr(import_module("IPython"), "get_ipython", None)
        shell = None if get_ipython is None else get_ipython()
    except Exception:
        return False
    return shell is not None and getattr(shell, "kernel", None) is not None


def _confirm_unsafe_copy(error: InspectionMemoryError) -> bool:
    """Ask the user whether to proceed; returns True only for explicit consent.

    Non-interactive sessions, end-of-input, and unusable stdin all decline so
    the original memory error is raised instead.
    """
    if not _interactive_stdin_available():
        return False
    prompt = (
        f"\n{error}\n\n"
        "Proceed with protected copying anyway? The copies may exhaust available "
        "memory. [y/N]: "
    )
    try:
        answer = input(prompt)
    except (EOFError, OSError):
        return False
    return answer.strip().lower() in {"y", "yes"}


def _apply_memory_safety_margin(size: int, safety_margin: float) -> int:
    try:
        adjusted = size * (1.0 + safety_margin)
    except OverflowError as exc:
        raise TypeError("memory_safety_margin is too large") from exc
    if not math.isfinite(adjusted):
        raise TypeError("memory_safety_margin is too large")
    return math.ceil(adjusted)


def preflight_protected_inspection(
    candidates: Sequence[InspectionCopyCandidate],
    *,
    node_name: str,
    function_name: str,
    safety_margin: float,
    pointer_reader: Callable[[OutputPointer], Any] | None = None,
) -> None:
    """Guard a protected inspection call when copying is estimated to be unsafe.

    This is a best-effort preflight, not a guarantee against out-of-memory
    errors. It runs before any protected copy or artifact materialization.
    When a check fails, interactive sessions are asked to confirm before the
    copy starts; declining, or running non-interactively, raises
    :class:`InspectionMemoryError`.
    """
    if not candidates:
        return
    estimator = _InspectionMemoryEstimator(pointer_reader)
    labels: list[str] = []
    for candidate in candidates:
        labels.append(candidate.parameter_name)
        try:
            estimator.add(
                candidate.parameter_name,
                candidate.value,
                materialize=_requires_materialization(candidate.value),
                expected_container_kind=candidate.expected_container_kind,
            )
        except MemoryError as exc:
            raise inspection_memory_failure(
                node_name=node_name,
                function_name=function_name,
                parameter_name=candidate.parameter_name,
                value=candidate.value,
                stage="estimating protected-copy memory",
                copy_started=False,
            ) from exc
    inputs = ", ".join(dict.fromkeys(labels))
    if estimator.uncertain:
        reasons = "; ".join(
            f"{name}: {reason}" for name, reason in estimator.uncertain
        )
        error = _inspection_memory_error(
            node_name=node_name,
            function_name=function_name,
            inputs=inputs,
            reason=(
                "the memory requirement could not be established reliably "
                f"({reasons})"
            ),
            extra_lines=(
                "A materially unbounded or unreliable estimate cannot be approved "
                "for protected copying.",
            ),
        )
        if _confirm_unsafe_copy(error):
            return
        raise error
    required_bytes = estimator.total_bytes
    if required_bytes > 0:
        required_bytes = _apply_memory_safety_margin(
            required_bytes,
            safety_margin,
        )
    cuda_required = {
        device: _apply_memory_safety_margin(size, safety_margin)
        for device, size in estimator.cuda_bytes.items()
        if size > 0
    }
    if required_bytes > 0:
        available, resource, details = _available_memory_bytes()
        if available is None:
            error = _inspection_memory_error(
                node_name=node_name,
                function_name=function_name,
                inputs=inputs,
                reason="available memory could not be determined reliably",
                extra_lines=(
                    "No usable system, container, or process memory measurement "
                    "is available.",
                ),
            )
            if _confirm_unsafe_copy(error):
                return
            raise error
        if required_bytes > available:
            error = _inspection_memory_error(
                node_name=node_name,
                function_name=function_name,
                inputs=inputs,
                reason=(
                    f"estimated additional memory "
                    f"{_format_bytes(required_bytes)} exceeds available memory "
                    f"{_format_bytes(available)} ({resource})"
                ),
                extra_lines=tuple(
                    f"memory check: {detail}" for detail in details
                ),
            )
            if _confirm_unsafe_copy(error):
                return
            raise error
    for device, required_cuda in cuda_required.items():
        free, cuda_resource = _available_cuda_bytes(device)
        if free is None:
            error = _inspection_memory_error(
                node_name=node_name,
                function_name=function_name,
                inputs=inputs,
                reason=(
                    f"available CUDA memory for device {device} could not be "
                    "determined reliably"
                ),
                extra_lines=(
                    f"Estimated additional CUDA memory: "
                    f"{_format_bytes(required_cuda)}",
                ),
            )
            if _confirm_unsafe_copy(error):
                return
            raise error
        if required_cuda > free:
            error = _inspection_memory_error(
                node_name=node_name,
                function_name=function_name,
                inputs=inputs,
                reason=(
                    f"estimated additional CUDA memory "
                    f"{_format_bytes(required_cuda)} exceeds available device "
                    f"memory {_format_bytes(free)} ({cuda_resource})"
                ),
            )
            if _confirm_unsafe_copy(error):
                return
            raise error


def _compute_investigation_value(value: Any, *, compute: bool) -> Any:
    if not compute:
        return value
    dd = cast(Any, sys.modules.get("dask.dataframe"))
    if dd is None:
        return value
    if isinstance(value, (dd.DataFrame, dd.Series)):
        return value.compute()
    return value


class PipelineInspection:
    """One-shot context holding values resolved for temporary investigation."""

    __slots__ = (
        "_active",
        "_compute",
        "_names",
        "_pipeline",
        "_priority",
        "_values",
    )

    def __init__(
        self,
        pipeline: Any,
        names: tuple[str, ...],
        *,
        compute: bool,
        priority: int | None,
    ) -> None:
        self._pipeline: Any | None = pipeline
        self._names = names
        self._compute = compute
        self._priority = priority
        self._values: dict[str, Any] = {}
        self._active = False

    def __enter__(self) -> Self:
        if self._pipeline is None:
            raise RuntimeError("Pipeline inspection contexts cannot be reused")
        if self._active:
            raise RuntimeError("Pipeline inspection context is already active")
        resolved: dict[str, Any] = {}
        try:
            visible_outputs: dict[str, Any] | None = None
            if self._priority is not None:
                selected = self._pipeline._select_node_at_or_below_priority(
                    self._priority
                )
                self._pipeline._log_selected_node(
                    selected,
                    operation="inspect",
                    requested_priority=self._priority,
                )
                visible_outputs = self._pipeline._visible_outputs_before_priority(
                    selected.execution_priority
                )
            for name in self._names:
                resolved[name] = _compute_investigation_value(
                    self._pipeline._resolve_investigation_input(
                        name,
                        "inspect",
                        visible_outputs=visible_outputs,
                        requested_priority=self._priority,
                    ),
                    compute=self._compute,
                )
        except BaseException:
            resolved.clear()
            self._pipeline = None
            raise
        self._values = resolved
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc_value, traceback
        self._values.clear()
        self._pipeline = None
        self._active = False
        return False

    def __getitem__(self, name: str) -> Any:
        return self._active_values()[name]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        values = self._active_values()
        try:
            return values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def _active_values(self) -> dict[str, Any]:
        if not self._active:
            raise RuntimeError("Pipeline inspection values are available only inside the with block")
        return self._values


class InspectionMixin:
    """Create temporary multi-value inspection contexts."""

    if TYPE_CHECKING:
        logger: Any = None
        project_root: Any = None
        _inspection_cache_bindings: dict[str, _InspectionCacheBinding] = {}
        _inspection_cache_entries: dict[_InspectionCacheKey, _InspectionCacheEntry] = {}
        _inspection_cache_lock: Any = None
        _inspection_copy_skipped: dict[str, tuple[int, Any | None]] = {}

        @staticmethod
        def _validate_integer_priority(priority: Any) -> int: ...
        def _resolve_investigation_copy_target(
            self, input_name: str, *, priority: int | None = None
        ) -> tuple[Any, Any, ResolutionSource]: ...
        def _iter_attached_pipelines(self) -> list[Any]: ...
        def _attempt_allocator_trim(self) -> None: ...

    def inspect(
        self,
        *names: str,
        compute: bool = False,
        priority: int | None = None,
    ) -> PipelineInspection:
        """Resolve several pipeline values for the lifetime of one ``with`` block."""
        if not names:
            raise ValueError("inspect() requires at least one variable name")
        invalid_names = [
            name for name in names if not isinstance(name, str) or not name.strip()
        ]
        if invalid_names:
            raise ValueError("inspect() variable names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("inspect() variable names must be unique")
        if not isinstance(compute, bool):
            raise TypeError("compute must be a boolean")
        if priority is not None:
            self._validate_integer_priority(priority)
        return PipelineInspection(self, names, compute=compute, priority=priority)

    def copy_for_inspection(
        self,
        object_names: list[str] | tuple[str, ...],
        *,
        compute: bool = True,
        priority: int | None = None,
    ) -> None:
        """Copy selected investigation values into memory for later inspection.

        Disk-backed values are loaded and cached; in-memory values are isolated
        with ``deepcopy`` and, when needed, a temporary serializer round trip.
        """
        names = self._validate_inspection_object_names(object_names)
        if not isinstance(compute, bool):
            raise TypeError("compute must be a boolean")
        if priority is not None:
            self._validate_integer_priority(priority)
        self._copy_inspection_objects(
            {name: (priority, compute) for name in names},
            operation="copy",
        )
        self.logger.info(
            "Copied inspection objects are shared and may become stale after pipeline "
            "or block execution. Call unload_inspection_objects() between executions, "
            "or refresh_copied_objects() to reload every copied object."
        )

    def load_for_inspection(
        self,
        object_names: list[str] | tuple[str, ...],
        *,
        compute: bool = True,
        priority: int | None = None,
    ) -> None:
        """Backward-compatible alias of :meth:`copy_for_inspection`."""
        self.copy_for_inspection(
            object_names,
            compute=compute,
            priority=priority,
        )

    def refresh_copied_objects(self) -> None:
        """Recopy every currently copied inspection object from current state."""
        with self._inspection_cache_lock:
            requests = {
                name: (binding.priority, binding.compute)
                for name, binding in self._inspection_cache_bindings.items()
            }
        if not requests:
            warnings.warn(
                "No inspection objects are currently loaded; nothing was refreshed",
                UserWarning,
                stacklevel=2,
            )
            return
        self._copy_inspection_objects(requests, operation="refresh")

    def refresh_loaded_objects(self) -> None:
        """Backward-compatible alias of :meth:`refresh_copied_objects`."""
        self.refresh_copied_objects()

    def _inspection_cache_has_entries(self) -> bool:
        with self._inspection_cache_lock:
            return bool(self._inspection_cache_bindings)

    def unload_inspection_objects(self) -> None:
        """Release this pipeline's references to all preloaded inspection objects."""
        with self._inspection_cache_lock:
            if not self._inspection_cache_bindings:
                warnings.warn(
                    "No inspection objects are currently loaded; nothing was unloaded",
                    UserWarning,
                    stacklevel=2,
                )
                return
            values = tuple(
                entry.value for entry in self._inspection_cache_entries.values()
            )
        estimate = self._format_inspection_object_estimate(
            values,
            label="released cached size",
        )
        del values
        gc.collect()
        self._attempt_allocator_trim()
        _release_native_allocators()
        before = _process_rss_bytes()
        self._unload_from_memory()
        gc.collect()
        self._attempt_allocator_trim()
        _release_native_allocators()
        after = _process_rss_bytes()
        try:
            self._log_inspection_memory_delta(
                "unload",
                None if before is None or after is None else after - before,
                estimate,
                phase="after cleanup",
            )
        except Exception:
            pass

    def _inspection_cached_value(
        self,
        input_name: str,
        *,
        artifact: ArtifactRecord | None,
        original: Any = None,
    ) -> Any:
        """Return ``(value, binding)`` for a cache hit, otherwise ``_MISSING``."""
        with self._inspection_cache_lock:
            binding = self._inspection_cache_bindings.get(input_name)
            if binding is None:
                return _MISSING
            if artifact is not None:
                if binding.kind != "artifact":
                    # An in-memory original may be replaced by a disk-backed
                    # generation. Release its strong identity guard on this
                    # miss just as we do for memory-to-memory replacements.
                    binding.original_guard = None
                    return _MISSING
                if binding.cache_key[0] != self._inspection_artifact_identity(artifact):
                    return _MISSING
            else:
                if binding.kind != "memory":
                    return _MISSING
                if not self._inspection_original_matches(binding, original):
                    if (
                        binding.original_ref is None
                        and binding.original_guard is not None
                    ):
                        # The original was replaced; release the guard so a
                        # large stale object does not stay alive.
                        binding.original_guard = None
                    return _MISSING
            entry = self._inspection_cache_entries.get(binding.cache_key)
            if entry is None:
                return _MISSING
            return entry.value, binding

    @staticmethod
    def _inspection_original_matches(
        binding: _InspectionCacheBinding,
        original: Any,
    ) -> bool:
        if binding.original_ref is not None:
            # A dead weakref definitively means the bound original is gone; do
            # not fall back to ``id`` because object ids can be reused.
            return binding.original_ref() is original
        if binding.original_guard is not None:
            return binding.original_guard is original
        return False

    def _copy_inspection_objects(
        self,
        requests: dict[str, tuple[int | None, bool]],
        *,
        operation: str,
    ) -> None:
        plans: dict[str, tuple[Any, Any, int | None, bool]] = {}
        for name, (priority, compute) in requests.items():
            owner, terminal, _ = self._resolve_investigation_copy_target(
                name,
                priority=priority,
            )
            plans[name] = (owner, terminal, priority, compute)

        # Return reusable free arenas to the OS before taking the baseline.
        # Otherwise a large new object can be satisfied from memory that is
        # already resident, making the RSS delta far smaller than the object.
        # ``_release_native_allocators`` covers pools such as PyArrow's, which
        # ``malloc_trim`` cannot reach.
        gc.collect()
        self._attempt_allocator_trim()
        _release_native_allocators()
        before = _process_rss_bytes()

        # One copier per batch so shared sub-objects across the requested names
        # keep their aliases, matching protected block inspection semantics.
        copier = InspectionCopier(self.logger, allow_mutable_objects=False)
        prepared: dict[str, _InspectionCacheBinding] = {}
        loaded: dict[_InspectionCacheKey, _InspectionCacheEntry] = {}
        failed: dict[_InspectionCacheKey, _InspectionValueNotCopyable] = {}
        skipped: list[tuple[str, _InspectionValueNotCopyable, Any]] = []
        for name, (owner, terminal, priority, compute) in plans.items():
            # Determine the identity first so aliased names share one load or
            # one copy instead of materialising duplicates before dedup.
            cache_key = self._inspection_copy_cache_key(terminal, compute)
            if cache_key not in loaded and cache_key not in failed:
                try:
                    value = self._prepare_inspection_copy(
                        owner,
                        terminal,
                        compute=compute,
                        copier=copier,
                    )
                except _InspectionValueNotCopyable as exc:
                    failed[cache_key] = exc
                else:
                    loaded[cache_key] = _InspectionCacheEntry(
                        value=value,
                        compute=compute,
                    )
            if cache_key in failed:
                skipped.append((name, failed[cache_key], terminal))
                continue
            prepared[name] = self._inspection_cache_binding(
                terminal,
                cache_key,
                priority=priority,
                compute=compute,
            )

        if operation == "refresh" and skipped:
            name, exc, _ = skipped[0]
            raise InspectionCopyError(
                f"Could not refresh copied inspection object '{name}': {exc}. "
                "The existing inspection cache was left unchanged."
            ) from exc

        with self._inspection_cache_lock:
            for name, _exc, _terminal in skipped:
                self._inspection_cache_bindings.pop(name, None)
            for name, binding in prepared.items():
                self._inspection_cache_bindings[name] = binding
                self._inspection_copy_skipped.pop(name, None)
            self._inspection_cache_entries.update(loaded)
            live_keys = {
                binding.cache_key
                for binding in self._inspection_cache_bindings.values()
            }
            for cache_key in tuple(self._inspection_cache_entries):
                if cache_key not in live_keys:
                    del self._inspection_cache_entries[cache_key]

        for name, exc, terminal in skipped:
            self._inspection_copy_skipped[name] = (
                id(terminal),
                self._weakref_or_none(terminal),
            )
            self.logger.warning(
                f"Inspection object '{name}' cannot be copied for isolation "
                f"({exc}). It will be inspected by reference and may be changed "
                "during inspection. To guarantee original objects are preserved, "
                "save the pipeline before inspection and reload it from backup "
                "afterwards."
            )

        # Sample after cleanup so the single number approximates the retained
        # cache growth rather than the copying peak: transient buffers are
        # already unreferenced, and allocator pools (PyArrow, glibc) would
        # otherwise keep them resident and inflate the delta.
        gc.collect()
        self._attempt_allocator_trim()
        _release_native_allocators()
        after = _process_rss_bytes()
        # Estimate after the sample so the estimator's lazy imports (pandas,
        # torch, dask) are not charged to the cache.
        try:
            estimate = self._format_inspection_object_estimate(
                tuple(entry.value for entry in loaded.values()),
                label="newly cached size",
            )
        except Exception:
            estimate = ""
        try:
            self._log_inspection_memory_delta(
                operation,
                None if before is None or after is None else after - before,
                estimate,
                phase="after cleanup",
            )
        except Exception:
            pass

    def _inspection_copy_cache_key(
        self,
        terminal: Any,
        compute: bool,
    ) -> _InspectionCacheKey:
        if isinstance(terminal, ArtifactRecord):
            return (self._inspection_artifact_identity(terminal), compute)
        return (("memory", id(terminal)), compute)

    def _inspection_cache_binding(
        self,
        terminal: Any,
        cache_key: _InspectionCacheKey,
        *,
        priority: int | None,
        compute: bool,
    ) -> _InspectionCacheBinding:
        if isinstance(terminal, ArtifactRecord):
            return _InspectionCacheBinding(
                cache_key=cache_key,
                kind="artifact",
                priority=priority,
                compute=compute,
            )
        original_ref = self._weakref_or_none(terminal)
        return _InspectionCacheBinding(
            cache_key=cache_key,
            kind="memory",
            priority=priority,
            compute=compute,
            original_ref=original_ref,
            original_guard=terminal if original_ref is None else None,
            original_id=id(terminal),
        )

    def _prepare_inspection_copy(
        self,
        owner: Any,
        terminal: Any,
        *,
        compute: bool,
        copier: InspectionCopier,
    ) -> Any:
        if isinstance(terminal, ArtifactRecord):
            if (
                terminal.serializer == OPTUNA_STUDY_SERIALIZER
                or terminal.metadata.get("optuna_type") == "sampler"
            ):
                raise ResolutionError(
                    "Optuna studies and samplers cannot be copied for inspection; "
                    "inspect them with allow_mutable_objects=True instead."
                )
            value = owner._materialize_stored_value(terminal, "")
            # Older pickle artifacts may not carry the Optuna type marker.
            if is_optuna_study(value) or is_optuna_sampler(value):
                raise ResolutionError(
                    "Optuna studies and samplers cannot be copied for inspection; "
                    "inspect them with allow_mutable_objects=True instead."
                )
            return _compute_investigation_value(value, compute=compute)

        self._reject_uncopyable_inspection_terminal(terminal)
        if InspectionCopier._is_known_immutable(terminal):
            # Immutable values cannot be mutated, so sharing the value is safe
            # and avoids a pointless serializer round trip (small ints and some
            # strings are interned, which would look like a failed copy).
            return terminal
        copied = self._isolate_inspection_value(terminal, copier)
        return _compute_investigation_value(copied, compute=compute)

    @staticmethod
    def _reject_uncopyable_inspection_terminal(terminal: Any) -> None:
        if isinstance(terminal, TorchStateArtifactRecord):
            raise ResolutionError(
                "Torch state artifacts cannot be copied for inspection isolation."
            )
        if isinstance(terminal, CallableValueReference):
            raise ResolutionError(
                "Callable references cannot be copied for inspection isolation."
            )
        if isinstance(
            terminal,
            (RuntimeValueReference, DataclassValueReference),
        ):
            raise ResolutionError(
                "Placeholder values cannot be copied for inspection isolation; "
                "restore or reset them first."
            )
        if is_optuna_study(terminal) or is_optuna_sampler(terminal):
            raise ResolutionError(
                "Optuna studies and samplers cannot be copied for inspection "
                "isolation; inspect them with allow_mutable_objects=True instead."
            )

    def _isolate_inspection_value(
        self,
        value: Any,
        copier: InspectionCopier,
    ) -> Any:
        # Reuse the protected-inspection copier so pandas object cells, NumPy
        # object arrays, Torch tensors, and Dask collections are isolated with
        # the same semantics as ``allow_mutable_objects=False`` inspection. The
        # copier is shared per batch, so aliases between requested names survive.
        previous_memo = copier._memo.copy()
        try:
            copied = copier.prepare(
                value,
                parameter_name="value",
                source=ResolutionSource(
                    kind="copy_for_inspection",
                    name="value",
                ),
            )[0]
        except InspectionCopyError as exc:
            cause = exc.__cause__
            isolation_error = (
                f"{type(cause).__name__}: {cause}"
                if cause is not None
                else "the value could not be isolated"
            )
        except Exception as exc:
            isolation_error = f"{type(exc).__name__}: {exc}"
        else:
            if copied is not value:
                return copied
            isolation_error = "copy isolation returned the original object"
        # A failed pandas copy may already have memoized an incomplete frame.
        # Never expose that partial frame through a later name in this batch.
        copier._memo.clear()
        copier._memo.update(previous_memo)
        try:
            serialized = self._copy_value_via_temporary_artifact(value)
            copier._memo[id(value)] = serialized
            return serialized
        except Exception as exc:
            raise _InspectionValueNotCopyable(
                f"copy isolation failed ({isolation_error}); temporary "
                f"serialization failed ({type(exc).__name__}: {exc})"
            ) from exc

    def _copy_value_via_temporary_artifact(self, value: Any) -> Any:
        if self.project_root is None:
            raise _InspectionValueNotCopyable(
                "no project root is available for temporary serialization"
            )
        temp_root = Path(self.project_root) / "inspection_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = temp_root / uuid4().hex
        temp_dir.mkdir()
        try:
            serializer = choose_serializer(value)
            path = temp_dir / f"value{extension_for(serializer)}"
            dump_value(value, serializer, path)
            copied = load_value(serializer, path, torch_weights_only=False)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        if copied is value:
            raise _InspectionValueNotCopyable(
                "temporary serialization returned the original object"
            )
        return copied

    def _clear_inspection_temp_root(self) -> None:
        for pipeline in self._iter_attached_pipelines():
            project_root = getattr(pipeline, "project_root", None)
            if project_root is None:
                continue
            temp_root = Path(project_root) / "inspection_tmp"
            if temp_root.exists():
                shutil.rmtree(temp_root, ignore_errors=True)

    @staticmethod
    def _weakref_or_none(value: Any) -> Any | None:
        try:
            return weakref.ref(value)
        except TypeError:
            return None

    def _log_inspection_cache_hit(
        self,
        input_name: str,
        binding: _InspectionCacheBinding,
        requested_priority: int | None,
    ) -> None:
        kind = (
            "copied in-memory value"
            if binding.kind == "memory"
            else "copied artifact"
        )
        message = (
            f"Inspection value '{input_name}' was served from copy_for_inspection "
            f"({kind}, copied priority={binding.priority})."
        )
        if (
            requested_priority is not None
            and requested_priority != binding.priority
        ):
            message += (
                f" Requested priority={requested_priority}; reuse "
                f"copy_for_inspection(['{input_name}'], "
                f"priority={binding.priority}) if the original object must be "
                "preserved."
            )
        self.logger.info(message)

    def _log_inspection_copy_nudge(
        self,
        input_name: str,
        value: Any,
        *,
        requested_priority: int | None,
    ) -> None:
        if isinstance(value, _IMMUTABLE_TYPES):
            return
        skipped = self._inspection_copy_skipped.get(input_name)
        if skipped is not None:
            original_id, original_ref = skipped
            if original_ref is not None:
                # Only the exact skipped original suppresses the reminder; a
                # dead weakref means the replacement deserves the reminder.
                if original_ref() is value:
                    return
            elif id(value) == original_id:
                return
        priority_suffix = (
            ""
            if requested_priority is None
            else f", priority={requested_priority}"
        )
        if isinstance(value, ArtifactRecord):
            if value.serializer == OPTUNA_STUDY_SERIALIZER:
                return
            self.logger.info(
                f"Inspection value '{input_name}' was loaded from disk for this "
                "inspection; repeated inspections reload it every time. Use "
                f"copy_for_inspection(['{input_name}']{priority_suffix}) to keep "
                "one copy in memory."
            )
            return
        if isinstance(
            value,
            (
                TorchStateArtifactRecord,
                CallableValueReference,
                RuntimeValueReference,
                DataclassValueReference,
            ),
        ):
            return
        if is_optuna_study(value) or is_optuna_sampler(value):
            return
        self.logger.info(
            f"Inspection value '{input_name}' is an in-memory object inspected by "
            "reference. If mutation is a concern, copy it first with "
            f"copy_for_inspection(['{input_name}']{priority_suffix})."
        )

    def _unload_from_memory(
        self,
        object_names: Sequence[str] | None = None,
    ) -> None:
        with self._inspection_cache_lock:
            if object_names is None:
                self._inspection_cache_bindings.clear()
                self._inspection_cache_entries.clear()
                return
            for name in object_names:
                self._inspection_cache_bindings.pop(name, None)
            live_keys = {
                binding.cache_key
                for binding in self._inspection_cache_bindings.values()
            }
            for cache_key in tuple(self._inspection_cache_entries):
                if cache_key not in live_keys:
                    del self._inspection_cache_entries[cache_key]

    def _log_inspection_memory_delta(
        self,
        operation: str,
        delta: int | None,
        estimate: str = "",
        *,
        phase: str,
    ) -> None:
        if delta is None:
            self.logger.warning(
                f"Inspection cache {operation} RSS delta is unavailable ({phase})"
                f"{estimate}; install the 'memory' extra to enable process-memory "
                "reporting"
            )
            return
        self.logger.info(
            f"Inspection cache {operation} net process RSS change ({phase}): "
            f"{_format_signed_bytes(delta)}{estimate}"
        )

    @staticmethod
    def _format_inspection_object_estimate(
        values: Sequence[Any],
        *,
        label: str,
    ) -> str:
        """Describe the estimated retained size of loaded objects.

        The estimate is storage-based where possible (DataFrame memory usage,
        array ``nbytes``, tensor element sizes), so it does not depend on
        allocator reuse. Bounded object-graph sampling marks the result as a
        lower bound.
        """
        if not values:
            return ""
        try:
            estimator = _InspectionMemoryEstimator(retained=True)
            for index, value in enumerate(values):
                estimator.add(f"inspection value {index}", value)
            parts: list[str] = []
            if estimator.total_bytes > 0:
                parts.append(f"~{_format_bytes(estimator.total_bytes)} host")
            for device, size in sorted(estimator.cuda_bytes.items()):
                if size > 0:
                    parts.append(f"~{_format_bytes(size)} CUDA device {device}")
            if not parts:
                parts.append("~0 B host")
            estimate = " + ".join(parts)
            if estimator.uncertain:
                estimate = f"at least {estimate}"
            return f"; estimated {label}: {estimate}"
        except Exception:
            return f"; estimated {label}: unavailable"

    @staticmethod
    def _inspection_artifact_identity(
        artifact: ArtifactRecord,
    ) -> _InspectionArtifactIdentity:
        return (
            str(Path(artifact.file_path).expanduser().resolve(strict=False)),
            artifact.serializer,
            bool(artifact.torch_load_weights_only),
            artifact.run_id,
            artifact.created_at,
        )

    @staticmethod
    def _validate_inspection_object_names(
        object_names: list[str] | tuple[str, ...],
    ) -> tuple[str, ...]:
        if not isinstance(object_names, (list, tuple)):
            raise TypeError("object_names must be a list or tuple of strings")
        names = tuple(object_names)
        if not names or any(
            not isinstance(name, str) or not name.strip() for name in names
        ):
            raise ValueError("object_names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("object_names must be unique")
        return names
