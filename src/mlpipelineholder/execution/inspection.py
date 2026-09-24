"""Temporary value and non-committing execution inspection helpers."""

from __future__ import annotations

import copy
import math
import os
import sys
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import date, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from types import BuiltinFunctionType, FunctionType
from typing import TYPE_CHECKING, Any, Final, Self
from uuid import UUID

from ..core.constants import _IMMUTABLE_TYPES
from ..core.models import ArtifactRecord, ResolutionSource
from ..exceptions import InspectionCopyError, InspectionMemoryError
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

_ARTIFACT_MEMORY_FACTORS: Final[dict[str, float]] = {
    "numpy": 1.1,
    "torch": 1.1,
    "json": 2.0,
    "pickle": 2.0,
    "feather": 4.0,
    "parquet": 4.0,
}


class InspectionCopier:
    """Copy resolved values while preserving aliases within one inspection."""

    def __init__(self, logger: Any, *, allow_mutable_objects: bool) -> None:
        self._logger = logger
        self._allow_mutable_objects = allow_mutable_objects
        self._memo: dict[int, Any] = {}

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
        if value is self._logger or isinstance(
            value,
            (FunctionType, BuiltinFunctionType),
        ):
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
        if value is self._logger or isinstance(
            value,
            (FunctionType, BuiltinFunctionType),
        ):
            return False
        if is_optuna_study(value) or is_optuna_sampler(value):
            return False
        if (
            isinstance(value, ArtifactRecord)
            and value.serializer == OPTUNA_STUDY_SERIALIZER
        ):
            return False
        return True

    @classmethod
    def _is_known_immutable(cls, value: Any) -> bool:
        if isinstance(
            value,
            (
                *_IMMUTABLE_TYPES,
                Path,
                date,
                datetime,
                time,
                timedelta,
                UUID,
                Enum,
                range,
                slice,
            ),
        ):
            return True
        if isinstance(value, (tuple, frozenset)):
            return all(cls._is_known_immutable(item) for item in value)
        return False

    def _copy_value(
        self,
        value: Any,
        parameter_name: str,
        source: ResolutionSource,
    ) -> Any:
        try:
            import pandas as pd  # type: ignore

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
        except ImportError:
            pass

        try:
            import numpy as np  # type: ignore

            if isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    return copy.deepcopy(value, self._memo)
                copied = value.copy()
                self._memo[id(value)] = copied
                return copied
        except ImportError:
            pass

        try:
            import dask.dataframe as dd  # type: ignore

            if isinstance(value, (dd.DataFrame, dd.Series)):
                copied = value.copy()
                self._memo[id(value)] = copied
                return copied
        except ImportError:
            pass

        try:
            import torch  # type: ignore

            if isinstance(value, torch.Tensor):
                copied = value.detach().clone()
                self._memo[id(value)] = copied
                return copied
        except ImportError:
            pass

        return copy.deepcopy(value, self._memo)

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


def _format_bytes(size: float) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def _path_logical_bytes(path: Path) -> int:
    try:
        if path.is_dir():
            total = 0
            for child in path.rglob("*"):
                if child.is_file():
                    try:
                        total += child.stat().st_size
                    except OSError:
                        continue
            return total
        return path.stat().st_size
    except OSError:
        return 0


def _parquet_directory_files(path: Path) -> list[Path]:
    try:
        return [
            child
            for child in path.rglob("*")
            if child.is_file() and child.suffix == ".parquet"
        ]
    except OSError:
        return []


def _slot_names(value_type: type) -> tuple[str, ...]:
    names: list[str] = []
    for cls in value_type.__mro__:
        slots = cls.__dict__.get("__slots__")
        if slots is None:
            continue
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name in ("__dict__", "__weakref__"):
                continue
            if name not in names:
                names.append(name)
    return tuple(names)


class _InspectionMemoryEstimator:
    """Best-effort recursive estimator for protected-copy allocations.

    Estimates are conservative rather than exact: actual peak memory depends on
    allocator behaviour, temporary objects, and the implementation of each copy
    operation. Values whose memory cannot be bounded conservatively are recorded
    as uncertain so the caller can reject the operation instead of trusting a
    fabricated number.
    """

    def __init__(
        self,
        pointer_reader: Callable[[OutputPointer], Any] | None = None,
    ) -> None:
        self._pointer_reader = pointer_reader
        self._seen: set[int] = set()
        self._bytes = 0
        self._cuda_bytes: dict[int, int] = {}
        self._uncertain: list[tuple[str, str]] = []

    @property
    def total_bytes(self) -> int:
        return self._bytes

    @property
    def cuda_bytes(self) -> dict[int, int]:
        return dict(self._cuda_bytes)

    @property
    def uncertain(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._uncertain)

    def add(self, label: str, value: Any, *, materialize: bool = False) -> None:
        size = self._estimate(value, label)
        if materialize:
            size *= 2
        self._bytes += size

    def _estimate(self, value: Any, label: str) -> int:
        identity = id(value)
        if identity in self._seen:
            return 0
        self._seen.add(identity)
        try:
            return self._estimate_value(value, label)
        except InspectionMemoryError:
            raise
        except Exception as exc:
            self._uncertain.append((label, f"{type(exc).__name__}: {exc}"))
            return 0

    def _estimate_value(self, value: Any, label: str) -> int:
        if isinstance(value, ArtifactRecord):
            return self._estimate_artifact_record(value, label)
        if isinstance(value, OutputPointer):
            return self._estimate_output_pointer(value, label)
        if isinstance(value, _IMMUTABLE_TYPES):
            return sys.getsizeof(value)
        if isinstance(
            value,
            (FunctionType, BuiltinFunctionType, type, types.ModuleType, types.MethodType),
        ):
            return sys.getsizeof(value)

        try:
            import pandas as pd  # type: ignore

            if isinstance(value, pd.DataFrame):
                return self._estimate_pandas_frame(value, label)
            if isinstance(value, pd.Series):
                return self._estimate_pandas_series(value, label)
        except ImportError:
            pass

        try:
            import numpy as np  # type: ignore

            if isinstance(value, np.ndarray):
                return self._estimate_numpy_array(value, label)
        except ImportError:
            pass

        try:
            import torch  # type: ignore

            if isinstance(value, torch.nn.Module):
                return self._estimate_torch_module(value, label)
            if isinstance(value, torch.Tensor):
                return self._estimate_torch_tensor(value)
        except ImportError:
            pass

        try:
            import dask.dataframe as dd  # type: ignore

            if isinstance(value, (dd.DataFrame, dd.Series)):
                return self._estimate_dask_collection(value)
        except ImportError:
            pass

        return self._estimate_generic(value, label)

    def _estimate_artifact_record(self, record: ArtifactRecord, label: str) -> int:
        serializer = record.serializer
        if serializer == OPTUNA_STUDY_SERIALIZER:
            return 0
        factor = _ARTIFACT_MEMORY_FACTORS.get(serializer)
        if factor is None:
            self._uncertain.append(
                (
                    label,
                    f"artifact serializer '{serializer}' has no conservative "
                    "memory estimator",
                )
            )
            return 0
        path = Path(record.file_path)
        if serializer == "parquet" and path.is_dir():
            size = self._estimate_parquet_directory(path)
        else:
            size = _path_logical_bytes(path)
        return int(size * factor)

    @staticmethod
    def _estimate_parquet_directory(path: Path) -> int:
        try:
            import dask.dataframe  # type: ignore  # noqa: F401
        except ImportError:
            return _path_logical_bytes(path)
        files = _parquet_directory_files(path)
        return 256 * 1024 + len(files) * 16 * 1024

    def _estimate_output_pointer(self, pointer: OutputPointer, label: str) -> int:
        if self._pointer_reader is None:
            self._uncertain.append(
                (label, "output pointer cannot be resolved for estimation")
            )
            return 0
        try:
            terminal = self._pointer_reader(pointer)
        except PointerResolutionError as exc:
            self._uncertain.append((label, f"output pointer is unresolvable: {exc}"))
            return 0
        return self._estimate(terminal, label)

    def _estimate_pandas_frame(self, frame: Any, label: str) -> int:
        total = int(frame.memory_usage(index=True, deep=False).sum())
        for column_index, dtype in enumerate(frame.dtypes):
            if dtype != object:
                continue
            for row_index in range(len(frame.index)):
                total += self._estimate(frame.iat[row_index, column_index], label)
        return total

    def _estimate_pandas_series(self, series: Any, label: str) -> int:
        total = int(series.memory_usage(index=True, deep=False))
        if series.dtype == object:
            for index in range(len(series.index)):
                total += self._estimate(series.iat[index], label)
        return total

    def _estimate_numpy_array(self, array: Any, label: str) -> int:
        total = int(array.nbytes)
        if array.dtype.hasobject:
            for item in array.flat:
                total += self._estimate(item, label)
        return total

    def _estimate_torch_tensor(self, tensor: Any) -> int:
        try:
            size = int(tensor.element_size() * tensor.nelement())
        except Exception:
            size = int(getattr(tensor, "nbytes", 0))
        if bool(tensor.is_cuda):
            index = tensor.device.index
            device = 0 if index is None else int(index)
            self._cuda_bytes[device] = self._cuda_bytes.get(device, 0) + size
            return 0
        return size

    def _estimate_torch_module(self, module: Any, label: str) -> int:
        total = sys.getsizeof(module)
        for parameter in module.parameters():
            total += self._estimate(parameter, label)
        for buffer in module.buffers():
            total += self._estimate(buffer, label)
        return total

    def _estimate_dask_collection(self, collection: Any) -> int:
        try:
            task_count = int(len(collection.__dask_graph__()))
        except Exception:
            try:
                task_count = int(collection.npartitions) * 8
            except Exception:
                task_count = 4096
        return 256 * 1024 + task_count * 512

    def _estimate_generic(self, value: Any, label: str) -> int:
        size = sys.getsizeof(value)
        if isinstance(value, Enum):
            return size
        if isinstance(value, bytearray):
            return size
        if isinstance(value, dict):
            for key, item in value.items():
                size += self._estimate(key, label)
                size += self._estimate(item, label)
            return size
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                size += self._estimate(item, label)
            return size
        traversed = False
        if is_dataclass(value) and not isinstance(value, type):
            for field_info in fields(value):
                if hasattr(value, field_info.name):
                    size += self._estimate(
                        getattr(value, field_info.name),
                        label,
                    )
            traversed = True
        elif hasattr(value, "__dict__"):
            try:
                for item in vars(value).values():
                    size += self._estimate(item, label)
                traversed = True
            except TypeError:
                pass
        for slot_name in _slot_names(type(value)):
            try:
                item = getattr(value, slot_name)
            except AttributeError:
                continue
            size += self._estimate(item, label)
            traversed = True
        if not traversed and size >= _MATERIAL_MEMORY_BYTES:
            self._uncertain.append(
                (
                    label,
                    f"opaque object of type '{type(value).__name__}' cannot be "
                    "bounded conservatively",
                )
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
    except (ValueError, OSError):
        return None, ""
    infinity = getattr(resource, "RLIM_INFINITY", -1)
    limit = soft if soft != infinity else hard
    if limit == infinity or limit <= 0:
        return None, ""
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
    parameter_name: str,
    value: Any = None,
) -> InspectionMemoryError:
    """Build the error used when copying raises ``MemoryError`` after preflight."""
    estimate_line = ""
    if value is not None:
        try:
            estimator = _InspectionMemoryEstimator()
            estimator.add(parameter_name, value)
            estimate_line = (
                f"Estimated size of the affected input: "
                f"{_format_bytes(estimator.total_bytes)}"
            )
        except Exception:
            estimate_line = ""
    lines = [
        f"Protected inspection of '{node_name}' failed while copying input "
        f"'{parameter_name}' for function '{function_name}': the process ran out "
        "of memory.",
        "A protected copy had already started; inspection does not modify "
        "pipeline state, but memory pressure may still affect the process.",
    ]
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
        reason="MemoryError raised during protected copying",
        copy_started=True,
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
    seen: set[int] = set()
    labels: list[str] = []
    for candidate in candidates:
        identity = id(candidate.value)
        if identity in seen:
            continue
        seen.add(identity)
        labels.append(candidate.parameter_name)
        estimator.add(
            candidate.parameter_name,
            candidate.value,
            materialize=_requires_materialization(candidate.value),
        )
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
        required_bytes = math.ceil(required_bytes * (1.0 + safety_margin))
    cuda_required = {
        device: math.ceil(size * (1.0 + safety_margin))
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
    try:
        import dask.dataframe as dd
    except ImportError:
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
        @staticmethod
        def _validate_integer_priority(priority: Any) -> int: ...

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
        if priority is not None:
            self._validate_integer_priority(priority)
        return PipelineInspection(self, names, compute=compute, priority=priority)
