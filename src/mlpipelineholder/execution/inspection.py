"""Temporary value and non-committing execution inspection helpers."""

from __future__ import annotations

import copy
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from types import BuiltinFunctionType, FunctionType
from typing import TYPE_CHECKING, Any, Self
from uuid import UUID

from ..core.constants import _IMMUTABLE_TYPES
from ..core.models import ResolutionSource
from ..exceptions import InspectionCopyError
from ..integrations.optuna.support import is_optuna_sampler, is_optuna_study

if TYPE_CHECKING:
    from types import TracebackType


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
        except InspectionCopyError:
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
