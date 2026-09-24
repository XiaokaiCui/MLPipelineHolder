"""Temporary, non-registering access to several pipeline values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from types import TracebackType


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
