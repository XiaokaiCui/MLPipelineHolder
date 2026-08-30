from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar


@dataclass(frozen=True, slots=True)
class OutputAddress:
    pipeline_name: str
    node_name: str
    output_name: str


@dataclass(frozen=True, slots=True)
class OutputPointer:
    destination: OutputAddress


class PointerResolutionError(Exception):
    pass


ValueT = TypeVar("ValueT")


def resolve_pointer_chain(
    start: OutputAddress,
    read: Callable[[OutputAddress], ValueT | OutputPointer],
) -> tuple[OutputAddress, ValueT]:
    current = start
    visited: set[OutputAddress] = set()
    while True:
        if current in visited:
            raise PointerResolutionError(
                f"Output pointer cycle detected at {current!r}"
            )
        visited.add(current)
        try:
            value = read(current)
        except KeyError as exc:
            raise PointerResolutionError(
                f"Output pointer destination does not exist: {current!r}"
            ) from exc
        if isinstance(value, OutputPointer):
            current = value.destination
            continue
        return current, value


def is_strictly_upstream(
    target_priority: tuple[float, ...],
    current_priority: tuple[float, ...],
) -> bool:
    return target_priority < current_priority
