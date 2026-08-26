from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Any, Final, Protocol, TypeAlias, TypeGuard

from .models import ArtifactRecord

SlotMapping: TypeAlias = dict[str, Any]
YesInput: TypeAlias = Callable[[str], str]

_YES_CHOICES: Final = frozenset({"y", "yes"})


class _LoggerProtocol(Protocol):
    def warning(self, message: str) -> None: ...


class _PipelineProtocol(Protocol):
    registration_name: str
    parent_pipeline: _PipelineProtocol | None
    project_root: Path
    pipeline_backup_root: Path | None
    config: Any
    para_value_dict: dict[str, Any]
    manual_values: dict[str, Any]
    artifact_registry: dict[str, ArtifactRecord]
    producer_outputs: dict[str, dict[str, Any]]
    logger: _LoggerProtocol

    def full_path(self) -> str: ...

    def _root_pipeline(self) -> _PipelineProtocol: ...

    def _iter_attached_pipelines(self) -> list[_PipelineProtocol]: ...

    def _find_declaring_node(self, variable_name: str) -> Any: ...

    def list_declared_outputs(self) -> set[str]: ...

    def _incoming_parent_outputs(self) -> dict[str, Any]: ...

    def _sorted_nodes(self) -> list[Any]: ...

    def _contains_missing_main_placeholder(self, value: Any) -> bool: ...

    def config_as_dict(self) -> dict[str, Any]: ...

    def _inject_recovered_value(self, variable_name: str, value: Any) -> None: ...


def _is_pipeline_node(node: object) -> TypeGuard[_PipelineProtocol]:
    return hasattr(node, "parent_pipeline") and hasattr(node, "para_value_dict")


@unique
class OwnerKind(StrEnum):
    MANUAL = "manual"
    LATEST_PRODUCER = "latest_producer"
    FALLBACK_PARA = "fallback_para"
    DECLARED = "declared"


@unique
class SlotKind(StrEnum):
    PARA = "para"
    MANUAL = "manual"
    LATEST_PRODUCER = "latest_producer"
    ARTIFACT_REGISTRY = "artifact_registry"
    CHILD_OUTPUT_MIRROR = "child_output_mirror"
    ANCESTOR_VISIBLE_MIRROR = "ancestor_visible_mirror"


@dataclass(frozen=True, slots=True)
class _StateSlot:
    pipeline_path: str
    slot_kind: SlotKind
    mapping: SlotMapping
    key: str
    producer_name: str | None = None


@dataclass(frozen=True, slots=True)
class _OwnedVariableState:
    pipeline_path: str
    owner_kind: OwnerKind
    update_slots: tuple[_StateSlot, ...]


@dataclass(frozen=True, slots=True)
class _VariableOwnershipInventory:
    variable_name: str
    owners: tuple[_OwnedVariableState, ...]
    mirror_slots: tuple[_StateSlot, ...]

    @property
    def affected_paths(self) -> tuple[str, ...]:
        return tuple(owner.pipeline_path for owner in self.owners)


@dataclass(frozen=True, slots=True)
class ImpactConfirmation:
    authorized: bool
    prompted: bool
    affected_paths: tuple[str, ...]


def discover_owned_variable_slots(
    root_pipeline: _PipelineProtocol,
    variable_name: str,
    *,
    scope_pipeline: _PipelineProtocol | None = None,
) -> _VariableOwnershipInventory:
    owners: list[_OwnedVariableState] = []
    mirrors: list[_StateSlot] = []
    if scope_pipeline is not None:
        pipelines = [scope_pipeline]
    else:
        pipelines = root_pipeline._iter_attached_pipelines()
    for pipeline in pipelines:
        owner = _discover_owner(pipeline, variable_name)
        if owner is None:
            continue
        owners.append(owner)
        mirrors.extend(_ancestor_mirror_slots(pipeline, variable_name))
    return _VariableOwnershipInventory(variable_name, tuple(owners), tuple(mirrors))


def confirm_recovery_impact(
    inventory: _VariableOwnershipInventory,
    logger: _LoggerProtocol,
    input_func: YesInput | None = None,
) -> ImpactConfirmation:
    if len(inventory.affected_paths) <= 1:
        return ImpactConfirmation(True, False, inventory.affected_paths)
    try:
        response = (input if input_func is None else input_func)(
            _impact_prompt(inventory.variable_name, inventory.affected_paths)
        )
    except EOFError:
        logger.warning(_refusal_warning(inventory.variable_name, inventory.affected_paths))
        return ImpactConfirmation(False, True, inventory.affected_paths)
    if response.strip().lower() in _YES_CHOICES:
        return ImpactConfirmation(True, True, inventory.affected_paths)
    logger.warning(_refusal_warning(inventory.variable_name, inventory.affected_paths))
    return ImpactConfirmation(False, True, inventory.affected_paths)


def _discover_owner(
    pipeline: _PipelineProtocol,
    variable_name: str,
) -> _OwnedVariableState | None:
    path = pipeline.full_path()
    slots = [_StateSlot(path, SlotKind.PARA, pipeline.para_value_dict, variable_name)]
    if variable_name in pipeline.manual_values:
        slots.append(_StateSlot(path, SlotKind.MANUAL, pipeline.manual_values, variable_name))
        slots.append(_StateSlot(path, SlotKind.ARTIFACT_REGISTRY, pipeline.artifact_registry, variable_name))
        return _OwnedVariableState(path, OwnerKind.MANUAL, tuple(slots))
    latest_name = _latest_local_non_child_producer_name(pipeline, variable_name)
    if latest_name is not None:
        slots.append(
            _StateSlot(
                path,
                SlotKind.LATEST_PRODUCER,
                pipeline.producer_outputs[latest_name],
                variable_name,
                latest_name,
            )
        )
        slots.append(_StateSlot(path, SlotKind.ARTIFACT_REGISTRY, pipeline.artifact_registry, variable_name))
        return _OwnedVariableState(path, OwnerKind.LATEST_PRODUCER, tuple(slots))
    if variable_name not in pipeline.para_value_dict:
        return None
    if _latest_child_mirror_name(pipeline, variable_name) is not None:
        return None
    if variable_name in pipeline._incoming_parent_outputs():
        return None
    slots.append(_StateSlot(path, SlotKind.ARTIFACT_REGISTRY, pipeline.artifact_registry, variable_name))
    return _OwnedVariableState(path, OwnerKind.FALLBACK_PARA, tuple(slots))


def _latest_local_non_child_producer_name(
    pipeline: _PipelineProtocol,
    variable_name: str,
) -> str | None:
    for node in reversed(pipeline._sorted_nodes()):
        outputs = pipeline.producer_outputs.get(node.registration_name)
        if outputs is None or variable_name not in outputs or _is_pipeline_node(node):
            continue
        return node.registration_name
    return None


def _latest_child_mirror_name(
    pipeline: _PipelineProtocol,
    variable_name: str,
) -> str | None:
    for node in reversed(pipeline._sorted_nodes()):
        outputs = pipeline.producer_outputs.get(node.registration_name)
        if outputs is None or variable_name not in outputs:
            continue
        return node.registration_name if _is_pipeline_node(node) else None
    return None


def _ancestor_mirror_slots(
    owner_pipeline: _PipelineProtocol,
    variable_name: str,
) -> tuple[_StateSlot, ...]:
    mirrors: list[_StateSlot] = []
    current = owner_pipeline
    while current.parent_pipeline is not None:
        parent = current.parent_pipeline
        child_outputs = parent.producer_outputs.get(current.registration_name)
        if child_outputs is not None and variable_name in child_outputs:
            mirrors.append(
                _StateSlot(
                    parent.full_path(),
                    SlotKind.CHILD_OUTPUT_MIRROR,
                    child_outputs,
                    variable_name,
                    current.registration_name,
                )
            )
        if variable_name in parent.para_value_dict:
            mirrors.append(
                _StateSlot(
                    parent.full_path(),
                    SlotKind.ANCESTOR_VISIBLE_MIRROR,
                    parent.para_value_dict,
                    variable_name,
                )
            )
        current = parent
    return tuple(mirrors)


def _impact_prompt(variable_name: str, affected_paths: tuple[str, ...]) -> str:
    return "\n".join(
        [f"Recover '{variable_name}' for these independently owned pipelines:", *affected_paths, "Type yes or y to continue: "]
    )


def _refusal_warning(variable_name: str, affected_paths: tuple[str, ...]) -> str:
    return f"Skipped backup recovery for '{variable_name}' across: {', '.join(affected_paths)}"
