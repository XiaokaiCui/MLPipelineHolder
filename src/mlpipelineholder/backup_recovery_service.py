from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_recovery import (
    _ArtifactRecoveryTransaction,
    _delete_unreferenced_artifact,
)
from .backup_recovery import (
    OwnerKind,
    SlotKind,
    _OwnedVariableState,
    _PipelineProtocol,
    _StateSlot,
    _VariableOwnershipInventory,
    _is_pipeline_node,
    confirm_recovery_impact,
    discover_owned_variable_slots,
)
from .backup_snapshot import _BackupSnapshot, read_backup_snapshot
from .backup_value_resolver import (
    resolve_saved_config_field,
    resolve_saved_root_variable,
)
from .exceptions import PersistenceError, RegistrationError, ResolutionError
from .models import ArtifactRecord, TorchStateArtifactRecord


@dataclass(frozen=True, slots=True)
class _SlotSnapshot:
    slot: _StateSlot
    existed: bool
    value: Any


@dataclass(frozen=True, slots=True)
class _DeclaredInjectionSnapshots:
    entry_snapshots: tuple[tuple[dict[str, Any] | None, str, bool, Any], ...]
    mapping_snapshots: tuple[
        tuple[dict[str, Any], str, bool, dict[str, Any] | None], ...
    ]


def recover_variable_from_backup(
    pipeline: _PipelineProtocol,
    name: str,
    *,
    pipeline_name: str | None = None,
) -> None:
    if pipeline.parent_pipeline is not None:
        root = pipeline._root_pipeline()
        raise RegistrationError(
            f"Variable recovery is only available from root pipeline '{root.full_path()}'; "
            f"call {root.registration_name}.recover_variable_from_backup(...)"
        )
    scope = pipeline
    if pipeline_name is not None:
        scope = _find_pipeline_by_name(pipeline, pipeline_name)
        if scope is None:
            raise ResolutionError(f"Unknown pipeline: {pipeline_name}")
    inventory = discover_owned_variable_slots(
        pipeline,
        name,
        scope_pipeline=scope if pipeline_name is not None else None,
    )
    declaring = scope
    declared_only = False
    if not inventory.owners:
        if pipeline_name is not None:
            declaring_node = scope._find_declaring_node(name)
            if declaring_node is None or _is_pipeline_node(declaring_node):
                raise ResolutionError(f"Unknown pipeline value: {name}")
        else:
            if name not in scope.list_declared_outputs():
                raise ResolutionError(f"Unknown pipeline value: {name}")
            declaring = _deepest_declaring_pipeline(scope, name)
        inventory = _declared_inventory(declaring, name)
        declared_only = True
    snapshot = read_backup_snapshot(pipeline)
    if declared_only or pipeline_name is not None:
        target_path = (
            declaring.full_path()
            if declared_only
            else scope.full_path()
        )
        payload = snapshot.payload_for_path(tuple(target_path.split("/")))
    else:
        payload = snapshot.state_payload
    resolved = resolve_saved_root_variable(
        payload,
        name,
        is_missing_main_placeholder=pipeline._contains_missing_main_placeholder,
    )
    snapshot.validate_selected_artifacts(resolved)
    confirmation = confirm_recovery_impact(inventory, pipeline.logger)
    if not confirmation.authorized:
        return
    transaction = _artifact_transaction(snapshot, pipeline.project_root)
    prepared = transaction.clone_value(resolved)
    snapshot.assert_unchanged()
    slot_snapshots = _snapshot_inventory(inventory)
    declared_snapshots = _snapshot_declared_injection(inventory, pipeline)
    try:
        if declared_snapshots is not None:
            _inject_declared_owners(pipeline, inventory, prepared)
        else:
            _assign_inventory(inventory, prepared)
        transaction.commit()
    except (KeyboardInterrupt, SystemExit):
        _restore_slots(slot_snapshots)
        if declared_snapshots is not None:
            _restore_declared_injection(declared_snapshots)
        transaction.rollback()
        raise
    except Exception as exc:
        _restore_slots(slot_snapshots)
        if declared_snapshots is not None:
            _restore_declared_injection(declared_snapshots)
        transaction.rollback()
        if isinstance(exc, PersistenceError):
            raise
        raise PersistenceError(
            f"Failed to recover pipeline value '{name}' from backup"
        ) from exc
    _cleanup_old_artifacts(pipeline, slot_snapshots)


def recover_config_from_backup(pipeline: _PipelineProtocol, name: str) -> None:
    current_config = pipeline.config_as_dict()
    if name not in current_config:
        raise ResolutionError(f"Unknown local config field: {name}")
    root = pipeline._root_pipeline()
    snapshot = read_backup_snapshot(root)
    payload = snapshot.payload_for_path(tuple(pipeline.full_path().split("/")))
    resolved = resolve_saved_config_field(
        payload,
        name,
        is_missing_main_placeholder=pipeline._contains_missing_main_placeholder,
    )
    snapshot.validate_selected_artifacts(resolved)
    transaction = _artifact_transaction(snapshot, pipeline.project_root)
    prepared = transaction.clone_value(resolved)
    staged_config = _staged_config(pipeline.config, name, prepared)
    snapshot.assert_unchanged()
    previous_config = pipeline.config
    try:
        pipeline.config = staged_config
        transaction.commit()
    except (KeyboardInterrupt, SystemExit):
        pipeline.config = previous_config
        transaction.rollback()
        raise
    except Exception as exc:
        pipeline.config = previous_config
        transaction.rollback()
        if isinstance(exc, PersistenceError):
            raise
        raise PersistenceError(
            f"Failed to recover config field '{name}' from backup"
        ) from exc
    _cleanup_config_artifact(pipeline, current_config[name])


def _artifact_transaction(
    snapshot: _BackupSnapshot,
    live_project_root: Path,
) -> _ArtifactRecoveryTransaction:
    saved_root = snapshot.state_payload.get("saved_project_root")
    if not isinstance(saved_root, str):
        raise PersistenceError("Backup payload has no valid saved project root")
    return _ArtifactRecoveryTransaction(
        saved_root,
        snapshot.backup_root,
        live_project_root,
    )


def _inventory_slots(
    inventory: _VariableOwnershipInventory,
) -> tuple[_StateSlot, ...]:
    slots: list[_StateSlot] = []
    seen: set[tuple[int, str]] = set()
    for owner in inventory.owners:
        for slot in owner.update_slots:
            key = (id(slot.mapping), slot.key)
            if key not in seen:
                seen.add(key)
                slots.append(slot)
    for slot in inventory.mirror_slots:
        key = (id(slot.mapping), slot.key)
        if key not in seen:
            seen.add(key)
            slots.append(slot)
    return tuple(slots)


def _snapshot_inventory(
    inventory: _VariableOwnershipInventory,
) -> tuple[_SlotSnapshot, ...]:
    return tuple(
        _SlotSnapshot(slot, slot.key in slot.mapping, slot.mapping.get(slot.key))
        for slot in _inventory_slots(inventory)
    )


def _find_pipeline_by_name(
    root: _PipelineProtocol,
    registration_name: str,
) -> _PipelineProtocol | None:
    for candidate in root._iter_attached_pipelines():
        if candidate.registration_name == registration_name:
            return candidate
    return None


def _declared_inventory(
    scope: _PipelineProtocol,
    variable_name: str,
) -> _VariableOwnershipInventory:
    owner = _OwnedVariableState(
        scope.full_path(),
        OwnerKind.DECLARED,
        (),
    )
    return _VariableOwnershipInventory(variable_name, (owner,), ())


def _deepest_declaring_pipeline(
    scope: _PipelineProtocol,
    variable_name: str,
) -> _PipelineProtocol:
    current = scope
    while True:
        node = current._find_declaring_node(variable_name)
        if node is None or not hasattr(node, "parent_pipeline"):
            return current
        current = node


def _inject_declared_owners(
    root: _PipelineProtocol,
    inventory: _VariableOwnershipInventory,
    value: Any,
) -> None:
    for owner in inventory.owners:
        if owner.owner_kind is not OwnerKind.DECLARED:
            continue
        target = _find_pipeline_by_path(root, owner.pipeline_path)
        if target is None:
            continue
        target._inject_recovered_value(inventory.variable_name, value)


def _snapshot_declared_injection(
    inventory: _VariableOwnershipInventory,
    root: _PipelineProtocol,
) -> _DeclaredInjectionSnapshots | None:
    owners = [
        owner
        for owner in inventory.owners
        if owner.owner_kind is OwnerKind.DECLARED
    ]
    if not owners:
        return None
    target = _find_pipeline_by_path(root, owners[0].pipeline_path)
    variable_name = inventory.variable_name
    if target is None:
        return None
    declaring_node = target._find_declaring_node(variable_name)
    while declaring_node is not None and _is_pipeline_node(declaring_node):
        target = declaring_node
        declaring_node = target._find_declaring_node(variable_name)
    if declaring_node is None:
        return None
    block_name = declaring_node.registration_name
    block_mapping = target.producer_outputs.get(block_name)
    entry_targets: list[tuple[dict[str, Any] | None, str]] = [
        (block_mapping, variable_name),
        (target.para_value_dict, variable_name),
        (target.artifact_registry, variable_name),
    ]
    mapping_targets = [(target.producer_outputs, block_name)]
    current = target
    while current.parent_pipeline is not None:
        parent = current.parent_pipeline
        child_mapping = parent.producer_outputs.get(current.registration_name)
        entry_targets.extend(
            [
                (child_mapping, variable_name),
                (parent.para_value_dict, variable_name),
                (parent.artifact_registry, variable_name),
            ]
        )
        mapping_targets.append(
            (parent.producer_outputs, current.registration_name)
        )
        current = parent
    entry_snapshots = tuple(
        (
            mapping,
            key,
            mapping is not None and key in mapping,
            mapping.get(key) if mapping is not None else None,
        )
        for mapping, key in entry_targets
    )
    mapping_snapshots = tuple(
        (parent_mapping, key, key in parent_mapping, parent_mapping.get(key))
        for parent_mapping, key in mapping_targets
    )
    return _DeclaredInjectionSnapshots(entry_snapshots, mapping_snapshots)


def _restore_declared_injection(snapshots: _DeclaredInjectionSnapshots) -> None:
    for mapping, key, existed, value in snapshots.mapping_snapshots:
        if existed:
            mapping[key] = value
        else:
            mapping.pop(key, None)
    for mapping, key, existed, value in snapshots.entry_snapshots:
        if mapping is None:
            continue
        if existed:
            mapping[key] = value
        else:
            mapping.pop(key, None)


def _find_pipeline_by_path(
    root: _PipelineProtocol,
    pipeline_path: str,
) -> _PipelineProtocol | None:
    for candidate in root._iter_attached_pipelines():
        if candidate.full_path() == pipeline_path:
            return candidate
    return None


def _assign_inventory(
    inventory: _VariableOwnershipInventory,
    value: Any,
) -> None:
    for slot in _inventory_slots(inventory):
        if slot.slot_kind is SlotKind.ARTIFACT_REGISTRY:
            if isinstance(value, ArtifactRecord):
                slot.mapping[slot.key] = value
            else:
                slot.mapping.pop(slot.key, None)
            continue
        slot.mapping[slot.key] = value


def _restore_slots(snapshots: tuple[_SlotSnapshot, ...]) -> None:
    for snapshot in snapshots:
        if snapshot.existed:
            snapshot.slot.mapping[snapshot.slot.key] = snapshot.value
        else:
            snapshot.slot.mapping.pop(snapshot.slot.key, None)


def _staged_config(config: Any, name: str, value: Any) -> Any:
    try:
        staged = deepcopy(config)
        if isinstance(staged, dict):
            staged[name] = value
        else:
            setattr(staged, name, value)
        return staged
    except Exception as exc:
        raise PersistenceError(
            f"Failed to stage config field '{name}' for recovery"
        ) from exc


def _cleanup_old_artifacts(
    root: _PipelineProtocol,
    snapshots: tuple[_SlotSnapshot, ...],
) -> None:
    old_records: dict[str, ArtifactRecord | TorchStateArtifactRecord] = {}
    for snapshot in snapshots:
        if isinstance(snapshot.value, (ArtifactRecord, TorchStateArtifactRecord)):
            old_records[snapshot.value.file_path] = snapshot.value
    live_values = _all_live_values(root)
    for record in old_records.values():
        try:
            _delete_unreferenced_artifact(record, live_values)
        except OSError as exc:
            root.logger.warning(
                f"Recovered value but could not delete superseded artifact '{record.file_path}': {exc}"
            )


def _cleanup_config_artifact(pipeline: _PipelineProtocol, old_value: Any) -> None:
    if not isinstance(old_value, (ArtifactRecord, TorchStateArtifactRecord)):
        return
    try:
        _delete_unreferenced_artifact(old_value, _all_live_values(pipeline._root_pipeline()))
    except OSError as exc:
        pipeline.logger.warning(
            f"Recovered config but could not delete superseded artifact '{old_value.file_path}': {exc}"
        )


def _all_live_values(root: _PipelineProtocol) -> list[object]:
    values: list[object] = []
    for pipeline in root._iter_attached_pipelines():
        values.extend(pipeline.para_value_dict.values())
        values.extend(pipeline.manual_values.values())
        values.extend(pipeline.artifact_registry.values())
        values.extend(pipeline.config_as_dict().values())
        for outputs in pipeline.producer_outputs.values():
            values.extend(outputs.values())
    return values
