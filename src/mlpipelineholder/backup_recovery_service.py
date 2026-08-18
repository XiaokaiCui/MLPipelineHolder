from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .artifact_recovery import (
    _ArtifactRecoveryTransaction,
    _delete_unreferenced_artifact,
)
from .backup_recovery import (
    SlotKind,
    _StateSlot,
    _VariableOwnershipInventory,
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

if TYPE_CHECKING:
    from .pipeline_handler import PipelineHandler


@dataclass(frozen=True, slots=True)
class _SlotSnapshot:
    slot: _StateSlot
    existed: bool
    value: Any


def recover_variable_from_backup(pipeline: PipelineHandler, name: str) -> None:
    if pipeline.parent_pipeline is not None:
        root = pipeline._root_pipeline()
        raise RegistrationError(
            f"Variable recovery is only available from root pipeline '{root.full_path()}'; "
            f"call {root.registration_name}.recover_variable_from_backup(...)"
        )
    inventory = discover_owned_variable_slots(pipeline, name)
    if not inventory.owners:
        raise ResolutionError(f"Unknown pipeline value: {name}")
    snapshot = read_backup_snapshot(pipeline)
    resolved = resolve_saved_root_variable(
        snapshot.state_payload,
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
    try:
        _assign_inventory(inventory, prepared)
        transaction.commit()
    except (KeyboardInterrupt, SystemExit):
        _restore_slots(slot_snapshots)
        transaction.rollback()
        raise
    except Exception as exc:
        _restore_slots(slot_snapshots)
        transaction.rollback()
        if isinstance(exc, PersistenceError):
            raise
        raise PersistenceError(
            f"Failed to recover pipeline value '{name}' from backup"
        ) from exc
    _cleanup_old_artifacts(pipeline, slot_snapshots)


def recover_config_from_backup(pipeline: PipelineHandler, name: str) -> None:
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
    root: PipelineHandler,
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


def _cleanup_config_artifact(pipeline: PipelineHandler, old_value: Any) -> None:
    if not isinstance(old_value, (ArtifactRecord, TorchStateArtifactRecord)):
        return
    try:
        _delete_unreferenced_artifact(old_value, _all_live_values(pipeline._root_pipeline()))
    except OSError as exc:
        pipeline.logger.warning(
            f"Recovered config but could not delete superseded artifact '{old_value.file_path}': {exc}"
        )


def _all_live_values(root: PipelineHandler) -> list[object]:
    values: list[object] = []
    for pipeline in root._iter_attached_pipelines():
        values.extend(pipeline.para_value_dict.values())
        values.extend(pipeline.manual_values.values())
        values.extend(pipeline.artifact_registry.values())
        values.extend(pipeline.config_as_dict().values())
        for outputs in pipeline.producer_outputs.values():
            values.extend(outputs.values())
    return values
