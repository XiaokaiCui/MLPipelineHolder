"""Output-graph state: Study ownership, pointer maintenance, and graph validation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.constants import _MISSING
from ..core.models import ArtifactRecord
from ..exceptions import PersistenceError, RegistrationError, ResolutionError
from ..integrations.optuna.api import OptunaStudy
from ..integrations.optuna.support import is_optuna_study
from ..state.output_pointers import (
    OutputAddress,
    OutputPointer,
    PointerDestinationMissingError,
    PointerResolutionError,
    resolve_pointer_chain,
)

_holder_base: type | None = None


def _register_state_holder_base(holder_base: type) -> None:
    """Register the holder class so state code can recognise child pipelines."""
    global _holder_base
    _holder_base = holder_base


def _is_child_pipeline(node: object) -> bool:
    return _holder_base is not None and isinstance(node, _holder_base)


class OutputGraphMixin:
    """Study ownership bookkeeping and output-pointer graph maintenance."""

    if TYPE_CHECKING:
        def _root_pipeline(self) -> Any: ...
        def _pipeline_by_name(self, pipeline_name: str) -> Any: ...
        def _priority_group(self, execution_priority: float | None) -> int: ...
        def _materialize_stored_value(self, value: Any, placeholder_error: str) -> Any: ...

    @staticmethod
    def _is_optuna_study_output(value: Any) -> bool:
        return is_optuna_study(value) or (
            isinstance(value, ArtifactRecord)
            and value.serializer == "optuna-study"
        )

    def _managed_study_owner(self, value: Any) -> tuple[str, str] | None:
        if isinstance(value, ArtifactRecord) and value.serializer == "optuna-study":
            owner_kind = value.metadata.get("study_owner_kind")
            owner_key = value.metadata.get("study_owner_key")
            if isinstance(owner_kind, str) and isinstance(owner_key, str):
                return owner_kind, owner_key
            return None
        if not is_optuna_study(value):
            return None
        return self._root_pipeline()._optuna_study_provenance.get(value)

    def _reject_managed_study_reuse(
        self,
        value: Any,
        permitted_owner: tuple[str, str] | None = None,
        permitted_owner_kinds: frozenset[str] = frozenset({"output"}),
    ) -> None:
        study_owner = self._managed_study_owner(value)
        if (
            study_owner is None
            or study_owner[0] in permitted_owner_kinds
            or study_owner == permitted_owner
        ):
            return
        owner_kind, owner_key = study_owner
        if owner_kind == "constant":
            constant_name = owner_key.rsplit(".", 1)[-1]
            raise RegistrationError(
                f"Study is already managed by constant '{owner_key}'. "
                "Constant values cannot hold a duplicate of a Study managed by "
                "another constant; seed the new constant from the original source "
                "(storage, pipeline output, or an unowned Study) instead, or replace "
                f"that constant's Study with set_constant_value('{constant_name}', ...) "
                "on its owning pipeline."
            )
        if owner_kind == "storage":
            raise RegistrationError(
                f"Study is already managed by stored object '{owner_key}'. "
                "Storage cannot hold duplicate Studies; replace that stored object "
                f"with update_storage(hash_id='{owner_key}', object_value=...) on the "
                "root pipeline, or store a fresh Study (e.g. from optuna.load_study)."
            )
        raise RegistrationError(
            f"Study is already managed by {owner_kind} '{owner_key}'"
        )

    def _optuna_study_value(
        self, value: Any
    ) -> OptunaStudy | None:
        if isinstance(value, ArtifactRecord):
            if value.serializer != "optuna-study":
                return None
            materialized = self._materialize_stored_value(value, "")
            if not is_optuna_study(materialized):
                raise PersistenceError("Optuna Study artifact did not load a Study")
            return materialized
        if is_optuna_study(value):
            return value
        return None

    def _remember_optuna_study_groups(self) -> None:
        root = self._root_pipeline()
        slots = root._public_output_slots()
        for address, value in slots.items():
            terminal = value
            if isinstance(value, OutputPointer):
                try:
                    _, terminal = resolve_pointer_chain(address, slots.__getitem__)
                except PointerResolutionError:
                    continue
            if not (
                isinstance(terminal, ArtifactRecord)
                and terminal.serializer == "optuna-study"
            ):
                continue
            root._optuna_study_group_addresses.setdefault(
                address.output_name,
                set(),
            ).add(address)
            study_name = terminal.metadata.get("study_name")
            original_name = terminal.metadata.get("original_study_name", study_name)
            if isinstance(study_name, str) and isinstance(original_name, str):
                root._optuna_study_group_names[address.output_name] = (
                    study_name,
                    original_name,
                )

    def _set_output_address(
        self, address: OutputAddress, value: Any
    ) -> None:
        pipeline = self._pipeline_by_name(address.pipeline_name)
        outputs = pipeline.producer_outputs.setdefault(address.node_name, {})
        outputs[address.output_name] = value

    def _output_address_priority(
        self, address: OutputAddress
    ) -> tuple[float, ...]:
        pipeline = self._pipeline_by_name(address.pipeline_name)
        return pipeline._priority_vector(address.node_name)

    def _same_priority_group_address(
        self,
        first: OutputAddress,
        second: OutputAddress,
    ) -> bool:
        root = self._root_pipeline()
        first_pipeline = root._pipeline_by_name(first.pipeline_name)
        second_pipeline = root._pipeline_by_name(second.pipeline_name)
        if first_pipeline is not second_pipeline:
            return False
        first_node = first_pipeline.nodes_by_name.get(first.node_name)
        second_node = first_pipeline.nodes_by_name.get(second.node_name)
        if first_node is None or second_node is None:
            return False
        return self._priority_group(
            first_node.execution_priority
        ) == self._priority_group(second_node.execution_priority)

    def _repair_pointer_gaps(
        self,
        invalidated: set[OutputAddress],
        slots: dict[OutputAddress, Any],
    ) -> None:
        for source, value in slots.items():
            if source in invalidated or not isinstance(value, OutputPointer):
                continue
            destination = value.destination
            if destination not in invalidated:
                continue
            visited: set[OutputAddress] = set()
            while destination in invalidated:
                if destination in visited:
                    raise ResolutionError(
                        f"Output pointer cycle detected while repairing {source!r}"
                    )
                visited.add(destination)
                next_value = slots.get(destination, _MISSING)
                if not isinstance(next_value, OutputPointer):
                    break
                destination = next_value.destination
            if destination not in invalidated and destination in slots:
                self._set_output_address(source, OutputPointer(destination))

    def _promote_pointer_values_before_removal(
        self,
        invalidated: set[OutputAddress],
        slots: dict[OutputAddress, Any],
    ) -> None:
        for owner in invalidated:
            value = slots.get(owner, _MISSING)
            if value is _MISSING or isinstance(value, OutputPointer):
                continue
            survivors: list[OutputAddress] = []
            for source in slots:
                if source in invalidated:
                    continue
                try:
                    terminal, _ = resolve_pointer_chain(source, slots.__getitem__)
                except PointerResolutionError:
                    continue
                if terminal == owner:
                    survivors.append(source)
            if not survivors:
                continue
            destination = max(survivors, key=self._output_address_priority)
            promoted_value = value
            if isinstance(value, ArtifactRecord):
                destination_pipeline = self._pipeline_by_name(
                    destination.pipeline_name
                )
                promoted_value = destination_pipeline.artifact_store.transfer(
                    value,
                    destination_pipeline.qualified_node_name(destination.node_name),
                )
            self._set_output_address(destination, promoted_value)
            if (
                isinstance(value, ArtifactRecord)
                and value.serializer == "optuna-study"
            ):
                for survivor in survivors:
                    if survivor != destination:
                        self._set_output_address(
                            survivor,
                            OutputPointer(destination),
                        )

    def _prepare_pointer_removal(
        self,
        invalidated: set[OutputAddress],
        slots: dict[OutputAddress, Any],
    ) -> None:
        self._repair_pointer_gaps(invalidated, slots)
        self._promote_pointer_values_before_removal(invalidated, slots)

    def _validate_runtime_output_pointers(self) -> None:
        slots = self._public_output_slots()
        for address, value in slots.items():
            if not isinstance(value, OutputPointer):
                continue
            try:
                resolve_pointer_chain(address, slots.__getitem__)
            except PointerResolutionError as exc:
                raise PersistenceError(
                    f"Saved output pointer graph is invalid: {exc}"
                ) from exc

    def _refresh_pointer_visible_state(self) -> None:
        root = self._root_pipeline()

        def refresh(pipeline: Any) -> None:
            pipeline._rebuild_visible_state(pipeline._incoming_parent_outputs())
            for child in pipeline._sorted_nodes():
                if _is_child_pipeline(child):
                    refresh(child)

        refresh(root)

    def _read_output_address(
        self, address: OutputAddress
    ) -> Any:
        pipelines = [
            pipeline
            for pipeline in self._root_pipeline()._iter_attached_pipelines()
            if pipeline.registration_name == address.pipeline_name
        ]
        if len(pipelines) != 1:
            raise KeyError(address)
        outputs = pipelines[0].producer_outputs.get(address.node_name)
        if outputs is None or address.output_name not in outputs:
            raise KeyError(address)
        return outputs[address.output_name]

    def _public_output_slots(
        self,
    ) -> dict[OutputAddress, Any]:
        slots: dict[OutputAddress, Any] = {}
        for pipeline in self._root_pipeline()._iter_attached_pipelines():
            if pipeline._is_atom:
                continue
            for node in pipeline._sorted_nodes():
                if _is_child_pipeline(node) and not node._is_atom:
                    continue
                node_outputs = pipeline.producer_outputs.get(
                    node.registration_name,
                    {},
                )
                for output_name, value in node_outputs.items():
                    address = OutputAddress(
                        pipeline.registration_name,
                        node.registration_name,
                        output_name,
                    )
                    slots[address] = value
        return slots

    def _all_output_slots(
        self,
    ) -> dict[OutputAddress, Any]:
        slots: dict[OutputAddress, Any] = {}
        for pipeline in self._root_pipeline()._iter_attached_pipelines():
            if pipeline._is_atom:
                continue
            for node in pipeline._sorted_nodes():
                node_outputs = pipeline.producer_outputs.get(
                    node.registration_name,
                    {},
                )
                for output_name, value in node_outputs.items():
                    slots[
                        OutputAddress(
                            pipeline.registration_name,
                            node.registration_name,
                            output_name,
                        )
                    ] = value
        return slots

    def _recover_loaded_output_pointers(self) -> None:
        root = self._root_pipeline()
        recovered = False

        def restore_artifact_owner(
            artifact: ArtifactRecord,
            owner: OutputAddress,
            sources: list[OutputAddress],
        ) -> None:
            owner_key = (
                f"{owner.pipeline_name}.{owner.node_name}.{owner.output_name}"
            )
            if artifact.serializer == "optuna-study":
                artifact.metadata.setdefault("study_owner_kind", "output")
                artifact.metadata.setdefault("study_owner_key", owner_key)
            root._set_output_address(owner, artifact)
            for source in sources:
                if source == owner:
                    continue
                if root._same_priority_group_address(source, owner):
                    root._set_output_address(source, None)
                    root.logger.warning(
                        f"Loaded output '{source.pipeline_name}.{source.node_name}."
                        f"{source.output_name}' as None because it shares an execution "
                        "priority group with its Study owner"
                    )
                else:
                    root._set_output_address(source, OutputPointer(owner))
            root.logger.warning(
                f"Recovered legacy output ownership at '{owner_key}'"
            )

        public_slots = root._public_output_slots()
        all_slots = root._all_output_slots()
        recoveries: dict[OutputAddress, tuple[ArtifactRecord, list[OutputAddress]]] = {}
        for source, value in public_slots.items():
            if not isinstance(value, OutputPointer):
                continue
            try:
                terminal, terminal_value = resolve_pointer_chain(
                    source,
                    all_slots.__getitem__,
                )
            except PointerResolutionError:
                continue
            if terminal in public_slots or not (
                isinstance(terminal_value, ArtifactRecord)
                and terminal_value.serializer == "optuna-study"
            ):
                continue
            terminal_pipeline = root._pipeline_by_name(terminal.pipeline_name)
            terminal_node = terminal_pipeline.nodes_by_name.get(terminal.node_name)
            if not _is_child_pipeline(terminal_node) or terminal_node._is_atom:
                continue
            existing = recoveries.get(terminal)
            if existing is None:
                recoveries[terminal] = (terminal_value, [source])
            else:
                existing[1].append(source)

        for artifact, sources in recoveries.values():
            owner = self._unique_legacy_artifact_owner(
                artifact, sources, public_slots
            )
            if owner is None:
                continue
            restore_artifact_owner(artifact, owner, sources)
            recovered = True

        public_slots = root._public_output_slots()
        for source, value in public_slots.items():
            if not isinstance(value, OutputPointer):
                continue
            try:
                resolve_pointer_chain(source, public_slots.__getitem__)
                continue
            except PointerDestinationMissingError:
                pass
            except PointerResolutionError:
                continue
            artifact, owner = self._find_orphan_legacy_artifact(
                source.output_name,
                public_slots,
            )
            if artifact is not None and owner is not None:
                restore_artifact_owner(artifact, owner, [source])
            else:
                root._set_output_address(source, None)
                root.logger.warning(
                    f"Loaded output '{source.pipeline_name}.{source.node_name}."
                    f"{source.output_name}' as None because its pointer "
                    "destination is missing from the saved pipeline"
                )
            recovered = True
        if recovered:
            root._refresh_pointer_visible_state()

    def _address_matches_legacy_artifact_owner(
        self,
        address: OutputAddress,
        artifact: ArtifactRecord,
    ) -> bool:
        if address.output_name != artifact.variable_name:
            return False
        qualified = self._root_pipeline()._pipeline_by_name(
            address.pipeline_name
        ).qualified_node_name(address.node_name)
        return (
            qualified == artifact.produced_by_block
            or artifact.produced_by_block.startswith(f"{qualified}/")
        )

    def _unique_legacy_artifact_owner(
        self,
        artifact: ArtifactRecord,
        sources: list[OutputAddress],
        public_slots: dict[OutputAddress, Any],
    ) -> OutputAddress | None:
        candidates = [
            address
            for address in public_slots
            if self._address_matches_legacy_artifact_owner(address, artifact)
        ]
        if len(candidates) != 1 or candidates[0] not in sources:
            return None
        return candidates[0]

    def _find_orphan_legacy_artifact(
        self,
        output_name: str,
        public_slots: dict[OutputAddress, Any],
    ) -> tuple[ArtifactRecord | None, OutputAddress | None]:
        root = self._root_pipeline()
        artifacts_by_block: dict[str, list[ArtifactRecord]] = {}
        for address, value in root._all_output_slots().items():
            if address in public_slots:
                continue
            if not (
                isinstance(value, ArtifactRecord)
                and value.variable_name == output_name
            ):
                continue
            block_artifacts = artifacts_by_block.setdefault(
                value.produced_by_block,
                [],
            )
            if all(existing is not value for existing in block_artifacts):
                block_artifacts.append(value)
        if not artifacts_by_block:
            return None, None
        qualified_by_address = {
            address: root._pipeline_by_name(
                address.pipeline_name
            ).qualified_node_name(address.node_name)
            for address in public_slots
        }
        owners = [
            address
            for address in public_slots
            if address.output_name == output_name
            and any(
                block == qualified_by_address[address]
                or block.startswith(f"{qualified_by_address[address]}/")
                for block in artifacts_by_block
            )
        ]
        if len(owners) != 1:
            return None, None
        owner = owners[0]
        owner_qualified = qualified_by_address[owner]
        artifacts: list[ArtifactRecord] = []
        for block_name, block_artifacts in artifacts_by_block.items():
            if block_name != owner_qualified and not block_name.startswith(
                f"{owner_qualified}/"
            ):
                continue
            for artifact in block_artifacts:
                if all(existing is not artifact for existing in artifacts):
                    artifacts.append(artifact)
        if len(artifacts) != 1:
            return None, None
        return artifacts[0], owner

    def _backfill_legacy_study_ownership(self) -> None:
        root = self._root_pipeline()
        slots = root._public_output_slots()
        for address, value in slots.items():
            terminal_address = address
            terminal_value = value
            if isinstance(value, OutputPointer):
                try:
                    terminal_address, terminal_value = resolve_pointer_chain(
                        address,
                        slots.__getitem__,
                    )
                except PointerResolutionError:
                    continue
            if not (
                isinstance(terminal_value, ArtifactRecord)
                and terminal_value.serializer == "optuna-study"
            ):
                continue
            terminal_value.metadata.setdefault("study_owner_kind", "output")
            terminal_value.metadata.setdefault(
                "study_owner_key",
                f"{terminal_address.pipeline_name}.{terminal_address.node_name}."
                f"{terminal_address.output_name}",
            )
        for pipeline in root._iter_attached_pipelines():
            for constant_name, value in pipeline.manual_values.items():
                if isinstance(value, ArtifactRecord) and value.serializer == "optuna-study":
                    value.metadata.setdefault("study_owner_kind", "constant")
                    value.metadata.setdefault(
                        "study_owner_key",
                        f"{pipeline.registration_name}.{constant_name}",
                    )
        for hash_id, record in root._stored_objects.items():
            artifact = record.artifact
            if isinstance(artifact, ArtifactRecord) and artifact.serializer == "optuna-study":
                artifact.metadata.setdefault("study_owner_kind", "storage")
                artifact.metadata.setdefault("study_owner_key", hash_id)

    def _replace_unloadable_persisted_values(self) -> None:
        root = self._root_pipeline()
        validity: dict[tuple[str, str], bool] = {}
        runtime_state_changed = False

        def artifact_loads(
            pipeline: Any,
            artifact: ArtifactRecord,
            owner_label: str,
        ) -> bool:
            key = (artifact.serializer, artifact.file_path)
            cached = validity.get(key)
            if cached is not None:
                return cached
            artifact_path = Path(artifact.file_path)
            if artifact.serializer == "torch" and (
                artifact_path.is_file() or artifact_path.is_dir()
            ):
                validity[key] = True
                return True
            try:
                pipeline.artifact_store.load(artifact)
            except PersistenceError as exc:
                validity[key] = False
                root.logger.warning(
                    f"Loaded {owner_label} as None because its persisted value is "
                    f"invalid: {exc}"
                )
                return False
            validity[key] = True
            return True

        for pipeline in root._iter_attached_pipelines():
            for node_name, outputs in pipeline.producer_outputs.items():
                for output_name, value in list(outputs.items()):
                    if isinstance(value, ArtifactRecord) and not artifact_loads(
                        pipeline,
                        value,
                        f"output '{pipeline.registration_name}.{node_name}.{output_name}'",
                    ):
                        outputs[output_name] = None
                        runtime_state_changed = True
            for constant_name, value in list(pipeline.manual_values.items()):
                if isinstance(value, ArtifactRecord) and not artifact_loads(
                    pipeline,
                    value,
                    f"constant '{pipeline.registration_name}.{constant_name}'",
                ):
                    pipeline.manual_values[constant_name] = None
                    runtime_state_changed = True

        for record in root._stored_objects.values():
            artifact = record.artifact
            if isinstance(artifact, ArtifactRecord) and not artifact_loads(
                root,
                artifact,
                f"stored object '{record.object_name}'",
            ):
                record.value = None
                record.value_is_loaded = True
                record.artifact = None
                record.dirty = True
        if runtime_state_changed:
            root._refresh_pointer_visible_state()
