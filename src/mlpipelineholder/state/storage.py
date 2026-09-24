"""Object storage: root-pipeline stored objects outside execution state."""

from __future__ import annotations

import pickle
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.models import ArtifactRecord
from ..exceptions import PersistenceError, RegistrationError
from ..integrations.optuna.api import StudyArtifactOptions
from ..integrations.optuna.support import (
    OPTUNA_STUDIES_DB_NAME,
    is_optuna_study,
)
from ..persistence.artifacts.serializers import choose_serializer
from ..persistence.artifacts.store import ArtifactStore
from ..persistence.object_storage import (
    StoredObjectRecord,
    create_record,
    metadata_frame,
    resolve_record,
    validate_object_description,
    validate_object_name,
    warn_for_pickle_fallback,
)

class StorageMixin:
    """Storing, updating, reading, and persisting root-pipeline stored objects."""

    if TYPE_CHECKING:
        _stored_objects: dict[str, StoredObjectRecord] = {}
        parent_pipeline: Any = None
        project_root: Any = None
        artifact_store: Any = None
        torch_load_weights_only: bool = False

        @property
        def optuna_studies_db_path(self) -> Any: ...
        def _reject_managed_study_reuse(
            self,
            value: Any,
            permitted_owner: tuple[str, str] | None = None,
            permitted_owner_kinds: frozenset[str] = frozenset({"output"}),
        ) -> None: ...
        def _optuna_study_value(self, value: Any) -> Any: ...
        def _materialize_stored_value(
            self, value: Any, placeholder_error: str
        ) -> Any: ...
        def _cleanup_replaced_artifact(self, previous_value: Any) -> None: ...
        @staticmethod
        def _normalized_path(path: Path) -> Path: ...

    def list_stored_objects(self) -> Any:
        """Return metadata for objects held by this root pipeline."""
        self._require_root_storage_owner()
        return metadata_frame(self._stored_objects)

    def save_to_storage(
        self,
        object_name: str,
        object_value: Any,
        object_description: str | None = None,
    ) -> str:
        """Keep an object outside pipeline execution state and return its hash ID."""
        self._require_root_storage_owner()
        self._reject_managed_study_reuse(
            object_value,
            permitted_owner_kinds=frozenset({"output", "constant"}),
        )
        study_value = self._optuna_study_value(object_value)
        stored_value = study_value if study_value is not None else object_value
        record = create_record(object_name, stored_value, object_description)
        if study_value is not None:
            record.artifact = self.artifact_store.save(
                variable_name=record.hash_id,
                value=study_value,
                block_name="storage",
                function_name="object",
                run_id=record.hash_id,
                optuna_db_path=self.optuna_studies_db_path,
                optuna_options=StudyArtifactOptions(
                    independent_copy=True,
                    owner_kind="storage",
                    owner_key=record.hash_id,
                ),
            )
            record.value = None
            record.value_is_loaded = False
            record.dirty = False
        self._stored_objects[record.hash_id] = record
        return record.hash_id

    def update_storage(
        self,
        hash_id: str,
        object_value: Any,
        object_name: str | None = None,
        object_description: str | None = None,
        to_disk: bool = False,
    ) -> None:
        """Replace a stored value and optionally persist it immediately."""
        self._require_root_storage_owner()
        record = resolve_record(self._stored_objects, hash_id=hash_id, object_name=None)
        self._reject_managed_study_reuse(
            object_value,
            permitted_owner=("storage", record.hash_id),
            permitted_owner_kinds=frozenset({"output", "constant"}),
        )
        study_value = self._optuna_study_value(object_value)
        if study_value is not None:
            if object_name is not None:
                validate_object_name(object_name)
            if object_description is not None:
                validate_object_description(object_description)
            previous_artifact = record.artifact
            previous_name = None
            original_name = None
            if (
                previous_artifact is not None
                and previous_artifact.serializer == "optuna-study"
            ):
                persisted_name = previous_artifact.metadata.get("study_name")
                lineage_name = previous_artifact.metadata.get("original_study_name")
                if isinstance(persisted_name, str):
                    previous_name = persisted_name
                if isinstance(lineage_name, str):
                    original_name = lineage_name
            replacement = self.artifact_store.save(
                variable_name=record.hash_id,
                value=study_value,
                block_name="storage",
                function_name="object",
                run_id=record.hash_id,
                optuna_db_path=self.optuna_studies_db_path,
                optuna_options=StudyArtifactOptions(
                    independent_copy=previous_name is None,
                    study_name=previous_name,
                    original_study_name=original_name,
                    owner_kind="storage",
                    owner_key=record.hash_id,
                ),
            )
            if object_name is not None:
                record.object_name = object_name
            if object_description is not None:
                record.object_description = object_description
            record.artifact = replacement
            record.object_type = type(study_value).__name__
            record.value = None
            record.value_is_loaded = False
            record.dirty = False
            record.last_modified_at_utc = datetime.now(UTC)
            self._cleanup_replaced_artifact(previous_artifact)
            return
        if object_name is not None:
            validate_object_name(object_name)
            record.object_name = object_name
        if object_description is not None:
            validate_object_description(object_description)
            record.object_description = object_description
        warn_for_pickle_fallback(object_value)
        record.object_type = type(object_value).__name__
        record.value = object_value
        record.value_is_loaded = True
        record.dirty = True
        record.last_modified_at_utc = datetime.now(UTC)
        if to_disk:
            self._persist_stored_object(record, self.project_root)

    def get_from_storage(
        self,
        hash_id: str | None = None,
        object_name: str | None = None,
    ) -> Any:
        """Return one stored object, preferring hash ID when both selectors exist."""
        self._require_root_storage_owner()
        record = resolve_record(
            self._stored_objects,
            hash_id=hash_id,
            object_name=object_name,
        )
        return self._load_stored_object_value(record)

    def _get_stored_object_by_name(
        self,
        object_name: str,
        *,
        cache: bool,
    ) -> Any:
        self._require_root_storage_owner()
        record = resolve_record(
            self._stored_objects,
            hash_id=None,
            object_name=object_name,
        )
        return self._load_stored_object_value(record, cache=cache)

    def remove_from_storage(
        self,
        hash_id: str | None = None,
        object_name: str | None = None,
    ) -> None:
        """Remove one stored object and its managed artifact, when present."""
        self._require_root_storage_owner()
        record = resolve_record(
            self._stored_objects,
            hash_id=hash_id,
            object_name=object_name,
        )
        if record.artifact is not None:
            self.artifact_store.delete(record.artifact)
        del self._stored_objects[record.hash_id]

    def _require_root_storage_owner(self) -> None:
        if self.parent_pipeline is not None:
            raise RegistrationError(
                "Object storage APIs are available only on the root pipeline"
            )

    def _load_stored_object_value(
        self,
        record: StoredObjectRecord,
        *,
        cache: bool = True,
    ) -> Any:
        if record.value_is_loaded:
            return record.value
        if record.artifact is None:
            raise PersistenceError(
                f"Stored object '{record.object_name}' has no persisted artifact"
            )
        value = self._materialize_stored_value(record.artifact, "")
        if cache:
            record.value = value
            record.value_is_loaded = True
        return value

    def _persist_stored_object(
        self,
        record: StoredObjectRecord,
        target_root: Path,
    ) -> ArtifactRecord | None:
        if record.artifact is not None and not record.dirty:
            return record.artifact
        previous_artifact = record.artifact
        value = self._load_stored_object_value(record)
        if not is_optuna_study(value) and choose_serializer(value) == "pickle":
            try:
                pickle.dumps(value)
            except Exception as exc:
                warnings.warn(
                    f"Stored object '{record.object_name}' could not be pickled and "
                    f"will not be included in the saved pipeline: {type(exc).__name__}: {exc}",
                    UserWarning,
                    stacklevel=3,
                )
                return None
        artifact = ArtifactStore(target_root).save(
            variable_name=record.hash_id,
            value=value,
            block_name="storage",
            function_name="object",
            run_id=record.hash_id,
            torch_load_weights_only=self.torch_load_weights_only,
            optuna_db_path=target_root / OPTUNA_STUDIES_DB_NAME,
        )
        if self._normalized_path(target_root) == self._normalized_path(self.project_root):
            record.artifact = artifact
            record.dirty = False
            self._cleanup_replaced_artifact(previous_artifact)
        return artifact

    def _serialize_stored_objects_for_save(
        self,
        target_root: Path,
    ) -> dict[str, dict[str, Any]]:
        payload: dict[str, dict[str, Any]] = {}
        for hash_id, record in self._stored_objects.items():
            artifact = self._persist_stored_object(record, target_root)
            if artifact is not None:
                payload[hash_id] = record.to_payload(artifact)
        return payload
