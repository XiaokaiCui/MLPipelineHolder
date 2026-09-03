from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final, cast
from uuid import uuid4

from .exceptions import PersistenceError, RegistrationError, ResolutionError
from .models import ArtifactRecord
from .optuna_support import is_optuna_sampler, is_optuna_study
from .serializers import choose_serializer

STORAGE_COLUMNS: Final = [
    "hash_id",
    "object_name",
    "object_type",
    "object_description",
    "created_at_utc",
    "last_modified_at_utc",
]


@dataclass
class StoredObjectRecord:
    hash_id: str
    object_name: str
    object_type: str
    object_description: str | None
    created_at_utc: datetime
    last_modified_at_utc: datetime
    value: Any
    artifact: ArtifactRecord | None = None
    value_is_loaded: bool = True
    dirty: bool = True

    def to_payload(self, artifact: ArtifactRecord) -> dict[str, Any]:
        return {
            "hash_id": self.hash_id,
            "object_name": self.object_name,
            "object_type": self.object_type,
            "object_description": self.object_description,
            "created_at_utc": self.created_at_utc,
            "last_modified_at_utc": self.last_modified_at_utc,
            "artifact": artifact,
        }


def create_record(
    object_name: str,
    object_value: Any,
    object_description: str | None,
) -> StoredObjectRecord:
    validate_object_name(object_name)
    validate_object_description(object_description)
    warn_for_pickle_fallback(object_value)
    now = datetime.now(UTC)
    return StoredObjectRecord(
        hash_id=uuid4().hex,
        object_name=object_name,
        object_type=type(object_value).__name__,
        object_description=object_description,
        created_at_utc=now,
        last_modified_at_utc=now,
        value=object_value,
    )


def record_from_payload(payload: object) -> StoredObjectRecord:
    if not isinstance(payload, dict):
        raise PersistenceError("Saved object storage contains an invalid record")
    record_payload = cast(dict[str, object], payload)
    required = (
        "hash_id",
        "object_name",
        "object_type",
        "object_description",
        "created_at_utc",
        "last_modified_at_utc",
        "artifact",
    )
    missing = [key for key in required if key not in record_payload]
    if missing:
        raise PersistenceError(
            f"Saved object storage record is missing key '{missing[0]}'"
        )
    artifact = record_payload["artifact"]
    if not isinstance(artifact, ArtifactRecord):
        raise PersistenceError("Saved object storage record has an invalid artifact")
    hash_id = _required_non_empty_string(record_payload, "hash_id")
    object_name = _required_non_empty_string(record_payload, "object_name")
    object_type = _required_non_empty_string(record_payload, "object_type")
    object_description = record_payload["object_description"]
    if object_description is not None and not isinstance(object_description, str):
        raise PersistenceError(
            "Saved object storage record has an invalid 'object_description'"
        )
    return StoredObjectRecord(
        hash_id=hash_id,
        object_name=object_name,
        object_type=object_type,
        object_description=object_description,
        created_at_utc=_required_utc_datetime(record_payload, "created_at_utc"),
        last_modified_at_utc=_required_utc_datetime(
            record_payload,
            "last_modified_at_utc",
        ),
        value=None,
        artifact=artifact,
        value_is_loaded=False,
        dirty=False,
    )


def resolve_record(
    records: dict[str, StoredObjectRecord],
    *,
    hash_id: str | None,
    object_name: str | None,
) -> StoredObjectRecord:
    if hash_id is not None:
        record = records.get(hash_id)
        if record is None:
            raise ResolutionError(f"No stored object has hash_id '{hash_id}'")
        return record
    if object_name is None:
        raise ResolutionError("Provide hash_id or object_name to select a stored object")
    matches = [record for record in records.values() if record.object_name == object_name]
    if not matches:
        raise ResolutionError(f"No stored object has object_name '{object_name}'")
    if len(matches) > 1:
        raise ResolutionError(
            f"Found multiple stored objects named '{object_name}'; use hash_id instead"
        )
    return matches[0]


def metadata_frame(records: dict[str, StoredObjectRecord]) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "list_stored_objects requires pandas; install mlpipelineholder[dataframe]"
        ) from exc
    rows = [
        {
            "hash_id": record.hash_id,
            "object_name": record.object_name,
            "object_type": record.object_type,
            "object_description": record.object_description,
            "created_at_utc": record.created_at_utc,
            "last_modified_at_utc": record.last_modified_at_utc,
        }
        for record in records.values()
    ]
    return pd.DataFrame(rows, columns=STORAGE_COLUMNS)


def validate_object_name(object_name: str) -> None:
    if not isinstance(object_name, str) or not object_name.strip():
        raise RegistrationError("object_name must be a non-empty string")


def validate_object_description(object_description: str | None) -> None:
    if object_description is not None and not isinstance(object_description, str):
        raise RegistrationError("object_description must be a string or None")


def warn_for_pickle_fallback(value: Any) -> None:
    if choose_serializer(value) != "pickle" or _known_pickle_value(value):
        return
    warnings.warn(
        f"Object type '{type(value).__name__}' has no dedicated serializer; "
        "falling back to pickle",
        UserWarning,
        stacklevel=3,
    )


def _known_pickle_value(value: Any) -> bool:
    if is_optuna_study(value) or is_optuna_sampler(value):
        return True
    try:
        import pandas as pd

        return isinstance(value, pd.DataFrame)
    except ImportError:
        return False


def _required_non_empty_string(payload: dict[str, object], field_name: str) -> str:
    value = payload[field_name]
    if not isinstance(value, str) or not value.strip():
        raise PersistenceError(
            f"Saved object storage record has an invalid '{field_name}'"
        )
    return value


def _required_utc_datetime(payload: dict[str, object], field_name: str) -> datetime:
    value = payload[field_name]
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise PersistenceError(
            f"Saved object storage record has an invalid UTC '{field_name}'"
        )
    return value
