from __future__ import annotations

import os
import pickle
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Final, TypeGuard, cast

from ...exceptions import PersistenceError
from .api import (
    OptunaApi,
    OptunaRuntimeValue,
    OptunaSampler,
    OptunaStudy,
    PersistedOptunaStudy,
    StudyArtifactOptions,
)
from .sqlite import populate_study_from_sqlite

OPTUNA_STUDY_SERIALIZER: Final = "optuna-study"
OPTUNA_STUDIES_DB_NAME: Final = "optuna_studies.db"


class StudyAlreadyExistsError(ValueError):
    pass


def _load_optuna() -> OptunaApi:
    try:
        return OptunaApi(import_module("optuna"))
    except ModuleNotFoundError as exc:
        if exc.name != "optuna":
            raise
        raise PersistenceError(
            "Optuna is required to save or load this artifact; install MLPipelineHolder with 'pip install mlpipelineholder[optuna]'"
        ) from exc


def is_optuna_study(value: OptunaRuntimeValue) -> TypeGuard[OptunaStudy]:
    try:
        optuna = OptunaApi(import_module("optuna"))
    except ModuleNotFoundError as exc:
        if exc.name != "optuna":
            raise
        return False
    return isinstance(value, optuna.study.Study)


def is_optuna_sampler(value: OptunaRuntimeValue) -> TypeGuard[OptunaSampler]:
    try:
        optuna = OptunaApi(import_module("optuna"))
    except ModuleNotFoundError as exc:
        if exc.name != "optuna":
            raise
        return False
    return isinstance(value, optuna.samplers.BaseSampler)


def _get_sqlite_storage_url(db_path: str | Path) -> str:
    resolved_path = Path(db_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{resolved_path.as_posix()}"


def save_study_to_db(
    study: OptunaStudy,
    db_path: str | Path,
    study_name: str | None = None,
    overwrite: bool = True,
) -> PersistedOptunaStudy:
    optuna = _load_optuna()
    storage = _get_sqlite_storage_url(db_path)
    persisted_name = study.study_name if study_name is None else study_name
    directions = tuple(study.directions)
    trials = tuple(study.get_trials(deepcopy=True))
    user_attrs = dict(study.user_attrs)
    existing_studies = optuna.get_all_study_names(storage=storage)
    previous_state = None
    if persisted_name in existing_studies:
        if not overwrite:
            raise StudyAlreadyExistsError(
                f"Study '{persisted_name}' already exists in:\n{Path(db_path).expanduser().resolve()}"
            )
        previous_study = optuna.load_study(
            study_name=persisted_name,
            storage=storage,
            sampler=study.sampler,
        )
        previous_state = (
            tuple(previous_study.directions),
            tuple(previous_study.get_trials(deepcopy=True)),
            dict(previous_study.user_attrs),
        )
        optuna.delete_study(study_name=persisted_name, storage=storage)
    created = False
    try:
        saved_study = optuna.create_study(
            study_name=persisted_name,
            storage=storage,
            directions=directions,
        )
        created = True
        saved_study.add_trials(trials)
        for key, value in user_attrs.items():
            saved_study.set_user_attr(key, value)
    except BaseException:
        if created:
            optuna.delete_study(study_name=persisted_name, storage=storage)
        if previous_state is not None:
            previous_directions, previous_trials, previous_user_attrs = previous_state
            restored_study = optuna.create_study(
                study_name=persisted_name,
                storage=storage,
                directions=previous_directions,
            )
            restored_study.add_trials(previous_trials)
            for key, value in previous_user_attrs.items():
                restored_study.set_user_attr(key, value)
        raise
    return saved_study


def copy_study_to_db(
    study: OptunaStudy,
    db_path: str | Path,
) -> PersistedOptunaStudy:
    optuna = _load_optuna()
    duplicate_error = optuna.duplicated_study_error
    suffix = 1
    while True:
        persisted_name = f"{study.study_name}_{suffix}"
        created = False
        try:
            optuna.create_study(
                study_name=persisted_name,
                storage=_get_sqlite_storage_url(db_path),
                directions=study.directions,
            )
            created = True
            if populate_study_from_sqlite(study, db_path, persisted_name):
                return optuna.load_study(
                    study_name=persisted_name,
                    storage=_get_sqlite_storage_url(db_path),
                    sampler=study.sampler,
                )
            optuna.delete_study(
                study_name=persisted_name,
                storage=_get_sqlite_storage_url(db_path),
            )
            created = False
            return save_study_to_db(
                study,
                db_path,
                persisted_name,
                overwrite=False,
            )
        except (
            StudyAlreadyExistsError,
            duplicate_error,
        ):
            suffix += 1
        except BaseException:
            if created:
                optuna.delete_study(
                    study_name=persisted_name,
                    storage=_get_sqlite_storage_url(db_path),
                )
            raise


def save_study_artifact(
    study: OptunaStudy,
    sampler_path: Path,
    db_path: str | Path,
    *,
    options: StudyArtifactOptions | None = None,
) -> dict[str, str]:
    optuna = _load_optuna()
    options = StudyArtifactOptions() if options is None else options
    if not isinstance(study, optuna.study.Study):
        raise PersistenceError("Optuna study persistence received a non-Study value")
    if not isinstance(study.sampler, optuna.samplers.BaseSampler):
        raise PersistenceError("Optuna study has an unsupported sampler")
    sampler_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = sampler_path.with_name(f"{sampler_path.name}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            pickle.dump(study.sampler, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if options.independent_copy:
            saved_study = copy_study_to_db(study, db_path)
        elif options.allocate_if_occupied:
            try:
                saved_study = save_study_to_db(
                    study,
                    db_path,
                    study.study_name,
                    overwrite=False,
                )
            except (StudyAlreadyExistsError, optuna.duplicated_study_error):
                saved_study = copy_study_to_db(study, db_path)
        else:
            saved_study = save_study_to_db(
                study,
                db_path,
                study.study_name if options.study_name is None else options.study_name,
                overwrite=True,
            )
        os.replace(temporary_path, sampler_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    metadata = {
        "study_name": saved_study.study_name,
        "original_study_name": (
            study.study_name
            if options.original_study_name is None
            else options.original_study_name
        ),
        "db_path": str(Path(db_path).expanduser().resolve()),
    }
    if options.owner_kind is not None:
        metadata["study_owner_kind"] = options.owner_kind
    if options.owner_key is not None:
        metadata["study_owner_key"] = options.owner_key
    return metadata


def load_study_artifact(
    sampler_path: Path,
    metadata: Mapping[str, str],
) -> OptunaStudy:
    optuna = _load_optuna()
    study_name = metadata.get("study_name")
    db_path = metadata.get("db_path")
    if not isinstance(study_name, str) or not isinstance(db_path, str):
        raise PersistenceError("Optuna study artifact metadata is incomplete")
    resolved_db_path = Path(db_path).expanduser().resolve()
    if not resolved_db_path.is_file():
        raise PersistenceError(
            f"Optuna study database artifact is missing: {resolved_db_path}"
        )
    try:
        with sampler_path.open("rb") as handle:
            sampler = cast(OptunaSampler, pickle.load(handle))
    except (AttributeError, EOFError, ImportError, OSError, pickle.UnpicklingError) as exc:
        raise PersistenceError(
            f"Failed to load Optuna sampler artifact: {sampler_path}"
        ) from exc
    if not isinstance(sampler, optuna.samplers.BaseSampler):
        raise PersistenceError("Optuna study artifact did not contain a BaseSampler")
    storage = _get_sqlite_storage_url(resolved_db_path)
    return optuna.load_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
    )
