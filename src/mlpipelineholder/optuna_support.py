from __future__ import annotations

import os
import pickle
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Final, Protocol, TypeAlias, TypeGuard, cast, final

from .exceptions import PersistenceError

OPTUNA_STUDY_SERIALIZER: Final = "optuna-study"
OPTUNA_STUDIES_DB_NAME: Final = "optuna_studies.db"


_JsonScalar: TypeAlias = str | int | float | bool | None
_JsonValue: TypeAlias = _JsonScalar | list["_JsonValue"] | dict[str, "_JsonValue"]


class _Sampler(Protocol):
    pass


class _Direction(Protocol):
    pass


class _FrozenTrial(Protocol):
    pass


class _RuntimeValue(Protocol):
    pass


class _Study(Protocol):
    study_name: str
    sampler: _Sampler
    directions: Sequence[_Direction]
    user_attrs: Mapping[str, _JsonValue]

    def get_trials(self, *, deepcopy: bool) -> Sequence[_FrozenTrial]: ...


class _PersistedStudy(_Study, Protocol):
    def add_trials(self, trials: Sequence[_FrozenTrial]) -> None: ...

    def set_user_attr(self, key: str, value: _JsonValue) -> None: ...


class _StudyNamespace(Protocol):
    Study: type[_Study]


class _SamplerNamespace(Protocol):
    BaseSampler: type[_Sampler]


class _GetAllStudyNames(Protocol):
    def __call__(self, *, storage: str) -> list[str]: ...


class _DeleteStudy(Protocol):
    def __call__(self, *, study_name: str, storage: str) -> None: ...


class _CreateStudy(Protocol):
    def __call__(
        self,
        *,
        study_name: str,
        storage: str,
        directions: Sequence[_Direction],
    ) -> _PersistedStudy: ...


class _LoadStudy(Protocol):
    def __call__(
        self,
        *,
        study_name: str,
        storage: str,
        sampler: _Sampler,
    ) -> _Study: ...


@final
class _OptunaApi:
    def __init__(self, module: ModuleType) -> None:
        self._module = module

    @property
    def study(self) -> _StudyNamespace:
        return cast(_StudyNamespace, getattr(self._module, "study"))

    @property
    def samplers(self) -> _SamplerNamespace:
        return cast(_SamplerNamespace, getattr(self._module, "samplers"))

    def get_all_study_names(self, *, storage: str) -> list[str]:
        function = cast(
            _GetAllStudyNames,
            getattr(self._module, "get_all_study_names"),
        )
        return function(storage=storage)

    def delete_study(self, *, study_name: str, storage: str) -> None:
        function = cast(_DeleteStudy, getattr(self._module, "delete_study"))
        function(study_name=study_name, storage=storage)

    def create_study(
        self,
        *,
        study_name: str,
        storage: str,
        directions: Sequence[_Direction],
    ) -> _PersistedStudy:
        function = cast(_CreateStudy, getattr(self._module, "create_study"))
        return function(
            study_name=study_name,
            storage=storage,
            directions=directions,
        )

    def load_study(
        self,
        *,
        study_name: str,
        storage: str,
        sampler: _Sampler,
    ) -> _Study:
        function = cast(_LoadStudy, getattr(self._module, "load_study"))
        return function(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
        )


class StudyAlreadyExistsError(ValueError):
    pass


def _load_optuna() -> _OptunaApi:
    try:
        return _OptunaApi(import_module("optuna"))
    except ModuleNotFoundError as exc:
        if exc.name != "optuna":
            raise
        raise PersistenceError(
            "Optuna is required to save or load this artifact; install MLPipelineHolder with 'pip install mlpipelineholder[optuna]'"
        ) from exc


def is_optuna_study(value: _RuntimeValue) -> TypeGuard[_Study]:
    try:
        optuna = _OptunaApi(import_module("optuna"))
    except ModuleNotFoundError as exc:
        if exc.name != "optuna":
            raise
        return False
    return isinstance(value, optuna.study.Study)


def is_optuna_sampler(value: _RuntimeValue) -> TypeGuard[_Sampler]:
    try:
        optuna = _OptunaApi(import_module("optuna"))
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
    study: _Study,
    db_path: str | Path,
    study_name: str | None = None,
    overwrite: bool = True,
) -> _PersistedStudy:
    optuna = _load_optuna()
    storage = _get_sqlite_storage_url(db_path)
    persisted_name = study.study_name if study_name is None else study_name
    directions = tuple(study.directions)
    trials = tuple(study.get_trials(deepcopy=True))
    user_attrs = dict(study.user_attrs)
    existing_studies = optuna.get_all_study_names(storage=storage)
    if persisted_name in existing_studies:
        if not overwrite:
            raise StudyAlreadyExistsError(
                f"Study '{persisted_name}' already exists in:\n{Path(db_path).expanduser().resolve()}"
            )
        optuna.delete_study(study_name=persisted_name, storage=storage)
    saved_study = optuna.create_study(
        study_name=persisted_name,
        storage=storage,
        directions=directions,
    )
    saved_study.add_trials(trials)
    for key, value in user_attrs.items():
        saved_study.set_user_attr(key, value)
    return saved_study


def save_study_artifact(
    study: _Study,
    sampler_path: Path,
    db_path: str | Path,
) -> dict[str, str]:
    optuna = _load_optuna()
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
        _ = save_study_to_db(
            study,
            db_path,
            study.study_name,
            overwrite=True,
        )
        os.replace(temporary_path, sampler_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return {
        "study_name": study.study_name,
        "db_path": str(Path(db_path).expanduser().resolve()),
    }


def load_study_artifact(
    sampler_path: Path,
    metadata: Mapping[str, str],
) -> _Study:
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
            sampler = cast(_Sampler, pickle.load(handle))
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
