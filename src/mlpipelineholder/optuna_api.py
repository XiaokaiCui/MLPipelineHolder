from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol, TypeAlias, cast, final


_JsonScalar: TypeAlias = str | int | float | bool | None
_JsonValue: TypeAlias = _JsonScalar | list["_JsonValue"] | dict[str, "_JsonValue"]


class OptunaSampler(Protocol):
    pass


class _Direction(Protocol):
    pass


class _FrozenTrial(Protocol):
    pass


class OptunaRuntimeValue(Protocol):
    pass


class OptunaStudy(Protocol):
    study_name: str
    sampler: OptunaSampler
    directions: Sequence[_Direction]
    user_attrs: Mapping[str, _JsonValue]

    def get_trials(self, *, deepcopy: bool) -> Sequence[_FrozenTrial]: ...


class PersistedOptunaStudy(OptunaStudy, Protocol):
    def add_trials(self, trials: Sequence[_FrozenTrial]) -> None: ...

    def set_user_attr(self, key: str, value: _JsonValue) -> None: ...


class _StudyNamespace(Protocol):
    Study: type[OptunaStudy]


class _SamplerNamespace(Protocol):
    BaseSampler: type[OptunaSampler]


class _ExceptionNamespace(Protocol):
    DuplicatedStudyError: type[Exception]


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
    ) -> PersistedOptunaStudy: ...


class _LoadStudy(Protocol):
    def __call__(
        self,
        *,
        study_name: str,
        storage: str,
        sampler: OptunaSampler,
    ) -> OptunaStudy: ...


@dataclass(frozen=True, slots=True)
class StudyArtifactOptions:
    independent_copy: bool = False
    allocate_if_occupied: bool = False
    study_name: str | None = None
    original_study_name: str | None = None
    owner_kind: str | None = None
    owner_key: str | None = None


@final
class OptunaApi:
    def __init__(self, module: ModuleType) -> None:
        self._module = module

    @property
    def study(self) -> _StudyNamespace:
        return cast(_StudyNamespace, getattr(self._module, "study"))

    @property
    def samplers(self) -> _SamplerNamespace:
        return cast(_SamplerNamespace, getattr(self._module, "samplers"))

    @property
    def duplicated_study_error(self) -> type[Exception]:
        exceptions = cast(_ExceptionNamespace, getattr(self._module, "exceptions"))
        return exceptions.DuplicatedStudyError

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
    ) -> PersistedOptunaStudy:
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
        sampler: OptunaSampler,
    ) -> OptunaStudy:
        function = cast(_LoadStudy, getattr(self._module, "load_study"))
        return function(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
        )
