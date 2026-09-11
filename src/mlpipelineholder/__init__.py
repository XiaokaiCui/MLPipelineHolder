from datetime import date, datetime, time
from importlib.metadata import version as distribution_version
from pathlib import Path
from tomllib import load
from typing import Final, TypeAlias

from .execution.atom_pipeline import AtomPipeline
from .execution.block import ExecutionBlock
from .exceptions import (
    ExecutionError,
    PersistenceError,
    PipelineError,
    RegistrationError,
    ResolutionError,
)
from .execution.function_registry import rename_args
from .execution.gate_block import GateBlock
from .presentation.logger import PipelineLogger
from .pipeline_holder import PipelineHandler, PipelineHolder

_TomlValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | date
    | datetime
    | time
    | list["_TomlValue"]
    | dict[str, "_TomlValue"]
)


def _read_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject_path.is_file():
        return distribution_version("mlpipelineholder")
    with pyproject_path.open("rb") as pyproject_file:
        pyproject: dict[str, _TomlValue] = load(pyproject_file)
    project = pyproject.get("project")
    if not isinstance(project, dict):
        raise TypeError("project metadata must be a table")
    version = project.get("version")
    if not isinstance(version, str):
        raise TypeError("project.version must be a string")
    return version


__version__: Final = _read_version()

__all__ = [
    "AtomPipeline",
    "ExecutionBlock",
    "ExecutionError",
    "GateBlock",
    "PipelineLogger",
    "PersistenceError",
    "PipelineError",
    "PipelineHandler",
    "PipelineHolder",
    "RegistrationError",
    "ResolutionError",
    "__version__",
    "rename_args",
]
