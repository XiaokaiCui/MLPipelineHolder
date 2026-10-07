from __future__ import annotations

import math
import numbers
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from ..exceptions import RegistrationError


_WINDOWS_RESERVED_NAMES = (
    {"CON", "PRN", "AUX", "NUL", "CLOCK$", "CONIN$", "CONOUT$"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


def validate_registration_name(name: Any, *, owner_label: str) -> str:
    """Require ``name`` to be exactly one safe path component.

    Registration names become filesystem path components (``children/<name>``,
    artifact directories, and log/metadata trees), so a name that can escape a
    directory or is invalid on a supported platform must be rejected before
    any path is constructed or any filesystem operation runs.
    """
    if not isinstance(name, str):
        raise RegistrationError(
            f"{owner_label.capitalize()} registration name must be a string, "
            f"got {type(name).__name__}"
        )
    if name == "" or name in (".", ".."):
        raise RegistrationError(
            f"{owner_label.capitalize()} registration name {name!r} must be a "
            "non-empty single path component"
        )
    if any(ord(character) < 32 for character in name) or any(
        character in '<>:"|?*/\\' for character in name
    ):
        raise RegistrationError(
            f"{owner_label.capitalize()} registration name {name!r} must not "
            "contain characters forbidden in filesystem components"
        )
    if PurePosixPath(name).is_absolute() or PureWindowsPath(name).is_absolute():
        raise RegistrationError(
            f"{owner_label.capitalize()} registration name {name!r} must not "
            "be an absolute path"
        )
    if name.endswith((".", " ")):
        raise RegistrationError(
            f"{owner_label.capitalize()} registration name {name!r} must not "
            "end with a dot or space"
        )
    if name.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise RegistrationError(
            f"{owner_label.capitalize()} registration name {name!r} is "
            "reserved by Windows"
        )
    return name


def validate_execution_priority(
    execution_priority: Any,
    *,
    owner_label: str,
    allow_none: bool = True,
) -> None:
    """Require a finite, non-negative number as an execution priority.

    ``None`` is the standalone-root default and is allowed only where a pipeline
    declares its own unspecified priority (``allow_none=True``, the default used
    by root construction). Registered nodes (blocks, child pipelines, and atoms)
    always sort and compare their priority against numbers, so callers that
    register a node pass ``allow_none=False`` to reject ``None`` before any state
    changes. Negative and non-finite priorities are rejected because priority
    groups are formed with ``int()``, which truncates toward zero and would merge
    negative and non-negative nodes into the same group.
    """
    if execution_priority is None:
        if allow_none:
            return
        raise RegistrationError(
            f"{owner_label.capitalize()} execution priority must be a finite "
            f"non-negative number, got NoneType"
        )
    if isinstance(execution_priority, bool) or not isinstance(
        execution_priority, numbers.Real
    ):
        raise RegistrationError(
            f"{owner_label.capitalize()} execution priority must be a finite "
            f"non-negative number, got {type(execution_priority).__name__}"
        )
    if not math.isfinite(float(execution_priority)) or execution_priority < 0:
        raise RegistrationError(
            f"{owner_label.capitalize()} execution priority must be a finite "
            f"non-negative number, got {execution_priority!r}"
        )
