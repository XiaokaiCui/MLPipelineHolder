"""Shared package-level sentinels and constants."""

from __future__ import annotations

import builtins

_MISSING = object()
_IMMUTABLE_TYPES = (int, float, complex, bool, str, bytes, type(None))

_RESERVED_BUILTIN_NAMES = {
    name
    for name in dir(builtins)
    if not name.startswith("_")
}
