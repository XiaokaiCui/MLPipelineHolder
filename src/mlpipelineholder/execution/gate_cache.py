"""Memo of per-level gate evaluations for one placeholder-recovery pass."""

from __future__ import annotations

from typing import Any


class _GateStatusCache:
    """Memo of per-level gate evaluations for one placeholder-recovery pass.

    Each gate-owning pipeline retains a snapshot of every value its gate can
    read (incoming parent outputs, config fields, and manual values), so a
    cached answer is reused only while those inputs remain deeply equal. The
    cache is short-lived: it is created at the entry point of one recovery
    pass and discarded when the pass completes.
    """

    __slots__ = ("_levels",)

    def __init__(self) -> None:
        self._levels: dict[
            int,
            tuple[tuple[tuple[str, Any], ...], tuple[str, str | None]],
        ] = {}
