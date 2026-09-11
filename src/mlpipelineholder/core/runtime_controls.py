"""Runtime controls: logger level, print capture, strict mode, invalidation toggles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..exceptions import RegistrationError


class RuntimeControlsMixin:
    """Runtime behaviour toggles shared by the PipelineHolder facade."""

    if TYPE_CHECKING:
        logger: Any = None
        parent_pipeline: Any = None
        _invalidation_forbidden: bool = False
        print_capture_mode: str = "tee"
        torch_load_weights_only: bool = False

        def _iter_attached_pipelines(self) -> list[Any]: ...

    def set_print_capture_mode(self, mode: str) -> None:
        if mode not in {"tee", "logger_only", "off"}:
            raise RegistrationError("print capture mode must be one of: tee, logger_only, off")
        self.print_capture_mode = mode

    def set_log_level(self, level: str) -> None:
        try:
            self.logger.set_level(level)
        except ValueError as exc:
            raise RegistrationError(str(exc)) from exc

    def set_torch_load_weights_only(self, enabled: bool) -> None:
        """Set whether torch artifacts saved by this pipeline load with weights_only=True."""
        self.torch_load_weights_only = bool(enabled)

    def set_strict_mode(self, enabled: bool) -> None:
        """Enable or disable strict-mode registration validation for this pipeline and all attached descendants."""
        for pipeline in self._iter_attached_pipelines():
            pipeline.strict_mode = bool(enabled)

    def _sync_invalidation_flag(self) -> None:
        """Copy this pipeline's invalidation flag to its whole attached subtree."""
        flag = self._invalidation_forbidden
        for pipeline in self._iter_attached_pipelines():
            pipeline._invalidation_forbidden = flag

    def forbid_invalidate_objects(self) -> None:
        """Suppress cascade invalidation after changes across the pipeline tree.

        Forced re-registrations anywhere from the root down to every descendant
        still erase the changed node's own outputs, but stop invalidating other
        upstream or downstream objects. Those retained values may be stale or
        inconsistent until this mode is lifted. The state is not persisted, only
        the root pipeline may toggle it, and every pipeline added later inherits
        the current state automatically; call `allow_invalidate_objects()` to
        restore normal behaviour.
        """
        if self.parent_pipeline is not None:
            raise RegistrationError(
                "forbid_invalidate_objects() must be called on the root pipeline"
            )
        if self._invalidation_forbidden:
            return
        self._invalidation_forbidden = True
        self._sync_invalidation_flag()
        self.logger.warning(
            "Object invalidation is now FORBIDDEN on the whole pipeline tree: forced "
            "re-registrations and structural changes will still erase each changed "
            "node's own outputs but will not invalidate other upstream or downstream "
            "outputs, so stale or inconsistent values may survive silently; call "
            "allow_invalidate_objects() to restore normal cascade invalidation"
        )

    def allow_invalidate_objects(self) -> None:
        """Restore normal erasure behaviour across the whole pipeline tree.

        Forced re-registrations and structural changes will invalidate affected
        upstream or downstream outputs again, which may discard previously
        computed results and require re-running blocks. Only the root pipeline
        may toggle this state; every pipeline already in the tree syncs
        immediately and later additions inherit the current state. Call
        `forbid_invalidate_objects()` to suppress cascade invalidation again.
        """
        if self.parent_pipeline is not None:
            raise RegistrationError(
                "allow_invalidate_objects() must be called on the root pipeline"
            )
        if not self._invalidation_forbidden:
            return
        self._invalidation_forbidden = False
        self._sync_invalidation_flag()
        self.logger.warning(
            "Object invalidation is now ALLOWED on the whole pipeline tree: forced "
            "re-registrations and structural changes will invalidate affected upstream "
            "or downstream outputs again and may discard previously computed results; "
            "call forbid_invalidate_objects() to suppress cascade invalidation again"
        )
