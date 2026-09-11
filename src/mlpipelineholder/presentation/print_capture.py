"""Print-capture helpers: tee stdout into pipeline logs."""

from __future__ import annotations

import sys
from contextlib import redirect_stdout
from io import StringIO
from typing import TYPE_CHECKING, Any


class _TeeStdout:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            flush = getattr(stream, "flush", None)
            if callable(flush):
                flush()


class PrintCaptureMixin:
    """Capture prints made by user functions into the pipeline log."""

    if TYPE_CHECKING:
        print_capture_mode: str = "tee"
        logger: Any = None

    def _capture_prints(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        if self.print_capture_mode == "off":
            return func(*args, **kwargs)

        # Convenience feature only: redirect_stdout is process-level state, so heavily
        # parallel print-heavy functions can still interleave output. Explicit logger usage
        # remains the safer option for important structured messages.
        buffer = StringIO()
        stdout_target: Any = (
            _TeeStdout(sys.stdout, buffer) if self.print_capture_mode == "tee" else buffer
        )
        try:
            with redirect_stdout(stdout_target):
                result = func(*args, **kwargs)
        finally:
            # Flush even when the function raises, so prints made before the
            # exception are still recorded in the pipeline log.
            self._flush_captured_prints(buffer)
        return result

    def _flush_captured_prints(self, buffer: StringIO) -> None:
        captured = buffer.getvalue()
        if not captured:
            return
        for line in captured.splitlines():
            if line:
                self.logger._write("PRINT", line, emit_console=False)
