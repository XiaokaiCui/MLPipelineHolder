from __future__ import annotations

from datetime import UTC, datetime
from io import StringIO, TextIOWrapper
from importlib import import_module
from pathlib import Path
from sys import stdout as sys_stdout
from threading import Lock
import traceback
from typing import Any


class PipelineLogger:
    """Small UTC logger that writes to disk and keeps in-memory RESULT history."""

    _LEVELS = {
        "DEBUG": 10,
        "INFO": 20,
        "PRINT": 30,
        "RESULT": 40,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50,
    }

    _VALID_LOG_LEVELS = frozenset(
        {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "RESULT", "PRINT"}
    )

    def __init__(
        self,
        log_file_path: str | Path,
        *,
        log_traceback_to_file: bool = True,
        show_traceback_locals: bool = False,
        use_rich_traceback_console: bool = True,
        truncate: bool = True,
    ) -> None:
        self.log_file_path = Path(log_file_path)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        if truncate and self.log_file_path.exists():
            self.log_file_path.unlink()
        self.log_file_path.touch()
        self._lock = Lock()
        self._result_history: list[str] = []
        self._min_level = self._LEVELS["DEBUG"]
        self._file_handle: TextIOWrapper | None = self.log_file_path.open("a", encoding="utf-8")
        self._file_logging_enabled = True
        self._file_logging_warning_printed = False
        self._pipeline: Any | None = None
        self._log_traceback_to_file = bool(log_traceback_to_file)
        self._show_traceback_locals = bool(show_traceback_locals)
        self._use_rich_traceback_console = bool(use_rich_traceback_console)

    def set_traceback_writing(self, enable: bool = True) -> None:
        """Enable or disable appending the traceback to the log file in log_exception()."""
        self._log_traceback_to_file = bool(enable)

    def set_show_traceback_locals(self, enable: bool = False) -> None:
        """Enable or disable showing local variables in Rich console tracebacks."""
        self._show_traceback_locals = bool(enable)

    def set_traceback_console_render(self, use_rich: bool = True) -> None:
        """Choose whether console tracebacks use Rich (default) or plain stdlib text."""
        self._use_rich_traceback_console = bool(use_rich)

    def get_traceback_settings(self) -> dict[str, bool]:
        """Return the current traceback settings as a dict of booleans."""
        return {
            "log_traceback_to_file": self._log_traceback_to_file,
            "show_traceback_locals": self._show_traceback_locals,
            "use_rich_traceback_console": self._use_rich_traceback_console,
        }

    def set_level(self, level: str) -> None:
        normalized = level.upper()
        if normalized not in self._LEVELS:
            raise ValueError(f"Unknown log level: {level}")
        self._min_level = self._LEVELS[normalized]

    def debug(self, message: str) -> None:
        self._write("DEBUG", message)

    def info(self, message: str) -> None:
        self._write("INFO", message)

    def warning(self, message: str) -> None:
        self._write("WARNING", message)

    def error(self, message: str) -> None:
        self._write("ERROR", message)

    def critical(self, message: str) -> None:
        self._write("CRITICAL", message)

    def log_exception(self, exc: BaseException, message: str | None = None) -> None:
        """Log an ERROR entry for `exc` with a rendered traceback.

        The log file always receives the plain stdlib traceback (never Rich
        formatting or ANSI codes). Writing the traceback body to the file can be
        disabled with `set_traceback_writing(False)`; the ERROR header line is
        still written.

        The console receives a Rich-rendered traceback by default (with local
        variables shown only when `set_show_traceback_locals(True)` was called),
        falling back to the plain stdlib traceback when Rich console rendering
        is disabled with `set_traceback_console_render(False)` or when Rich is
        not installed (Rich is an optional dependency).
        """
        timestamp = datetime.now(UTC).strftime("%H:%M:%S.%f")[:-3]
        header = f"{timestamp} ERROR {message if message is not None else str(exc)}"
        stdlib_text = "".join(traceback.format_exception(exc)).rstrip("\n")
        with self._lock:
            if self._file_logging_enabled and self._file_handle is not None:
                try:
                    self._file_handle.write(header + "\n")
                    if self._log_traceback_to_file:
                        self._file_handle.write(stdlib_text + "\n")
                    self._file_handle.flush()
                except OSError:
                    self._file_logging_enabled = False
                    self._close_file_handle()
                    if not self._file_logging_warning_printed:
                        self._file_logging_warning_printed = True
                        print(
                            "PipelineLogger file logging disabled after OSError while writing pipeline.log",
                            file=sys_stdout,
                        )
        if self._LEVELS["ERROR"] >= self._min_level:
            console_text = self._console_traceback_text(exc, stdlib_text)
            print(self._colorize("ERROR", header), file=sys_stdout)
            print(console_text, file=sys_stdout)

    def _console_traceback_text(self, exc: BaseException, stdlib_text: str) -> str:
        if not self._use_rich_traceback_console:
            return stdlib_text
        return self._render_rich_traceback(exc, stdlib_text)

    def _render_rich_traceback(self, exc: BaseException, stdlib_text: str) -> str:
        """Render a Rich console traceback, falling back to plain stdlib text.

        Rich is an optional dependency, so this degrades gracefully to the
        stdlib traceback when it is not installed.
        """
        try:
            from rich.console import Console
            from rich.traceback import Traceback
        except ImportError:
            return stdlib_text

        buffer = StringIO()
        console = Console(file=buffer, force_terminal=True, highlight=False)
        console.print(
            Traceback.from_exception(
                type(exc),
                exc,
                exc.__traceback__,
                show_locals=self._show_traceback_locals,
                locals_hide_dunder=True,
                locals_max_string=200,
                suppress=("concurrent.futures",),
            )
        )
        return buffer.getvalue().rstrip("\n")

    def result(self, message: str) -> None:
        self._write("RESULT", message)

    def print(self, message: str) -> None:
        self._write("PRINT", message)

    def get_result_history(self) -> list[str]:
        return list(self._result_history)

    def clear_result_history(self) -> None:
        with self._lock:
            self._result_history.clear()

    def flush(self) -> None:
        with self._lock:
            if self._file_handle is not None and not self._file_handle.closed:
                self._file_handle.flush()

    def close(self) -> None:
        with self._lock:
            self._close_file_handle()

    def disable_file_logging(self) -> None:
        with self._lock:
            self._file_logging_enabled = False
            self._close_file_handle()

    def enable_file_logging(self) -> None:
        with self._lock:
            if self._file_handle is None or self._file_handle.closed:
                self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
                self._file_handle = self.log_file_path.open("a", encoding="utf-8")
            self._file_logging_enabled = True
            self._file_logging_warning_printed = False

    def rebind_path(self, log_file_path: str | Path) -> None:
        with self._lock:
            self._close_file_handle()
            self.log_file_path = Path(log_file_path)
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_file_path.exists():
                self.log_file_path.touch()
            self._file_handle = self.log_file_path.open("a", encoding="utf-8")
            self._file_logging_enabled = True

    def _root_pipeline(self) -> Any | None:
        pipeline = self._pipeline
        while pipeline is not None and pipeline.parent_pipeline is not None:
            pipeline = pipeline.parent_pipeline
        return pipeline

    def _history_log_root(self) -> Path | None:
        root = self._root_pipeline()
        if root is None:
            return None
        return root.project_root / "history_logs"

    def _normalized_log_level(self, log_level: str | None) -> str | None:
        if log_level is None:
            return None
        normalized = log_level.upper()
        if normalized not in self._VALID_LOG_LEVELS:
            raise ValueError(f"Unknown log level: {log_level}")
        return normalized

    @staticmethod
    def _line_matches_level(entry: str, log_level: str | None) -> bool:
        if log_level is None:
            return True
        parts = entry.split(" ", 2)
        return len(parts) >= 2 and parts[1].upper() == log_level

    def show_recent_logs(self, lines: int = 5, log_level: str | None = None) -> None:
        """Print the latest `lines` entries of the current root pipeline log file.

        When `log_level` is given, only entries at that level are shown and the
        `lines` limit applies after the level filter.
        """
        if lines < 1:
            raise ValueError("lines must be >= 1")
        level = self._normalized_log_level(log_level)
        root = self._root_pipeline()
        log_path = root.logger.log_file_path if root is not None else self.log_file_path
        try:
            log_lines = log_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            log_lines = []
        matching = [line for line in log_lines if self._line_matches_level(line, level)]
        for entry in matching[-lines:]:
            print(entry)

    def list_history_logs(self) -> list[Path]:
        """Return the log snapshots saved in the root pipeline's history_logs folder."""
        history_root = self._history_log_root()
        if history_root is None or not history_root.is_dir():
            return []
        return sorted(history_root.glob("*.log"))

    def show_history_log(
        self,
        file_name: str | Path,
        line_starts: int = 0,
        line_ends: int | None = None,
        log_level: str | None = None,
    ) -> None:
        """Print lines [line_starts, line_ends) of a history log snapshot.

        When `log_level` is given, only entries at that level within the range are
        shown; the range always refers to file line numbers.
        """
        level = self._normalized_log_level(log_level)
        history_root = self._history_log_root()
        path = Path(file_name)
        if not path.is_absolute() and history_root is not None:
            path = history_root / path
        try:
            log_lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No history log found at '{path}'. Use list_history_logs() to see available snapshots."
            ) from None
        for entry in log_lines[line_starts:line_ends]:
            if self._line_matches_level(entry, level):
                print(entry)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def print_result_history(self) -> None:
        for entry in self._result_history:
            print(self._colorize("RESULT", entry))

    def _write(self, level: str, message: str, *, emit_console: bool = True) -> None:
        timestamp = datetime.now(UTC).strftime("%H:%M:%S.%f")[:-3]
        entry = f"{timestamp} {level} {message}"
        with self._lock:
            if self._file_logging_enabled and self._file_handle is not None:
                try:
                    self._file_handle.write(entry + "\n")
                    self._file_handle.flush()
                except OSError:
                    self._file_logging_enabled = False
                    self._close_file_handle()
                    if not self._file_logging_warning_printed:
                        self._file_logging_warning_printed = True
                        print(
                            "PipelineLogger file logging disabled after OSError while writing pipeline.log",
                            file=sys_stdout,
                        )
            if level == "RESULT":
                self._result_history.append(entry)
        if emit_console and self._LEVELS[level] >= self._min_level:
            console_text = message if level == "PRINT" else self._colorize(level, entry)
            print(console_text, file=sys_stdout)

    def _close_file_handle(self) -> None:
        if self._file_handle is not None and not self._file_handle.closed:
            self._file_handle.close()
        self._file_handle = None

    def _colorize(self, level: str, entry: str) -> str:
        color_map = {
            "DEBUG": "cyan",
            "INFO": "blue",
            "WARNING": "magenta",
            "ERROR": "red",
            "CRITICAL": "red",
            "RESULT": "green",
            "PRINT": "cyan",
        }
        return import_module("termcolor").colored(
            entry,
            color_map.get(level, "blue"),
            force_color=True,
        )
