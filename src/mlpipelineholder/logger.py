from __future__ import annotations

from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from sys import stdout as sys_stdout
from threading import Lock


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

    def __init__(self, log_file_path: str | Path) -> None:
        self.log_file_path = Path(log_file_path)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        if self.log_file_path.exists():
            self.log_file_path.unlink()
        self.log_file_path.touch()
        self._lock = Lock()
        self._result_history: list[str] = []
        self._min_level = self._LEVELS["DEBUG"]

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

    def result(self, message: str) -> None:
        self._write("RESULT", message)

    def print(self, message: str) -> None:
        self._write("PRINT", message)

    def get_result_history(self) -> list[str]:
        return list(self._result_history)

    def clear_result_history(self) -> None:
        with self._lock:
            self._result_history.clear()

    def print_result_history(self) -> None:
        for entry in self._result_history:
            print(self._colorize("RESULT", entry))

    def _write(self, level: str, message: str, *, emit_console: bool = True) -> None:
        timestamp = datetime.now(UTC).strftime("%H:%M:%S.%f")[:-3]
        entry = f"{timestamp} {level} {message}"
        with self._lock:
            with self.log_file_path.open("a", encoding="utf-8") as handle:
                handle.write(entry + "\n")
            if level == "RESULT":
                self._result_history.append(entry)
        if emit_console and self._LEVELS[level] >= self._min_level:
            console_text = message if level == "PRINT" else self._colorize(level, entry)
            print(console_text, file=sys_stdout)

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
