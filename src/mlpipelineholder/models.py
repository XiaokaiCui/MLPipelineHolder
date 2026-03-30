from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ArtifactRecord:
    variable_name: str
    serializer: str
    file_path: str
    produced_by_block: str
    produced_by_function: str
    run_id: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def path(self) -> Path:
        return Path(self.file_path)


@dataclass(slots=True)
class FunctionRegistration:
    function_name: str
    import_path: str | None
    callable_obj: Any
    input_names: list[str]
    output_names: list[str]
    save_to_disk: set[str]


@dataclass(slots=True)
class FunctionExecutionResult:
    function_name: str
    outputs: dict[str, Any]
    loaded_artifact_inputs: list[str]


@dataclass(slots=True)
class RunRecord:
    run_id: str
    mode: str
    executed_blocks: list[str]
    started_at: str
    finished_at: str | None = None
    status: str = "running"
    error_message: str | None = None
    config_snapshot_path: str | None = None
    produced_outputs: list[str] = field(default_factory=list)
