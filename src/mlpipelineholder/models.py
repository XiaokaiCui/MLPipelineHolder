from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ArtifactRecord:
    """Metadata for one disk-backed pipeline value."""

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
    """Captured registration metadata used for execution, charting, and persistence."""

    function_name: str
    import_path: str | None
    callable_obj: Any
    input_names: list[str]
    output_names: list[str]
    save_to_disk: set[str]
    param_mapping: dict[str, str] = field(default_factory=dict)
    var_pos_name: str | None = None
    var_kw_name: str | None = None


@dataclass(slots=True)
class BlockArgsRegistration:
    """Block-scoped ordered items used to build *args for specific functions."""

    name: str
    ordered_items: list[str]


@dataclass(slots=True)
class BlockKwargsRegistration:
    """Block-scoped name mapping used to build **kwargs for specific functions."""

    name: str
    mapping_dct: dict[str, str]


@dataclass(slots=True)
class FunctionExecutionResult:
    """Normalized result of one registered function invocation."""

    function_name: str
    outputs: dict[str, Any]
    loaded_artifact_inputs: list[str]


@dataclass(slots=True)
class RunRecord:
    """Execution summary for one pipeline or sub-pipeline run."""

    run_id: str
    mode: str
    executed_blocks: list[str]
    started_at: str
    finished_at: str | None = None
    status: str = "running"
    error_message: str | None = None
    config_snapshot_path: str | None = None
    produced_outputs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RuntimeValueReference:
    """Fallback reference used when a runtime value cannot be safely persisted."""

    type_name: str
    repr_text: str
    reason: str


@dataclass(slots=True)
class TorchStateArtifactRecord:
    """Artifact metadata for torch objects restored from state-dict style persistence."""

    variable_name: str
    file_path: str
    object_kind: str
    metadata: dict[str, Any] = field(default_factory=dict)
