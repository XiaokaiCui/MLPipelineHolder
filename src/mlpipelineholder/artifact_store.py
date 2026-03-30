from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import ArtifactRecord
from .serializers import choose_serializer, dump_value, extension_for, load_value


class ArtifactStore:
    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root)
        self.artifact_root = self.project_root / "artifacts"
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        variable_name: str,
        value: Any,
        block_name: str,
        function_name: str,
        run_id: str,
    ) -> ArtifactRecord:
        serializer = choose_serializer(value)
        suffix = extension_for(serializer)
        safe_block = block_name.replace("/", "_")
        safe_function = function_name.replace("/", "_")
        safe_variable = variable_name.replace("/", "_")
        artifact_path = self.artifact_root / safe_block / f"{safe_function}__{safe_variable}__{run_id}__{uuid4().hex}{suffix}"
        dump_value(value, serializer, artifact_path)
        return ArtifactRecord(
            variable_name=variable_name,
            serializer=serializer,
            file_path=str(artifact_path),
            produced_by_block=block_name,
            produced_by_function=function_name,
            run_id=run_id,
        )

    def load(self, artifact: ArtifactRecord) -> Any:
        return load_value(artifact.serializer, Path(artifact.file_path))

    def delete(self, artifact: ArtifactRecord) -> None:
        path = Path(artifact.file_path)
        if path.exists():
            path.unlink()
