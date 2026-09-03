from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from .exceptions import PersistenceError
from .models import ArtifactRecord
from .optuna_support import (
    OPTUNA_STUDIES_DB_NAME,
    OPTUNA_STUDY_SERIALIZER,
    is_optuna_study,
    load_study_artifact,
    save_study_artifact,
)
from .serializers import choose_serializer, dump_value, extension_for, load_value


class ArtifactStore:
    """Stores disk-backed pipeline outputs under the project artifact tree."""

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
        torch_load_weights_only: bool = False,
        optuna_db_path: str | Path | None = None,
    ) -> ArtifactRecord:
        if is_optuna_study(value):
            return self._save_optuna_study(
                variable_name,
                value,
                block_name,
                function_name,
                run_id,
                optuna_db_path,
            )
        serializer = choose_serializer(value)
        suffix = extension_for(serializer)
        safe_block = block_name.replace("/", "_")
        safe_function = function_name.replace("/", "_")
        safe_variable = variable_name.replace("/", "_")
        artifact_path = (
            self.artifact_root
            / safe_block
            / f"{safe_function}__{safe_variable}__{run_id}__{uuid4().hex}{suffix}"
        )
        dump_value(value, serializer, artifact_path)
        return ArtifactRecord(
            variable_name=variable_name,
            serializer=serializer,
            file_path=str(artifact_path),
            produced_by_block=block_name,
            produced_by_function=function_name,
            run_id=run_id,
            torch_load_weights_only=torch_load_weights_only,
        )

    def _save_optuna_study(
        self,
        variable_name: str,
        value: Any,
        block_name: str,
        function_name: str,
        run_id: str,
        optuna_db_path: str | Path | None,
    ) -> ArtifactRecord:
        safe_block = block_name.replace("/", "_")
        safe_function = function_name.replace("/", "_")
        safe_variable = variable_name.replace("/", "_")
        sampler_path = (
            self.artifact_root
            / safe_block
            / f"{safe_function}__{safe_variable}__{run_id}__{uuid4().hex}.pkl"
        )
        db_path = (
            self.project_root / OPTUNA_STUDIES_DB_NAME
            if optuna_db_path is None
            else Path(optuna_db_path)
        )
        metadata = save_study_artifact(value, sampler_path, db_path)
        return ArtifactRecord(
            variable_name=variable_name,
            serializer=OPTUNA_STUDY_SERIALIZER,
            file_path=str(sampler_path),
            produced_by_block=block_name,
            produced_by_function=function_name,
            run_id=run_id,
            metadata=metadata,
        )

    def load(self, artifact: ArtifactRecord) -> Any:
        if artifact.serializer == OPTUNA_STUDY_SERIALIZER:
            return load_study_artifact(
                Path(artifact.file_path),
                getattr(artifact, "metadata", {}),
            )
        return load_value(
            artifact.serializer,
            Path(artifact.file_path),
            torch_weights_only=getattr(artifact, "torch_load_weights_only", False),
        )

    def _assert_managed_artifact_path(self, path: Path) -> Path:
        resolved = path.resolve()
        artifact_root = self.artifact_root.resolve()
        if artifact_root not in resolved.parents:
            raise PersistenceError(
                f"Refusing to operate on artifact outside artifact root: {resolved}"
            )
        return resolved

    def delete(self, artifact: ArtifactRecord) -> None:
        path = Path(artifact.file_path).resolve()
        if not path.exists():
            return
        managed_path = self._assert_managed_artifact_path(path)
        if managed_path.is_dir():
            shutil.rmtree(managed_path)
        else:
            managed_path.unlink()

    def transfer(
        self,
        artifact: ArtifactRecord,
        block_name: str,
    ) -> ArtifactRecord:
        source = self._assert_managed_artifact_path(Path(artifact.file_path))
        if not source.is_file():
            raise PersistenceError(
                f"Cannot transfer missing artifact: {source}"
            )
        safe_block = block_name.replace("/", "_")
        destination_dir = self.artifact_root / safe_block
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"promoted__{uuid4().hex}{source.suffix}"
        os.replace(source, destination)
        return ArtifactRecord(
            variable_name=artifact.variable_name,
            serializer=artifact.serializer,
            file_path=str(destination),
            produced_by_block=block_name,
            produced_by_function=artifact.produced_by_function,
            run_id=artifact.run_id,
            created_at=artifact.created_at,
            torch_load_weights_only=artifact.torch_load_weights_only,
            metadata=dict(getattr(artifact, "metadata", {})),
        )
