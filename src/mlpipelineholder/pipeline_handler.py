from __future__ import annotations

import inspect
import pickle
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .artifact_store import ArtifactStore
from .exceptions import ExecutionError, PersistenceError, RegistrationError, ResolutionError
from .function_registry import default_map, resolve_callable
from .models import ArtifactRecord, FunctionRegistration, RunRecord


class PipelineHandler:
    def __init__(self, registration_name: str, configuration: Any, local_folder_path: str | Path) -> None:
        self.registration_name = registration_name
        self.config = configuration
        self.project_root = Path(local_folder_path)
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.blocks: list[Any] = []
        self.blocks_by_name: dict[str, Any] = {}
        self.para_value_dict: dict[str, Any] = {}
        self.artifact_registry: dict[str, ArtifactRecord] = {}
        self.produced_by_variable: dict[str, str] = {}
        self.run_history: list[RunRecord] = []
        self.artifact_store = ArtifactStore(self.project_root)
        self.metadata_root = self.project_root / "metadata"
        self.metadata_root.mkdir(parents=True, exist_ok=True)

    def add_block(self, registration_name: str, execution_priority: int):
        from .execution_block import ExecutionBlock

        block = ExecutionBlock(self, registration_name, execution_priority)
        self._register_block(block)
        return block

    def run_all(self, overrides: dict[str, Any] | None = None) -> RunRecord:
        return self._execute_blocks(self._sorted_blocks(), mode="run_all", overrides=overrides)

    def run_until(self, block_name: str, overrides: dict[str, Any] | None = None) -> RunRecord:
        block = self.blocks_by_name[block_name]
        selected = [candidate for candidate in self._sorted_blocks() if candidate.execution_priority <= block.execution_priority]
        return self._execute_blocks(selected, mode=f"run_until:{block_name}", overrides=overrides)

    def run_from(self, block_name: str, overrides: dict[str, Any] | None = None) -> RunRecord:
        block = self.blocks_by_name[block_name]
        self._invalidate_from_priority(block.execution_priority)
        selected = [candidate for candidate in self._sorted_blocks() if candidate.execution_priority >= block.execution_priority]
        return self._execute_blocks(selected, mode=f"run_from:{block_name}", overrides=overrides)

    def run_block(self, block_name: str, overrides: dict[str, Any] | None = None) -> RunRecord:
        block = self.blocks_by_name[block_name]
        self._invalidate_from_priority(block.execution_priority, include_target=False)
        return self._execute_blocks([block], mode=f"run_block:{block_name}", overrides=overrides)

    def save_project(self, path: str | Path) -> Path:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        self._persist_config_snapshot(target / "config.pkl")

        try:
            block_payload = []
            for block in self.blocks:
                function_payload = []
                for registration in block.functions:
                    if registration.import_path is None:
                        raise PersistenceError(
                            f"Function '{registration.function_name}' is not importable; save/load requires importable callables"
                        )
                    function_payload.append(
                        {
                            "import_path": registration.import_path,
                            "output_names": registration.output_names,
                            "save_to_disk": sorted(registration.save_to_disk),
                        }
                    )
                block_payload.append(
                    {
                        "registration_name": block.registration_name,
                        "execution_priority": block.execution_priority,
                        "functions": function_payload,
                    }
                )

            state_payload = {
                "registration_name": self.registration_name,
                "blocks": block_payload,
                "para_value_dict": self.para_value_dict,
                "artifact_registry": self.artifact_registry,
                "produced_by_variable": self.produced_by_variable,
                "run_history": self.run_history,
            }
            with (target / "pipeline_state.pkl").open("wb") as handle:
                pickle.dump(state_payload, handle)
        except Exception as exc:
            if isinstance(exc, PersistenceError):
                raise
            raise PersistenceError("Failed to save pipeline project") from exc
        return target

    @classmethod
    def load_project(cls, path: str | Path) -> "PipelineHandler":
        target = Path(path)
        try:
            with (target / "config.pkl").open("rb") as handle:
                config = pickle.load(handle)
            with (target / "pipeline_state.pkl").open("rb") as handle:
                state_payload = pickle.load(handle)
        except Exception as exc:
            raise PersistenceError("Failed to load pipeline project") from exc

        pipeline = cls(
            registration_name=state_payload["registration_name"],
            configuration=config,
            local_folder_path=target,
        )
        for block_data in state_payload["blocks"]:
            block = pipeline.add_block(block_data["registration_name"], block_data["execution_priority"])
            for function_data in block_data["functions"]:
                block.register_function(
                    function_data["import_path"],
                    function_data["output_names"],
                    function_data["save_to_disk"],
                )
        pipeline.para_value_dict = state_payload["para_value_dict"]
        pipeline.artifact_registry = state_payload["artifact_registry"]
        pipeline.produced_by_variable = state_payload["produced_by_variable"]
        pipeline.run_history = state_payload["run_history"]
        return pipeline

    def _execute_blocks(
        self,
        blocks: list[Any],
        mode: str,
        overrides: dict[str, Any] | None = None,
    ) -> RunRecord:
        run_id = uuid4().hex
        run_record = RunRecord(
            run_id=run_id,
            mode=mode,
            executed_blocks=[],
            started_at=datetime.now(UTC).isoformat(),
        )
        config_snapshot_path = self.metadata_root / f"config__{run_id}.pkl"
        run_record.config_snapshot_path = str(config_snapshot_path)
        self._persist_config_snapshot(config_snapshot_path)
        self.run_history.append(run_record)

        try:
            for block in blocks:
                produced_outputs = block.execute(run_id=run_id, overrides=overrides)
                run_record.executed_blocks.append(block.registration_name)
                run_record.produced_outputs.extend(produced_outputs)
            run_record.status = "success"
        except Exception as exc:
            run_record.status = "failed"
            run_record.error_message = str(exc)
            if isinstance(exc, (ExecutionError, ResolutionError)):
                raise
            raise ExecutionError("Pipeline execution failed") from exc
        finally:
            run_record.finished_at = datetime.now(UTC).isoformat()
        return run_record

    def _persist_config_snapshot(self, path: Path) -> None:
        with path.open("wb") as handle:
            pickle.dump(self.config, handle)

    def _sorted_blocks(self) -> list[Any]:
        return sorted(self.blocks, key=lambda block: (block.execution_priority, block.registration_name))

    def _register_block(self, block: Any) -> None:
        if block.registration_name in self.blocks_by_name:
            existing = self.blocks_by_name[block.registration_name]
            if existing is not block:
                raise RegistrationError(f"Block already registered: {block.registration_name}")
            return
        self.blocks.append(block)
        self.blocks_by_name[block.registration_name] = block

    def _validate_new_outputs(self, block: Any, registration: FunctionRegistration) -> None:
        existing_outputs = {
            output_name: candidate.registration_name
            for candidate in self.blocks
            for function in candidate.functions
            for output_name in function.output_names
        }
        for output_name in registration.output_names:
            existing_block_name = existing_outputs.get(output_name)
            if existing_block_name and existing_block_name != block.registration_name:
                raise RegistrationError(
                    f"Output '{output_name}' is already registered by block '{existing_block_name}'"
                )

    def _resolve_arguments(
        self,
        registration: FunctionRegistration,
        overrides: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        defaults = default_map(registration.callable_obj)
        resolved: dict[str, Any] = {}
        loaded_artifacts: list[str] = []

        for input_name in registration.input_names:
            if input_name in overrides:
                value = overrides[input_name]
            elif input_name in self.para_value_dict:
                value = self.para_value_dict[input_name]
            elif self._config_has_field(input_name):
                value = self._config_value(input_name)
            elif input_name in defaults:
                value = defaults[input_name]
            else:
                raise ResolutionError(
                    f"Cannot resolve argument '{input_name}' for function '{registration.function_name}'"
                )

            if isinstance(value, ArtifactRecord):
                value = self.artifact_store.load(value)
                loaded_artifacts.append(input_name)

            resolved[input_name] = value
        return resolved, loaded_artifacts

    def _config_has_field(self, field_name: str) -> bool:
        if is_dataclass(self.config) and not isinstance(self.config, type):
            return any(field.name == field_name for field in self.config.__dataclass_fields__.values())
        if isinstance(self.config, dict):
            return field_name in self.config
        return hasattr(self.config, field_name)

    def _config_value(self, field_name: str) -> Any:
        if is_dataclass(self.config) and not isinstance(self.config, type):
            return getattr(self.config, field_name)
        if isinstance(self.config, dict):
            return self.config[field_name]
        return getattr(self.config, field_name)

    def _invalidate_from_priority(self, priority: int, include_target: bool = True) -> None:
        for block in self._sorted_blocks():
            if block.execution_priority < priority:
                continue
            if block.execution_priority == priority and not include_target:
                continue
            for registration in block.functions:
                for output_name in registration.output_names:
                    existing_value = self.para_value_dict.pop(output_name, None)
                    artifact = self.artifact_registry.pop(output_name, None)
                    self.produced_by_variable.pop(output_name, None)
                    if isinstance(existing_value, ArtifactRecord):
                        artifact = existing_value
                    if artifact is not None:
                        self.artifact_store.delete(artifact)

    def config_as_dict(self) -> dict[str, Any]:
        if is_dataclass(self.config) and not isinstance(self.config, type):
            return asdict(self.config)
        if isinstance(self.config, dict):
            return dict(self.config)
        if hasattr(self.config, "__dict__"):
            return dict(vars(self.config))
        raise PersistenceError("Configuration object is not serializable to dict")
