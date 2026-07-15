from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
from typing import TYPE_CHECKING, Any

from .exceptions import ExecutionError, RegistrationError, ResolutionError
from .function_registry import inspect_input_names, rename_args, resolve_callable
from .models import FunctionExecutionResult, FunctionRegistration

if TYPE_CHECKING:
    from .pipeline_handler import PipelineHandler


class ExecutionBlock:
    """Represents one priority level whose registered functions run in parallel."""

    def __init__(
        self, parent: PipelineHandler, registration_name: str, execution_priority: int
    ) -> None:
        self.parent = parent
        self.registration_name = registration_name
        self.execution_priority = execution_priority
        self.functions: list[FunctionRegistration] = []

    def register_function(
        self,
        function_or_path: Any,
        output_variable_names: str | list[str] | tuple[str, ...],
        save_to_disk: list[str] | tuple[str, ...] | set[str] | None = None,
        kw_mapping: dict[str, str] | None = None,
        pos_mapping: dict[int, int] | None = None,
        var_pos_name: str | None = None,
        var_kw_name: str | None = None,
    ) -> FunctionRegistration:
        if pos_mapping:
            raise RegistrationError("pos_mapping is not supported")
        output_names = (
            [output_variable_names]
            if isinstance(output_variable_names, str)
            else list(output_variable_names)
        )
        if not output_names:
            raise RegistrationError("At least one output variable name is required")
        if len(set(output_names)) != len(output_names):
            raise RegistrationError("Duplicate output variable names are not allowed")

        existing_local_outputs = {
            output_name
            for registration in self.functions
            for output_name in registration.output_names
        }
        overlap = existing_local_outputs.intersection(output_names)
        if overlap:
            raise RegistrationError(
                f"Duplicate output names inside block '{self.registration_name}': {sorted(overlap)}"
            )

        disk_names = set(save_to_disk or [])
        if not disk_names.issubset(set(output_names)):
            raise RegistrationError(
                "Disk-saved output names must be a subset of output variable names"
            )

        callable_obj, import_path, function_name = resolve_callable(function_or_path)
        signature = inspect.signature(callable_obj)
        if (
            any(
                parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                for parameter in signature.parameters.values()
            )
            or kw_mapping
            or var_pos_name
            or var_kw_name
        ):
            callable_obj = rename_args(
                callable_obj,
                kw_mapping=kw_mapping,
                var_pos_name=var_pos_name,
                var_kw_name=var_kw_name,
            )
        input_names = inspect_input_names(callable_obj)
        conflicting_inputs = disk_names.intersection(set(input_names))
        if conflicting_inputs:
            raise RegistrationError(
                f"Output names cannot overlap input names within the same function: {sorted(conflicting_inputs)}"
            )

        registration = FunctionRegistration(
            function_name=function_name,
            import_path=import_path,
            callable_obj=callable_obj,
            input_names=input_names,
            output_names=output_names,
            save_to_disk=disk_names,
            kw_mapping=dict(kw_mapping or {}),
            var_pos_name=var_pos_name,
            var_kw_name=var_kw_name,
        )
        self.functions.append(registration)
        self.parent._register_node(self)
        return registration

    def remove_function(self, function_name: str) -> None:
        matches = [
            registration
            for registration in self.functions
            if registration.function_name == function_name
        ]
        if not matches:
            raise RegistrationError(
                f"Function not registered in block '{self.registration_name}': {function_name}"
            )
        if len(matches) > 1:
            raise RegistrationError(
                f"Multiple functions named '{function_name}' exist in block '{self.registration_name}'"
            )

        self.functions.remove(matches[0])
        self.parent._invalidate_from_priority(self.execution_priority)

    def declared_outputs(self) -> set[str]:
        return {
            output_name
            for registration in self.functions
            for output_name in registration.output_names
        }

    def execute(
        self,
        run_id: str,
        visible_outputs: dict[str, Any],
        overrides: dict[str, Any] | None = None,
        parent_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.functions:
            return {}

        block_output_names = self.declared_outputs()
        for registration in self.functions:
            same_block_dependencies = block_output_names.difference(
                registration.output_names
            ).intersection(registration.input_names)
            if same_block_dependencies:
                raise ExecutionError(
                    f"Function '{registration.function_name}' depends on outputs from the same block, "
                    f"which cannot be resolved during parallel execution: {sorted(same_block_dependencies)}"
                )

        futures = []
        with ThreadPoolExecutor(max_workers=len(self.functions)) as executor:
            for registration in self.functions:
                futures.append(
                    executor.submit(
                        self._execute_function,
                        registration,
                        run_id,
                        dict(visible_outputs),
                        overrides or {},
                        parent_config or {},
                    )
                )

        produced_outputs: dict[str, Any] = {}
        for future in futures:
            result = future.result()
            for output_name, output_value in result.outputs.items():
                if output_name in result.outputs and output_name in produced_outputs:
                    raise ExecutionError(
                        f"Duplicate output '{output_name}' produced inside block '{self.registration_name}'"
                    )
                if output_name in self.functions_output_disk_names():
                    output_value = self.parent.artifact_store.save(
                        variable_name=output_name,
                        value=output_value,
                        block_name=self.parent.qualified_node_name(self.registration_name),
                        function_name=result.function_name,
                        run_id=run_id,
                    )
                produced_outputs[output_name] = output_value
        return produced_outputs

    def functions_output_disk_names(self) -> set[str]:
        output_names: set[str] = set()
        for registration in self.functions:
            output_names.update(registration.save_to_disk)
        return output_names

    def _execute_function(
        self,
        registration: FunctionRegistration,
        run_id: str,
        visible_outputs: dict[str, Any],
        overrides: dict[str, Any],
        parent_config: dict[str, Any],
    ) -> FunctionExecutionResult:
        del run_id
        positional_args, keyword_args, loaded_artifacts = self.parent._prepare_call_arguments(
            registration,
            overrides,
            visible_outputs,
            parent_config,
        )
        try:
            result = self.parent._capture_prints(
                registration.callable_obj,
                *positional_args,
                **keyword_args,
            )
        except ResolutionError:
            raise
        except Exception as exc:
            raise ExecutionError(
                f"Function '{registration.function_name}' in block '{self.registration_name}' failed"
            ) from exc

        outputs = self._normalize_outputs(registration, result)
        return FunctionExecutionResult(
            function_name=registration.function_name,
            outputs=outputs,
            loaded_artifact_inputs=loaded_artifacts,
        )

    @staticmethod
    def _normalize_outputs(registration: FunctionRegistration, result: Any) -> dict[str, Any]:
        if len(registration.output_names) == 1:
            return {registration.output_names[0]: result}

        if not isinstance(result, (tuple, list)):
            raise ExecutionError(
                f"Function '{registration.function_name}' must return tuple/list matching declared outputs"
            )
        if len(result) != len(registration.output_names):
            raise ExecutionError(
                f"Function '{registration.function_name}' returned {len(result)} values but "
                f"{len(registration.output_names)} outputs were declared"
            )
        return dict(zip(registration.output_names, result, strict=True))
