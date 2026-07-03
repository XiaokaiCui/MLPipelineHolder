from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
from typing import TYPE_CHECKING, Any

from .exceptions import ExecutionError, RegistrationError, ResolutionError
from .function_registry import infer_declared_output_count, inspect_input_names, rename_args, resolve_callable
from .function_registry import inspect_exposed_input_names
from .models import (
    BlockArgsRegistration,
    BlockKwargsRegistration,
    FunctionExecutionResult,
    FunctionRegistration,
)

if TYPE_CHECKING:
    from .pipeline_handler import PipelineHandler


class ExecutionBlock:
    """Represents one priority level whose registered functions run in parallel."""

    def __init__(
        self, parent: PipelineHandler, registration_name: str, execution_priority: float
    ) -> None:
        self.parent = parent
        self.registration_name = registration_name
        self.execution_priority = execution_priority
        self.functions: list[FunctionRegistration] = []
        self.registered_args: dict[str, BlockArgsRegistration] = {}
        self.registered_kwargs: dict[str, BlockKwargsRegistration] = {}

    def register_args(
        self, name: str, ordered_items: tuple[str, ...] | list[str], forced: bool = False
    ) -> BlockArgsRegistration | None:
        try:
            if name in self.registered_args and not forced:
                raise RegistrationError(
                    f"Args helper '{name}' is already registered in block '{self.registration_name}'"
                )
            registration = BlockArgsRegistration(name=name, ordered_items=list(ordered_items))
            self.registered_args[name] = registration
            return registration
        except RegistrationError as exc:
            self.parent.logger.warning(
                f"Skipped args helper registration in block '{self.registration_name}': {exc}"
            )
            return None

    def register_kwargs(
        self, name: str, mapping_dct: dict[str, str], forced: bool = False
    ) -> BlockKwargsRegistration | None:
        try:
            if name in self.registered_kwargs and not forced:
                raise RegistrationError(
                    f"Kwargs helper '{name}' is already registered in block '{self.registration_name}'"
                )
            registration = BlockKwargsRegistration(name=name, mapping_dct=dict(mapping_dct))
            self.registered_kwargs[name] = registration
            return registration
        except RegistrationError as exc:
            self.parent.logger.warning(
                f"Skipped kwargs helper registration in block '{self.registration_name}': {exc}"
            )
            return None

    def register_function(
        self,
        function_or_path: Any,
        output_variable_names: str | list[str] | tuple[str, ...] | None,
        save_to_disk: list[str] | tuple[str, ...] | set[str] | None = None,
        param_mapping: dict[str, str] | None = None,
        var_pos_name: str | None = None,
        var_kw_name: str | None = None,
        forced: bool = False,
    ) -> Any:
        _, _, function_name = resolve_callable(function_or_path)
        existing_registration = next(
            (
                registration
                for registration in self.functions
                if registration.function_name == function_name
            ),
            None,
        )
        if existing_registration is not None and not forced:
            raise RegistrationError(
                f"Function '{function_name}' is already registered in block '{self.registration_name}'"
            )
        if existing_registration is not None and forced:
            self.remove_function(function_name)
        try:
            registration = self._register_function_strict(
                function_or_path,
                output_variable_names,
                save_to_disk=save_to_disk,
                param_mapping=param_mapping,
                var_pos_name=var_pos_name,
                var_kw_name=var_kw_name,
                forced=forced,
            )
        except RegistrationError as exc:
            self.parent.logger.warning(
                f"Skipped function registration in block '{self.registration_name}': {exc}"
            )
            return None
        return registration

    def _register_function_strict(
        self,
        function_or_path: Any,
        output_variable_names: str | list[str] | tuple[str, ...] | None,
        save_to_disk: list[str] | tuple[str, ...] | set[str] | None = None,
        param_mapping: dict[str, str] | None = None,
        var_pos_name: str | None = None,
        var_kw_name: str | None = None,
        forced: bool = False,
    ) -> FunctionRegistration:
        del forced
        if output_variable_names is None:
            output_names: list[str] = []
        elif isinstance(output_variable_names, str):
            output_names = [output_variable_names]
        else:
            output_names = list(output_variable_names)
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
        self.parent._validate_output_names_against_config(output_names)

        callable_obj, import_path, function_name = resolve_callable(function_or_path)
        declared_output_count = infer_declared_output_count(callable_obj)
        input_names = inspect_exposed_input_names(
            callable_obj,
            param_mapping=param_mapping,
            var_pos_name=var_pos_name,
            var_kw_name=var_kw_name,
        )
        if not output_names and declared_output_count is not None and declared_output_count > 0:
            self.parent.logger.warning(
                f"Function '{function_name}' in block '{self.registration_name}' declares {declared_output_count} output(s), but output_variable_names=None was used; any returned value will be ignored"
            )
        if (
            declared_output_count is not None
            and output_names
            and declared_output_count != len(output_names)
        ):
            raise RegistrationError(
                f"Function '{function_name}' declares {declared_output_count} output(s), but {len(output_names)} output name(s) were registered"
            )
        registration = FunctionRegistration(
            function_name=function_name,
            import_path=import_path,
            callable_obj=callable_obj,
            input_names=input_names,
            output_names=output_names,
            save_to_disk=disk_names,
            param_mapping=dict(param_mapping or {}),
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
            block=self,
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
            callable_label = registration.import_path or registration.function_name
            raise ExecutionError(
                f"Function '{registration.function_name}' ({callable_label}) in block '{self.registration_name}' failed: {type(exc).__name__}: {exc}"
            ) from exc

        outputs = self._normalize_outputs(registration, result)
        return FunctionExecutionResult(
            function_name=registration.function_name,
            outputs=outputs,
            loaded_artifact_inputs=loaded_artifacts,
        )

    @staticmethod
    def _normalize_outputs(registration: FunctionRegistration, result: Any) -> dict[str, Any]:
        if len(registration.output_names) == 0:
            return {}
        if len(registration.output_names) == 1:
            return {registration.output_names[0]: result}

        callable_label = registration.import_path or registration.function_name
        if not isinstance(result, (tuple, list)):
            raise ExecutionError(
                f"Function '{registration.function_name}' ({callable_label}) declared multiple outputs {registration.output_names} "
                f"but returned {type(result).__name__}: {ExecutionBlock._preview_value(result)}"
            )
        if len(result) != len(registration.output_names):
            raise ExecutionError(
                f"Function '{registration.function_name}' ({callable_label}) returned {len(result)} values but "
                f"{len(registration.output_names)} outputs were declared: {registration.output_names}"
            )
        return dict(zip(registration.output_names, result, strict=True))

    @staticmethod
    def _preview_value(value: Any, max_length: int = 200) -> str:
        preview = repr(value)
        if len(preview) > max_length:
            return preview[: max_length - 3] + "..."
        return preview
