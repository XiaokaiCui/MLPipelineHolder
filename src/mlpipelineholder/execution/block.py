from __future__ import annotations

import ast
import builtins
import inspect
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Final

from ..exceptions import (
    ExecutionError,
    InspectionCopyError,
    PersistenceError,
    RegistrationError,
    ResolutionError,
)
from ..integrations.optuna.support import (
    OPTUNA_STUDY_SERIALIZER,
    is_optuna_sampler,
    is_optuna_study,
)
from .inspection import (
    DEFAULT_INSPECTION_MEMORY_SAFETY_MARGIN,
    InspectionCopier,
    InspectionCopyCandidate,
    inspection_memory_failure,
    preflight_protected_inspection,
)
from .function_registry import (
    callable_identity_matches,
    callable_signature,
    default_map,
    effective_variadic_names,
    infer_declared_output_count,
    inspect_exposed_input_names,
    inspect_input_names,
    normalize_renamed_registration,
    rename_args,
    resolve_callable,
)
from ..core.models import (
    ArtifactRecord,
    BlockArgsRegistration,
    BlockKwargsRegistration,
    ExpressionRegistration,
    FunctionExecutionResult,
    FunctionRegistration,
    ResolutionSource,
    ResolvedInspectionCall,
)
from ..core.naming import validate_registration_name
from ..state.output_pointers import (
    OutputPointer,
    resolve_pointer_chain,
)

if TYPE_CHECKING:
    from ..pipeline_holder import PipelineHolder


_ALLOWED_EXPRESSION_BUILTIN_NAMES: Final[set[str]] = {
    name
    for name in dir(builtins)
    if not name.startswith("_")
}


class ExecutionBlock:
    """Represents one priority level whose registered functions run in parallel."""

    def __init__(
        self, parent: PipelineHolder, registration_name: str, execution_priority: float
    ) -> None:
        self.parent = parent
        self.registration_name = validate_registration_name(
            registration_name,
            owner_label="block",
        )
        self.execution_priority = execution_priority
        self.functions: list[FunctionRegistration | ExpressionRegistration] = []
        self.registered_args: dict[str, BlockArgsRegistration] = {}
        self.registered_kwargs: dict[str, BlockKwargsRegistration] = {}

    def register_expression(
        self,
        code: str,
        *,
        output_variable_name: str | None = None,
        save_to_disk: bool = False,
        forced: bool = True,
        warn_on_input_mutation: bool = False,
        overridden_outputs: dict[str, tuple[str, str]] | None = None,
    ) -> Any:
        self.parent._assert_mutable("accept new expressions")
        return self._register_expression_strict(
            code,
            output_variable_name=output_variable_name,
            save_to_disk=save_to_disk,
            forced=forced,
            warn_on_input_mutation=warn_on_input_mutation,
            overridden_outputs=overridden_outputs,
        )

    def _register_expression_strict(
        self,
        code: str,
        *,
        output_variable_name: str | None,
        save_to_disk: bool,
        forced: bool,
        warn_on_input_mutation: bool,
        overridden_outputs: dict[str, tuple[str, str]] | None = None,
    ) -> ExpressionRegistration:
        if any(
            isinstance(registration, FunctionRegistration)
            for registration in self.functions
        ):
            raise RegistrationError(
                f"Block '{self.registration_name}' already contains registered functions; "
                "functions and expressions cannot be mixed in one block"
            )
        code = self._normalize_expression_code(code)
        expression = self._parse_expression(code)
        inferred_output, input_names, printing_only = self._analyze_expression(
            expression,
            ignored_names=self.parent._expression_runtime_defined_names(),
        )
        final_output = output_variable_name if output_variable_name is not None else inferred_output
        if printing_only and final_output is not None:
            raise RegistrationError("Printing expressions cannot declare an output variable")
        if not printing_only and final_output is None:
            raise RegistrationError("Assignment expressions must resolve to exactly one output variable")
        if inferred_output is not None and output_variable_name is not None and inferred_output != output_variable_name:
            raise RegistrationError(
                f"Expression output '{inferred_output}' does not match declared output '{output_variable_name}'"
            )
        output_names = [] if final_output is None else [final_output]
        if save_to_disk and not output_names:
            raise RegistrationError("save_to_disk=True requires an output variable")
        if output_names:
            self.parent._validate_output_names_against_config(output_names)
        normalized_overrides = self.parent._normalize_overridden_outputs(
            output_names,
            overridden_outputs,
            current_node_name=self.registration_name,
            current_priority=self.execution_priority,
        )
        identity = final_output or code
        existing = next(
            (
                registration
                for registration in self.functions
                if isinstance(registration, ExpressionRegistration)
                and ((registration.output_names[0] if registration.output_names else registration.code) == identity)
            ),
            None,
        )
        other_expressions = [
            registration
            for registration in self.functions
            if isinstance(registration, ExpressionRegistration)
            and registration is not existing
        ]
        if other_expressions:
            if not forced:
                raise RegistrationError(
                    f"Block '{self.registration_name}' already contains an expression; "
                    "at most one expression may be registered per block"
                )
            if len(other_expressions) > 1:
                raise RegistrationError(
                    f"Block '{self.registration_name}' contains multiple expressions; "
                    "cannot determine which one to override"
                )
            existing = other_expressions[0]
        if existing is not None and not forced:
            raise RegistrationError(
                f"Expression '{identity}' is already registered in block '{self.registration_name}'"
            )
        if existing is not None and forced:
            new_save_to_disk = (
                {final_output} if save_to_disk and final_output is not None else set()
            )
            if (
                existing.code == code
                and existing.output_names == output_names
                and existing.save_to_disk == new_save_to_disk
                and existing.warn_on_input_mutation == warn_on_input_mutation
                and existing.overridden_outputs == normalized_overrides
            ):
                return existing
            if self.parent._invalidation_forbidden:
                self.parent.logger.warning(
                    f"Expression in block '{self.registration_name}' was overridden with a "
                    "different expression; erasing the block's own outputs while retaining "
                    "other cached outputs because cascade invalidation is forbidden"
                )
            else:
                self.parent.logger.warning(
                    f"Expression in block '{self.registration_name}' was overridden with a "
                    "different expression; invalidating its outputs and downstream dependents"
                )
            self.parent._erase_overridden_node_outputs(
                self.registration_name,
                self.execution_priority,
                self.execution_priority,
                existing.output_names,
                output_names,
            )
            self.functions.remove(existing)
        registration = ExpressionRegistration(
            code=code,
            input_names=input_names,
            output_names=output_names,
            save_to_disk={final_output} if save_to_disk and final_output is not None else set(),
            warn_on_input_mutation=warn_on_input_mutation,
            overridden_outputs=normalized_overrides,
        )
        if warn_on_input_mutation and input_names:
            self.parent.logger.info(
                f"Expression registration in block '{self.registration_name}' is intended for one line and one variable change; if more than one variable may be mutated, prefer a registered function or multiple expressions"
            )
        self.functions.append(registration)
        self.parent._register_node(self)
        return registration

    @staticmethod
    def _normalize_expression_code(code: str) -> str:
        normalized = dedent(code).strip()
        if not normalized:
            raise RegistrationError("Expression code cannot be empty")
        return normalized

    def _parse_expression(self, code: str) -> ast.Module:
        if ";" in code:
            raise RegistrationError("Expressions may not contain semicolons")
        try:
            parsed = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            location = ""
            if exc.lineno is not None:
                location = f" at line {exc.lineno}"
                if exc.offset is not None:
                    location += f", column {exc.offset}"
            message = f"Invalid expression syntax{location}: {exc.msg}"
            code_lines = code.splitlines()
            if exc.lineno is not None and 1 <= exc.lineno <= len(code_lines):
                error_index = exc.lineno - 1
                context_start = max(0, error_index - 2)
                context_end = min(len(code_lines), error_index + 2)
                number_width = len(str(context_end))
                context: list[str] = []
                for index in range(context_start, context_end):
                    source_line = code_lines[index]
                    prefix = ">" if index == error_index else " "
                    gutter = f"{prefix} {index + 1:>{number_width}} | "
                    context.append(f"{gutter}{source_line.expandtabs()}")
                    if index == error_index and exc.offset is not None:
                        prefix_width = len(source_line[: exc.offset - 1].expandtabs())
                        end_offset = exc.end_offset or exc.offset + 1
                        marker_width = max(1, end_offset - exc.offset)
                        marker = " " * prefix_width + "^" * marker_width
                        context.append(f"{' ' * len(gutter)}{marker}")
                message += "\n" + "\n".join(context)
            raise RegistrationError(message) from exc
        if len(parsed.body) != 1:
            raise RegistrationError("Expressions must contain exactly one statement")
        for node in ast.walk(parsed):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise RegistrationError("Expressions may not contain import statements")
            if isinstance(node, ast.NamedExpr):
                raise RegistrationError("Expressions may not use the walrus operator")
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                raise RegistrationError(
                    "Expressions may not use comprehensions or generator expressions"
                )
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef, ast.Lambda)):
                raise RegistrationError(
                    "Expressions may not define functions, classes, or lambdas"
                )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "__import__"}:
                raise RegistrationError("Expression may not call eval, exec, or __import__")
        return parsed

    def _analyze_expression(
        self,
        parsed: ast.Module,
        *,
        ignored_names: set[str],
    ) -> tuple[str | None, list[str], bool]:
        stmt = parsed.body[0]
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise RegistrationError("Expressions must assign to exactly one plain variable name")
            input_names = self._extract_loaded_names(stmt.value, ignored_names=ignored_names)
            return stmt.targets[0].id, input_names, False
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            is_print = isinstance(call.func, ast.Name) and call.func.id == "print"
            is_logger = (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "logger"
            )
            if not (is_print or is_logger):
                raise RegistrationError("Expression statements must be print(...) or logger.xxx(...)")
            input_names = self._extract_loaded_names(stmt.value, ignored_names=ignored_names)
            return None, input_names, True
        raise RegistrationError("Expressions must be either NAME = EXPR or a print/logger call")

    def _extract_loaded_names(self, node: ast.AST, *, ignored_names: set[str]) -> list[str]:
        names: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if (
                    child.id in {"print", "logger"}
                    or child.id in _ALLOWED_EXPRESSION_BUILTIN_NAMES
                    or child.id in ignored_names
                ):
                    continue
                if child.id not in names:
                    names.append(child.id)
        return names

    def _effective_expression_input_names(
        self,
        registration: ExpressionRegistration,
    ) -> list[str]:
        parsed = self._parse_expression(registration.code)
        _, input_names, _ = self._analyze_expression(
            parsed,
            ignored_names=self.parent._expression_runtime_defined_names(),
        )
        return input_names

    def register_args(
        self, name: str, ordered_items: tuple[str, ...] | list[str], forced: bool = False
    ) -> BlockArgsRegistration | None:
        self.parent._assert_mutable("accept new args helpers")
        try:
            if name in self.registered_args:
                if not forced:
                    raise RegistrationError(
                        f"Args helper '{name}' is already registered in block '{self.registration_name}'"
                    )
                existing = self.registered_args[name]
                if list(existing.ordered_items) == list(ordered_items):
                    return existing
            registration = BlockArgsRegistration(name=name, ordered_items=list(ordered_items))
            self.registered_args[name] = registration
            self._invalidate_helper_consumers(name, is_args=True)
            return registration
        except RegistrationError as exc:
            self.parent.logger.warning(
                f"Skipped args helper registration in block '{self.registration_name}': {exc}"
            )
            return None

    def register_kwargs(
        self, name: str, mapping_dct: dict[str, str], forced: bool = False
    ) -> BlockKwargsRegistration | None:
        self.parent._assert_mutable("accept new kwargs helpers")
        try:
            if name in self.registered_kwargs:
                if not forced:
                    raise RegistrationError(
                        f"Kwargs helper '{name}' is already registered in block '{self.registration_name}'"
                    )
                existing = self.registered_kwargs[name]
                if dict(existing.mapping_dct) == dict(mapping_dct):
                    return existing
            registration = BlockKwargsRegistration(name=name, mapping_dct=dict(mapping_dct))
            previous = self.registered_kwargs.get(name)
            self.registered_kwargs[name] = registration
            try:
                for consumer in self.functions:
                    if not isinstance(consumer, FunctionRegistration):
                        continue
                    _, effective_kw_name = effective_variadic_names(
                        consumer.callable_obj,
                        var_pos_name=consumer.var_pos_name,
                        var_kw_name=consumer.var_kw_name,
                    )
                    if effective_kw_name == name:
                        self._strict_validate_registration(consumer)
            except RegistrationError:
                if previous is None:
                    del self.registered_kwargs[name]
                else:
                    self.registered_kwargs[name] = previous
                raise
            self._invalidate_helper_consumers(name, is_args=False)
            return registration
        except RegistrationError as exc:
            self.parent.logger.warning(
                f"Skipped kwargs helper registration in block '{self.registration_name}': {exc}"
            )
            return None

    def _invalidate_helper_consumers(self, name: str, *, is_args: bool) -> None:
        consumers: list[FunctionRegistration] = []
        for registration in self.functions:
            if not isinstance(registration, FunctionRegistration):
                continue
            effective_pos_name, effective_kw_name = effective_variadic_names(
                registration.callable_obj,
                var_pos_name=registration.var_pos_name,
                var_kw_name=registration.var_kw_name,
            )
            effective_name = effective_pos_name if is_args else effective_kw_name
            if effective_name == name:
                consumers.append(registration)
        output_names = [
            output_name
            for registration in consumers
            for output_name in registration.produced_output_names
        ]
        for registration in consumers:
            if is_args:
                registration.args_registration_state = list(
                    self.registered_args[name].ordered_items
                )
            else:
                registration.kwargs_registration_state = dict(
                    self.registered_kwargs[name].mapping_dct
                )
            registration.input_names = self._function_input_names(registration)
        if not output_names:
            return
        self.parent._erase_overridden_node_outputs(
            self.registration_name,
            self.execution_priority,
            self.execution_priority,
            output_names,
            output_names,
            erase_output_names=output_names,
        )

    def register_function(
        self,
        function_or_path: Any,
        output_variable_names: str | list[str] | tuple[str, ...] | None,
        save_to_disk: list[str] | tuple[str, ...] | set[str] | None = None,
        param_mapping: dict[str, str | None] | None = None,
        var_pos_name: str | None = None,
        var_kw_name: str | None = None,
        forced: bool = False,
        overridden_outputs: dict[str, tuple[str, str]] | None = None,
    ) -> Any:
        self.parent._assert_mutable("accept new functions")
        if "_" in set(save_to_disk or []):
            raise RegistrationError(
                "Ignored output marker '_' cannot be included in save_to_disk"
            )
        function_or_path, param_mapping, var_pos_name, var_kw_name = (
            normalize_renamed_registration(
                function_or_path,
                param_mapping,
                var_pos_name,
                var_kw_name,
            )
        )
        callable_obj, import_path, function_name = resolve_callable(function_or_path)
        existing_registration = next(
            (
                registration
                for registration in self.functions
                if isinstance(registration, FunctionRegistration)
                and registration.function_name == function_name
            ),
            None,
        )
        if existing_registration is not None and not forced:
            raise RegistrationError(
                f"Function '{function_name}' is already registered in block '{self.registration_name}'"
            )
        if existing_registration is not None and forced:
            if self._function_registration_matches(
                existing_registration,
                callable_obj,
                import_path,
                output_variable_names,
                save_to_disk,
                param_mapping,
                var_pos_name,
                var_kw_name,
                overridden_outputs,
            ):
                return existing_registration
        try:
            registration = self._register_function_strict(
                function_or_path,
                output_variable_names,
                save_to_disk=save_to_disk,
                param_mapping=param_mapping,
                var_pos_name=var_pos_name,
                var_kw_name=var_kw_name,
                forced=forced,
                overridden_outputs=overridden_outputs,
                replacing=existing_registration,
                commit=existing_registration is None,
            )
        except RegistrationError as exc:
            if self.parent.strict_mode:
                raise
            self.parent.logger.warning(
                f"Skipped function registration in block '{self.registration_name}': {exc}"
            )
            return None
        if existing_registration is not None:
            self.parent._erase_overridden_node_outputs(
                self.registration_name,
                self.execution_priority,
                self.execution_priority,
                list(existing_registration.produced_output_names),
                list(registration.produced_output_names),
            )
            index = self.functions.index(existing_registration)
            self.functions[index] = registration
            self.parent._register_node(self)
        return registration

    def _function_registration_matches(
        self,
        existing: FunctionRegistration,
        callable_obj: Any,
        import_path: str | None,
        output_variable_names: str | list[str] | tuple[str, ...] | None,
        save_to_disk: list[str] | tuple[str, ...] | set[str] | None,
        param_mapping: dict[str, str | None] | None,
        var_pos_name: str | None,
        var_kw_name: str | None,
        overridden_outputs: dict[str, tuple[str, str]] | None,
    ) -> bool:
        if not callable_identity_matches(
            existing.import_path,
            existing.callable_obj,
            import_path,
            callable_obj,
        ):
            return False
        if output_variable_names is None:
            new_output_names: list[str] = []
        elif isinstance(output_variable_names, str):
            new_output_names = [output_variable_names]
        else:
            new_output_names = list(output_variable_names)
        new_produced_output_names = [
            output_name for output_name in new_output_names if output_name != "_"
        ]
        effective_pos_name, effective_kw_name = effective_variadic_names(
            callable_obj,
            var_pos_name=var_pos_name,
            var_kw_name=var_kw_name,
        )
        args_state, kwargs_state = self._variadic_registration_state(
            effective_pos_name,
            effective_kw_name,
        )
        normalized_overrides = self.parent._normalize_overridden_outputs(
            new_produced_output_names,
            overridden_outputs,
            current_node_name=self.registration_name,
            current_priority=self.execution_priority,
        )
        return (
            existing.output_names == new_output_names
            # Public registrations use discard semantics. A legacy registration
            # that treated "_" as a real output must be replaced, not reused.
            and existing.ignore_underscore_outputs
            and existing.save_to_disk == set(save_to_disk or [])
            and existing.param_mapping == dict(param_mapping or {})
            and existing.var_pos_name == var_pos_name
            and existing.var_kw_name == var_kw_name
            and existing.args_registration_state == args_state
            and existing.kwargs_registration_state == kwargs_state
            and existing.overridden_outputs == normalized_overrides
        )

    def _variadic_registration_state(
        self,
        var_pos_name: str | None,
        var_kw_name: str | None,
    ) -> tuple[list[str] | None, dict[str, str] | None]:
        args_registration = (
            None
            if var_pos_name is None
            else self.registered_args.get(var_pos_name)
        )
        kwargs_registration = (
            None
            if var_kw_name is None
            else self.registered_kwargs.get(var_kw_name)
        )
        return (
            None
            if args_registration is None
            else list(args_registration.ordered_items),
            None
            if kwargs_registration is None
            else dict(kwargs_registration.mapping_dct),
        )

    def _register_function_strict(
        self,
        function_or_path: Any,
        output_variable_names: str | list[str] | tuple[str, ...] | None,
        save_to_disk: list[str] | tuple[str, ...] | set[str] | None = None,
        param_mapping: dict[str, str | None] | None = None,
        var_pos_name: str | None = None,
        var_kw_name: str | None = None,
        forced: bool = False,
        replacing: FunctionRegistration | None = None,
        commit: bool = True,
        overridden_outputs: dict[str, tuple[str, str]] | None = None,
        ignore_underscore_outputs: bool = True,
    ) -> FunctionRegistration:
        del forced
        if any(
            isinstance(registration, ExpressionRegistration)
            for registration in self.functions
        ):
            raise RegistrationError(
                f"Block '{self.registration_name}' already contains an expression; "
                "functions and expressions cannot be mixed in one block"
            )
        if output_variable_names is None:
            output_names: list[str] = []
        elif isinstance(output_variable_names, str):
            output_names = [output_variable_names]
        else:
            output_names = list(output_variable_names)
        produced_output_names = (
            [output_name for output_name in output_names if output_name != "_"]
            if ignore_underscore_outputs
            else list(output_names)
        )
        if len(set(produced_output_names)) != len(produced_output_names):
            raise RegistrationError("Duplicate output variable names are not allowed")

        existing_local_outputs = {
            output_name
            for registration in self.functions
            if registration is not replacing
            and isinstance(registration, FunctionRegistration)
            for output_name in registration.produced_output_names
        }
        overlap = existing_local_outputs.intersection(produced_output_names)
        if overlap:
            raise RegistrationError(
                f"Duplicate output names inside block '{self.registration_name}': {sorted(overlap)}"
            )

        disk_names = set(save_to_disk or [])
        if ignore_underscore_outputs and "_" in disk_names:
            raise RegistrationError(
                "Ignored output marker '_' cannot be included in save_to_disk"
            )
        if not disk_names.issubset(set(produced_output_names)):
            raise RegistrationError(
                "Disk-saved output names must be a subset of output variable names"
            )
        self.parent._validate_output_names_against_config(produced_output_names)
        normalized_overrides = self.parent._normalize_overridden_outputs(
            produced_output_names,
            overridden_outputs,
            current_node_name=self.registration_name,
            current_priority=self.execution_priority,
        )

        callable_obj, import_path, function_name = resolve_callable(function_or_path)
        declared_output_count = infer_declared_output_count(callable_obj)
        effective_pos_name, effective_kw_name = effective_variadic_names(
            callable_obj,
            var_pos_name=var_pos_name,
            var_kw_name=var_kw_name,
        )
        args_state, kwargs_state = self._variadic_registration_state(
            effective_pos_name,
            effective_kw_name,
        )
        if not output_names and declared_output_count is not None and declared_output_count > 0:
            if not getattr(self.parent, "suppress_registration_advisories", False):
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
            input_names=[],
            output_names=output_names,
            save_to_disk=disk_names,
            ignore_underscore_outputs=ignore_underscore_outputs,
            param_mapping=dict(param_mapping or {}),
            var_pos_name=var_pos_name,
            var_kw_name=var_kw_name,
            args_registration_state=args_state,
            kwargs_registration_state=kwargs_state,
            overridden_outputs=normalized_overrides,
        )
        registration.input_names = self._function_input_names(registration)
        self._strict_validate_registration(registration)
        self._warn_on_unmapped_resolvable_inputs(callable_obj, registration)
        self._warn_on_disk_backed_input_persistence_pitfall(registration)
        if commit:
            self.functions.append(registration)
            self.parent._register_node(self)
        return registration

    def _warn_on_unmapped_resolvable_inputs(
        self,
        callable_obj: Any,
        registration: FunctionRegistration,
    ) -> None:
        if self.parent.strict_mode:
            return
        try:
            signature = callable_signature(callable_obj)
            visible_names = (
                set(self.parent._visible_config_names())
                | set(self.parent._visible_outputs_before_priority(self.execution_priority))
                | set(
                    self.parent._declared_output_names_before_priority(
                        self.execution_priority
                    )
                )
                | set(self.parent.manual_values)
                | set(self.parent._ancestor_manual_values())
            )
            suspicious: list[str] = []
            for parameter in signature.parameters.values():
                if parameter.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                if parameter.name == "logger":
                    continue
                if parameter.name in registration.param_mapping:
                    continue
                if parameter.name in visible_names:
                    suspicious.append(parameter.name)
            if suspicious:
                if not getattr(self.parent, "suppress_registration_advisories", False):
                    self.parent.logger.warning(
                        f"Function '{registration.function_name}' in block '{self.registration_name}' has unmapped input(s) {sorted(suspicious)} that match visible config, output, or manual value names and may be resolved implicitly"
                    )
        except Exception:
            return

    def _refresh_function_input_names(self) -> None:
        for registration in self.functions:
            if not isinstance(registration, FunctionRegistration):
                continue
            registration.input_names = self._function_input_names(registration)

    def _function_input_names(
        self,
        registration: FunctionRegistration,
    ) -> list[str]:
        strict = self.parent.strict_mode
        input_names = inspect_exposed_input_names(
            registration.callable_obj,
            param_mapping=registration.param_mapping,
            var_pos_name=None if strict else registration.var_pos_name,
            var_kw_name=None if strict else registration.var_kw_name,
            strict_mode=strict,
        )
        effective_pos_name, effective_kw_name = effective_variadic_names(
            registration.callable_obj,
            var_pos_name=registration.var_pos_name,
            var_kw_name=registration.var_kw_name,
        )
        variadic_dependencies = (
            (effective_pos_name, registration.args_registration_state),
            (
                effective_kw_name,
                None
                if registration.kwargs_registration_state is None
                else list(registration.kwargs_registration_state.values()),
            ),
        )
        for helper_name, dependencies in variadic_dependencies:
            if dependencies is None:
                continue
            if helper_name in input_names:
                input_names.remove(helper_name)
            for input_name in dependencies:
                if input_name not in input_names:
                    input_names.append(input_name)
        return input_names

    def _warn_on_disk_backed_input_persistence_pitfall(
        self,
        registration: FunctionRegistration,
    ) -> None:
        try:
            mapped_inputs = set(registration.param_mapping.values())
            mapped_inputs.update(
                name
                for name in registration.input_names
                if name not in registration.param_mapping and name != "logger"
            )
            risky = sorted(
                name
                for name in mapped_inputs
                if name
                in self.parent._registration_disk_backed_names(
                    self.execution_priority
                )
                and name not in registration.produced_output_names
            )
            if risky:
                if not getattr(self.parent, "suppress_registration_advisories", False):
                    self.parent.logger.info(
                        f"Function '{registration.function_name}' in block '{self.registration_name}' reads disk-backed input(s) {risky} without declaring them as outputs; in-function mutations will not persist unless those names are also outputs"
                    )
        except Exception:
            return

    def _strict_validate_registration(
        self,
        registration: FunctionRegistration,
        *,
        force_strict: bool = False,
        visible_names: set[str] | None = None,
    ) -> None:
        """Run strict-mode checks 7, 8, 10, 11, 12 for a function registration.

        In strict mode (or when force_strict is set) a violation raises
        RegistrationError; otherwise it is logged as a warning. Checks are
        skipped while loading a saved pipeline. When visible_names is given
        (attach-time revalidation) it is used instead of this pipeline's own
        visible names, so the new parent's objects are taken into account.
        """
        if self.parent._suppress_strict_validation:
            return
        strict = self.parent.strict_mode or force_strict
        signature = callable_signature(registration.callable_obj)
        explicit_params = {
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }
        has_var_keyword = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

        # Check 7: kwargs_dct used but the function has no **kwargs parameter.
        if registration.var_kw_name is not None and not has_var_keyword:
            self._strict_violation(
                f"kwargs_dct is used (var_kw_name='{registration.var_kw_name}') but function "
                f"'{registration.function_name}' has no **kwargs parameter",
                strict,
            )

        # Check 8: a kwargs_dct key conflicts with an explicit function argument.
        _, effective_kw_name = effective_variadic_names(
            registration.callable_obj,
            var_pos_name=registration.var_pos_name,
            var_kw_name=registration.var_kw_name,
        )
        kwargs_registration = (
            None
            if effective_kw_name is None
            else self.registered_kwargs.get(effective_kw_name)
        )
        if kwargs_registration is not None:
            for key in kwargs_registration.mapping_dct:
                if key in explicit_params:
                    self._strict_violation(
                        f"kwargs_dct key '{key}' conflicts with explicit parameter '{key}' "
                        f"of function '{registration.function_name}'",
                        strict,
                    )

        # Check 12: a param_mapping key is not a function argument.
        for key in registration.param_mapping:
            if key not in explicit_params:
                self._strict_violation(
                    f"param_mapping key '{key}' is not a parameter of function "
                    f"'{registration.function_name}'",
                    strict,
                )

        # Check 10: a param_mapping value is not found in config, visible or
        # self-declared output values, or visible manual values. Literal None
        # mapping and 'logger' are always resolvable.
        if visible_names is None:
            visible_names = self._registration_visible_names()
        resolvable_names = visible_names | set(registration.produced_output_names)
        for key, value in registration.param_mapping.items():
            if value is None or value == "logger":
                continue
            if value not in resolvable_names:
                self._strict_violation(
                    f"param_mapping value '{value}' for parameter '{key}' of function "
                    f"'{registration.function_name}' is not found in config, visible output values, or visible manual values",
                    strict,
                )

        # Check 11: a kwargs_dct value is not found in config, visible output
        # values, or visible manual values.
        if kwargs_registration is not None:
            for key, value in kwargs_registration.mapping_dct.items():
                if value == "logger":
                    continue
                if value not in visible_names:
                    self._strict_violation(
                        f"kwargs_dct value '{value}' for key '{key}' of function "
                        f"'{registration.function_name}' is not found in config, visible output values, or visible manual values",
                        strict,
                    )

    def _registration_visible_names(self) -> set[str]:
        return (
            set(self.parent._visible_config_names())
            | set(self.parent._visible_outputs_before_priority(self.execution_priority))
            | set(
                self.parent._declared_output_names_before_priority(
                    self.execution_priority
                )
            )
            | set(self.parent.manual_values)
            | set(self.parent._ancestor_manual_values())
        )

    def _strict_violation(self, message: str, strict: bool) -> None:
        if strict:
            raise RegistrationError(message)
        self.parent.logger.warning(message)

    def remove_function(self, function_name: str) -> None:
        self.parent._assert_mutable("remove functions")
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
        self.parent._erase_node_outputs(self.registration_name)
        if not self.parent._invalidation_forbidden:
            self.parent._invalidate_from_priority(self.execution_priority)
        if self.parent.parent_pipeline is not None:
            self.parent._resync_mirror_to_parent()

    def declared_outputs(self) -> set[str]:
        return {
            output_name
            for registration in self.functions
            for output_name in (
                registration.output_names
                if isinstance(registration, ExpressionRegistration)
                else registration.produced_output_names
            )
        }

    def inspect(
        self,
        function_name: str | None = None,
        overrides: dict[str, Any] | None = None,
        strict_mode: bool | None = None,
        resolve_only: bool = False,
        allow_mutable_objects: bool = True,
        memory_safety_margin: float = DEFAULT_INSPECTION_MEMORY_SAFETY_MARGIN,
    ) -> Any:
        """Resolve and optionally invoke one registration without committing outputs."""
        return self._inspect_registration(
            function_name=function_name,
            overrides=overrides,
            strict_mode=strict_mode,
            resolve_only=resolve_only,
            allow_mutable_objects=allow_mutable_objects,
            memory_safety_margin=memory_safety_margin,
            node_name=self.registration_name,
        )

    def _inspect_registration(
        self,
        *,
        function_name: str | None,
        overrides: dict[str, Any] | None,
        strict_mode: bool | None,
        resolve_only: bool,
        allow_mutable_objects: bool,
        memory_safety_margin: float = DEFAULT_INSPECTION_MEMORY_SAFETY_MARGIN,
        node_name: str,
        upstream_outputs: dict[str, Any] | None = None,
        previous_outputs: dict[str, Any] | None = None,
    ) -> Any:
        if self.parent.nodes_by_name.get(self.registration_name) is not self:
            raise RegistrationError(
                f"Block '{self.registration_name}' is no longer attached to pipeline "
                f"'{self.parent.registration_name}'"
            )
        if strict_mode is not None and not isinstance(strict_mode, bool):
            raise TypeError("strict_mode must be a boolean or None")
        if not isinstance(resolve_only, bool):
            raise TypeError("resolve_only must be a boolean")
        if not isinstance(allow_mutable_objects, bool):
            raise TypeError("allow_mutable_objects must be a boolean")
        try:
            normalized_memory_safety_margin = float(memory_safety_margin)
        except (TypeError, ValueError, OverflowError):
            normalized_memory_safety_margin = float("nan")
        if (
            isinstance(memory_safety_margin, bool)
            or not isinstance(memory_safety_margin, (int, float))
            or not math.isfinite(normalized_memory_safety_margin)
            or normalized_memory_safety_margin < 0
        ):
            raise TypeError(
                "memory_safety_margin must be a finite non-negative number"
            )
        memory_safety_margin = normalized_memory_safety_margin
        inspection_overrides = dict(overrides or {})
        registration = self._select_inspection_registration(
            function_name,
            inspection_overrides,
        )
        effective_strict = (
            self.parent._root_pipeline().strict_mode
            if strict_mode is None
            else strict_mode
        )
        self._validate_selected_inspection_dependencies(
            registration,
            inspection_overrides,
            strict_mode=effective_strict,
        )
        visible_outputs, visible_constants, previous_names, parent_config = (
            self._build_inspection_environment(
                upstream_outputs=upstream_outputs,
                previous_outputs=previous_outputs,
            )
        )
        copier = InspectionCopier(
            self.parent.logger,
            allow_mutable_objects=allow_mutable_objects,
        )
        if isinstance(registration, FunctionRegistration):
            resolved = self._resolve_inspected_registration(
                registration,
                inspection_overrides,
                strict_mode=effective_strict,
                allow_mutable_objects=allow_mutable_objects,
                memory_safety_margin=memory_safety_margin,
                visible_outputs=visible_outputs,
                visible_constants=visible_constants,
                previous_names=previous_names,
                parent_config=parent_config,
                copier=copier,
                node_name=node_name,
            )
            if resolve_only:
                return resolved
            try:
                return self.parent._capture_prints(
                    registration.callable_obj,
                    *resolved.args,
                    **resolved.kwargs,
                )
            except MemoryError as exc:
                raise inspection_memory_failure(
                    node_name=node_name,
                    function_name=registration.function_name,
                    parameter_name=None,
                    stage="executing the inspected function",
                    copy_started=copier.copy_started,
                ) from exc
            except ResolutionError:
                raise
            except Exception as exc:
                callable_label = registration.import_path or registration.function_name
                raise ExecutionError(
                    f"Function '{registration.function_name}' ({callable_label}) in "
                    f"block '{self.registration_name}' failed during inspection: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        resolved, namespace = self._resolve_inspected_expression(
            registration,
            inspection_overrides,
            strict_mode=effective_strict,
            allow_mutable_objects=allow_mutable_objects,
            memory_safety_margin=memory_safety_margin,
            visible_outputs=visible_outputs,
            visible_constants=visible_constants,
            previous_names=previous_names,
            parent_config=parent_config,
            copier=copier,
            node_name=node_name,
        )
        if resolve_only:
            return resolved
        try:
            self.parent._capture_prints(
                self._run_expression_code,
                registration.code,
                namespace,
            )
        except MemoryError as exc:
            raise inspection_memory_failure(
                node_name=node_name,
                function_name=registration.function_name,
                parameter_name=None,
                stage="executing the inspected expression",
                copy_started=copier.copy_started,
            ) from exc
        except ResolutionError:
            raise
        except Exception as exc:
            raise ExecutionError(
                f"Expression in block '{self.registration_name}' failed during "
                f"inspection: {type(exc).__name__}: {exc}"
            ) from exc
        values = [namespace[name] for name in registration.output_names]
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return tuple(values)

    def _resolve_inspected_registration(
        self,
        registration: FunctionRegistration,
        overrides: dict[str, Any],
        *,
        strict_mode: bool,
        allow_mutable_objects: bool,
        memory_safety_margin: float,
        visible_outputs: dict[str, Any],
        visible_constants: dict[str, Any],
        previous_names: set[str],
        parent_config: dict[str, Any],
        copier: InspectionCopier,
        node_name: str,
    ) -> ResolvedInspectionCall:
        if not allow_mutable_objects:
            candidates: list[InspectionCopyCandidate] = []
            self._resolve_inspection_callable(
                registration,
                overrides,
                strict_mode=strict_mode,
                allow_mutable_objects=allow_mutable_objects,
                visible_outputs=visible_outputs,
                visible_constants=visible_constants,
                previous_names=previous_names,
                parent_config=parent_config,
                copier=copier,
                node_name=node_name,
                materialize=False,
                planning=True,
                candidates=candidates,
            )
            self._preflight_inspection_memory(
                candidates,
                node_name=node_name,
                function_name=registration.function_name,
                safety_margin=memory_safety_margin,
            )
        return self._resolve_inspection_callable(
            registration,
            overrides,
            strict_mode=strict_mode,
            allow_mutable_objects=allow_mutable_objects,
            visible_outputs=visible_outputs,
            visible_constants=visible_constants,
            previous_names=previous_names,
            parent_config=parent_config,
            copier=copier,
            node_name=node_name,
        )

    def _resolve_inspected_expression(
        self,
        registration: ExpressionRegistration,
        overrides: dict[str, Any],
        *,
        strict_mode: bool,
        allow_mutable_objects: bool,
        memory_safety_margin: float,
        visible_outputs: dict[str, Any],
        visible_constants: dict[str, Any],
        previous_names: set[str],
        parent_config: dict[str, Any],
        copier: InspectionCopier,
        node_name: str,
    ) -> tuple[ResolvedInspectionCall, dict[str, Any]]:
        runtime_namespace: dict[str, Any] | None = None
        if not allow_mutable_objects:
            runtime_namespace = self.parent._build_expression_runtime_namespace(
                cache=False
            )
            candidates: list[InspectionCopyCandidate] = []
            self._resolve_inspection_expression(
                registration,
                overrides,
                strict_mode=strict_mode,
                allow_mutable_objects=allow_mutable_objects,
                visible_outputs=visible_outputs,
                visible_constants=visible_constants,
                previous_names=previous_names,
                parent_config=parent_config,
                copier=copier,
                node_name=node_name,
                materialize=False,
                planning=True,
                candidates=candidates,
                runtime_namespace=runtime_namespace,
            )
            self._preflight_inspection_memory(
                candidates,
                node_name=node_name,
                function_name=registration.function_name,
                safety_margin=memory_safety_margin,
            )
        return self._resolve_inspection_expression(
            registration,
            overrides,
            strict_mode=strict_mode,
            allow_mutable_objects=allow_mutable_objects,
            visible_outputs=visible_outputs,
            visible_constants=visible_constants,
            previous_names=previous_names,
            parent_config=parent_config,
            copier=copier,
            node_name=node_name,
            runtime_namespace=runtime_namespace,
        )

    def _preflight_inspection_memory(
        self,
        candidates: list[InspectionCopyCandidate],
        *,
        node_name: str,
        function_name: str,
        safety_margin: float,
    ) -> None:
        preflight_protected_inspection(
            candidates,
            node_name=node_name,
            function_name=function_name,
            safety_margin=safety_margin,
            pointer_reader=self._inspection_pointer_terminal,
        )

    def _inspection_pointer_terminal(self, pointer: OutputPointer) -> Any:
        _, terminal = resolve_pointer_chain(
            pointer.destination,
            self.parent._root_pipeline()._read_output_address,
        )
        return terminal

    def _select_inspection_registration(
        self,
        function_name: str | None,
        overrides: dict[str, Any],
    ) -> FunctionRegistration | ExpressionRegistration:
        if not self.functions:
            raise RegistrationError(
                f"Block '{self.registration_name}' has no registered functions or expressions"
            )
        if len(self.functions) == 1:
            registration = self.functions[0]
            if function_name is not None and registration.function_name != function_name:
                raise RegistrationError(
                    f"Block '{self.registration_name}' has no registration named "
                    f"'{function_name}'"
                )
            self._validate_inspection_override_names(registration, overrides)
            return registration
        if function_name is None:
            names = [registration.function_name for registration in self.functions]
            raise RegistrationError(
                f"Block '{self.registration_name}' contains multiple registered functions "
                f"{names}; function_name is required"
            )
        matches = [
            registration
            for registration in self.functions
            if registration.function_name == function_name
        ]
        if not matches:
            raise RegistrationError(
                f"Block '{self.registration_name}' has no registration named "
                f"'{function_name}'"
            )
        if len(matches) == 1:
            self._validate_inspection_override_names(matches[0], overrides)
            return matches[0]
        function_matches = [
            match for match in matches if isinstance(match, FunctionRegistration)
        ]
        if len(function_matches) != len(matches):
            raise RegistrationError(
                f"Registration name '{function_name}' is ambiguous in block "
                f"'{self.registration_name}'"
            )
        callable_obj = function_matches[0].callable_obj
        if any(
            match.callable_obj is not callable_obj
            for match in function_matches[1:]
        ):
            raise RegistrationError(
                f"Multiple registrations named '{function_name}' refer to different "
                "callable objects"
            )
        self._validate_inspection_override_names(function_matches[0], overrides)
        descriptors = [
            self._inspection_binding_descriptors(match, overrides)
            for match in function_matches
        ]
        differing = sorted(
            parameter_name
            for parameter_name in descriptors[0]
            if any(
                descriptor.get(parameter_name) != descriptors[0].get(parameter_name)
                for descriptor in descriptors[1:]
            )
        )
        if differing:
            raise RegistrationError(
                f"Multiple registrations named '{function_name}' have different input "
                f"bindings for {differing}; override those original callable parameters "
                "to select deterministically"
            )
        return function_matches[0]

    def _validate_inspection_override_names(
        self,
        registration: FunctionRegistration | ExpressionRegistration,
        overrides: dict[str, Any],
    ) -> None:
        if isinstance(registration, FunctionRegistration):
            accepted = set(callable_signature(registration.callable_obj).parameters)
        else:
            accepted = set(self._effective_expression_input_names(registration))
        unknown = sorted(set(overrides).difference(accepted))
        if unknown:
            raise ResolutionError(
                f"Unknown inspection override(s) {unknown} for "
                f"'{registration.function_name}'; accepted names are {sorted(accepted)}"
            )

    def _inspection_binding_descriptors(
        self,
        registration: FunctionRegistration,
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        descriptors: dict[str, Any] = {}
        signature = callable_signature(registration.callable_obj)
        effective_pos_name, effective_kw_name = effective_variadic_names(
            registration.callable_obj,
            var_pos_name=registration.var_pos_name,
            var_kw_name=registration.var_kw_name,
        )
        for parameter in signature.parameters.values():
            if parameter.name in overrides:
                descriptors[parameter.name] = ("inspection_override",)
            elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                helper = (
                    self.registered_args.get(effective_pos_name)
                    if effective_pos_name is not None
                    else None
                )
                descriptors[parameter.name] = (
                    "registered_args",
                    None if helper is None else tuple(helper.ordered_items),
                    effective_pos_name,
                )
            elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
                helper = (
                    self.registered_kwargs.get(effective_kw_name)
                    if effective_kw_name is not None
                    else None
                )
                descriptors[parameter.name] = (
                    "registered_kwargs",
                    None
                    if helper is None
                    else tuple(sorted(helper.mapping_dct.items())),
                    effective_kw_name,
                )
            elif parameter.name in registration.param_mapping:
                descriptors[parameter.name] = (
                    "mapped",
                    registration.param_mapping[parameter.name],
                )
            else:
                descriptors[parameter.name] = ("implicit", parameter.name)
        return descriptors

    def _validate_selected_inspection_dependencies(
        self,
        registration: FunctionRegistration | ExpressionRegistration,
        overrides: dict[str, Any],
        *,
        strict_mode: bool,
    ) -> None:
        if isinstance(registration, ExpressionRegistration):
            input_names = set(self._effective_expression_input_names(registration))
            input_names.difference_update(overrides)
            output_names = set(registration.output_names)
        else:
            input_names = self._inspection_pipeline_input_names(
                registration,
                overrides,
                strict_mode=strict_mode,
            )
            output_names = set(registration.produced_output_names)
        same_block_dependencies = self.declared_outputs().difference(
            output_names
        ).intersection(input_names)
        if same_block_dependencies:
            raise ExecutionError(
                f"Function '{registration.function_name}' depends on outputs from the "
                "same block, which cannot be resolved during parallel execution: "
                f"{sorted(same_block_dependencies)}"
            )

    def _inspection_pipeline_input_names(
        self,
        registration: FunctionRegistration,
        overrides: dict[str, Any],
        *,
        strict_mode: bool,
    ) -> set[str]:
        names: set[str] = set()
        signature = callable_signature(registration.callable_obj)
        effective_pos_name, effective_kw_name = effective_variadic_names(
            registration.callable_obj,
            var_pos_name=registration.var_pos_name,
            var_kw_name=registration.var_kw_name,
        )
        for parameter in signature.parameters.values():
            if parameter.name in overrides:
                continue
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                helper = (
                    self.registered_args.get(effective_pos_name)
                    if effective_pos_name is not None
                    else None
                )
                if helper is not None:
                    names.update(helper.ordered_items)
                elif not strict_mode:
                    names.add(effective_pos_name or parameter.name)
                continue
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                helper = (
                    self.registered_kwargs.get(effective_kw_name)
                    if effective_kw_name is not None
                    else None
                )
                if helper is not None:
                    names.update(helper.mapping_dct.values())
                elif not strict_mode:
                    names.add(effective_kw_name or parameter.name)
                continue
            if parameter.name in registration.param_mapping:
                mapped = registration.param_mapping[parameter.name]
                if mapped is not None:
                    names.add(mapped)
            elif not strict_mode and parameter.name != "logger":
                names.add(parameter.name)
        return names

    def _build_inspection_environment(
        self,
        *,
        upstream_outputs: dict[str, Any] | None,
        previous_outputs: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], set[str], dict[str, Any]]:
        visible_outputs = self.parent._visible_outputs_before_priority(
            self.execution_priority,
            upstream_outputs=upstream_outputs,
        )
        visible_constants = self.parent._visible_manual_values_before_priority(
            self.execution_priority
        )
        for constant_name in visible_constants:
            visible_outputs.pop(constant_name, None)
        previous = dict(
            self.parent.producer_outputs.get(self.registration_name, {})
            if previous_outputs is None
            else previous_outputs
        )
        visible_outputs.update(previous)
        return (
            visible_outputs,
            visible_constants,
            set(previous),
            self.parent._ancestor_config_values(),
        )

    def _resolve_inspection_callable(
        self,
        registration: FunctionRegistration,
        overrides: dict[str, Any],
        *,
        strict_mode: bool,
        allow_mutable_objects: bool,
        visible_outputs: dict[str, Any],
        visible_constants: dict[str, Any],
        previous_names: set[str],
        parent_config: dict[str, Any],
        copier: InspectionCopier,
        node_name: str,
        materialize: bool = True,
        planning: bool = False,
        candidates: list[InspectionCopyCandidate] | None = None,
    ) -> ResolvedInspectionCall:
        signature = callable_signature(registration.callable_obj)
        parameters = list(signature.parameters.values())
        effective_var_pos_name, effective_var_kw_name = effective_variadic_names(
            registration.callable_obj,
            var_pos_name=registration.var_pos_name,
            var_kw_name=registration.var_kw_name,
        )
        resolver_defaults = (
            {} if strict_mode else default_map(registration.callable_obj)
        )
        var_pos_index = next(
            (
                index
                for index, parameter in enumerate(parameters)
                if parameter.kind == inspect.Parameter.VAR_POSITIONAL
            ),
            None,
        )
        positional_args: list[Any] = []
        keyword_args: dict[str, Any] = {}
        sources: dict[str, Any] = {}
        loaded_artifacts: list[str] = []
        declared_output_names = set(visible_outputs).union(
            self.parent.list_declared_outputs(),
            self.declared_outputs(),
        )

        for index, parameter in enumerate(parameters):
            if parameter.name in overrides:
                value = overrides[parameter.name]
                source = ResolutionSource(
                    kind="inspect_override",
                    name=parameter.name,
                    mapped_from=parameter.name,
                )
                if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                    if not isinstance(value, (list, tuple)):
                        raise ResolutionError(
                            f"Inspection override for variadic parameter "
                            f"'{parameter.name}' must be a list or tuple"
                        )
                    positional_args.extend(value)
                    sources[parameter.name] = source
                    continue
                if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                    if not isinstance(value, dict):
                        raise ResolutionError(
                            f"Inspection override for variadic parameter "
                            f"'{parameter.name}' must be a dictionary"
                        )
                    overlap = set(value).intersection(keyword_args)
                    if overlap:
                        raise ResolutionError(
                            f"Inspection override for variadic parameter "
                            f"'{parameter.name}' conflicts with explicit arguments: "
                            f"{sorted(overlap)}"
                        )
                    keyword_args.update(value)
                    sources[parameter.name] = source
                    continue
                sources[parameter.name] = source
            elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                helper = (
                    self.registered_args.get(effective_var_pos_name)
                    if effective_var_pos_name is not None
                    else None
                )
                if helper is not None:
                    positional_values: list[Any] = []
                    positional_sources: list[ResolutionSource] = []
                    for input_name in helper.ordered_items:
                        value, item_source = self._resolve_and_copy_inspection_input(
                            input_name,
                            parameter.name,
                            registration,
                            visible_outputs,
                            visible_constants,
                            previous_names,
                            parent_config,
                            resolver_defaults,
                            loaded_artifacts,
                            declared_output_names,
                            copier,
                            node_name=node_name,
                            materialize=materialize,
                            planning=planning,
                            candidates=candidates,
                        )
                        positional_values.append(value)
                        positional_sources.append(
                            replace(item_source, kind="registered_args")
                        )
                    positional_args.extend(positional_values)
                    sources[parameter.name] = tuple(positional_sources)
                elif strict_mode:
                    sources[parameter.name] = tuple()
                else:
                    input_name = effective_var_pos_name or parameter.name
                    value, source = self._resolve_and_copy_inspection_input(
                        input_name,
                        parameter.name,
                        registration,
                        visible_outputs,
                        visible_constants,
                        previous_names,
                        parent_config,
                        resolver_defaults,
                        loaded_artifacts,
                        declared_output_names,
                        copier,
                        node_name=node_name,
                        materialize=materialize,
                        planning=planning,
                        candidates=candidates,
                        expected_container_kind="sequence",
                        allow_missing=True,
                        missing_value=[],
                    )
                    if planning and isinstance(value, (ArtifactRecord, OutputPointer)):
                        sources[parameter.name] = source
                        continue
                    if not isinstance(value, (list, tuple)):
                        raise ResolutionError(
                            f"Variadic positional argument '{input_name}' for function "
                            f"'{registration.function_name}' must resolve to a list or tuple"
                        )
                    positional_args.extend(value)
                    sources[parameter.name] = source
                continue
            elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
                helper = (
                    self.registered_kwargs.get(effective_var_kw_name)
                    if effective_var_kw_name is not None
                    else None
                )
                if helper is not None:
                    keyword_values: dict[str, Any] = {}
                    keyword_sources: dict[str, ResolutionSource] = {}
                    for key, input_name in helper.mapping_dct.items():
                        value, item_source = self._resolve_and_copy_inspection_input(
                            input_name,
                            parameter.name,
                            registration,
                            visible_outputs,
                            visible_constants,
                            previous_names,
                            parent_config,
                            resolver_defaults,
                            loaded_artifacts,
                            declared_output_names,
                            copier,
                            node_name=node_name,
                            materialize=materialize,
                            planning=planning,
                            candidates=candidates,
                        )
                        keyword_values[key] = value
                        keyword_sources[key] = replace(
                            item_source,
                            kind="registered_kwargs",
                        )
                    overlap = set(keyword_values).intersection(keyword_args)
                    if overlap:
                        raise ResolutionError(
                            f"Variadic keyword argument '{parameter.name}' conflicts "
                            f"with explicit arguments: {sorted(overlap)}"
                        )
                    keyword_args.update(keyword_values)
                    sources[parameter.name] = keyword_sources
                elif strict_mode:
                    sources[parameter.name] = {}
                else:
                    input_name = effective_var_kw_name or parameter.name
                    value, source = self._resolve_and_copy_inspection_input(
                        input_name,
                        parameter.name,
                        registration,
                        visible_outputs,
                        visible_constants,
                        previous_names,
                        parent_config,
                        resolver_defaults,
                        loaded_artifacts,
                        declared_output_names,
                        copier,
                        node_name=node_name,
                        materialize=materialize,
                        planning=planning,
                        candidates=candidates,
                        expected_container_kind="mapping",
                        allow_missing=True,
                        missing_value={},
                    )
                    if planning and isinstance(value, (ArtifactRecord, OutputPointer)):
                        sources[parameter.name] = source
                        continue
                    if not isinstance(value, dict):
                        raise ResolutionError(
                            f"Variadic keyword argument '{input_name}' for function "
                            f"'{registration.function_name}' must resolve to a dict"
                        )
                    overlap = set(value).intersection(keyword_args)
                    if overlap:
                        raise ResolutionError(
                            f"Variadic keyword argument '{input_name}' conflicts with "
                            f"explicit arguments: {sorted(overlap)}"
                        )
                    keyword_args.update(value)
                    sources[parameter.name] = source
                continue
            elif parameter.name in registration.param_mapping:
                input_name = registration.param_mapping[parameter.name]
                if input_name is None:
                    value = None
                    source = ResolutionSource(
                        kind="registered_value",
                        name=parameter.name,
                        mapped_from=parameter.name,
                    )
                else:
                    value, source = self._resolve_and_copy_inspection_input(
                        input_name,
                        parameter.name,
                        registration,
                        visible_outputs,
                        visible_constants,
                        previous_names,
                        parent_config,
                        resolver_defaults,
                        loaded_artifacts,
                        declared_output_names,
                        copier,
                        node_name=node_name,
                        materialize=materialize,
                        planning=planning,
                        candidates=candidates,
                    )
                sources[parameter.name] = source
            elif strict_mode:
                if parameter.name == "logger":
                    source = ResolutionSource(kind="logger", name="logger")
                    value, source = self._prepare_inspection_value(
                        self.parent.logger,
                        parameter_name=parameter.name,
                        source=source,
                        copier=copier,
                        node_name=node_name,
                        function_name=registration.function_name,
                        planning=planning,
                        candidates=candidates,
                    )
                elif parameter.default is not inspect.Parameter.empty:
                    source = ResolutionSource(
                        kind="function_default",
                        name=parameter.name,
                    )
                    value, source = self._prepare_inspection_value(
                        parameter.default,
                        parameter_name=parameter.name,
                        source=source,
                        copier=copier,
                        node_name=node_name,
                        function_name=registration.function_name,
                        planning=planning,
                        candidates=candidates,
                    )
                else:
                    raise ResolutionError(
                        f"Cannot resolve argument '{parameter.name}' for function "
                        f"'{registration.function_name}': strict mode requires an "
                        "explicit mapping, inspection override, or callable default"
                    )
                sources[parameter.name] = source
            else:
                value, source = self._resolve_and_copy_inspection_input(
                    parameter.name,
                    parameter.name,
                    registration,
                    visible_outputs,
                    visible_constants,
                    previous_names,
                    parent_config,
                    resolver_defaults,
                    loaded_artifacts,
                    declared_output_names,
                    copier,
                    node_name=node_name,
                    materialize=materialize,
                    planning=planning,
                    candidates=candidates,
                )
                sources[parameter.name] = source

            if parameter.kind == inspect.Parameter.POSITIONAL_ONLY or (
                parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                and var_pos_index is not None
                and index < var_pos_index
            ):
                positional_args.append(value)
            else:
                keyword_args[parameter.name] = value

        try:
            bound = signature.bind(*positional_args, **keyword_args)
            bound.apply_defaults()
        except TypeError as exc:
            raise ResolutionError(
                f"Cannot bind resolved arguments for function "
                f"'{registration.function_name}' in block "
                f"'{self.registration_name}': {exc}"
            ) from exc
        return ResolvedInspectionCall(
            args=tuple(positional_args),
            kwargs=keyword_args,
            arguments=dict(bound.arguments),
            sources=sources,
            callable=registration.callable_obj,
            function_name=registration.function_name,
            block_name=self.registration_name,
            node_name=node_name,
            strict_mode=strict_mode,
            allow_mutable_objects=allow_mutable_objects,
        )

    def _prepare_inspection_value(
        self,
        value: Any,
        *,
        parameter_name: str,
        source: ResolutionSource,
        copier: InspectionCopier,
        node_name: str,
        function_name: str,
        planning: bool,
        candidates: list[InspectionCopyCandidate] | None,
        caller_owned: bool = False,
        expected_container_kind: str | None = None,
    ) -> tuple[Any, ResolutionSource]:
        if planning:
            if (
                is_optuna_study(value)
                or is_optuna_sampler(value)
                or (
                    isinstance(value, ArtifactRecord)
                    and (
                        value.serializer == OPTUNA_STUDY_SERIALIZER
                        or value.metadata.get("optuna_type") == "sampler"
                    )
                )
                ):
                raise InspectionCopyError(
                    "This inspection input contains Optuna state (a study or sampler) "
                    "whose shared state cannot be isolated by protected copying. "
                    "To inspect it using its original state, call "
                    "inspect(..., allow_mutable_objects=True)."
                )
            if candidates is not None and copier.requires_copy(
                value,
                caller_owned=caller_owned,
            ):
                candidates.append(
                    InspectionCopyCandidate(
                        parameter_name=parameter_name,
                        value=value,
                        expected_container_kind=expected_container_kind,
                    )
                )
            return value, source
        try:
            return copier.prepare(
                value,
                parameter_name=parameter_name,
                source=source,
                caller_owned=caller_owned,
            )
        except MemoryError as exc:
            raise inspection_memory_failure(
                node_name=node_name,
                function_name=function_name,
                parameter_name=parameter_name,
                value=value,
                stage="copying",
                copy_started=copier.copy_started,
            ) from exc

    def _resolve_and_copy_inspection_input(
        self,
        input_name: str,
        parameter_name: str,
        registration: FunctionRegistration | str,
        visible_outputs: dict[str, Any],
        visible_constants: dict[str, Any],
        previous_names: set[str],
        parent_config: dict[str, Any],
        defaults: dict[str, Any],
        loaded_artifacts: list[str],
        declared_output_names: set[str],
        copier: InspectionCopier,
        *,
        node_name: str = "",
        materialize: bool = True,
        planning: bool = False,
        candidates: list[InspectionCopyCandidate] | None = None,
        expected_container_kind: str | None = None,
        allow_missing: bool = False,
        missing_value: Any = None,
    ) -> tuple[Any, ResolutionSource]:
        function_name = (
            registration.function_name
            if isinstance(registration, FunctionRegistration)
            else registration
        )
        inspection_outputs = visible_outputs
        pointer_materialized = False
        try:
            if input_name in previous_names:
                previous_value = visible_outputs.get(input_name)
                if isinstance(previous_value, OutputPointer) and materialize:
                    inspection_outputs = dict(visible_outputs)
                    inspection_outputs[input_name] = (
                        self.parent._materialize_stored_value(previous_value, "")
                    )
                    pointer_materialized = True
            value, source = self.parent._resolve_named_input_with_source(
                input_name,
                function_name,
                {},
                inspection_outputs,
                parent_config,
                defaults,
                loaded_artifacts,
                declared_output_names,
                allow_missing=allow_missing,
                missing_value=missing_value,
                visible_constants=visible_constants,
                same_node_previous_outputs=previous_names,
                materialize=materialize,
            )
        except PersistenceError as exc:
            if not isinstance(exc.__cause__, MemoryError):
                raise
            raise inspection_memory_failure(
                node_name=node_name,
                function_name=function_name,
                parameter_name=parameter_name,
                stage="resolving or materialising",
                copy_started=copier.copy_started,
            ) from exc.__cause__
        except MemoryError as exc:
            raise inspection_memory_failure(
                node_name=node_name,
                function_name=function_name,
                parameter_name=parameter_name,
                stage="resolving or materialising",
                copy_started=copier.copy_started,
            ) from exc
        source = replace(
            source,
            mapped_from=parameter_name,
            materialized=source.materialized or pointer_materialized,
        )
        return self._prepare_inspection_value(
            value,
            parameter_name=parameter_name,
            source=source,
            copier=copier,
            node_name=node_name,
            function_name=function_name,
            planning=planning,
            candidates=candidates,
            expected_container_kind=expected_container_kind,
        )

    def _resolve_inspection_expression(
        self,
        registration: ExpressionRegistration,
        overrides: dict[str, Any],
        *,
        strict_mode: bool,
        allow_mutable_objects: bool,
        visible_outputs: dict[str, Any],
        visible_constants: dict[str, Any],
        previous_names: set[str],
        parent_config: dict[str, Any],
        copier: InspectionCopier,
        node_name: str,
        materialize: bool = True,
        planning: bool = False,
        candidates: list[InspectionCopyCandidate] | None = None,
        runtime_namespace: dict[str, Any] | None = None,
    ) -> tuple[ResolvedInspectionCall, dict[str, Any]]:
        namespace = (
            self.parent._build_expression_runtime_namespace(cache=False)
            if runtime_namespace is None
            else dict(runtime_namespace)
        )
        arguments: dict[str, Any] = {}
        sources: dict[str, ResolutionSource] = {}
        loaded_artifacts: list[str] = []
        declared_output_names = set(visible_outputs).union(
            self.parent.list_declared_outputs(),
            self.declared_outputs(),
        )
        for input_name in self._effective_expression_input_names(registration):
            if input_name in overrides:
                value = overrides[input_name]
                source = ResolutionSource(
                    kind="inspect_override",
                    name=input_name,
                    mapped_from=input_name,
                )
            else:
                value, source = self._resolve_and_copy_inspection_input(
                    input_name,
                    input_name,
                    "expression",
                    visible_outputs,
                    visible_constants,
                    previous_names,
                    parent_config,
                    {},
                    loaded_artifacts,
                    declared_output_names,
                    copier,
                    node_name=node_name,
                    materialize=materialize,
                    planning=planning,
                    candidates=candidates,
                )
            arguments[input_name] = value
            sources[input_name] = source
            namespace[input_name] = value
        namespace["logger"] = self.parent.logger
        return (
            ResolvedInspectionCall(
                args=(),
                kwargs={},
                arguments=arguments,
                sources=sources,
                callable=None,
                function_name="expression",
                block_name=self.registration_name,
                node_name=node_name,
                strict_mode=strict_mode,
                allow_mutable_objects=allow_mutable_objects,
            ),
            namespace,
        )

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
            input_names = (
                self._effective_expression_input_names(registration)
                if isinstance(registration, ExpressionRegistration)
                else registration.input_names
            )
            registration_output_names = (
                registration.output_names
                if isinstance(registration, ExpressionRegistration)
                else registration.produced_output_names
            )
            same_block_dependencies = block_output_names.difference(
                registration_output_names
            ).intersection(input_names)
            if same_block_dependencies:
                raise ExecutionError(
                    f"Function '{registration.function_name}' depends on outputs from the same block, "
                    f"which cannot be resolved during parallel execution: {sorted(same_block_dependencies)}"
                )

        results: list[FunctionExecutionResult] = []
        if len(self.functions) == 1:
            registration = self.functions[0]
            results.append(
                self._execute_function(
                    registration,
                    run_id,
                    dict(visible_outputs),
                    overrides or {},
                    parent_config or {},
                    True,
                )
            )
        else:
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
                            False,
                        )
                    )
            for future in futures:
                results.append(future.result())

        produced_outputs: dict[str, Any] = {}
        for result in results:
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
                        torch_load_weights_only=self.parent.torch_load_weights_only,
                        optuna_db_path=self.parent._optuna_studies_db_path_for_storage(),
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
        registration: FunctionRegistration | ExpressionRegistration,
        run_id: str,
        visible_outputs: dict[str, Any],
        overrides: dict[str, Any],
        parent_config: dict[str, Any],
        capture_prints: bool,
    ) -> FunctionExecutionResult:
        match registration:
            case FunctionRegistration():
                return self._execute_callable_registration(
                    registration,
                    visible_outputs,
                    overrides,
                    parent_config,
                    capture_prints,
                )
            case ExpressionRegistration():
                return self._execute_expression_registration(
                    registration,
                    visible_outputs,
                    overrides,
                    parent_config,
                    capture_prints,
                )
            case _:
                del run_id
                raise ExecutionError(
                    f"Unsupported registration type in block '{self.registration_name}'"
                )

    def _execute_callable_registration(
        self,
        registration: FunctionRegistration,
        visible_outputs: dict[str, Any],
        overrides: dict[str, Any],
        parent_config: dict[str, Any],
        capture_prints: bool,
    ) -> FunctionExecutionResult:
        positional_args, keyword_args, loaded_artifacts = self.parent._prepare_call_arguments(
            registration,
            overrides,
            visible_outputs,
            parent_config,
            block=self,
        )
        try:
            if capture_prints:
                result = self.parent._capture_prints(
                    registration.callable_obj,
                    *positional_args,
                    **keyword_args,
                )
            else:
                result = registration.callable_obj(*positional_args, **keyword_args)
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

    def _execute_expression_registration(
        self,
        registration: ExpressionRegistration,
        visible_outputs: dict[str, Any],
        overrides: dict[str, Any],
        parent_config: dict[str, Any],
        capture_prints: bool,
    ) -> FunctionExecutionResult:
        loaded_artifacts: list[str] = []
        declared_output_names = set(visible_outputs).union(self.parent.list_declared_outputs())
        declared_output_names.update(self.declared_outputs())
        namespace = self.parent._build_expression_runtime_namespace()
        input_names = self._effective_expression_input_names(registration)
        namespace.update({
            input_name: self.parent._resolve_named_input(
                input_name,
                registration.function_name,
                overrides,
                visible_outputs,
                parent_config,
                {},
                loaded_artifacts,
                declared_output_names,
            )
            for input_name in input_names
        })
        namespace["logger"] = self.parent.logger
        try:
            if capture_prints:
                self.parent._capture_prints(self._run_expression_code, registration.code, namespace)
            else:
                self._run_expression_code(registration.code, namespace)
        except ResolutionError:
            raise
        except Exception as exc:
            raise ExecutionError(
                f"Expression in block '{self.registration_name}' failed: {type(exc).__name__}: {exc}"
            ) from exc

        outputs = {
            output_name: namespace[output_name]
            for output_name in registration.output_names
        }
        return FunctionExecutionResult(
            function_name=registration.function_name,
            outputs=outputs,
            loaded_artifact_inputs=loaded_artifacts,
        )

    @staticmethod
    def _run_expression_code(code: str, namespace: dict[str, Any]) -> None:
        exec(compile(code, "<pipeline-expression>", "exec"), {}, namespace)

    @staticmethod
    def _normalize_outputs(registration: FunctionRegistration, result: Any) -> dict[str, Any]:
        if len(registration.output_names) == 0:
            return {}
        if len(registration.output_names) == 1:
            output_name = registration.output_names[0]
            return (
                {}
                if registration.ignore_underscore_outputs and output_name == "_"
                else {output_name: result}
            )

        callable_label = registration.import_path or registration.function_name
        if not isinstance(result, (tuple, list)):
            raise ExecutionError(
                f"Function '{registration.function_name}' ({callable_label}) declared multiple output slots {registration.output_names} "
                f"but returned {type(result).__name__}: {ExecutionBlock._preview_value(result)}"
            )
        if len(result) != len(registration.output_names):
            raise ExecutionError(
                f"Function '{registration.function_name}' ({callable_label}) returned {len(result)} values but "
                f"{len(registration.output_names)} output slots were declared: {registration.output_names}"
            )
        return {
            output_name: output_value
            for output_name, output_value in zip(
                registration.output_names,
                result,
                strict=True,
            )
            if not registration.ignore_underscore_outputs or output_name != "_"
        }

    @staticmethod
    def _preview_value(value: Any, max_length: int = 200) -> str:
        preview = repr(value)
        if len(preview) > max_length:
            return preview[: max_length - 3] + "..."
        return preview
