from __future__ import annotations

import ast
import builtins
from concurrent.futures import ThreadPoolExecutor
import inspect
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Final

from .exceptions import ExecutionError, RegistrationError, ResolutionError
from .function_registry import callable_signature, infer_declared_output_count, inspect_input_names, rename_args, resolve_callable
from .function_registry import callable_identity_matches
from .function_registry import inspect_exposed_input_names
from .models import (
    BlockArgsRegistration,
    BlockKwargsRegistration,
    ExpressionRegistration,
    FunctionExecutionResult,
    FunctionRegistration,
)

if TYPE_CHECKING:
    from .pipeline_handler import PipelineHandler


_ALLOWED_EXPRESSION_BUILTIN_NAMES: Final[set[str]] = {
    name
    for name in dir(builtins)
    if not name.startswith("_")
}


class ExecutionBlock:
    """Represents one priority level whose registered functions run in parallel."""

    def __init__(
        self, parent: PipelineHandler, registration_name: str, execution_priority: float
    ) -> None:
        self.parent = parent
        self.registration_name = registration_name
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
        forced: bool = False,
        warn_on_input_mutation: bool = False,
    ) -> Any:
        if self.parent._is_atom:
            raise RegistrationError(
                f"Atom pipeline '{self.parent.registration_name}' is immutable "
                "and cannot accept new expressions"
            )
        return self._register_expression_strict(
            code,
            output_variable_name=output_variable_name,
            save_to_disk=save_to_disk,
            forced=forced,
            warn_on_input_mutation=warn_on_input_mutation,
        )

    def _register_expression_strict(
        self,
        code: str,
        *,
        output_variable_name: str | None,
        save_to_disk: bool,
        forced: bool,
        warn_on_input_mutation: bool,
    ) -> ExpressionRegistration:
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
            ):
                return existing
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
            raise RegistrationError(f"Invalid expression syntax: {exc}") from exc
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
        if self.parent._is_atom:
            raise RegistrationError(
                f"Atom pipeline '{self.parent.registration_name}' is immutable "
                "and cannot accept new args helpers"
            )
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
        if self.parent._is_atom:
            raise RegistrationError(
                f"Atom pipeline '{self.parent.registration_name}' is immutable "
                "and cannot accept new kwargs helpers"
            )
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
        param_mapping: dict[str, str | None] | None = None,
        var_pos_name: str | None = None,
        var_kw_name: str | None = None,
        forced: bool = False,
    ) -> Any:
        if self.parent._is_atom:
            raise RegistrationError(
                f"Atom pipeline '{self.parent.registration_name}' is immutable "
                "and cannot accept new functions"
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
                list(existing_registration.output_names),
                list(registration.output_names),
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
        args_state, kwargs_state = self._variadic_registration_state(
            var_pos_name,
            var_kw_name,
        )
        return (
            existing.output_names == new_output_names
            and existing.save_to_disk == set(save_to_disk or [])
            and existing.param_mapping == dict(param_mapping or {})
            and existing.var_pos_name == var_pos_name
            and existing.var_kw_name == var_kw_name
            and existing.args_registration_state == args_state
            and existing.kwargs_registration_state == kwargs_state
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
            if registration is not replacing
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
        args_state, kwargs_state = self._variadic_registration_state(
            var_pos_name,
            var_kw_name,
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
            input_names=input_names,
            output_names=output_names,
            save_to_disk=disk_names,
            param_mapping=dict(param_mapping or {}),
            var_pos_name=var_pos_name,
            var_kw_name=var_kw_name,
            args_registration_state=args_state,
            kwargs_registration_state=kwargs_state,
        )
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
                if name in self.parent._registration_disk_backed_names()
                and name not in registration.output_names
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
        kwargs_registration = None
        if registration.var_kw_name is not None:
            kwargs_registration = self.registered_kwargs.get(registration.var_kw_name)
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

        # Check 10: a param_mapping value is not found in config, visible
        # output values, or visible manual values. Literal None mapping and
        # 'logger' are always resolvable.
        if visible_names is None:
            visible_names = self._registration_visible_names()
        for key, value in registration.param_mapping.items():
            if value is None or value == "logger":
                continue
            if value not in visible_names:
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
        if self.parent._is_atom:
            raise RegistrationError(
                f"Atom pipeline '{self.parent.registration_name}' is immutable "
                "and cannot remove functions"
            )
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
            input_names = (
                self._effective_expression_input_names(registration)
                if isinstance(registration, ExpressionRegistration)
                else registration.input_names
            )
            same_block_dependencies = block_output_names.difference(
                registration.output_names
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
