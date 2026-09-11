"""Expression runtime definitions and per-level gate input digests."""

from __future__ import annotations

import ast
import builtins
from textwrap import dedent
from typing import TYPE_CHECKING, Any

from ..exceptions import PersistenceError, RegistrationError


class ExpressionRuntimeMixin:
    """Validating, building, and exposing the restricted expression runtime."""

    if TYPE_CHECKING:
        config: Any = None
        parent_pipeline: Any = None
        manual_values: dict[str, Any] = {}

        def _incoming_parent_outputs(self) -> dict[str, Any]: ...
        def _ancestor_manual_values(self) -> dict[str, Any]: ...
        @staticmethod
        def _config_name_mapping(config_obj: Any) -> dict[str, Any]: ...
        @classmethod
        def _is_mutable_value(cls, value: Any) -> bool: ...
        @staticmethod
        def _copy_value(value: Any) -> Any: ...

    @staticmethod
    def _validate_expression_runtime_code(code: str) -> None:
        try:
            parsed = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            raise RegistrationError(f"Invalid expression runtime syntax: {exc}") from exc
        for node in parsed.body:
            if isinstance(node, ast.Import):
                continue
            if isinstance(node, ast.ImportFrom):
                if node.level != 0 or node.module is None:
                    raise RegistrationError(
                        "Expression runtime only supports absolute imports"
                    )
                if any(alias.name == "*" for alias in node.names):
                    raise RegistrationError(
                        "Expression runtime does not support wildcard imports"
                    )
                continue
            raise RegistrationError(
                "Expression runtime only supports import and from-import statements"
            )

    @staticmethod
    def _normalize_expression_runtime_code(code: str) -> str:
        normalized = dedent(code).strip()
        if not normalized:
            raise RegistrationError("Expression runtime code cannot be empty")
        return normalized

    def _effective_expression_runtime_owner(self) -> Any:
        current: Any = self
        while current is not None:
            if current.expression_runtime_code is not None:
                return current
            current = current.parent_pipeline
        return None

    def _expression_runtime_defined_names(self) -> set[str]:
        owner = self._effective_expression_runtime_owner()
        if owner is None or owner.expression_runtime_code is None:
            return set()
        if owner._expression_runtime_defined_names_cache is not None:
            return set(owner._expression_runtime_defined_names_cache)
        parsed = ast.parse(owner.expression_runtime_code, mode="exec")
        names: set[str] = set()
        for node in parsed.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split(".")[0])
                continue
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names.add(alias.asname or alias.name)
        owner._expression_runtime_defined_names_cache = names
        return set(names)

    def _build_expression_runtime_namespace(self) -> dict[str, Any]:
        owner = self._effective_expression_runtime_owner()
        if owner is None or owner.expression_runtime_code is None:
            return {}
        if owner._expression_runtime_namespace_cache is not None:
            return dict(owner._expression_runtime_namespace_cache)
        globals_namespace: dict[str, Any] = {"__builtins__": builtins.__dict__}
        try:
            exec(owner.expression_runtime_code, globals_namespace, globals_namespace)
        except Exception as exc:
            raise PersistenceError(
                f"Failed to build expression runtime for pipeline '{owner.registration_name}': {type(exc).__name__}: {exc}"
            ) from exc
        runtime_namespace = {
            key: value
            for key, value in globals_namespace.items()
            if key != "__builtins__"
        }
        owner._expression_runtime_namespace_cache = runtime_namespace
        return dict(runtime_namespace)

    def _gate_level_input_digest(self) -> tuple[tuple[str, Any], ...] | None:
        """Snapshot every state this pipeline's own gate can read.

        Captures the incoming parent outputs, own and ancestor config fields,
        and own and ancestor manual values. Mutable values are copied so an
        in-place change invalidates the cached gate result. If a value cannot
        be copied safely, caching is disabled for that gate level.
        """
        entries: list[tuple[str, Any]] = []

        def append_snapshot(name: str, value: Any) -> bool:
            if not self._is_mutable_value(value):
                entries.append((name, value))
                return True
            try:
                snapshot = self._copy_value(value)
            except Exception:
                return False
            entries.append((name, snapshot))
            return True

        for name, value in self._incoming_parent_outputs().items():
            if not append_snapshot(name, value):
                return None
        for name, value in self._config_name_mapping(self.config).items():
            if not append_snapshot(f"config:{name}", value):
                return None
        ancestor = self.parent_pipeline
        while ancestor is not None:
            for name, value in self._config_name_mapping(ancestor.config).items():
                if not append_snapshot(
                    f"ancestor_config:{ancestor.registration_name}:{name}",
                    value,
                ):
                    return None
            ancestor = ancestor.parent_pipeline
        for name, value in self.manual_values.items():
            if not append_snapshot(f"manual:{name}", value):
                return None
        for name, value in self._ancestor_manual_values().items():
            if not append_snapshot(f"ancestor_manual:{name}", value):
                return None
        return tuple(sorted(entries, key=lambda item: item[0]))
