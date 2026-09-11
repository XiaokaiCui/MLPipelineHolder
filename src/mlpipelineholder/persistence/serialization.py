"""Config value and dataclass serialization helpers for pipeline persistence."""

from __future__ import annotations

import pickle
import sys
import warnings
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from ..core.models import (
    CallableValueReference,
    RuntimeCallableReference,
    RuntimeValueReference,
)
from ..exceptions import PersistenceError, RegistrationError
from ..execution.function_registry import resolve_callable


class SerializationMixin:
    """Serialize and reconstruct configuration values and dataclasses."""

    if TYPE_CHECKING:
        para_value_dict: dict[str, Any] = {}
        manual_values: dict[str, Any] = {}
        producer_outputs: dict[str, dict[str, Any]] = {}
        parent_pipeline: Any = None

    @staticmethod
    def _callable_reference_round_trips(value: Any, import_path: str) -> bool:
        try:
            resolved_callable, _, _ = resolve_callable(import_path)
        except Exception:
            return False
        return resolved_callable is value

    @staticmethod
    def _restore_callable_value(reference: CallableValueReference) -> Any:
        callable_obj, _, _ = resolve_callable(reference.import_path)
        return callable_obj

    @staticmethod
    def _restore_runtime_callable(
        reference: RuntimeCallableReference,
        owner_label: str,
    ) -> Any:
        main_module = sys.modules.get("__main__")
        callable_obj = (
            None
            if main_module is None
            else getattr(main_module, reference.callable_name, None)
        )
        if not callable(callable_obj):
            raise PersistenceError(
                f"Required runtime callable '{reference.callable_name}' for {owner_label} "
                "is unavailable in __main__; import or define it before loading the pipeline"
            )
        return callable_obj

    @staticmethod
    def _serialize_config_for_save(config: Any) -> Any:
        return SerializationMixin._serialize_config_value(config)

    @staticmethod
    def _serialize_config_value(value: Any) -> Any:
        if callable(value):
            try:
                _, import_path, callable_name = resolve_callable(value)
            except RegistrationError:
                import_path = None
                callable_name = getattr(value, "__name__", type(value).__name__)
            if import_path is not None and import_path.startswith("__main__."):
                return RuntimeCallableReference(callable_name=callable_name)
            if (
                import_path is not None
                and SerializationMixin._callable_reference_round_trips(value, import_path)
            ):
                return CallableValueReference(
                    callable_name=callable_name,
                    import_path=import_path,
                )
            warnings.warn(
                f"Callable config value '{callable_name}' is not importable; saving a reference placeholder instead.",
                stacklevel=2,
            )
            return RuntimeValueReference(
                type_name=type(value).__name__,
                repr_text=repr(value),
                reason="callable config value is not importable during save_pipeline",
            )
        if isinstance(value, dict):
            return {
                "__pipeline_serialized_config__": True,
                "kind": "dict",
                "data": {
                    key: SerializationMixin._serialize_config_value(item)
                    for key, item in value.items()
                },
            }
        if is_dataclass(value) and not isinstance(value, type):
            return {
                "__pipeline_serialized_config__": True,
                "kind": "namespace",
                "class_name": type(value).__name__,
                "module": type(value).__module__,
                "data": {
                    key: SerializationMixin._serialize_config_value(item)
                    for key, item in SerializationMixin._config_object_as_dict(value).items()
                },
            }
        if hasattr(value, "__dict__"):
            return {
                "__pipeline_serialized_config__": True,
                "kind": "namespace",
                "class_name": type(value).__name__,
                "module": type(value).__module__,
                "data": {
                    key: SerializationMixin._serialize_config_value(item)
                    for key, item in vars(value).items()
                },
            }
        if isinstance(value, list):
            return [SerializationMixin._serialize_config_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(SerializationMixin._serialize_config_value(item) for item in value)
        return value

    @staticmethod
    def _deserialize_saved_config(
        saved_config: Any,
        *,
        verbose: bool = False,
        warn: Any | None = None,
    ) -> Any:
        return SerializationMixin._deserialize_config_value(
            saved_config,
            verbose=verbose,
            warn=warn,
        )

    @staticmethod
    def _deserialize_config_value(
        value: Any,
        *,
        verbose: bool = False,
        warn: Any | None = None,
    ) -> Any:
        if isinstance(value, CallableValueReference):
            return SerializationMixin._restore_callable_value(value)
        if isinstance(value, RuntimeCallableReference):
            return SerializationMixin._restore_runtime_callable(
                value,
                "configuration value",
            )
        if not (
            isinstance(value, dict)
            and value.get("__pipeline_serialized_config__") is True
        ):
            if isinstance(value, dict):
                return {
                    key: SerializationMixin._deserialize_config_value(
                        item,
                        verbose=verbose,
                        warn=warn,
                    )
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [
                    SerializationMixin._deserialize_config_value(
                        item,
                        verbose=verbose,
                        warn=warn,
                    )
                    for item in value
                ]
            if isinstance(value, tuple):
                return tuple(
                    SerializationMixin._deserialize_config_value(
                        item,
                        verbose=verbose,
                        warn=warn,
                    )
                    for item in value
                )
            return value
        kind = value.get("kind")
        if kind == "dict":
            return {
                key: SerializationMixin._deserialize_config_value(
                    item,
                    verbose=verbose,
                    warn=warn,
                )
                for key, item in dict(value.get("data", {})).items()
            }
        if kind == "namespace":
            data = {
                key: SerializationMixin._deserialize_config_value(
                    item,
                    verbose=verbose,
                    warn=warn,
                )
                for key, item in dict(value.get("data", {})).items()
            }
            return SerializationMixin._reconstruct_dataclass(
                value.get("class_name"),
                data,
                verbose=verbose,
                module_name=value.get("module"),
                warn=warn,
            )
        return value

    @classmethod
    def _serialize_dataclass_field_value(cls, value: Any) -> Any:
        """Return a picklable structured representation of one dataclass field value.

        Picklable values are kept as-is; unpicklable values are converted into
        reconstructable references (callables, nested dataclasses, dict-like
        objects) or a last-resort placeholder.
        """
        if callable(value):
            return cls._serialize_config_value(value)
        try:
            pickle.dumps(value)
            return value
        except Exception:
            pass
        if isinstance(value, dict):
            return {
                key: cls._serialize_dataclass_field_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._serialize_dataclass_field_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._serialize_dataclass_field_value(item) for item in value)
        serialized = cls._serialize_config_value(value)
        if isinstance(
            serialized,
            (CallableValueReference, RuntimeCallableReference, RuntimeValueReference),
        ):
            return serialized
        try:
            pickle.dumps(serialized)
            return serialized
        except Exception:
            return RuntimeValueReference(
                type_name=type(value).__name__,
                repr_text=repr(value),
                reason="dataclass field not directly serializable during save_pipeline",
            )

    @staticmethod
    def _find_dataclass_class(
        class_name: str,
        module_name: str | None = None,
    ) -> type | None:
        """Locate an importable pure dataclass by name.

        When the saved module is known, the class is looked up there first;
        otherwise every loaded module's attributes are scanned. ``__main__``
        definitions (notebook-local classes) are preferred over ambiguous
        same-name matches.
        """
        if module_name is not None:
            module = sys.modules.get(module_name)
            if module is not None:
                candidate = getattr(module, class_name, None)
                if (
                    isinstance(candidate, type)
                    and is_dataclass(candidate)
                    and candidate.__name__ == class_name
                ):
                    return candidate
        candidates: list[type] = []
        for module in sys.modules.values():
            candidate = getattr(module, class_name, None)
            if (
                isinstance(candidate, type)
                and is_dataclass(candidate)
                and candidate.__name__ == class_name
            ):
                candidates.append(candidate)
        main_candidates = [
            candidate
            for candidate in candidates
            if getattr(candidate, "__module__", None) == "__main__"
        ]
        if main_candidates:
            return main_candidates[0]
        if candidates:
            return candidates[0]
        return None

    def _find_pipeline_dataclass_class(
        self,
        class_name: str,
        module_name: str | None = None,
    ) -> type | None:
        """Locate a dataclass class among pipeline-visible runtime values.

        Placeholder recovery re-runs blocks, which can produce dynamically
        generated classes (for example a factory function returning a new
        dataclass) that exist only as pipeline values after load. Checks this
        pipeline's visible state and producer outputs, walking up to ancestor
        pipelines, for a dataclass class matching ``class_name`` (and
        ``module_name`` when known).
        """
        current: Any = self
        while current is not None:
            for mapping in (current.para_value_dict, current.manual_values):
                for value in mapping.values():
                    candidate = SerializationMixin._match_dataclass_class_value(
                        value,
                        class_name,
                        module_name,
                    )
                    if candidate is not None:
                        return candidate
            for outputs in current.producer_outputs.values():
                for value in outputs.values():
                    candidate = SerializationMixin._match_dataclass_class_value(
                        value,
                        class_name,
                        module_name,
                    )
                    if candidate is not None:
                        return candidate
            current = current.parent_pipeline
        return None

    @staticmethod
    def _match_dataclass_class_value(
        value: Any,
        class_name: str,
        module_name: str | None,
    ) -> type | None:
        """Return ``value`` when it is a dataclass class matching the criteria."""
        if not isinstance(value, type) or not is_dataclass(value):
            return None
        if value.__name__ != class_name:
            return None
        if module_name is not None and getattr(value, "__module__", None) != module_name:
            return None
        return value

    @staticmethod
    def _reconstruct_dataclass(
        class_name: str | None,
        data: dict[str, Any],
        *,
        verbose: bool,
        module_name: str | None = None,
        warn: Any | None = None,
        pipeline: Any | None = None,
    ) -> Any:
        """Rebuild a saved pure dataclass from its structured fields.

        Returns a real dataclass instance when the class is importable and can be
        constructed from the saved fields; otherwise falls back to a
        ``SimpleNamespace`` (with a verbose-gated warning when ``warn`` is given).
        When ``pipeline`` is given, classes visible as pipeline runtime values
        (for example dynamically generated classes produced by a block that was
        re-run during placeholder recovery) are also considered.
        """
        if class_name is not None:
            candidate_class = SerializationMixin._find_dataclass_class(
                class_name,
                module_name=module_name,
            )
            if candidate_class is None and pipeline is not None:
                candidate_class = pipeline._find_pipeline_dataclass_class(
                    class_name,
                    module_name=module_name,
                )
            if candidate_class is not None:
                init_field_names = {
                    field.name for field in fields(candidate_class) if field.init
                }
                try:
                    return candidate_class(
                        **{
                            key: value
                            for key, value in data.items()
                            if key in init_field_names
                        }
                    )
                except TypeError:
                    pass
        if verbose and warn is not None:
            warn(
                f"Saved dataclass '{class_name}' could not be reconstructed at load "
                "(class is not importable as a pure dataclass, its fields changed, or the "
                "saved fields cannot be passed to its constructor); "
                "using a SimpleNamespace fallback instead."
            )
        return SimpleNamespace(**data)

    @staticmethod
    def _config_object_as_dict(config_obj: Any) -> dict[str, Any]:
        # Shallow field extraction: nested dataclass values stay real instances so
        # the recursive value serializer can preserve their identity (asdict()
        # would collapse them into plain dicts).
        config_dict = {
            field.name: getattr(config_obj, field.name)
            for field in fields(config_obj)
        }
        extra_attrs = {
            key: value
            for key, value in vars(config_obj).items()
            if key not in config_dict
        }
        config_dict.update(extra_attrs)
        return config_dict
