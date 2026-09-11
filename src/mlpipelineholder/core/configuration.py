"""Configuration access, backup recovery, and runtime controls."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

from ..exceptions import PersistenceError, RegistrationError, ResolutionError
from .models import DataclassValueReference, RuntimeValueReference


class ConfigurationMixin:
    """Configuration access and runtime-control APIs for PipelineHolder."""

    if TYPE_CHECKING:
        _is_atom: bool = False
        parent_pipeline: Any = None
        logger: Any = None
        manual_values: Any = None
        config: Any = None
        _invalidation_forbidden: bool = False

        def _require_owned_config(self) -> None: ...
        def list_declared_outputs(self) -> set[str]: ...
        def _ancestor_manual_values(self) -> dict[str, Any]: ...
        def _ancestor_config_values(self) -> dict[str, Any]: ...
        def _set_config_value(self, field_name: str, value: Any) -> None: ...
        @staticmethod
        def _validate_config_value_picklable(field_name: str, value: Any) -> None: ...
        @staticmethod
        def _validate_builtin_name_conflict(name: str, owner_label: str) -> None: ...
        def _visible_config_names(self) -> set[str]: ...
        def _iter_attached_pipelines(self) -> list[Any]: ...

        print_capture_mode: str = "tee"
        torch_load_weights_only: bool = False

    def set_config(self, config_name: str, new_config_value: Any) -> None:
        self.set_configs({config_name: new_config_value})

    def set_configs(self, overrides: dict[str, Any]) -> None:
        self._require_owned_config()
        declared_outputs = self.list_declared_outputs()
        manual_names = set(self.manual_values) | set(self._ancestor_manual_values())
        for field_name, value in overrides.items():
            self._validate_builtin_name_conflict(field_name, owner_label="configuration")
            self._validate_config_value_picklable(field_name, value)
            if field_name in declared_outputs:
                self.logger.warning(
                    f"Skipped config update for '{field_name}' because it conflicts with a declared output"
                )
                continue
            if field_name in manual_names:
                self.logger.warning(
                    f"Skipped config update for '{field_name}' because it conflicts with a manual value"
                )
                continue
            self._set_config_value(field_name, value)

    def update_config(self, config_name: str, new_config_value: Any) -> None:
        self.update_configs({config_name: new_config_value})

    def update_configs(self, overrides: dict[str, Any]) -> None:
        self._require_owned_config()
        config_names = self.get_full_config()
        declared_outputs = self.list_declared_outputs()
        manual_names = set(self.manual_values) | set(self._ancestor_manual_values())
        for field_name, value in overrides.items():
            self._validate_builtin_name_conflict(field_name, owner_label="configuration")
            if field_name not in config_names:
                raise ResolutionError(f"Unknown config field: {field_name}")
            self._validate_config_value_picklable(field_name, value)
            if field_name in declared_outputs:
                self.logger.warning(
                    f"Skipped config update for '{field_name}' because it conflicts with a declared output"
                )
                continue
            if field_name in manual_names:
                self.logger.warning(
                    f"Skipped config update for '{field_name}' because it conflicts with a manual value"
                )
                continue
            self._set_config_value(field_name, value)

    def recover_variable_from_backup(
        self: Any,
        name: str,
        *,
        pipeline_name: str | None = None,
    ) -> None:
        from ..persistence.backup.service import recover_variable_from_backup

        recover_variable_from_backup(self, name, pipeline_name=pipeline_name)

    def get_full_config(self) -> dict[str, Any]:
        if self._is_atom and self.parent_pipeline is not None:
            return self.parent_pipeline.get_full_config()
        return dict(self._ancestor_config_values(), **self.config_as_dict())

    def get_config_value(self, field_name: str) -> Any:
        config = self.get_full_config()
        if field_name not in config:
            raise ResolutionError(f"Unknown config field: {field_name}")
        value = config[field_name]
        if isinstance(value, (RuntimeValueReference, DataclassValueReference)):
            raise ResolutionError(
                f"Cannot get config field '{field_name}': it was saved as a placeholder "
                f"({value.reason}) and cannot be restored"
            )
        return value

    def get_config(self, config_name: str) -> Any:
        return self.get_config_value(config_name)

    def list_visible_config(self) -> set[str]:
        """Return a detached set of visible configuration field names.

        Local plus ancestor config keys count by stored key, including ``None``
        values, without materialization. Atom pipelines delegate to their
        parent's visible config names. Sibling and descendant configs are
        excluded.
        """
        return self._visible_config_names()

    def has_visible_config(self, config_name: str) -> bool:
        """Report key-based config visibility without materialization or side effects."""
        return config_name in self._visible_config_names()

    def recover_config_from_backup(self: Any, name: str) -> None:
        self._require_owned_config()
        from ..persistence.backup.service import recover_config_from_backup

        recover_config_from_backup(self, name)

    def config_as_dict(self) -> dict[str, Any]:
        if is_dataclass(self.config) and not isinstance(self.config, type):
            config_dict = {
                field.name: getattr(self.config, field.name)
                for field in fields(self.config)
            }
            extra_attrs = {
                key: value
                for key, value in vars(self.config).items()
                if key not in config_dict
            }
            config_dict.update(extra_attrs)
            return config_dict
        if isinstance(self.config, dict):
            return dict(self.config)
        if hasattr(self.config, "__dict__"):
            return dict(vars(self.config))
        raise PersistenceError("Configuration object is not serializable to dict")

    def _config_has_field(self, config_obj: Any, field_name: str) -> bool:
        if is_dataclass(config_obj) and not isinstance(config_obj, type):
            return any(
                field.name == field_name for field in config_obj.__dataclass_fields__.values()
            )
        if isinstance(config_obj, dict):
            return field_name in config_obj
        return hasattr(config_obj, field_name)

    def _config_value(self, config_obj: Any, field_name: str) -> Any:
        if is_dataclass(config_obj) and not isinstance(config_obj, type):
            return getattr(config_obj, field_name)
        if isinstance(config_obj, dict):
            return config_obj[field_name]
        return getattr(config_obj, field_name)
