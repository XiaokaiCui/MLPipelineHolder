from __future__ import annotations

import importlib
import importlib.util
import unittest
from io import BytesIO

LEGACY_FLAT_MODULES = (
    "artifact_recovery",
    "artifact_store",
    "backup_recovery",
    "backup_recovery_service",
    "backup_snapshot",
    "backup_value_resolver",
    "code_comparison",
    "execution_block",
    "function_registry",
    "gate_block",
    "logger",
    "models",
    "naming",
    "object_storage",
    "optuna_api",
    "optuna_sqlite",
    "optuna_support",
    "output_pointers",
    "pipeline_handler",
    "serializers",
)

CANONICAL_IMPORTS = (
    ("mlpipelineholder.core.models", "ArtifactRecord"),
    ("mlpipelineholder.execution.block", "ExecutionBlock"),
    ("mlpipelineholder.execution.gate_block", "GateBlock"),
    ("mlpipelineholder.state.output_pointers", "OutputPointer"),
    ("mlpipelineholder.persistence.artifacts.store", "ArtifactStore"),
    ("mlpipelineholder.presentation.logger", "PipelineLogger"),
)


class CompatibilityContractTests(unittest.TestCase):
    def test_root_alias_identity(self) -> None:
        from mlpipelineholder import PipelineHandler, PipelineHolder

        self.assertIs(PipelineHandler, PipelineHolder)

    def test_canonical_subpackage_imports_resolve(self) -> None:
        for module_name, attribute in CANONICAL_IMPORTS:
            module = importlib.import_module(module_name)
            self.assertTrue(
                hasattr(module, attribute),
                f"{module_name} does not expose {attribute}",
            )

    def test_legacy_flat_modules_are_intentionally_absent(self) -> None:
        for legacy_name in LEGACY_FLAT_MODULES:
            module_name = f"mlpipelineholder.{legacy_name}"
            self.assertIsNone(
                importlib.util.find_spec(module_name),
                f"{module_name} should have been removed in 0.3.10",
            )

    def test_legacy_unpickle_map_resolves_canonical_classes(self) -> None:
        from mlpipelineholder.core.models import ArtifactRecord
        from mlpipelineholder.persistence.pickle_io import _MissingClassUnpickler
        from mlpipelineholder.pipeline_holder import PipelineHolder

        unpickler = _MissingClassUnpickler(BytesIO(b""))
        self.assertIs(
            unpickler.find_class("mlpipelineholder.models", "ArtifactRecord"),
            ArtifactRecord,
        )
        self.assertIs(
            unpickler.find_class("mlpipelineholder.pipeline_handler", "PipelineHolder"),
            PipelineHolder,
        )
        # Pipelines saved while the package was imported as src.mlpipelineholder still load.
        self.assertIs(
            unpickler.find_class("src.mlpipelineholder.models", "ArtifactRecord"),
            ArtifactRecord,
        )
        self.assertIs(
            unpickler.find_class("src.mlpipelineholder.pipeline_handler", "PipelineHolder"),
            PipelineHolder,
        )


if __name__ == "__main__":
    _ = unittest.main()
