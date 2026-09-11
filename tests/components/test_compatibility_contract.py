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
    ("src.mlpipelineholder.core.models", "ArtifactRecord"),
    ("src.mlpipelineholder.execution.block", "ExecutionBlock"),
    ("src.mlpipelineholder.execution.gate_block", "GateBlock"),
    ("src.mlpipelineholder.state.output_pointers", "OutputPointer"),
    ("src.mlpipelineholder.persistence.artifacts.store", "ArtifactStore"),
    ("src.mlpipelineholder.presentation.logger", "PipelineLogger"),
)


class CompatibilityContractTests(unittest.TestCase):
    def test_root_alias_identity_for_src_root(self) -> None:
        from src.mlpipelineholder import PipelineHandler, PipelineHolder

        self.assertIs(PipelineHandler, PipelineHolder)

    def test_root_alias_identity_for_public_root(self) -> None:
        try:
            import mlpipelineholder
        except ModuleNotFoundError:
            self.skipTest("public package name is not importable in this environment")
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
            for root in ("src.mlpipelineholder", "mlpipelineholder"):
                module_name = f"{root}.{legacy_name}"
                self.assertIsNone(
                    importlib.util.find_spec(module_name),
                    f"{module_name} should have been removed in 0.3.10",
                )

    def test_legacy_unpickle_map_resolves_canonical_classes(self) -> None:
        from src.mlpipelineholder.core.models import ArtifactRecord
        from src.mlpipelineholder.persistence.pickle_io import _MissingClassUnpickler
        from src.mlpipelineholder.pipeline_holder import PipelineHolder

        unpickler = _MissingClassUnpickler(BytesIO(b""))
        self.assertIs(
            unpickler.find_class("mlpipelineholder.models", "ArtifactRecord"),
            ArtifactRecord,
        )
        self.assertIs(
            unpickler.find_class("mlpipelineholder.pipeline_handler", "PipelineHolder"),
            PipelineHolder,
        )


if __name__ == "__main__":
    _ = unittest.main()
