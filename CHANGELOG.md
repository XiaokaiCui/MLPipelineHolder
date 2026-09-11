# Changelog

All notable changes to MLPipelineHolder are documented here.

## 0.3.10 - 2026-09-11

### Breaking changes

The package was reorganised into subpackages, and the old flat module paths were removed. Importing them now
raises `ModuleNotFoundError`. Import from the new locations instead:

| removed import | import instead |
| --- | --- |
| `mlpipelineholder.pipeline_handler` | `mlpipelineholder.pipeline_holder` |
| `mlpipelineholder.models` | `mlpipelineholder.core.models` |
| `mlpipelineholder.naming` | `mlpipelineholder.core.naming` |
| `mlpipelineholder.execution_block` | `mlpipelineholder.execution.block` |
| `mlpipelineholder.gate_block` | `mlpipelineholder.execution.gate_block` |
| `mlpipelineholder.function_registry` | `mlpipelineholder.execution.function_registry` |
| `mlpipelineholder.code_comparison` | `mlpipelineholder.execution.code_comparison` |
| `mlpipelineholder.output_pointers` | `mlpipelineholder.state.output_pointers` |
| `mlpipelineholder.artifact_store` | `mlpipelineholder.persistence.artifacts.store` |
| `mlpipelineholder.serializers` | `mlpipelineholder.persistence.artifacts.serializers` |
| `mlpipelineholder.artifact_recovery` | `mlpipelineholder.persistence.artifact_recovery` |
| `mlpipelineholder.object_storage` | `mlpipelineholder.persistence.object_storage` |
| `mlpipelineholder.backup_recovery` | `mlpipelineholder.persistence.backup.recovery` |
| `mlpipelineholder.backup_recovery_service` | `mlpipelineholder.persistence.backup.service` |
| `mlpipelineholder.backup_snapshot` | `mlpipelineholder.persistence.backup.snapshot` |
| `mlpipelineholder.backup_value_resolver` | `mlpipelineholder.persistence.backup.value_resolver` |
| `mlpipelineholder.optuna_api` | `mlpipelineholder.integrations.optuna.api` |
| `mlpipelineholder.optuna_support` | `mlpipelineholder.integrations.optuna.support` |
| `mlpipelineholder.optuna_sqlite` | `mlpipelineholder.integrations.optuna.sqlite` |
| `mlpipelineholder.logger` | `mlpipelineholder.presentation.logger` |

- **Still works:** `from mlpipelineholder import PipelineHandler` - the same class as `PipelineHolder`.
- **Still works:** pipelines saved by earlier releases load unchanged; historical module names are mapped to their
  current homes while unpickling.
- `PipelineHolder` is the canonical class name; `PipelineHandler` remains an alias.

### Other changes

- Unit tests run on Python 3.11, 3.12, 3.13, and 3.14.

## 0.4.0 (planned)

The next feature release, cut after the current hardening cycle (base type for internal holder checks, further
extraction out of the main class, mixin contract test, and configuration/runtime-control split) is fully verified.
