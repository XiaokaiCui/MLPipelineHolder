# MLPipelineHolder

Lightweight Python library for building, running, recording, and modifying small machine-learning pipelines.

The project is centered on two concepts:

- `PipelineHandler`: owns config, block registration, execution state, artifacts, run history, persistence, and logging
- `ExecutionBlock`: owns a priority and one or more registered functions that run in parallel inside that block

## Current status

This project is implemented and tested.

Current verified behavior includes:

- ordered block execution by unique priority
- parallel function execution inside a block
- automatic argument binding from runtime overrides, pipeline values, config fields, and function defaults
- in-memory outputs plus disk-backed artifacts for selected outputs
- partial reruns with downstream invalidation
- single-block execution
- project save/load
- config updates from dictionaries
- resolved value access for disk-backed artifacts
- colorful pipeline chart output
- colorful UTC logger output with custom `result` level
- result-history cleanup without deleting persisted logs
- safe block removal with state cleanup
- safe function removal inside a block with state cleanup
- json and numpy artifact serializers

## Installation and environment

This project uses **Poetry**.

The current project metadata is in `pyproject.toml`.

### Use the experiment environment Poetry

```bash
pip install poetry
poetry install --no-interaction
```

### Main dependency

- `termcolor` for colorful logger and chart output
- `numpy` for ndarray artifact serialization

## Project layout

```text
MLPipelineHolder/
├── examples/
│   └── simple_pipeline.py
├── src/
│   ├── __init__.py
│   ├── main.py
│   └── mlpipelineholder/
│       ├── __init__.py
│       ├── artifact_store.py
│       ├── exceptions.py
│       ├── execution_block.py
│       ├── function_registry.py
│       ├── logger.py
│       ├── models.py
│       ├── pipeline_handler.py
│       └── serializers.py
├── tests/
│   ├── test_execution_block.py
│   ├── test_pipeline_handler.py
│   └── test_save_load.py
└── pyproject.toml
```

## Public API

Exports from `src.mlpipelineholder`:

- `PipelineHandler`
- `ExecutionBlock`
- `PipelineLogger`
- `PipelineError`
- `RegistrationError`
- `ResolutionError`
- `ExecutionError`
- `PersistenceError`

## Core concepts

### 1. PipelineHandler

Construct with:

- `registration_name`
- `configuration`
- `local_folder_path`

It manages:

- registered blocks
- `para_value_dict`
- artifact registry
- run history
- metadata directory
- logger

### 2. ExecutionBlock

Each block has:

- a block name
- a unique execution priority
- one or more registered functions

All functions inside the same block run in parallel.

### 3. Argument resolution

When executing a registered function, inputs are resolved in this order:

1. explicit runtime overrides
2. `para_value_dict`
3. config fields
4. function defaults

Special case:

- if a function declares an argument named `logger`, the pipeline logger is injected automatically

### 4. Outputs

Function outputs can be:

- stored directly in memory
- stored on disk if listed in `save_to_disk`

Disk-backed outputs are represented in memory by `ArtifactRecord`, but can be resolved back to real values using `get_value(...)`.

Current serializer behavior:

- JSON-serializable values use `json`
- `numpy.ndarray` values use `.npy`
- torch tensors/modules use `torch`
- pandas DataFrames use `feather` when available
- everything else falls back to `pickle`

## Main features

### Register blocks and functions

Example:

```python
from dataclasses import dataclass
from pathlib import Path

from src.mlpipelineholder import PipelineHandler


@dataclass
class Config:
    raw_value: int = 5
    multiplier: int = 3


def create_seed(raw_value: int) -> int:
    return raw_value + 1


def create_features(seed: int) -> tuple[int, int]:
    return seed * 2, seed * 4


pipeline = PipelineHandler("demo", Config(), Path("demo_run"))

setup = pipeline.add_block("setup", 1)
setup.register_function(create_seed, ["seed"])

features = pipeline.add_block("features", 2)
features.register_function(create_features, ["feature_a", "feature_b"])
```

### Run modes

Available execution methods:

- `run_all()`
- `run_until(block_name)`
- `run_from(block_name)`
- `run_block(block_name)`

### Update config

```python
pipeline.update_config({"multiplier": 10})
```

Unknown fields raise `ResolutionError`.

### Access values safely

```python
value = pipeline.get_value("model_blob")
```

If the value is disk-backed, the true object is loaded and returned.

### Remove a block safely

```python
pipeline.remove_block("feature_generation")
```

This removes the block and invalidates outputs from the removed block and all downstream blocks so pipeline state stays consistent.

### Remove a function safely

```python
block.remove_function("feature_step")
```

This removes the named function from the block and invalidates outputs from that function and all downstream block outputs.

### Save and load a project

```python
pipeline.save_project()
loaded = PipelineHandler.load_project("demo_run")
```

`save_project()` defaults to `local_folder_path` when no path is given.

### Print the pipeline chart

```python
print(pipeline.describe_pipeline())
print(pipeline)
print(repr(pipeline))
```

Current chart format includes:

- block name
- priority
- function name
- argument names
- output names
- `*` marker for disk-backed outputs

### Logging

The pipeline creates a logger automatically.

Supported methods:

- `debug(...)`
- `info(...)`
- `warning(...)`
- `error(...)`
- `critical(...)`
- `result(...)`

Behavior:

- every log line includes a UTC timestamp
- all log lines are appended to `metadata/pipeline.log`
- `result(...)` messages are kept in a separate in-memory history list

Logger helpers:

```python
history = pipeline.get_result_history()
pipeline.print_result_history()
pipeline.clear_result_history()
```

`clear_result_history()` only clears in-memory result history. It does not modify `metadata/pipeline.log`.

## Example script

Run:

```bash
poetry run python examples/simple_pipeline.py
```

The example demonstrates:

- config-backed execution
- multiple blocks
- disk artifact storage
- chart rendering
- injected logger usage
- result history collection

## Persistence model

Projects are saved as:

- `config.pkl`
- `pipeline_state.pkl`
- artifact files under `artifacts/`
- log file under `metadata/pipeline.log`
- config snapshots under `metadata/`

## Rules and safeguards

- block priorities must be unique
- duplicate output names across blocks are rejected
- `*args` and `**kwargs` are not supported for registered functions
- functions inside one block cannot depend on outputs from the same block
- non-importable callables cannot be saved for load/replay

## Save/load limitation

Saved pipelines currently preserve **import paths**, not historical function snapshots.

That means:

- if a source function changes later, a loaded pipeline may use the new behavior
- if a transitive dependency of that function changes later, behavior may also change

Because of that, `save_project()` and `load_project()` emit warnings explaining this limitation.

This is intentionally deferred because reliable historical behavior preservation would require a much heavier code and environment snapshot system.

## Tested behavior

Current test coverage verifies:

- full pipeline execution
- partial reruns
- disk artifact save/load behavior
- duplicate output rejection
- duplicate priority rejection
- config override behavior
- resolved artifact loading via `get_value`
- chart generation
- `__str__` and `__repr__`
- logger injection and result history
- default save path
- save/load warnings
- safe block removal
- safe function removal
- json artifact serialization
- numpy ndarray serialization
- result-history cleanup
- importable vs non-importable save behavior

## Run tests

Using Poetry:

```bash
poetry run python -m unittest discover -s tests -v
```

Using Python directly:

```bash
python -m unittest discover -s tests -v
```

## Current limitations

- no historical function snapshotting
- no DAG beyond priority-based ordering
- no retry scheduler
- no distributed execution
- no separate per-run log file yet
- colors are optimized for notebook/CLI readability, but exact rendering still depends on the terminal frontend

## Recommended next steps

- add a small CLI wrapper
- add README examples for `run_until`, `run_from`, and `remove_block`
- add optional per-run logs
- add richer artifact serializers if needed
