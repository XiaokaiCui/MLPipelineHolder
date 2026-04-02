# MLPipelineHolder

Lightweight Python library for building, running, recording, and modifying small machine-learning pipelines.

The project is centered on two concepts:

- `PipelineHandler`: owns config, block registration, execution state, artifacts, run history, persistence, and logging
- `ExecutionBlock`: owns a priority and one or more registered functions that run in parallel inside that block

Pipelines can also be nested, so a `PipelineHandler` may be registered inside another `PipelineHandler` as a child execution node.

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
- optional capture of `print(...)` output into the pipeline log with default tee behavior
- result-history cleanup without deleting persisted logs
- safe block removal with state cleanup
- safe function removal inside a block with state cleanup
- json and numpy artifact serializers
- nested child pipelines with shared upstream/downstream outputs
- optional gate block for conditional pipeline skipping
- parent-level output override reporting
- renamed keyword and variadic function inputs for safer registration

## Installation and environment

This project uses **Poetry**.

The current project metadata is in `pyproject.toml`.

### Install dependencies with Poetry

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
- `GateBlock`
- `PipelineLogger`
- `rename_args`
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
- child pipelines can use upstream parent outputs from earlier parent-level nodes
- child config values override same-named parent config values
- child config values are not exposed to parent blocks

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

from mlpipelineholder import PipelineHandler


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

Registration UX notes:

- invalid `add_block(...)` requests are skipped with a warning log instead of raising
- invalid `register_function(...)` requests are skipped with a warning log instead of raising
- `output_variable_names=None` is allowed when a function should run only for side effects

### Rename function inputs and use variadics safely

When a function uses generic names like `obj`, or uses `*args` / `**kwargs`, you can expose safer pipeline-facing names during registration.

```python
def mapped_variadic(obj: int, *more_values: int, scale: int = 1, **extra_values: int) -> int:
    return (obj + sum(more_values) + sum(extra_values.values())) * scale


block.register_function(
    mapped_variadic,
    ["result"],
    kw_mapping={"obj": "payload", "scale": "scale_value"},
    var_pos_name="extra_args",
    var_kw_name="extra_kwargs",
)
```

This lets the pipeline resolve:

- `payload` → original `obj`
- `scale_value` → original `scale`
- `extra_args` → original `*more_values`
- `extra_kwargs` → original `**extra_values`

Rules:

- `pos_mapping` is intentionally not supported
- renamed variadic positional values must resolve to a `list` or `tuple`
- renamed variadic keyword values must resolve to a `dict`
- mapping metadata is preserved on save/load

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

Rules:

- non-conflicting new fields may be added
- updates that would conflict with declared output names are skipped with a warning

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

### Register a child pipeline

```python
parent.add_child_pipeline(child_pipeline, 3)
```

Full example:

```python
from dataclasses import dataclass
from pathlib import Path

from mlpipelineholder import PipelineHandler


@dataclass
class ParentConfig:
    raw_value: int = 5
    multiplier: int = 3


@dataclass
class ChildConfig:
    raw_value: int = 99
    bias: int = 7


def create_seed(raw_value: int) -> int:
    return raw_value + 1


def allow_child(seed: int) -> bool:
    return seed > 0


def child_feature(seed: int, raw_value: int, bias: int) -> int:
    return seed + raw_value + bias


def final_metric(child_score: int, multiplier: int) -> int:
    return child_score * multiplier


parent = PipelineHandler("parent", ParentConfig(), Path("nested_run"))
setup = parent.add_block("setup", 1)
setup.register_function(create_seed, ["seed"])

child = PipelineHandler("child", ChildConfig(), Path("child_original"))
child.set_gate_block(allow_child)
child_block = child.add_block("feature", 1)
child_block.register_function(child_feature, ["child_score"])

parent.add_child_pipeline(child, 2)

final = parent.add_block("final", 3)
final.register_function(final_metric, ["final_metric"])

parent.run_all()
```

Behavior:

- the child pipeline participates in the parent priority order as one parent-level execution node
- parent upstream outputs are visible inside the child pipeline
- child outputs are visible to later parent nodes
- later parent-level nodes override earlier outputs with the same name
- the parent logger is used for future child execution
- if an attached child pipeline is run directly, its current outputs are synced back into the parent visible state and downstream parent outputs are invalidated

### Add a gate block

```python
pipeline.set_gate_block(should_run)
```

Minimal example:

```python
def should_run(seed: int) -> bool:
    return seed > 0


pipeline.set_gate_block(should_run)
```

Rules:

- one gate block per pipeline
- the gate block runs before every other node
- it must return `True` or `False`
- when `False`, the whole pipeline is skipped
- skipping does not overwrite an existing upstream value with `None`; it only exposes `None` for unique outputs introduced by that skipped pipeline

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
- child pipeline hierarchy
- gate block
- function name
- only argument names that are actually supplied by visible configs or earlier outputs
- output names
- `*` marker for disk-backed outputs

### Output conflicts and overrides

Duplicate output names across different parent-level blocks or child pipelines are allowed.

- later parent-level nodes override earlier parent-level nodes
- child internal override chains are not expanded in the parent conflict report

Helpers:

```python
conflicts = pipeline.get_output_conflicts()
print(pipeline.describe_output_conflicts())
```

### Logging

The pipeline creates a logger automatically.

Supported methods:

- `debug(...)`
- `info(...)`
- `warning(...)`
- `error(...)`
- `critical(...)`
- `result(...)`
- `print(...)`

Behavior:

- every log line includes a UTC timestamp
- all log lines are appended to `metadata/pipeline.log`
- `result(...)` messages are kept in a separate in-memory history list
- `print(...)` inside registered functions can also be captured into the logger

Logger helpers:

```python
history = pipeline.get_result_history()
pipeline.print_result_history()
pipeline.clear_result_history()
pipeline.set_print_capture_mode("tee")
```

`clear_result_history()` only clears in-memory result history. It does not modify `metadata/pipeline.log`.

Print capture modes:

- `tee` (default): send `print(...)` output to both normal stdout and the pipeline log
- `logger_only`: capture `print(...)` output only into the pipeline log
- `off`: leave normal `print(...)` behavior unchanged

## Example script

Run:

```bash
poetry run python examples/simple_pipeline.py
poetry run python examples/nested_pipeline.py
```

The example demonstrates:

- config-backed execution
- multiple blocks
- disk artifact storage
- chart rendering
- injected logger usage
- result history collection

The nested example demonstrates:

- parent/child pipeline registration
- child gate block behavior
- child config overriding parent config inside the child only
- parent consumption of child outputs
- hierarchical chart output

## Persistence model

Projects are saved as:

- `config.pkl`
- `pipeline_state.pkl`
- artifact files under `artifacts/`
- log file under `metadata/pipeline.log`
- config snapshots under `metadata/`

## Rules and safeguards

- block priorities must be unique
- child pipeline priorities must also be unique at the parent level
- duplicate outputs inside the same parallel block are rejected
- duplicate outputs across different parent-level nodes are allowed and resolved by execution order
- renamed keyword arguments are supported during registration
- renamed `*args` / `**kwargs` are supported during registration
- `pos_mapping` is not supported
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
- duplicate output override behavior
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
- child pipeline registration and visibility rules
- gate-block skip behavior
- parent-level output conflict reporting
- stale-output protection for individual reruns of earlier nodes
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
- save bundles currently preserve metadata/state references for disk-backed artifacts rather than creating a fully self-contained artifact snapshot
- print capture uses process-level stdout redirection, so heavily parallel print-heavy functions may still interleave output

Nested pipeline notes:

- nested pipelines are persisted recursively
- child runtime files are moved under the parent project path on registration
- child result-history display continues to read from the child’s historical result log file after registration

## Caveats

### 1. Save/load is not a full artifact snapshot

`save_project()` saves the pipeline definition, config, runtime metadata, and artifact references, but it does **not** package every disk-backed file into a fully portable bundle.

What this means in practice:

- disk-backed outputs are restored through saved `ArtifactRecord` file paths
- metadata such as historical log paths and config snapshot paths also remain path-based
- the normal workflow is safe when the pipeline continues to live under the same `project_root`
- portability is weaker when saving to a different folder as an export bundle, or when old project files are moved/deleted later

In other words, the current save/load behavior is best treated as:

- good for restoring pipeline state in the same project tree
- not yet a guaranteed self-contained archive format

### 2. Print capture under parallel threaded execution is not fully robust

The pipeline can capture `print(...)` output from registered functions and send it into the logger.

Current implementation details:

- print capture uses `redirect_stdout(...)`
- blocks can still run multiple functions in parallel using threads

This is usually fine for normal usage, but it has an important caveat:

- `stdout` is still process-level state
- print-heavy concurrent functions may interleave output or attribute lines less cleanly than explicit logger calls

Recommendation:

- use the logger directly for important structured messages
- treat captured `print(...)` as a convenience feature, not the strongest concurrency-safe logging path

### 3. Child result history after attachment is intentionally asymmetric

When a child pipeline is attached to a parent pipeline:

- future execution uses the **parent logger**
- the child pipeline’s historical result-history reader still points at the child’s historical result log path

This preserves access to pre-attachment child history, but it also means there are two concepts in play:

- current runtime logging ownership: parent logger
- historical child result display: child historical log source

So the behavior is workable, but conceptually brittle. It is best understood as a compatibility-oriented compromise rather than a fully unified nested logging model.

## Recommended next steps

- add a small CLI wrapper
- add README examples for `run_until`, `run_from`, and `remove_block`
- add optional per-run logs
- add richer artifact serializers if needed
