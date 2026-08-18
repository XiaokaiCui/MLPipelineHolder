# MLPipelineHolder

**MLPipelineHolder is a lightweight Python framework for reproducible machine-learning pipelines in Jupyter notebooks.**

It helps you persist intermediate results, organise modelling workflows, run reproducible experiments, manage large DataFrames, and track pipeline structure without turning an exploratory project into a full MLOps system.

## Installation

Install from [PyPI](https://pypi.org/project/mlpipelineholder/):

```bash
pip install mlpipelineholder
```

Install optional integrations as needed:

```bash
pip install "mlpipelineholder[dataframe]"
pip install "mlpipelineholder[torch]"
pip install "mlpipelineholder[memory]"
pip install "mlpipelineholder[all]"
```

Each extra adds:

- `dataframe`: pandas, PyArrow, and Dask DataFrame support
- `torch`: PyTorch model, tensor, and optimiser persistence
- `memory`: `psutil`-based memory profiling logs
- `all`: all optional features listed above

The core package requires Python 3.11 or later and includes `termcolor`, NumPy, and Rich.

## At a glance

### Typical use cases

- I run many experiments with different parameters in a notebook and want an easy way to run, record, and compare them consistently.
- I have a modelling notebook and want to organise it into a safer structure so accidental changes are less likely to break my work.
- I use the same notebook across different days and do not want to rerun expensive steps every time I reopen it.
- I have limited RAM and need a convenient way to load and offload DataFrames while exploring the data.
- I want to focus on analysis and modelling, with logging and pipeline visualisation handled more simply.

### Key features

MLPipelineHolder organises workflows into explicit execution blocks and nested child pipelines while keeping their runtime structure easy to inspect and modify. It helps you:

- Organise data flows and manage variables, configurations, outputs, and dependencies through clearly defined scopes.
- Persist intermediate and final outputs to disk, then reload them quickly and easily after a kernel restart.
- Automatically run independent functions in parallel within the same execution block, while keeping cross-block execution explicit and ordered by priority.
- Reduce RAM usage by storing large artifacts on disk without sacrificing pipeline usability. Enable `memory_saving_mode` to release objects that are no longer needed.
- Track logs, results, and pipeline state with minimal effort.
- Improve the stability and reproducibility of modelling and analysis outputs while retaining full flexibility over the pipeline structure.

## Fastest way to start: collaborate with an LLM on your notebook

If you already have a data modelling or analysis notebook, the fastest way to get started is to ask an LLM to convert it into a pipeline-managed workflow.

This repository includes a notebook-oriented guide for the LLM agent:

- [`PIPELINE_CREATOR_SKILL.md`](PIPELINE_CREATOR_SKILL.md)

The guide explains how to inspect your notebook, design pipeline scopes, identify persistence and memory requirements, and produce the converted code.

### Low-pressure onboarding workflow

You do not need a perfect prompt or complete answers. Responses such as "not sure", "suggest for me", or "use sensible defaults" are fine.

The agent will inspect your code, propose a scope plan and likely disk-backed outputs, and ask only a short batch of blocking or high-impact questions.

### Recommended prompt template

Copy and paste this prompt, filling in what you can:

```text
I have a Jupyter notebook for data modelling or analysis that I want to convert into a pipeline-managed notebook using MLPipelineHolder.

Repository:
https://github.com/XiaokaiCui/MLPipelineHolder

Please read the repository and the file `PIPELINE_CREATOR_SKILL.md` first. Then use that guidance to convert my notebook.

Here are my workflow details (rough answers are fine, or say "not sure" / "suggest for me"):
- Group 1 (Intent and Context): Are we creating a new pipeline from scratch, or extending an existing saved parent pipeline? If extending, what is the parent path or structure?
- Group 2 (Scope and Prefix): Which parts are shared vs workflow-local, and what prefix should we use for workflow-local config/values?
- Group 3 (Large Objects): Which produced objects are expected to be large and should use save_to_disk? (Answer yes/no/not sure; suggest candidates for me.)
- Group 4 (Memory Settings): Do you want memory_saving_mode or memory_profile_logging enabled? (Answer yes/no/not sure; suggest defaults for me.)
- Group 5 (Persistence and Callables): Do you intend to save and reload the pipeline later, and where are your callables defined?

Please inspect my code, propose likely large outputs and a scope plan, and then ask only a short batch of blocking or high-impact questions before writing the code.

Your final output must include:
1. Structure Explanation: A short explanation of the proposed pipeline structure.
2. Concise Conversion Manifest: The manifest detailing hierarchy, stage types, priorities, inputs/outputs, prefixes, and disk-backed outputs.
3. Converted Code: The complete converted notebook code as one contiguous deliverable. Use one fenced code block if the script is reasonably short; otherwise, create one `.py` file and provide its path or link. Do not split the script into blocks that I must combine manually.
4. Assumptions and Questions: Any assumptions made or blocking questions.
5. Verification and Persistence Notes: Notes on how to verify the pipeline and any persistence limitations.

I will now provide my notebook/code.
```

### Understanding large objects and memory settings

**Large objects and disk backing**

Large objects are usually large DataFrames, arrays, models, or expensive intermediates. Storing them on disk lowers the RAM retained during execution, but it adds serialisation and I/O cost. The `save_to_disk` option applies only to declared produced outputs of a block, not to arbitrary inputs set via `set_constant_value`.

**Memory saving and profiling**

- `memory_saving_mode` performs best-effort cleanup of intermediate values after each block finishes.
- `memory_profile_logging` logs memory usage after computation and cleanup for each block.

Attached child pipelines inherit both settings from their parent, so configure the policy on the owning or root pipeline.

## API reference

The full public API is documented in a standalone, pandas/numpy-style reference page:

- [`doc/api_reference.html`](doc/api_reference.html) — complete documentation of every public class, function, decorator, exception, and data model in the package. It is a local, self-contained HTML file (no external dependencies) and can be opened directly in any browser.

It covers:

- main classes: `PipelineHandler`, `ExecutionBlock`, `GateBlock`, `PipelineLogger`
- functions and decorators: `rename_args`
- exceptions: `PipelineError`, `RegistrationError`, `ResolutionError`, `ExecutionError`, `PersistenceError`
- data models in `mlpipelineholder.models` (e.g. `ArtifactRecord`, `RunRecord`, function/expression/block registrations, runtime value and callable references)

The sections below summarise the key behaviours and concepts; for signatures, parameters, and full details, open the HTML reference above.

## Core concepts

### 1. PipelineHandler

Construct with:

- `registration_name`
- optional `configuration`
- optional `local_folder_path`
- optional `pipeline_backup_directory`

It manages:

- registered blocks
- `para_value_dict`
- artifact registry
- run history
- metadata directory
- logger
- optional backup metadata used by backup-aware save/load

If `configuration` is omitted, the pipeline starts with an empty config.

If `local_folder_path` is omitted, the pipeline automatically uses a temporary staging root. This is useful when building a child pipeline in a notebook before attaching it to a parent. Once attached, the child moves under the parent project tree and the temporary root is no longer its working location.

This staging root is a notebook-friendly convenience:

- unattached temporary-root pipelines are cleaned up during normal object or runtime cleanup
- if such a pipeline is attached to a parent, it is relocated under the parent tree
- if such a pipeline is saved to an explicit path, its `project_root` is materialised in that directory
- for a durable standalone location from the beginning, pass an explicit `local_folder_path`

### 2. ExecutionBlock

Each block has:

- a block name
- a numeric execution priority
- one or more registered functions

Independent functions registered in the same block run in parallel. Blocks remain explicit execution boundaries and run according to priority.

Parent-level execution can also use float priorities for branch groups. For example, `5.1`, `5.3`, and `5.9` all belong to group `5`. Once one node in that integer-priority group actually executes, later nodes in the same group are skipped automatically.

### 3. Argument resolution

When executing a registered function, inputs are resolved in this order:

1. explicit runtime overrides
2. `para_value_dict`
3. config fields
4. function defaults

Special cases:

- if a function declares an argument named `logger`, the pipeline logger is injected automatically
- child pipelines can use upstream parent outputs from earlier parent-level nodes
- child config values override same-named parent config values
- child config values are not exposed to parent blocks

### 4. Outputs

Function outputs can be:

- stored directly in memory
- stored on disk if listed in `save_to_disk`

Disk-backed outputs are represented in memory by `ArtifactRecord`, but can be resolved back to real values using `get_value(...)`.

Current serialisation behaviour:

- JSON-serialisable values use `json`
- `numpy.ndarray` values use `.npy`
- PyTorch tensors and modules use `torch`
- pandas DataFrames use Feather by default
- pandas DataFrames with more than 3 million rows use a single Parquet file
- Dask DataFrames use Parquet directories and remain Dask on reload
- everything else falls back to `pickle`

Torch artifact loading can be hardened per pipeline. By default torch artifacts are loaded with `weights_only=False` (required for modules saved with full state). Enable the safer `weights_only=True` mode per pipeline:

```python
pipeline = PipelineHandler(..., torch_load_weights_only=True)   # at creation
pipeline.set_torch_load_weights_only(True)                      # or at runtime
```

The setting is applied to every torch artifact produced by that pipeline (stamped on the artifact record at save time) and survives save/load. Each pipeline keeps its own setting — children do not inherit it from their parent, and artifacts always load with the setting of the pipeline that produced them, regardless of who reads them later. Note that `weights_only=True` cannot load torch modules saved with full state; it is intended for pure tensor artifacts.

### 5. Rename function inputs and use variadics safely

When a function uses generic names like `obj`, or uses `*args` / `**kwargs`, you can expose safer pipeline-facing names during registration.

```python
def mapped_variadic(obj: int, *more_values: int, scale: int = 1, **extra_values: int) -> int:
    return (obj + sum(more_values) + sum(extra_values.values())) * scale


block.register_function(
    mapped_variadic,
    ["result"],
    param_mapping={"obj": "payload", "scale": "scale_value"},
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

- renamed variadic positional values must resolve to a `list` or `tuple`
- renamed variadic keyword values must resolve to a `dict`
- mapping metadata is preserved on save/load
- if the same function is already registered in a block, use `forced=True` to replace it

### 6. Run modes

Available execution methods:

- `run_all()`
- `run_until(block_name)`
- `run_from(block_name)`
- `run_block(block_name)`

Nested targeting is also supported by path:

```python
pipeline.run_until("modeling_pipeline", "predictor_components")
pipeline.run_from("modeling_pipeline", "predictor_training_pipeline", "predictor_saving_pipeline")
pipeline.run_block("modeling_pipeline", "predictor_components")
```

### 7. Manage configuration

```python
pipeline.set_config({"multiplier": 10})
```

Rules:

- the pipeline may be created with `configuration=None`, which is treated as an empty config
- `set_config(...)` adds new fields or updates existing ones
- `update_config(...)` updates existing fields only
- config writes that would conflict with declared output names are rejected or skipped depending on the method used

```python
full_config = pipeline.get_full_config()
model_cls = pipeline.get_config_value("model_cls")
```

Behaviour:

- `get_full_config()` returns the visible merged config for the pipeline
- parent configs are included recursively for nested child pipelines
- current pipeline config overrides same-named parent values
- `get_config_value(name)` raises if the key does not exist
- child configs are not propagated upward to parents or siblings

### 8. Access values safely

The pipeline keeps two separate value namespaces:

- **produced values** — outputs produced by registered functions during execution
- **pipeline constants** — fixed values you set directly on a pipeline

```python
value = pipeline.get_value("model_blob")            # produced value
pipeline.set_constant_value("learning_rate", 0.01)  # set a constant
constant = pipeline.get_constant_value("learning_rate")
```

If the value is disk-backed, the true object is loaded and returned.

To modify produced values:

```python
pipeline.update_value("existing_name", 10)
pipeline.set_value("new_or_existing_name", 20)
```

Behaviour:

- `update_value(...)` updates an existing produced value only
- `set_value(...)` updates an existing produced value, or injects a value for a declared-but-cleared produced name
- both raise `ResolutionError` for unknown names
- `get_value(...)` raises `ResolutionError` if the name is a pipeline constant — use `get_constant_value(...)` instead
- `update_value(...)` and `set_value(...)` likewise refuse names owned by pipeline constants

To set and read pipeline constants:

```python
pipeline.set_constant_value("learning_rate", 0.01)
rate = pipeline.get_constant_value("learning_rate")
```

Constant behaviour:

- `set_constant_value(...)` creates or replaces a constant; `get_constant_value(...)` raises `ResolutionError` for unknown names
- constants are visible to the pipeline, its descendants, and downstream siblings through the parent visibility model
- when multiple pipelines define the same constant name, the nearest definition wins: sibling (by priority) over parent, parent over grandparent
- constants survive save/load and backup recovery, and may hold callables or `ArtifactRecord` values

Name conflicts are prevented across the whole pipeline tree in both directions:

- `set_constant_value(...)` raises `RegistrationError` if the name is already a declared output or produced value anywhere in the tree
- registering an output whose name matches a constant anywhere in the tree is rejected or skipped with a warning, depending on the registration path

### 9. Save and load projects

```python
pipeline.save_pipeline()
loaded = PipelineHandler.load_pipeline("demo_run")
```

Without a path, `save_pipeline()` saves to the current `project_root`.

Passing a different path creates a restart-safe copy of the current project tree before writing fresh state and metadata. The copy includes nested child directories and disk-backed artifacts. The target must not overlap the current project tree.

You can also configure a backup directory when creating the pipeline:

```python
pipeline = PipelineHandler(
    "demo",
    config,
    Path("demo_run"),
    pipeline_backup_directory=Path("demo_backup"),
)
```

Save and load behaviour:

- if `pipeline_backup_directory is None`, an in-place save updates only the working tree
- if `pipeline_backup_directory` is set, an in-place save also refreshes the backup copy
- `load_pipeline(path)` reads lightweight metadata before loading the full state
- if `path` is the canonical working directory, the pipeline loads directly
- if `path` differs from the canonical working directory, the library restores the saved tree to the canonical directory before loading it
- `load_pipeline(path, forced_deleting=False)` asks for keyboard confirmation with `yes` or `y` before deleting a non-empty canonical working directory during restore
- `load_pipeline(path, forced_deleting=True)` deletes the canonical working directory directly during restore

Saved projects contain:

- `config.pkl`
- `pipeline_state.pkl`
- disk-backed outputs under `artifacts/`
- logs and configuration snapshots under `metadata/`

Every `save_pipeline(...)` call also archives a timestamped copy of the current `pipeline.log` into `history_logs/` inside the project root, named `yyyy-mm-dd_hh-mm-ss.mmm.log` (UTC, e.g. `history_logs/2026-08-18_16-38-14.857.log`). This preserves each save-time log snapshot even though loading a pipeline starts a fresh `pipeline.log`. When a `pipeline_backup_directory` is configured, the refreshed backup copy also receives the updated `history_logs/` folder, and loading a saved project restores its `history_logs/` folder into the project root.

Saved pipelines preserve callable references rather than historical source code. Importable callables are restored from their import paths. Notebook-local functions and other runtime-only callables, including callable values supplied through `set_constant_value(...)`, `set_value(...)`, or upstream outputs, must be defined or imported under the same name before calling `load_pipeline(...)`. Missing runtime callables raise during loading instead of being passed to a stage as inert placeholders. A saved project copies pipeline data, not Python source, installed packages, or the runtime environment.

Compatibility aliases `save_project()` and `load_project()` still exist.

### Recover one value or config field from backup

A pipeline with a configured `pipeline_backup_directory` can restore one current item from its latest complete in-place backup:

```python
# Both methods update in place and return None.
pipeline.recover_variable_from_backup(name="output_df")
modeling_pipeline.recover_config_from_backup(name="learning_rate")
```

Recovery requires `pipeline_state.pkl`, `config.pkl`, and `pipeline_meta.pkl` plus every saved artifact referenced by the backup. The requested name must exist both in the live pipeline and in the matching saved pipeline state. The backup is inspected read-only; recovery never replaces the complete working tree.

Variable recovery rules:

- call `recover_variable_from_backup(...)` on the root pipeline only
- if the same name is independently owned by multiple root/child pipelines, one prompt lists every affected pipeline; only `yes` or `y` applies the recovery to all of them
- declining the prompt leaves all values unchanged and returns `None`
- disk-backed values are copied into the live project, so they remain usable if the backup is later removed

Config recovery rules:

- call `recover_config_from_backup(...)` on the pipeline that locally owns the field
- descendants inherit the restored field through normal config visibility unless they define their own local override
- successful recovery preserves the config type and unrelated fields, but replaces the pipeline's config object with a staged copy

Import or define runtime-only callables under their saved names before recovery, including functions used by `functools.partial`. Recovery resolves supported references to live objects and raises `PersistenceError` rather than installing an unresolved placeholder. Missing live or saved names raise `ResolutionError`; invalid, incomplete, unsafe, or unresolvable backup data raises `PersistenceError`. Failures leave current state unchanged. Recovery replaces only the requested item and does not invalidate or rerun derived outputs.

### 10. Print the pipeline chart

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

Additional chart behaviour:

- child pipelines gated off by config are greyed out when the current config value does not match the gate’s expected value
- the root pipeline is never greyed out this way

Gate lines do not show `-> bool`, and chart symbols such as `()` and `->` use the same colour family as priority markers for readability.

Gate skip cleanup confirmation:

When a gate does not pass, the run is skipped. By default the pipeline also invalidates the produced values of the skipped run and deletes their disk artifacts. If you want to be asked before that cleanup happens, enable confirmation on the pipeline:

```python
pipeline.gate_cleanup_confirmation = True
```

With confirmation enabled, a skipped run first asks (via `input()`):

- `yes` / `y` → invalidate the produced values and delete their disk artifacts (the default behaviour)
- anything else → keep the current values and artifact files; the run is skipped non-destructively

The prompt lists the gate, the run mode, the number of affected produced values, and notes that downstream blocks and child pipelines consuming them would receive `None` after cleanup. No prompt is shown when there is nothing to clean up. This setting is per pipeline and is not persisted by `save_pipeline`.

### 11. Output conflicts and overrides

Duplicate output names across different parent-level blocks or child pipelines are allowed.

- later parent-level nodes override earlier parent-level nodes
- child internal override chains are not expanded in the parent conflict report

Helpers:

```python
conflicts = pipeline.get_output_conflicts()
print(pipeline.describe_output_conflicts())
```

### 12. Logging

The pipeline creates a logger automatically.

Supported methods:

- `debug(...)`
- `info(...)`
- `warning(...)`
- `error(...)`
- `critical(...)`
- `result(...)`
- `print(...)`

Behaviour:

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

# Optional: pause and resume file logging.
pipeline.logger.disable_file_logging()
pipeline.logger.enable_file_logging()

# View the current log and archived snapshots (all resolved from the root pipeline).
pipeline.logger.show_recent_logs(20)                      # last 20 lines of metadata/pipeline.log
pipeline.logger.show_recent_logs(10, "ERROR")             # last 10 ERROR lines only
snapshots = pipeline.logger.list_history_logs()           # snapshot files in history_logs/
pipeline.logger.show_history_log(snapshots[0].name)       # print one snapshot (full or a line range)
pipeline.logger.show_history_log(snapshots[0].name, log_level="result")  # only RESULT lines
```

`clear_result_history()` only clears in-memory result history. It does not modify `metadata/pipeline.log`.

Normal notebook use does not require manual logger cleanup. `disable_file_logging()` pauses writes to `metadata/pipeline.log` while console output and in-memory result history continue. `enable_file_logging()` resumes appending without clearing existing entries. Attached children share their parent logger and do not manage its lifetime.

Log viewing helpers (resolved from the root pipeline):

- `show_recent_logs(lines=5, log_level=None)` prints the latest `lines` entries of the current `metadata/pipeline.log`; with `log_level` set, only entries at that level are shown and the limit applies after the filter
- `list_history_logs()` returns the timestamped snapshots saved in `history_logs/`, sorted by name
- `show_history_log(file_name, line_starts=0, line_ends=None, log_level=None)` prints lines `[line_starts, line_ends)` of one snapshot; `line_ends=None` prints to the end of the file, and with `log_level` set only entries at that level within the range are shown

`log_level` accepts the normal levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) plus `RESULT` and `PRINT`, case-insensitive; unknown levels raise `ValueError`.

Print capture modes:

- `tee` (default): send `print(...)` output to both normal stdout and the pipeline log
- `logger_only`: capture `print(...)` output only into the pipeline log
- `off`: leave normal `print(...)` behaviour unchanged

Print capture applies to blocks with one registered function. These blocks run on the caller thread so manually interrupting them restores notebook stdout before returning control. Output buffered when the function is interrupted may not be added to the pipeline log. Blocks with multiple functions remain parallel and use normal stdout without adding their `print(...)` output to the pipeline log. Setting capture to `off` disables automatic print capture without disabling the pipeline logger.

Traceback logging:

When a run fails, the logger writes an `ERROR` entry for the exception plus a rendered traceback.

- the log file always receives the plain stdlib traceback (never Rich formatting or ANSI codes)
- the console receives a Rich-rendered traceback by default (box-drawing guides, optional local variables), falling back to the plain stdlib traceback when Rich console rendering is disabled
- `show_traceback_locals` (default `False`) controls whether local variables appear in the Rich console traceback

Configure at pipeline creation or directly on the logger:

```python
pipeline = PipelineHandler(
    ...,
    log_traceback_to_file=True,       # append the stdlib traceback to pipeline.log (default)
    show_traceback_locals=False,      # show locals in the Rich console traceback (default)
    use_rich_traceback_console=True,  # render console tracebacks with Rich (default)
)

# Or adjust at runtime on the logger:
pipeline.logger.set_traceback_writing(False)        # stop appending the traceback to the file
pipeline.logger.set_show_traceback_locals(True)     # include locals in Rich console tracebacks
pipeline.logger.set_traceback_console_render(False) # use plain stdlib text on the console
```

- `set_traceback_writing(enable=True)` controls whether the traceback body is appended to the log file; the `ERROR` header line is always written
- `set_show_traceback_locals(enable=False)` controls local-variable display in Rich console tracebacks
- `set_traceback_console_render(use_rich=True)` selects Rich (default) or plain stdlib text for console tracebacks
- the three settings are saved and restored with the pipeline

### 13. Memory options

You can enable optional runtime memory tools when creating a pipeline:

```python
pipeline = PipelineHandler(
    ...,
    memory_saving_mode=True,
    memory_profile_logging=True,
)
```

Behaviour:

- `memory_saving_mode=True` runs a best-effort cleanup after each block finishes
- `memory_profile_logging=True` logs memory after computation and cleanup for each block
- attached child pipelines inherit these settings from their parent

## Example notebook

Open and run:

```text
examples/comprehensive_pipeline.ipynb
```

The notebook writes its runtime data under:

- `examples/example_run/`
