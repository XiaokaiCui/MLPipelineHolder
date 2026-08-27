# MLPipelineHolder

**MLPipelineHolder is a lightweight Python framework for reproducible machine-learning pipelines in Jupyter notebooks.**

It helps you persist intermediate results, organise modelling workflows, run reproducible experiments, manage large DataFrames, and track pipeline structure without turning an exploratory project into a full MLOps system.

Although this is currently a personal project that I originally developed for my own quantitative investing experiments, I’d be very happy to see it become useful to a wider group of people. Thank you for your interest in the project — any ideas, issues, suggestions, or feedback are very welcome!

**v0.3.0 will be the first full release**, but **v0.2.14 and later versions are already very close to the planned full version**. If you are currently using an earlier release, I recommend upgrading to v0.2.14 or newer.

## Installation

Install from [PyPI](https://pypi.org/project/mlpipelineholder/):

```bash
pip install mlpipelineholder
```

Install optional integrations as needed:

```bash
pip install "mlpipelineholder[dataframe]"
pip install "mlpipelineholder[torch]"
pip install "mlpipelineholder[rich]"
pip install "mlpipelineholder[memory]"
pip install "mlpipelineholder[all]"
```

Extras can be combined in a single install. For example, to add DataFrame
and PyTorch support together:

```bash
pip install "mlpipelineholder[dataframe,torch]"
```

Each extra adds:

- `dataframe`: pandas, PyArrow, and Dask DataFrame support
- `torch`: PyTorch model, tensor, and optimiser persistence
- `rich`: Rich-rendered console tracebacks (console tracebacks fall back to plain stdlib text when Rich is not installed)
- `memory`: `psutil`-based memory profiling logs (useful when you utilise disk-backed features)
- `all`: all optional features listed above (**recommended**)

The core package requires Python 3.11 or later and includes `termcolor` and NumPy.

## At a glance

### Typical use cases

- I run many experiments with different parameters in a notebook and want an easy way to run, record, and compare them consistently.
- I have a modelling notebook and want to organise it into a safer structure so accidental changes are less likely to break my work (use `strict_mode`).
- I use the same notebook across different days and do not want to rerun expensive steps every time I reopen it.
- I have limited RAM and need a convenient way to load and offload DataFrames while exploring the data.
- I want to focus on analysis and modelling, with logging and pipeline visualisation handled more simply.

### Key features

MLPipelineHolder organises workflows into explicit execution blocks and nested child pipelines while keeping their runtime structure easy to inspect and modify. It helps you:

- Organise data flows and manage variables, configurations, outputs, and dependencies through clearly defined scopes.
- Persist intermediate and final outputs to disk, then reload them quickly and easily after a kernel restart.
- Run independent functions concurrently using threads within the same execution block, while keeping cross-block execution explicit and ordered by priority.
- Reduce RAM usage by storing large artifacts on disk without sacrificing pipeline usability. Enable `memory_saving_mode` to release objects that are no longer needed(more effective when used together with `save_to_disk`).
- Track logs, results, and pipeline state with minimal effort.
- Improve the stability and reproducibility of modelling and analysis outputs while retaining full flexibility over the pipeline structure.

### Example notebook

Please refer to the [example notebook](https://githubtocolab.com/XiaokaiCui/MLPipelineHolder/blob/master/examples/comprehensive_pipeline.ipynb) to see the details.

## Fastest way to start: collaborate with an LLM on your notebook

If you already have a data modelling or analysis notebook, the fastest way to get started is to ask an LLM to convert it into a pipeline-managed workflow.

This repository includes a notebook-oriented guide for the LLM agent:

- [`PIPELINE_CREATOR_SKILL.md`](https://github.com/XiaokaiCui/MLPipelineHolder/blob/master/PIPELINE_CREATOR_SKILL.md)

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

## API reference

The full public API is documented in a standalone reference page:

- [docs/api_reference.html](https://mlpipelineholder.readthedocs.io/en/latest/api_reference.html) — complete documentation of every public class, function, decorator, exception, and data model in the package. It is a local, self-contained HTML file (no external dependencies) and can be opened directly in any browser.
  
It covers:

- main classes: `PipelineHandler`, `ExecutionBlock`, `GateBlock`, `PipelineLogger`
- functions and decorators: `rename_args`
- exceptions: `PipelineError`, `RegistrationError`, `ResolutionError`, `ExecutionError`, `PersistenceError`
- data models in `mlpipelineholder.models` (e.g. `ArtifactRecord`, `RunRecord`, function/expression/block registrations, runtime value and callable references)

