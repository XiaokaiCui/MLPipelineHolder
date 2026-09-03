# MLPipelineHolder

**MLPipelineHolder is a straightforward framework for reproducible machine-learning experimentation in Jupyter notebooks. It helps you progressively turn mature parts of an exploratory workflow into stable pipeline stages, while keeping configurations, hyperparameters, and pipeline structure flexible and easy to change.**

I originally developed MLPipelineHolder for my own quantitative investing experiments because I found that the available alternatives were either more complex than I needed or not quite flexible enough for active experimentation.

If those are not concerns for you, I would also recommend looking at [Apache Hamilton](https://hamilton.apache.org/) and [Kedro](https://github.com/kedro-org/kedro) if you have not already considered them. They are mature and capable frameworks with different strengths.

If you have similar concerns to mine, however, MLPipelineHolder may be worth trying. It is designed to be adopted at almost any stage of experimentation without requiring you to rebuild your workflow from scratch. In many cases, an existing notebook can be converted into a pipeline-managed structure in less than 30 minutes while keeping most of its original organisation and logic.

The idea is not to force an exploratory notebook into a rigid workflow, but to let you progressively turn the parts that have become mature into stable pipeline stages while keeping configurations, hyperparameters, functions, and pipeline structure easy to experiment with and change.

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
pip install "mlpipelineholder[optuna]"
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
- `optuna`: SQLite-backed Optuna `Study` artifacts and pickle-backed `BaseSampler` artifacts
- `all`: all optional features listed above (**recommended**)

The core package requires Python 3.11 or later and includes `termcolor` and NumPy.

## At a glance

### Key features

MLPipelineHolder organises workflows into explicit execution blocks and nested child pipelines while keeping their runtime structure easy to inspect and modify. It helps you:

- Organise data flows and manage variables, configurations, outputs, and dependencies through clearly defined scopes. (use `strict_mode` to make it even safer)
- Build a pipeline around your existing Jupyter notebook in a straightforward way while retaining most of its original structure and logic.
- Reduce RAM usage by storing large artifacts on disk without sacrificing pipeline usability. Enable `memory_saving_mode` to release objects that are no longer needed (more effective when used together with `save_to_disk`).
- Track and record logs, results, and pipeline state with minimal effort by using built-in logger.

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
