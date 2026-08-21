# Pipeline Creator Skill

Use this skill when an LLM agent is helping a first-time user convert a normal Jupyter notebook into a notebook managed by `MLPipelineHolder`.

This file is a compact conversion playbook for the agent.

## Operational Preflight

Before writing any pipeline code, read these files in the repository:
1. `README.md`
2. `src/mlpipelineholder/pipeline_handler.py`
3. `src/mlpipelineholder/execution_block.py`
4. `tests/test_pipeline_handler.py`
5. `tests/test_execution_block.py`
6. `tests/test_save_load.py`

Verify that you understand:
- `PipelineHandler` owns config, values, child pipelines, save/load, and logging.
- `ExecutionBlock` runs registered functions or expressions in parallel inside one block.
- Priorities control execution order.
- Child pipelines inherit parent visibility.
- Expressions are restricted one-statement Python code.
- `define_expression_runtime(...)` provides import-only helpers for expressions.

## Strict Mode Requirement

**Always create the root pipeline with `strict_mode=True`.** This is mandatory for
every converted pipeline:

```python
pipeline = PipelineHandler(..., strict_mode=True)
```

Strict mode enforces fail-fast registration validation that catches mistakes early
instead of surfacing them as confusing run-time failures:

- `kwargs_dct` used without a `**kwargs` parameter in the target function raises at registration.
- A `kwargs_dct` key that conflicts with an explicit function argument raises at registration.
- A `gate_config` name not found in config, visible output values, or visible manual values raises at registration.
- A `param_mapping` value not found in config, visible output values, or visible manual values raises at registration.
- A `kwargs_dct` value not found in config, visible output values, or visible manual values raises at registration.
- A `param_mapping` key that is not a function parameter raises at registration.
- Attaching a child whose config/manual values/outputs cross-collide with the parent's objects raises, as does attaching a child whose registrations violate any of the checks above.

Consequences to account for when converting with strict mode:
- Input names must be resolvable at registration time: use config fields, manual values (`set_constant_value`), or already-visible output values. Do not rely on forward references to outputs that will only be produced later.
- Since strict mode validates values against the current visible state, register functions after the values they consume are defined (config set, constants set, producing blocks registered and run).
- All attached children inherit strict mode from the root pipeline, so the same rules apply throughout the tree.

## Classification and Clarification Policy

Classify all input requirements into four categories:
- **Explicit**: Requirements stated directly by the user. Preserve these exactly.
- **Inferred**: Safe assumptions based on the codebase or notebook. Disclose these clearly.
- **Contradictory**: Conflicting instructions in the prompt or code.
- **Missing**: Blocking facts needed to proceed.

Contradictions and blocking missing facts require concise questions. For example, if the user's prose assigns configs or values to `analysis_pipeline`, but the desired structure places profit-local configs or values on `profit_analysis_pipeline`, you must ask or state a delegated choice. Avoid choosing silently.

Distinguish consequential choices from cosmetic defaults:
- **Consequential**: Scope, prefix, signatures, persistence, and disk or memory policy. Ask the user about any ambiguity here.
- **Cosmetic**: Comments, formatting, and logger output. Use sensible defaults for these. Don't pressure the user over cosmetic choices.

## Conversion Manifest

Draft the conversion manifest internally while analyzing the notebook. If you need clarification on blockers or high-impact choices, ask the user in one concise batch. Otherwise, do not force an intermediate manifest message. Include the final manifest with the converted code.

The manifest must include:
- **Hierarchy and Scope**: The pipeline structure and parent/child relationships.
- **Stage Type**: Whether each stage is a direct value, expression, direct function, or wrapper.
- **Priority**: The numeric priority for each block.
- **Inputs and Outputs**: The variables consumed and produced.
- **Prefix**: The prefix used for local config and values.
- **Disk-backed Outputs**: Which outputs are saved to disk.
- **Assumptions and Questions**: Any inferences made or blocking questions.

## Scope Rules

Organize variables and configurations by their scope:
- **Shared Scope**: Shared preparation steps and shared outputs belong to the parent analysis or container pipeline.
- **Workflow Scope**: Workflow-only configurations, values, and outputs belong to the workflow child pipeline. They must use the designated prefix, subject to user confirmation.

## Disk and Memory Intake

Handle large objects and memory settings based on these rules:
- **Large Objects**: Use a proposal and confirmation workflow for disk backing. Identify likely candidates based on object type, estimated size, reuse frequency, and recomputation cost. Explain the RAM-vs-I/O trade-off and ask the user to confirm. Never infer actual size without evidence. The `save_to_disk` option applies only to declared produced outputs of a block, not to arbitrary inputs set via `set_constant_value`.
- **Memory Settings**: An attached child pipeline's settings are overwritten by and inherit the existing parent settings. Changing the root parent affects the attached tree, so do not silently enable memory settings. Ask the user first. If the user is not sure, recommend a default based on the observed workload and disclose it.
- **Implicit Promotion**: Do not silently promote memory flags, disk-backed outputs, exact priorities, comments, or logger output to requirements unless they are explicitly requested or delegated.

## Expression and Persistence Constraints

Classify each notebook step into one of these categories:
- **Direct Value**: Stored using `set_constant_value`. Read back with `get_constant_value`, not `get_value`. Large objects that should be disk-backed must be produced by a function stage instead, because `save_to_disk` applies only to declared produced outputs.
- **Expression**: Registered using `register_expression`.
- **Direct Function / Atom Child**: Registered using `register_function` or `create_atom_child_pipeline`.
- **Wrapper-Required**: Needs a wrapper function due to hidden globals or unrepresentable setup.
- **Unresolved**: Cannot be mapped without clarification.

**Direct Function Rules**
Inspect the real signatures of functions. Apply actual keyword parameter names in `param_mapping`. Do not guess. Mapping to a literal `None` is supported. Use wrappers only for hidden globals or unrepresentable setup.

If a user-defined function declares a parameter named `logger`, the pipeline injects its own logger automatically when executing that function. Do not pass the logger through `param_mapping`, do not wrap the function to supply it, and do not register it as an input. Just leave the `logger` parameter in the signature and let the pipeline fill it.

**Expression Rules**
Expressions must represent exactly one statement. Multiline formatting is allowed.
Restrictions:
- No semicolons, imports, `eval`, `exec`, or `__import__`.
- No comprehensions or generator expressions.
- No walrus operator.
- No nested function, class, or lambda definitions.
- No multiple assignment targets or unpacking.
- Statement form must be an assignment (`NAME = EXPR`) or a print/logger call.

If a step uses a comprehension, you cannot register it via `register_expression`. Preserve it using a normal importable function stage, or ask for approval to use an equivalent supported expression.

**Persistence Preflight**
Registered stages with an import path are restored from that path. Registered stages without one are saved by callable name, and the loading notebook or script must import or define that callable under the same name in `__main__` before calling `load_pipeline(...)`.

`target_function=functools.partial` supports save/load when `partial` is imported into the loading runtime first. Notebook-local functions and runtime-only callable values stored through `set_constant_value(...)`, `set_value(...)`, or produced upstream require an equivalent callable bound under the saved name before loading. Nested functions, lambdas, closures, and bound callables that have no reliable runtime name remain non-portable placeholders. These references are runtime dependencies, not historical code snapshots. Do not make unsupported claims about portability.

When the user requests a restart-safe backup, `save_pipeline(backup_path)` copies the current project tree, including nested disk-backed artifacts, into a non-overlapping target. Loading restores that backup into the canonical working directory. This copies project data, not Python source, installed packages, or the runtime environment.

**Single-Item Backup Recovery**

Use `root_pipeline.recover_variable_from_backup(name=...)` only when the root has a configured backup directory and a complete in-place save. The name must exist in both current and saved state. If independently owned same-name values exist across the attached tree, the method lists all affected pipeline paths and requires `yes` or `y`; refusal is a no-op. The method updates in place, returns `None`, clones disk-backed data into the live project, and does not rerun or invalidate derived outputs.

Use `selected_pipeline.recover_config_from_backup(name=...)` only for a field locally owned by that selected pipeline in both current and saved state. Existing config inheritance applies to descendants, while descendant local overrides remain unchanged. The method returns `None` and swaps in a staged config copy after validation.

Before either recovery call, import or define runtime-only callables under their saved names. This includes functions supplied to `functools.partial`. Never claim that recovery can reconstruct `RuntimeValueReference` or missing-class placeholders: unresolved selected objects raise `PersistenceError` and leave live state untouched. Recovery requires current-format `pipeline_state.pkl`, `config.pkl`, and `pipeline_meta.pkl`; it reads the backup without restoring the whole project tree.

## Final Checks and Output Contract

Before returning code, verify:
- The operational preflight files were read.
- Prefixes are applied consistently.
- Direct functions use real parameter names from `inspect.signature`.
- Functions that need the pipeline logger declare a `logger` parameter and are not wired to it through `param_mapping` or wrappers.
- Imports needed by expressions are in `define_expression_runtime`.
- Data and parameter tables are stored with `set_constant_value`.
- No dependent expressions share a block.
- `None` arguments use `param_mapping={...: None}`.
- Long expressions use triple-quoted strings.
- Wrappers exist only where genuinely needed.
- If extending a parent pipeline, the container vs workflow child structure is clear.

Your final output must include:
1. **Structure Explanation**: A short explanation of the proposed pipeline structure.
2. **Concise Conversion Manifest**: The manifest detailing hierarchy, stage types, priorities, inputs/outputs, prefixes, and disk-backed outputs.
3. **Converted Code**: The complete converted notebook code as one contiguous deliverable. If the whole script is reasonably short, output it in a single fenced code block. If it is too long for a convenient response, create one `.py` file for the user and provide its path or link. Never split the script across multiple sequential code blocks that the user must combine manually.
4. **Assumptions and Questions**: Any assumptions made or blocking questions.
5. **Verification and Persistence Notes**: Notes on how to verify the pipeline and any persistence limitations.
