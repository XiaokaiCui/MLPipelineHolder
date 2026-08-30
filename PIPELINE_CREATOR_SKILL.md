# Pipeline Creator Skill

Use this skill when an LLM agent is helping a first-time user convert a normal Jupyter notebook into a notebook managed by `MLPipelineHolder`.

This file is a compact conversion playbook for the agent.

## Operational Preflight

Before writing pipeline code, read `README.md`, the relevant implementation in
`pipeline_handler.py` and `execution_block.py`, and the tests for the APIs you will use
(especially execution, save/load, and recovery tests).

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

Strict mode fails registration when mappings do not match the callable signature, mapped
inputs or gates are not currently visible, kwargs helpers conflict with explicit
parameters, or an attached child cross-collides with parent config, values, or outputs.

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
- **Output Pointer Chains**: When successive expression, function, or atom stages intentionally replace the same large output, propose <code>overridden_outputs={"name": ("pipeline_name", "upstream_node")}</code>. This keeps one terminal completed object or artifact after the full suffix runs while preserving addressable upstream stages. Never point to constants, config, ordinary pipeline containers, atom internals, detached trees, or downstream nodes.
- **Implicit Promotion**: Do not silently promote memory flags, disk-backed outputs, exact priorities, comments, or logger output to requirements unless they are explicitly requested or delegated.

## Expression and Persistence Constraints

Classify each notebook step into one of these categories:
- **Direct Value**: Stored using `set_constant_value`. Read back with `get_constant_value`, not `get_value`. Mutable values are stored as a deep copy by default, so later in-place mutation of the original object does not affect the pipeline. Large objects that should be disk-backed can use `set_constant_value(..., to_disk=True)`; alternatively produce them via a function stage with `save_to_disk`.
- **Expression**: Registered using `register_expression`.
- **Direct Function / Atom Child**: Registered using `register_function` or `create_atom_child_pipeline`.
- **Wrapper-Required**: Needs a wrapper function due to hidden globals or unrepresentable setup.
- **Unresolved**: Cannot be mapped without clarification.

**Direct Function Rules**
Inspect the real signatures of functions. Apply actual keyword parameter names in `param_mapping`. Do not guess. Mapping to a literal `None` is supported. Use wrappers only for hidden globals or unrepresentable setup.

Pipeline configuration must be a dict or a pure fields-only dataclass, and every config
field must be picklable. Put non-picklable inputs in pipeline constants instead.

**Atom Child Rules**
`create_atom_child_pipeline` creates an immutable child: structural mutation (blocks,
children, functions, expressions, args/kwargs helpers, gates, or removals) raises
`RegistrationError`; config, constants, execution, and persistence remain available.

**Output Override Rules**
Use the exact spelling `overridden_outputs`. Each key must be one of the current stage's declared outputs and each value must be an existing globally-upstream `(pipeline_name, node_name)` that declares the same output. A declaration is metadata until the downstream stage commits successfully. Public reads and downstream inputs resolve the terminal real object automatically. Normal invalidation promotes a surviving owner; forbidden invalidation may splice a removed middle link; save/load preserves both declarations and active pointers. Explain that an incomplete partial rerun may temporarily retain multiple generations, while the complete missing suffix converges to one retained terminal value. If `set_value` reports that the selected slot is a pointer, follow its terminal owner message and set the value on that pipeline instead of bypassing the guard.

Forced re-registration of an atom, function, or expression is a no-op when its definition
is unchanged. Atom equality covers priority, gate, inner registrations/helpers, and only
the effective config fields the atom consumes; unused config is ignored. Function equality
covers callable identity, outputs, disk settings, mappings, variadic helper state, and
`functools.partial` bindings. A real change erases the overridden node's outputs and
invalidates from the earliest downstream consumer of any old or new output. A later
same-name producer shields consumers after it; descendant and ancestor mirrors are kept in
sync.

If a user-defined function declares a parameter named `logger`, the pipeline injects its own logger automatically when executing that function. Do not pass the logger through `param_mapping`, do not wrap the function to supply it, and do not register it as an input. Just leave the `logger` parameter in the signature and let the pipeline fill it.

**Expression Rules**
Expressions must contain exactly one assignment (`NAME = EXPR`) or one print/logger call.
A block may hold at most one expression. Semicolons, imports, `eval`, `exec`,
`__import__`, comprehensions, generators, walrus expressions, definitions, lambdas,
unpacking, and multiple targets are unsupported. Use an importable function stage when a
step needs any of them.

**Persistence Preflight**
Importable stages restore from their import path. Runtime-only stages require an equivalent
callable under the saved name in `__main__` before loading. Partials are saved as their
wrapped callable plus bound arguments; the callable must follow the same rule and every
bound value must be serializable or saving raises `PersistenceError`. Saved projects contain data and callable references, not
source code or an environment snapshot. Do not claim that nested functions, lambdas,
closures, or arbitrary callable instances are portable.

When the user requests a restart-safe backup, `save_pipeline(backup_path)` copies the current project tree, including nested disk-backed artifacts, into a non-overlapping target. Loading fully preflights the saved payload and callable references before restoring that backup into the canonical working directory, so preflight failure leaves the existing tree untouched. This copies project data, not Python source, installed packages, or the runtime environment.

By default, `load_pipeline(...)` re-runs the producer of each placeholder output without
invalidating downstream values and records the recovery in run history. Unrecoverable
placeholders raise on use. Importable dataclasses are reconstructed; otherwise a
`SimpleNamespace` is used. Mention this only when the workflow relies on non-picklable
outputs or dataclass configs.

**Single-Item Backup Recovery**

Use root `recover_variable_from_backup(...)` for values and the locally owning pipeline's
`recover_config_from_backup(...)` for config. Recovery needs a complete current-format
backup, preserves unrelated state, and does not rerun or invalidate derived outputs.
Disambiguate duplicate value owners with `pipeline_name`; multi-owner recovery requires
confirmation. Runtime callables must already be available, and unresolved placeholders
raise `PersistenceError` without changing live state.

## Final Checks and Output Contract

Before returning code, verify:
- The operational preflight files were read.
- Prefixes are applied consistently.
- Direct functions use real parameter names from `inspect.signature`.
- Functions that need the pipeline logger declare a `logger` parameter and are not wired to it through `param_mapping` or wrappers.
- Imports needed by expressions are in `define_expression_runtime`.
- Data and parameter tables are stored with `set_constant_value`.
- Repeated same-name large outputs use `overridden_outputs` only when the target is a verified public upstream block or atom.
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
