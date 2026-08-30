# Pipeline Creator Skill

Use this guide when an LLM is helping a first-time user convert an existing Jupyter notebook into a notebook managed by `MLPipelineHolder`.

The goal is to preserve the notebook's working logic while introducing clear pipeline scopes, reproducible stages, and only the persistence or memory features the workflow actually needs. This is a conversion playbook, not an API reference; consult `docs/api_reference.html` for complete signatures and class behaviour.

## 1. Inspect Before Converting

Before writing pipeline code:

- Read `README.md` and the relevant sections of `docs/api_reference.html`.
- Inspect the complete notebook, including helper functions, hidden globals, data-loading cells, and expected final outputs.
- Inspect the real signatures of every function you plan to register. Never guess parameter names.
- Identify which values are configuration, direct input data, produced outputs, or temporary notebook state.
- Identify whether the user is creating a new pipeline or extending a saved parent pipeline.

Ask one concise batch of questions only for choices that materially affect the result:

- Which steps and values are shared, and which belong to a workflow-specific child pipeline?
- What prefix should workflow-local config and values use?
- Which objects are genuinely large or expensive enough to save to disk?
- Will the pipeline be saved and loaded later, and where are its callables defined?
- Should memory-saving or memory-profile logging be enabled?

If the user is unsure, recommend sensible defaults from the observed notebook and state the assumptions. Do not turn cosmetic choices into blockers.

## 2. Propose a Small Conversion Manifest

Before producing the final code, decide:

- **Hierarchy and scope**: root, children, and shared versus local values.
- **Stage type**: direct constant, expression, function block, or atom child.
- **Priority**: execution order and any intentional alternative priority groups.
- **Inputs and outputs**: names consumed and produced by each stage.
- **Prefix**: naming for workflow-local config and values.
- **Disk-backed outputs**: only confirmed large or expensive outputs.

Include this manifest with the final conversion. An intermediate approval message is needed only when a high-impact decision remains ambiguous.

## 3. Build with Safe Defaults

Always create the root with strict mode:

```python
pipeline = PipelineHandler(..., strict_mode=True)
```

Strict mode turns unsafe registrations into immediate `RegistrationError`s. Register upstream producers before downstream consumers so their outputs are already declared; producers do **not** need to run before downstream registration. Attached children inherit strict mode from the root.

Use these structure rules:

- Put shared preparation and shared outputs on the parent pipeline.
- Put workflow-only config, constants, and outputs on the workflow child, using the agreed prefix.
- Functions inside one `ExecutionBlock` run in parallel, so dependent functions must use separate blocks.
- Integer priority groups represent alternatives: priorities such as `5.1` and `5.2` are in group `5`, and only one alternative executes.
- Use `set_constant_value` for direct notebook inputs and `get_constant_value` to read them. Use `get_value` for produced outputs.
- Pipeline configuration must be a dict or a fields-only dataclass whose field values are picklable. Put non-picklable direct inputs in constants instead.

## 4. Choose the Simplest Stage Type

- **Direct constant**: existing input data or a user-controlled value stored with `set_constant_value`.
- **Expression**: one simple assignment or print/logger call registered with `register_expression`.
- **Function block**: a normal importable function registered with `register_function`.
- **Atom child**: one reusable function that should appear as a child pipeline.
- **Wrapper**: only when the original step depends on hidden globals or setup that cannot be represented directly.

For function stages:

- Use actual callable parameter names in `param_mapping`.
- Mapping a parameter to literal `None` is supported.
- A parameter named `logger` is injected automatically; do not map or wrap it.
- Use `register_args` and `register_kwargs` only for real variadic inputs.

Keep expressions short. They support one assignment or one print/logger call. Put imports in `define_expression_runtime`; use a normal function when the step needs imports, comprehensions, lambdas, definitions, unpacking, or multiple statements.

Atom children are structurally immutable after creation. Use them only when that constraint matches the intended notebook structure.

## 5. Add Disk and Memory Features Deliberately

Do not infer object size without evidence. Propose likely disk-backed candidates using type, expected size, reuse frequency, and recomputation cost, explain the RAM-versus-I/O trade-off, and ask the user to confirm.

- Use `save_to_disk` only for outputs declared by a block.
- Use `set_constant_value(..., to_disk=True)` for a confirmed large direct input.
- Ask before enabling `memory_saving_mode` or `memory_profile_logging`; attached children inherit the root's settings.
- Use `overridden_outputs` only when successive stages intentionally replace the same large output. The target must be an existing globally upstream block or atom that declares the same output; alternatives in the same integer priority group cannot point to each other.

For partial reruns, use `run_from(...)` directly, including nested paths. Earlier producers remain available, the target may consume its previous output, and later producers stay hidden until they rerun; do not manually reconstruct upstream state.

## 6. Plan for Save and Load

Importable functions restore by import path. Notebook-only runtime functions must exist under the same `__main__` name before loading. Saved projects contain pipeline data and callable references, not source code, installed packages, or an environment snapshot.

When restart safety is required, save to a non-overlapping path with `save_pipeline(...)`. Normal saves use `cleanup="auto"` to remove obsolete framework-generated artifact generations and orphaned child trees; use `cleanup="none"` only when the user explicitly wants those old generated files retained.

`load_pipeline(...)` attempts to recover placeholder outputs by rerunning their producers by default. Mention placeholder and dataclass recovery only when the workflow contains non-picklable values or depends on that behaviour.

## 7. Final Verification and Handoff

Before returning the conversion, verify that:

- every input name resolves from config, constants, an upstream declared output, or a function default;
- dependent stages are in separate blocks;
- prefixes and parent/child ownership are consistent;
- disk-backed outputs and memory flags match the user's decision;
- expressions use only supported syntax and required runtime imports;
- persistence limitations for notebook-local callables are stated;
- the pipeline executes through the intended happy path and at least one partial rerun;
- save/load is exercised when persistence is part of the workflow.

The final response must include:

1. **Structure Explanation**: a short explanation of the pipeline hierarchy and scope choices.
2. **Concise Conversion Manifest**: hierarchy, stage types, priorities, inputs, outputs, prefixes, and disk-backed outputs.
3. **Converted Code**: one complete, contiguous deliverable. Use one fenced block when practical; otherwise create one `.py` file and provide its path.
4. **Assumptions and Questions**: disclosed assumptions and any unresolved blockers.
5. **Verification and Persistence Notes**: what was executed, how to rerun it, and any save/load limitations.
