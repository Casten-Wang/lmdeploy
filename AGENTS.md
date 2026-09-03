# GPU Validation Task

This branch exists only to validate the proposed LMDeploy fix for issue #4933.
The production candidate is commit `68b34205`; commit `c3d21386` adds the
validation script and must not be included in an upstream pull request.

## Objective

Run the cold/warm/cold GPU experiment documented in
`verify/README-lmdeploy-4933.md` and determine whether different
`input_embeddings` values are isolated in the prefix-cache identity.

## Required behavior

- Use an NVIDIA Linux host with CUDA and the PyTorch backend.
- Run from branch `validation/lmdeploy-4933-gpu` without rebasing or modifying
  the production fix.
- Prefer `Qwen/Qwen2.5-0.5B-Instruct` unless a local equivalent is explicitly
  supplied.
- Preserve the generated `lmdeploy-4933-result.json` and the terminal log.
- Report exact hardware, CUDA, PyTorch, LMDeploy commit, model revision,
  workload, cache-hit counts, generated tokens, and every failed check.
- A nonzero exit is a validation failure to investigate, not permission to
  weaken or remove a check.

## Public-action safety

- Do not open, update, comment on, close, or reopen an issue or pull request.
- Do not push any branch or commit.
- Do not modify the official `InternLM/lmdeploy` repository or remote.
- Return the result artifact to the user for review. Public submission requires
  a separate, explicit approval after validation.

## Success criteria

The run passes only when all three conditions are true:

1. the two different embedding payloads produce observably different cold
   outputs;
2. the warm output for the second payload equals its cold output;
3. the second request reuses only the complete token block before the changed
   embedding span, rather than blocks whose KV depends on that span.

Do not claim a performance improvement. This experiment validates correctness,
not throughput or latency.
