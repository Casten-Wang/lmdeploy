# LMDeploy #4933 GPU validation

## What this checks

LMDeploy's prefix cache previously keyed `input_embeddings` requests by token
IDs and span boundaries but not by embedding content. A warm request could
therefore reuse KV computed from a different embedding tensor. The candidate
fix adds the embedding digest to the existing exact block identity.

The validation performs three deterministic greedy runs:

1. embedding payload E1 with a cold cache;
2. different payload E2 while E1's prefix is cached;
3. E2 again after evicting trie-owned KV blocks.

The harness also proves that runs 1 and 3 are actually cold, that every run
generates tokens, that run 2 is identical to run 3, and that run 2 stops its
prefix hit at the complete block before the changed embedding span.

## Exact branch and commits

- Repository: `https://github.com/Casten-Wang/lmdeploy`
- Branch: `validation/lmdeploy-4933-gpu`
- Production fix: `68b3420515cbf87655fd87fc4c3e341fd962945c`
- Validation harness: `c3d2138692df22518a0453f0cdad6f38815d8329`
- Upstream base: `d9888113e862806fe06d10c007eb231acb99ddbb`

Verify the checkout before running:

```bash
git status --short --branch
git rev-parse HEAD
git merge-base --is-ancestor c3d2138692df22518a0453f0cdad6f38815d8329 HEAD
```

The final command must exit successfully, the branch must be
`validation/lmdeploy-4933-gpu`, and the worktree must be clean. Documentation
commits may exist after the validation-harness commit.

## Recommended environment

- Linux with an NVIDIA CUDA GPU
- Python 3.12 in a fresh Conda environment
- Enough disk space for LMDeploy dependencies and the 0.5B validation model
- Network access to Hugging Face, unless the model is already stored locally

## One-time setup

```bash
git clone --branch validation/lmdeploy-4933-gpu \
  https://github.com/Casten-Wang/lmdeploy.git
cd lmdeploy

conda create -n lmdeploy-4933 python=3.12 -y
conda activate lmdeploy-4933

DISABLE_TURBOMIND=1 pip install -e .
```

`DISABLE_TURBOMIND=1` keeps this PyTorch-only correctness check from compiling
the unrelated TurboMind backend.

## Run

```bash
python verify/lmdeploy_4933_embeddings.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output lmdeploy-4933-result.json \
  2>&1 | tee lmdeploy-4933-run.log
```

For a pre-downloaded model, replace the model ID with its absolute local path.
To pin a Hub revision, also pass `--revision REVISION`.

## Expected result

The command exits with status zero and the JSON ends with:

```json
{
  "checks": {
    "first_run_is_cold": true,
    "third_run_is_cold_after_eviction": true,
    "all_runs_generated_tokens": true,
    "embedding_changes_output": true,
    "warm_matches_cold_for_e2": true,
    "different_embedding_stops_before_span": true
  },
  "passed": true
}
```

The exact generated token IDs may differ across GPU architectures or software
versions. Equality is evaluated only within the same process and environment.
The JSON also records every visible GPU, the NVIDIA driver, CUDA and PyTorch
versions, the branch and commit, and whether the checkout was dirty.

## Return these artifacts

- `lmdeploy-4933-result.json`
- `lmdeploy-4933-run.log`
- If installation or execution fails, the complete error output and the result
  of `nvidia-smi`

Do not commit generated logs or result JSON. Do not open or update any public
issue or pull request from this validation branch.
