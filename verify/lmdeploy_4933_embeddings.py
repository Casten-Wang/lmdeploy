"""GPU end-to-end check for input-embedding prefix-cache identity.

This file belongs only to the validation branch.  It is intentionally excluded
from the minimal upstream patch.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess

import numpy as np
import torch
from transformers import AutoTokenizer

import lmdeploy
from lmdeploy.messages import GenerationConfig, PytorchEngineConfig, ResponseType
from lmdeploy.pytorch.engine.engine import Engine
from lmdeploy.pytorch.engine.request import RequestType
from lmdeploy.pytorch.messages import InputEmbeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local path or Hugging Face model id")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", default="lmdeploy-4933-result.json")
    return parser.parse_args()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_with_embeddings(instance, session_id, token_ids, embedding, start, end, config):
    original_send_async = instance.req_sender.send_async

    def send_async(request_type, data):
        if request_type is RequestType.ADD_MESSAGE:
            data = dict(data)
            data["input_embeddings"] = [
                InputEmbeddings(embedding.copy(), start, end)
            ]
        return original_send_async(request_type, data)

    instance.req_sender.send_async = send_async
    try:
        outputs = list(instance.stream_infer(session_id, token_ids, gen_config=config))
    finally:
        instance.req_sender.send_async = original_send_async

    generated = []
    cached_tokens = None
    for output in outputs:
        if output.status not in (ResponseType.SUCCESS, ResponseType.FINISH):
            raise RuntimeError(f"inference failed with status {output.status}")
        generated.extend(output.token_ids)
        if output.req_metrics is not None:
            cached_tokens = output.req_metrics.cached_tokens
    return generated, cached_tokens


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this validation requires an NVIDIA CUDA GPU")

    engine_config = PytorchEngineConfig(
        tp=1,
        eager_mode=True,
        enable_prefix_caching=True,
        block_size=64,
        max_batch_size=1,
        revision=args.revision,
    )
    engine = Engine(args.model, engine_config=engine_config)
    engine.start()
    instance = engine.create_instance()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, revision=args.revision, trust_remote_code=False
        )
        encoded = tokenizer.encode(" cache identity", add_special_tokens=False)
        token_id = encoded[0] if encoded else (tokenizer.bos_token_id or 1)
        block_size = engine.cache_config.block_size
        token_ids = [token_id] * (3 * block_size + 1)
        hidden_size = int(engine.model_config.hidden_size)
        rng = np.random.default_rng(4933)
        first = rng.standard_normal((block_size, hidden_size), dtype=np.float32)
        second = -first
        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            ignore_eos=True,
        )

        first_tokens, first_hit = run_with_embeddings(
            instance,
            493301,
            token_ids,
            first,
            block_size,
            2 * block_size,
            generation_config,
        )
        warm_tokens, warm_hit = run_with_embeddings(
            instance,
            493302,
            token_ids,
            second,
            block_size,
            2 * block_size,
            generation_config,
        )
        evicted_blocks = engine.scheduler.block_trie.evict(1 << 30)
        cold_tokens, cold_hit = run_with_embeddings(
            instance,
            493303,
            token_ids,
            second,
            block_size,
            2 * block_size,
            generation_config,
        )
    finally:
        engine.close()

    result = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "lmdeploy": getattr(lmdeploy, "__version__", None),
            "git_revision": git_revision(),
            "model": args.model,
            "model_revision": args.revision,
        },
        "workload": {
            "block_size": block_size,
            "prompt_tokens": len(token_ids),
            "embedding_span": [block_size, 2 * block_size],
            "embedding_shape": list(first.shape),
            "embedding_dtype": str(first.dtype),
            "max_new_tokens": args.max_new_tokens,
        },
        "runs": {
            "first_e1_cold": {"tokens": first_tokens, "cached_tokens": first_hit},
            "second_e2_warm": {"tokens": warm_tokens, "cached_tokens": warm_hit},
            "third_e2_cold": {"tokens": cold_tokens, "cached_tokens": cold_hit},
            "evicted_blocks": evicted_blocks,
        },
        "checks": {
            "embedding_changes_output": first_tokens != cold_tokens,
            "warm_matches_cold_for_e2": warm_tokens == cold_tokens,
            "different_embedding_stops_before_span": warm_hit == block_size,
        },
    }
    result["passed"] = all(result["checks"].values())
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
