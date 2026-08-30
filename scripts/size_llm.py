#!/usr/bin/env python3
"""Size the vLLM sidecar for the GPU that is actually present.

The compose defaults (max_num_seqs=1, max_model_len=1536) were chosen for an
8 GB consumer card. On a 24 GB board they leave most of the GPU idle and
serialise every chat: a hundred concurrent requests queued behind a batch of
one, and two thirds of them aged out at the proxy.

Nothing here is a lookup table of card names. Two measured facts decide it:

  * how much memory the GPU reports, via nvidia-smi;
  * the model's own shape, read from the config.json in its snapshot --
    layers, KV heads and head dimension are what a token of KV cache costs.

From those, the KV budget is whatever vLLM's memory fraction leaves after the
weights, and the number of sequences that fit is that budget divided by the
cost of one full-length sequence. A GPU big enough for more concurrency gets
it; a small one keeps today's conservative numbers, because the arithmetic
says so rather than because a threshold was written down.

Prints shell assignments on stdout:  VLLM_MAX_MODEL_LEN=... VLLM_MAX_NUM_SEQS=...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Bytes per stored value. vLLM serves these weights in bfloat16.
DTYPE_BYTES = 2
# Both halves of the attention cache, K and V.
KV_TENSORS = 2
# vLLM's own working set: activations, CUDA graphs, fragmentation. Measured as
# roughly a gigabyte on this model class; taken off the top so the KV estimate
# does not claim memory the runtime needs for itself.
RUNTIME_OVERHEAD_BYTES = 1024**3
# Context window this app can actually fill. The prompt is capped in
# assistant/settings.py (llm_prompt_context_chars + llm_prompt_query_chars,
# plus the system prompt) and the reply by llm_max_tokens_rag, so anything
# beyond this is cache reserved for tokens that never arrive.
# tests/test_size_llm.py asserts it still covers those caps.
DEFAULT_CONTEXT_TOKENS = 4096
# Wall-clock for one generation at this prompt and reply size, measured on
# the deployed 3090: an unloaded chat turn returns in about twelve seconds.
# Only used to convert concurrency into a sustained per-IP rate.
SECONDS_PER_GENERATION = 12


def gpu_memory_bytes() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    first = next((line.strip() for line in out.splitlines() if line.strip()), "")
    try:
        return int(float(first)) * 1024 * 1024
    except ValueError:
        return None


def weights_bytes(snapshot: Path) -> int:
    """Actual size of the weight shards on disk, not an estimate from params."""
    total = 0
    for pattern in ("*.safetensors", "*.bin"):
        for f in snapshot.glob(pattern):
            try:
                total += f.stat().st_size
            except OSError:
                continue
    return total


def text_config(cfg: dict) -> dict:
    """The language-model half of the config.

    A multimodal checkpoint keeps the transformer's shape under `text_config`
    and describes the vision tower separately, so reading the top level found
    nothing and sizing silently fell back to the 8 GB defaults.
    """
    inner = cfg.get("text_config")
    return inner if isinstance(inner, dict) else cfg


def is_multimodal(cfg: dict) -> bool:
    """Whether this checkpoint can accept images at all."""
    return isinstance(cfg.get("vision_config"), dict)


def kv_bytes_per_token(cfg: dict) -> int | None:
    cfg = text_config(cfg)
    layers = cfg.get("num_hidden_layers")
    hidden = cfg.get("hidden_size")
    heads = cfg.get("num_attention_heads")
    kv_heads = cfg.get("num_key_value_heads", heads)
    if not all(isinstance(v, int) and v > 0 for v in (layers, hidden, heads, kv_heads)):
        return None
    head_dim = cfg.get("head_dim") or hidden // heads
    return KV_TENSORS * layers * kv_heads * head_dim * DTYPE_BYTES


def plan(
    snapshot: Path,
    gpu_util: float,
    floor_len: int,
    floor_seqs: int,
    want_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
) -> dict:
    result = {
        "max_model_len": floor_len,
        "max_num_seqs": floor_seqs,
        "reason": "defaults kept",
    }
    total = gpu_memory_bytes()
    cfg_path = snapshot / "config.json"
    if total is None or not cfg_path.is_file():
        result["reason"] = "no GPU reading or no model config; keeping defaults"
        return result
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        result["reason"] = "unreadable model config; keeping defaults"
        return result

    per_token = kv_bytes_per_token(cfg)
    weights = weights_bytes(snapshot)
    if not per_token or not weights:
        result["reason"] = "model shape or weight size unknown; keeping defaults"
        return result

    budget = int(total * gpu_util) - weights - RUNTIME_OVERHEAD_BYTES
    if budget <= 0:
        result["reason"] = "weights fill the GPU; keeping defaults"
        return result

    # Never promise more context than the model was trained to attend over.
    ceiling = text_config(cfg).get("max_position_embeddings") or floor_len
    tokens_total = budget // per_token

    # Give the context window what this app can actually fill, and spend
    # everything left on concurrency. The prompt is capped in settings before
    # it ever reaches the sidecar, so a window past that serves nobody: sizing
    # a 24 GiB card for its full 32k context yielded two sequences and left
    # every chat queued behind one generation. Concurrency is the scarce thing
    # for a chat service; context past the cap is not.
    max_len = max(floor_len, min(ceiling, want_context_tokens))

    seqs = int(tokens_total // max_len)
    while seqs < floor_seqs and max_len > floor_len:
        max_len //= 2
        seqs = int(tokens_total // max_len)

    seqs = max(floor_seqs, seqs)
    result.update(
        max_model_len=int(max_len),
        max_num_seqs=seqs,
        reason=(
            f"gpu={total // 1024**3}GiB weights={weights // 1024**3}GiB "
            f"kv_per_token={per_token}B budget_tokens={tokens_total}"
        ),
    )
    result.update(lb_limits(seqs))
    # Vision was pinned off with --limit-mm-per-prompt image=0 on every GPU,
    # with a comment about saving KV cache on 8 GB cards. On a card with room
    # to spare that silently threw away a feature the checkpoint supports: a
    # parent could upload a photo of a rash and be answered from the caption
    # alone. Enable it when the checkpoint has a vision tower AND the sizing
    # above found spare capacity, which is the condition that comment was
    # really about.
    result["limit_mm_image"] = 1 if (is_multimodal(cfg) and seqs > floor_seqs) else 0
    return result


def lb_limits(max_num_seqs: int) -> dict:
    """Per-IP admission at the proxy, sized to what the sidecar can serve.

    The proxy's burst was a fixed 5 whatever the hardware, so a card able to
    run twenty-two sequences still turned away the sixth caller: a hundred
    concurrent requests came back as ninety-two rejections. Admitting one full
    batch matches the queue the sidecar can actually work on.

    Sustained rate is throughput, not a preference: a batch of `max_num_seqs`
    takes about one generation to clear, so the service drains that many
    requests per generation and no faster. Both stay per-IP, so this is still
    abuse protection -- it is just sized to the machine instead of to a guess.
    """
    return {
        "lb_chat_burst": max(1, max_num_seqs),
        "lb_chat_rps": max(1, round(max_num_seqs / SECONDS_PER_GENERATION)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--floor-model-len", type=int, default=1536)
    ap.add_argument("--floor-num-seqs", type=int, default=1)
    ap.add_argument("--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS)
    args = ap.parse_args(argv)

    p = plan(
        args.snapshot,
        args.gpu_util,
        args.floor_model_len,
        args.floor_num_seqs,
        args.context_tokens,
    )
    print(f"VLLM_MAX_MODEL_LEN={p['max_model_len']}")
    print(f"VLLM_MAX_NUM_SEQS={p['max_num_seqs']}")
    if "lb_chat_burst" in p:
        print(f"NESTLING_LB_CHAT_BURST={p['lb_chat_burst']}")
        print(f"NESTLING_LB_CHAT_RPS={p['lb_chat_rps']}")
    if "limit_mm_image" in p:
        print(f"VLLM_LIMIT_MM_IMAGE={p['limit_mm_image']}")
    print(f"# {p['reason']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
