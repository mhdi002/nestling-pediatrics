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


# What each generation of card can actually compute in, expressed as the
# compute capability where support begins rather than as a list of card
# names -- a list would be wrong for the next card, and this is checked
# against the hardware in front of us.
#
# fp8 arithmetic for weights arrived with Ada (8.9) and Hopper (9.0). vLLM
# will also keep the KV cache in fp8 from Ampere (8.0) onwards, where the
# conversion is done in software around bf16 math and still halves the cache.
# Below that -- Turing, 7.5 -- neither exists.
FP8_WEIGHTS_MIN_CAP = (8, 9)
FP8_KV_MIN_CAP = (8, 0)


def compute_capability() -> tuple[int, int] | None:
    """The CUDA compute capability of the card, e.g. (7, 5) for a 2080 Ti."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    first = next((line.strip() for line in out.splitlines() if line.strip()), "")
    try:
        major, _, minor = first.partition(".")
        return int(major), int(minor or 0)
    except ValueError:
        return None


def precision_for(cap: tuple[int, int] | None) -> dict:
    """Which quantisations this card can run.

    The compose file pinned fp8 for both weights and KV cache, sized for the
    Ada card it was first deployed on. On a Turing card vLLM refuses to start
    at all -- the sidecar never comes up, the app falls back to extractive
    RAG, and the deploy looks like it succeeded. Asking the driver what the
    card supports costs one subprocess and makes the same deploy correct on
    hardware nobody has tested it on yet.

    Unknown capability is treated as the oldest case: bf16 runs everywhere,
    and being slower is recoverable in a way that not starting is not.
    """
    if cap is None:
        return {
            "quantization": "none",
            "kv_cache_dtype": "auto",
            "note": "compute capability unknown -- bf16 weights and cache",
        }
    weights = "fp8" if cap >= FP8_WEIGHTS_MIN_CAP else "none"
    kv = "fp8" if cap >= FP8_KV_MIN_CAP else "auto"
    return {
        "quantization": weights,
        "kv_cache_dtype": kv,
        "note": f"sm_{cap[0]}{cap[1]}: weights={weights} kv={kv}",
    }


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
    precision = precision_for(compute_capability())
    result = {
        "max_model_len": floor_len,
        "max_num_seqs": floor_seqs,
        "quantization": precision["quantization"],
        "kv_cache_dtype": precision["kv_cache_dtype"],
        # Emitted even when every other number falls back to the compose
        # default, because this one is not a tuning choice -- it decides
        # whether the sidecar starts.
        "reason": "defaults kept",
    }
    total = gpu_memory_bytes()
    cfg_path = snapshot / "config.json"
    if total is None or not cfg_path.is_file():
        result["reason"] = (
            "no GPU reading or no model config; keeping defaults; " + precision["note"]
        )
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
            f"kv_per_token={per_token}B budget_tokens={tokens_total}; "
            + precision["note"]
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
    # The app widens its own request concurrency to match what the sidecar can
    # batch. The app reads VLLM_MAX_NUM_SEQS on its own (app/concurrency.py),
    # so this is emitted mainly to make the derived value explicit in .env and
    # overridable there; the two are the same number by construction.
    result["app_concurrency"] = seqs
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
    print(f"VLLM_QUANTIZATION={p['quantization']}")
    print(f"VLLM_KV_CACHE_DTYPE={p['kv_cache_dtype']}")
    if "lb_chat_burst" in p:
        print(f"NESTLING_LB_CHAT_BURST={p['lb_chat_burst']}")
        print(f"NESTLING_LB_CHAT_RPS={p['lb_chat_rps']}")
    if "limit_mm_image" in p:
        print(f"VLLM_LIMIT_MM_IMAGE={p['limit_mm_image']}")
    if "app_concurrency" in p:
        print(f"NESTLING_LLM_MAX_CONCURRENCY={p['app_concurrency']}")
    print(f"# {p['reason']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
