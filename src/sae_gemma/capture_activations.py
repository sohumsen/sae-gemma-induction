"""
Phase 4.1 — Capture top-20 activating text snippets for every SAE feature.

Streams ~1M tokens from the held-out C4 dataset (different from training data),
runs each batch through Gemma + SAE, and records the top-20 snippets per feature
by peak activation value. Results serialised to results/top_snippets.parquet.

Schema of top_snippets.parquet:
    feature_id  int32
    rank        int8    (0 = highest activation)
    activation  float32
    token       str     (the activating token)
    context     str     (±30 chars of surrounding text)
    token_pos   int32   (position in the sequence)
    seq_id      int64   (global sequence counter for reproducibility)

Usage:
    python src/sae_gemma/capture_activations.py \
        --sae-path models/sae_main \
        --n-tokens 1000000 \
        --top-k 20
"""

import argparse
import heapq
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

from sae_gemma.model_utils import load_model
from sae_gemma.paths import HOOK_NAME, REPO_ROOT, SAE_MAIN_DIR, TOP_SNIPPETS_PATH

CONTEXT_CHARS = 120  # chars of context around activating token


def load_sae_local(sae_path: Path, device: str):
    """Load SAE weights saved by LanguageModelSAETrainingRunner."""
    from sae_lens.saes.sae import SAE
    sae = SAE.load_from_disk(str(sae_path), device=device)
    sae.eval()
    return sae


class TopKBuffer:
    """Per-feature min-heap of (activation, counter, metadata) — keeps top-k by activation."""

    def __init__(self, n_features: int, top_k: int):
        self.n_features = n_features
        self.top_k = top_k
        # heaps[f] = min-heap of (activation_value, counter, metadata_dict)
        # counter breaks ties so dict comparison is never needed
        self.heaps: list[list] = [[] for _ in range(n_features)]
        self._counter = 0

    def update(self, feature_id: int, activation: float, metadata: dict):
        heap = self.heaps[feature_id]
        entry = (activation, self._counter, metadata)
        self._counter += 1
        if len(heap) < self.top_k:
            heapq.heappush(heap, entry)
        elif activation > heap[0][0]:
            heapq.heapreplace(heap, entry)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for fid, heap in enumerate(self.heaps):
            sorted_items = sorted(heap, key=lambda x: -x[0])  # descending
            for rank, (act, _cnt, meta) in enumerate(sorted_items):
                rows.append({
                    "feature_id": fid,
                    "rank": rank,
                    "activation": float(act),
                    "token": meta.get("token", ""),
                    "context": meta.get("context", ""),
                    "token_pos": meta.get("token_pos", -1),
                    "seq_id": meta.get("seq_id", -1),
                })
        if not rows:
            return pd.DataFrame(columns=["feature_id", "rank", "activation", "token", "context", "token_pos", "seq_id"]).astype({
                "feature_id": "int32", "rank": "int8", "activation": "float32",
                "token_pos": "int32", "seq_id": "int64",
            })
        return pd.DataFrame(rows).astype({
            "feature_id": "int32",
            "rank": "int8",
            "activation": "float32",
            "token_pos": "int32",
            "seq_id": "int64",
        })


def run_capture(
    sae_path: Path,
    n_tokens: int,
    top_k: int,
    output_path: Path,
    batch_size: int = 16,
    context_size: int = 512,
    device: str = "cuda",
):
    print(f"[capture] Loading Gemma-2-2B ...", flush=True)
    model = load_model(device=device)

    print(f"[capture] Loading SAE from {sae_path} ...", flush=True)
    sae = load_sae_local(sae_path, device)
    n_features = sae.cfg.d_sae

    print(f"[capture] n_features={n_features}, hook={HOOK_NAME}", flush=True)
    buf = TopKBuffer(n_features, top_k)

    tokenizer = model.tokenizer

    print("[capture] Streaming C4 ...", flush=True)
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

    tokens_seen = 0
    seq_id = 0
    t0 = time.monotonic()

    token_batch = []  # list of 1-D token tensors

    for example in dataset:
        text = example["text"]
        toks = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]
        # Chunk into context_size windows
        for start in range(0, len(toks) - context_size + 1, context_size):
            chunk = toks[start: start + context_size]
            token_batch.append((seq_id, chunk, text))
            seq_id += 1
            tokens_seen += len(chunk)

            if len(token_batch) >= batch_size:
                _process_batch(token_batch, model, sae, buf, tokenizer, device)
                token_batch = []

                if tokens_seen % 100_000 < batch_size * context_size:
                    elapsed = time.monotonic() - t0
                    rate = tokens_seen / elapsed
                    eta = (n_tokens - tokens_seen) / rate if rate > 0 else 0
                    print(
                        f"[capture] {tokens_seen:,}/{n_tokens:,} tokens  "
                        f"rate={rate/1000:.1f}k tok/s  ETA={eta/60:.0f}m",
                        flush=True,
                    )

        if tokens_seen >= n_tokens:
            break

    # Process remaining partial batch
    if token_batch:
        _process_batch(token_batch, model, sae, buf, tokenizer, device)

    print(f"[capture] Processed {tokens_seen:,} tokens. Building DataFrame ...", flush=True)
    df = buf.to_dataframe()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[capture] Saved {len(df):,} rows to {output_path}", flush=True)


@torch.no_grad()
def _process_batch(
    batch: list[tuple],
    model: HookedTransformer,
    sae,
    buf: TopKBuffer,
    tokenizer,
    device: str,
):
    """Run a batch of sequences through Gemma + SAE, update top-k buffer."""
    seq_ids = [item[0] for item in batch]
    chunks = [item[1] for item in batch]
    texts = [item[2] for item in batch]

    # Pad to same length
    max_len = max(len(c) for c in chunks)
    padded = torch.zeros(len(chunks), max_len, dtype=torch.long)
    for i, c in enumerate(chunks):
        padded[i, :len(c)] = c
    padded = padded.to(device)

    # Hook to capture residual stream at target layer
    residuals = {}

    def hook_fn(value, hook):
        residuals[hook.name] = value.detach()

    model.run_with_hooks(
        padded,
        fwd_hooks=[(HOOK_NAME, hook_fn)],
    )

    acts = residuals[HOOK_NAME]  # (batch, seq, d_model)
    B, S, _ = acts.shape

    # Run SAE encoder to get feature activations
    acts_flat = acts.reshape(B * S, -1).float()  # SAE trained in float32
    feature_acts = sae.encode(acts_flat)  # (B*S, n_features)
    feature_acts = feature_acts.reshape(B, S, -1).cpu().float().numpy()

    # For each sequence position, update top-k per feature
    for i, (sid, chunk, text) in enumerate(zip(seq_ids, chunks, texts)):
        # Decode tokens for context extraction
        token_strs = tokenizer.convert_ids_to_tokens(chunk.tolist())
        token_strs_decoded = [tokenizer.decode([tid]) for tid in chunk.tolist()]

        for pos in range(len(chunk)):
            act_vec = feature_acts[i, pos]  # (n_features,)
            # Only update non-zero activations (sparse — most are 0)
            nonzero_fids = np.where(act_vec > 0)[0]
            for fid in nonzero_fids:
                activation = float(act_vec[fid])
                # Build context string (surrounding decoded text)
                ctx_start = max(0, pos - 10)
                ctx_end = min(len(chunk), pos + 10)
                context = tokenizer.decode(chunk[ctx_start:ctx_end].tolist())
                tok_str = token_strs_decoded[pos]
                meta = {
                    "token": tok_str,
                    "context": context[:CONTEXT_CHARS * 2],
                    "token_pos": pos,
                    "seq_id": sid,
                }
                buf.update(int(fid), activation, meta)


def main():
    parser = argparse.ArgumentParser(description="Capture top-20 activating snippets per SAE feature")
    parser.add_argument("--sae-path", type=Path, default=SAE_MAIN_DIR)
    parser.add_argument("--n-tokens", type=int, default=1_000_000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=Path, default=TOP_SNIPPETS_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    run_capture(
        sae_path=args.sae_path,
        n_tokens=args.n_tokens,
        top_k=args.top_k,
        output_path=args.output,
        batch_size=args.batch_size,
        context_size=args.context_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
