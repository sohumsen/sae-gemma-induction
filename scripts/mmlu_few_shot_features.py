"""
Verify that the top induction SAE feature (F15289 in v9c) — auto-labelled
"second occurrence of a repeated word or phrase" — generalises from our
synthetic induction probes to natural in-context learning on MMLU.

For each of N MMLU questions we build:
  - a real 4-shot prompt (4 same-subject Q/A examples + target Q with "Answer:")
  - a control prompt with the 4 example answers SHUFFLED (same vocabulary,
    no consistent Q->A mapping, so the induction signal is broken)

We then run Gemma-2-2B + the v9c SAE and record the activation of
TARGET_FEATURES at the final token position. If F15289 is a genuine
induction feature, its mean activation on real few-shot prompts should
be much larger than on the shuffled controls.

Output: results/mmlu_feature_activations.json

Usage:
    python scripts/mmlu_few_shot_features.py \
        --sae-path models/sae_main \
        --n-questions 200
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

from sae_gemma.model_utils import load_model
from sae_gemma.paths import HOOK_NAME, REPO_ROOT, RESULTS_DIR, SAE_MAIN_DIR

TARGET_FEATURES = [15289, 11606, 14740, 7467]
MAX_CONTEXT_LEN = 1024
CHOICE_LETTERS = ["A", "B", "C", "D"]
OUTPUT_PATH = RESULTS_DIR / "mmlu_feature_activations.json"


def load_sae_local(sae_path: Path, device: str):
    from sae_lens.saes.sae import SAE
    sae = SAE.load_from_disk(str(sae_path), device=device)
    sae.eval()
    return sae


def _format_question_block(row, include_answer: bool, answer_override: str | None = None) -> str:
    """Render an MMLU row as `Question: ... Answer: X` (optionally without the answer)."""
    q = row["question"].strip()
    choices = row["choices"]
    lines = [f"Question: {q}"]
    for letter, choice in zip(CHOICE_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    if include_answer:
        letter = answer_override if answer_override is not None else CHOICE_LETTERS[row["answer"]]
        lines.append(f"Answer: {letter}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def _build_prompt(examples: list[dict], target: dict, answer_letters: list[str]) -> str:
    """4-shot prompt: 4 example blocks (with given answer letters) + target Q without answer."""
    blocks = [
        _format_question_block(ex, include_answer=True, answer_override=letter)
        for ex, letter in zip(examples, answer_letters)
    ]
    blocks.append(_format_question_block(target, include_answer=False))
    return "\n\n".join(blocks)


@torch.no_grad()
def _final_pos_features(
    model: HookedTransformer,
    sae,
    hook_name: str,
    token_seqs: list[list[int]],
    feature_ids: list[int],
    device: str,
    batch_size: int = 4,
) -> np.ndarray:
    """
    Run Gemma + SAE on each (possibly variable-length) sequence. Return only the
    requested feature activations at the final position. Shape: (n_seqs, n_feats).
    """
    n_seqs = len(token_seqs)
    n_feats = len(feature_ids)
    feat_idx = torch.tensor(feature_ids, dtype=torch.long, device=device)
    out = np.zeros((n_seqs, n_feats), dtype=np.float32)

    for i in range(0, n_seqs, batch_size):
        batch = token_seqs[i: i + batch_size]
        max_len = max(len(s) for s in batch)
        seq_lens = [len(s) for s in batch]

        padded = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        for j, s in enumerate(batch):
            padded[j, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)

        residuals = {}

        def hook_fn(value, hook):
            residuals[hook.name] = value.detach()

        model.run_with_hooks(padded, fwd_hooks=[(hook_name, hook_fn)])
        acts = residuals[hook_name]  # (batch, seq, d_model)

        for j, slen in enumerate(seq_lens):
            final_act = acts[j, slen - 1, :].unsqueeze(0)  # (1, d_model)
            feat_acts = sae.encode(final_act.float()).squeeze(0)  # (d_sae,)
            out[i + j] = feat_acts.index_select(0, feat_idx).cpu().float().numpy()

        if (i // batch_size) % 10 == 0:
            n_done = min(i + batch_size, n_seqs)
            print(f"  {n_done}/{n_seqs} sequences", flush=True)

    return out


def main():
    parser = argparse.ArgumentParser(description="MMLU few-shot vs shuffled control on SAE features")
    parser.add_argument("--sae-path", type=Path, default=SAE_MAIN_DIR)
    parser.add_argument("--n-questions", type=int, default=200)
    parser.add_argument("--n-shots", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260518)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    print("[mmlu] Loading Gemma-2-2B ...", flush=True)
    model = load_model(device=args.device)
    tokenizer = model.tokenizer

    print(f"[mmlu] Loading SAE from {args.sae_path} ...", flush=True)
    sae = load_sae_local(args.sae_path, args.device)
    print(f"[mmlu] d_sae={sae.cfg.d_sae}, hook={HOOK_NAME}", flush=True)
    for fid in TARGET_FEATURES:
        if fid >= sae.cfg.d_sae:
            raise ValueError(f"feature id {fid} out of range (d_sae={sae.cfg.d_sae})")

    print("[mmlu] Loading MMLU test split ...", flush=True)
    ds = load_dataset("cais/mmlu", "all", split="test")
    print(f"[mmlu] {len(ds)} MMLU test questions loaded", flush=True)

    # Bucket questions by subject so few-shot exemplars come from the same subject.
    by_subject: dict[str, list[int]] = defaultdict(list)
    for idx, subj in enumerate(ds["subject"]):
        by_subject[subj].append(idx)

    rng = random.Random(args.seed)

    # Sample target questions: only from subjects with >= n_shots+1 questions.
    eligible_subjects = [s for s, ids in by_subject.items() if len(ids) >= args.n_shots + 1]
    eligible_idxs = [i for s in eligible_subjects for i in by_subject[s]]
    target_idxs = rng.sample(eligible_idxs, args.n_questions)

    few_shot_seqs: list[list[int]] = []
    shuffled_seqs: list[list[int]] = []
    n_truncated = 0

    print(f"[mmlu] Building {args.n_questions} prompts ({args.n_shots}-shot) ...", flush=True)
    for tgt_idx in target_idxs:
        target = ds[int(tgt_idx)]
        subj = target["subject"]
        pool = [i for i in by_subject[subj] if i != tgt_idx]
        ex_idxs = rng.sample(pool, args.n_shots)
        examples = [ds[int(i)] for i in ex_idxs]

        true_letters = [CHOICE_LETTERS[ex["answer"]] for ex in examples]

        # Shuffled control: same vocabulary of answer letters, but reassigned
        # to different questions. Force a derangement so the few-shot pattern
        # is broken (unless all 4 letters are identical, in which case shuffling
        # cannot break it — we still try a few times then accept).
        shuffled_letters = list(true_letters)
        for _ in range(20):
            rng.shuffle(shuffled_letters)
            if any(a != b for a, b in zip(shuffled_letters, true_letters)):
                break

        real_prompt = _build_prompt(examples, target, true_letters)
        ctrl_prompt = _build_prompt(examples, target, shuffled_letters)

        real_ids = tokenizer.encode(real_prompt, add_special_tokens=False)
        ctrl_ids = tokenizer.encode(ctrl_prompt, add_special_tokens=False)

        if len(real_ids) > MAX_CONTEXT_LEN:
            real_ids = real_ids[-MAX_CONTEXT_LEN:]
            n_truncated += 1
        if len(ctrl_ids) > MAX_CONTEXT_LEN:
            ctrl_ids = ctrl_ids[-MAX_CONTEXT_LEN:]

        few_shot_seqs.append(real_ids)
        shuffled_seqs.append(ctrl_ids)

    if n_truncated:
        print(f"[mmlu] {n_truncated}/{args.n_questions} prompts truncated to {MAX_CONTEXT_LEN} tokens", flush=True)

    t0 = time.monotonic()
    print("[mmlu] Computing features on real few-shot prompts ...", flush=True)
    real_acts = _final_pos_features(
        model, sae, HOOK_NAME, few_shot_seqs, TARGET_FEATURES, args.device, args.batch_size
    )
    print("[mmlu] Computing features on shuffled-control prompts ...", flush=True)
    ctrl_acts = _final_pos_features(
        model, sae, HOOK_NAME, shuffled_seqs, TARGET_FEATURES, args.device, args.batch_size
    )
    elapsed = time.monotonic() - t0
    print(f"[mmlu] Activations computed in {elapsed:.0f}s", flush=True)

    few_shot_mean = real_acts.mean(axis=0)
    shuffled_mean = ctrl_acts.mean(axis=0)
    eps = 1e-6
    ratio = few_shot_mean / np.maximum(shuffled_mean, eps)

    result = {
        "n_questions": int(args.n_questions),
        "n_shots": int(args.n_shots),
        "seed": int(args.seed),
        "hook": HOOK_NAME,
        "sae_path": str(args.sae_path),
        "target_features": [int(f) for f in TARGET_FEATURES],
        "few_shot_mean": [float(x) for x in few_shot_mean],
        "shuffled_mean": [float(x) for x in shuffled_mean],
        "ratio": [float(x) for x in ratio],
        "n_truncated": int(n_truncated),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[mmlu] Saved results to {OUTPUT_PATH}", flush=True)

    print("\n[mmlu] Per-feature summary:")
    print(f"{'Feature':>8} {'FewShot':>10} {'Shuffled':>10} {'Ratio':>10}")
    for fid, fs, sh, r in zip(TARGET_FEATURES, few_shot_mean, shuffled_mean, ratio):
        print(f"{fid:>8} {fs:>10.4f} {sh:>10.4f} {r:>10.2f}")


if __name__ == "__main__":
    main()
