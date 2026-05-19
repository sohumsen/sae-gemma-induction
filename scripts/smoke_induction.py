"""Induction smoke test — verify Gemma-2-2B does token-copying induction.

Build sequences like [random tokens][A][B][random tokens][A] and check that the
top-1 logit at the final position is B. Reports accuracy over N probes.

This is the critical Phase 1 gate: if baseline ICL accuracy is <50%, the project's
core assumption (Gemma-2-2B has functional induction) is wrong and we must escalate.
"""

import sys

import torch
from transformer_lens import HookedTransformer

N_PROBES = 200
SEQ_LEN_BEFORE_AB = 30
GAP_TOKENS = 80
MIN_VOCAB_ID = 1000  # avoid special tokens
MAX_VOCAB_ID = 50000


def build_probe(rng: torch.Generator, vocab_lo: int, vocab_hi: int) -> torch.Tensor:
    prefix = torch.randint(vocab_lo, vocab_hi, (SEQ_LEN_BEFORE_AB,), generator=rng)
    a = torch.randint(vocab_lo, vocab_hi, (1,), generator=rng)
    b = torch.randint(vocab_lo, vocab_hi, (1,), generator=rng)
    while b.item() == a.item():
        b = torch.randint(vocab_lo, vocab_hi, (1,), generator=rng)
    gap = torch.randint(vocab_lo, vocab_hi, (GAP_TOKENS,), generator=rng)
    # ensure A doesn't reappear inside the gap (would confuse the induction)
    gap = torch.where(gap == a.item(), (gap + 1).clamp(max=vocab_hi - 1), gap)
    return torch.cat([prefix, a, b, gap, a]), a.item(), b.item()


def main() -> int:
    model = HookedTransformer.from_pretrained("google/gemma-2-2b", dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()

    vocab_lo = MIN_VOCAB_ID
    vocab_hi = min(MAX_VOCAB_ID, model.cfg.d_vocab - 1)
    rng = torch.Generator().manual_seed(0)

    correct = 0
    top5_correct = 0
    with torch.no_grad():
        for i in range(N_PROBES):
            seq, a, b = build_probe(rng, vocab_lo, vocab_hi)
            seq = seq.unsqueeze(0).to("cuda")
            logits = model(seq)[0, -1]  # logits over vocab at final position
            top5 = torch.topk(logits, 5).indices.tolist()
            if top5[0] == b:
                correct += 1
            if b in top5:
                top5_correct += 1
            if i < 5:
                print(f"  probe {i}: A={a} B={b} top5={top5}")

    acc = correct / N_PROBES
    top5_acc = top5_correct / N_PROBES
    print(f"\nInduction top-1 accuracy: {acc:.1%} ({correct}/{N_PROBES})")
    print(f"Induction top-5 accuracy: {top5_acc:.1%} ({top5_correct}/{N_PROBES})")

    if acc < 0.5:
        print("\nFAIL: baseline ICL <50% — induction may not be working on Gemma-2-2B", file=sys.stderr)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
