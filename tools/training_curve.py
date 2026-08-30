"""
Extract the training/evaluation curve from a Trainer run.

Every checkpoint directory contains trainer_state.json, which holds the
full log_history: each logged training loss and each evaluation. This
pulls out the evaluation rows so convergence can be inspected rather than
assumed, and writes a CSV for plotting in the paper.

Usage (from the HalalGuard folder):
    python tools/training_curve.py
    python tools/training_curve.py models/bert-ner
"""

import csv
import json
import os
import sys


def find_state(root: str) -> str | None:
    """Prefer the top-level state; otherwise the highest-numbered checkpoint."""
    direct = os.path.join(root, "trainer_state.json")
    if os.path.exists(direct):
        return direct

    ckpts = []
    if os.path.isdir(root):
        for d in os.listdir(root):
            if d.startswith("checkpoint-"):
                try:
                    ckpts.append((int(d.split("-")[1]), d))
                except (IndexError, ValueError):
                    continue
    for _, d in sorted(ckpts, reverse=True):
        p = os.path.join(root, d, "trainer_state.json")
        if os.path.exists(p):
            return p
    return None


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "models/bert-ner"
    path = find_state(root)
    if not path:
        print(f"No trainer_state.json found under {root}")
        print("Check the folder name, or pass the path explicitly.")
        return 1

    print(f"Reading: {path}\n")
    with open(path, encoding="utf-8") as f:
        state = json.load(f)

    history = state.get("log_history", [])
    evals = [h for h in history if "eval_f1" in h]
    trains = [h for h in history if "loss" in h and "eval_loss" not in h]

    if not evals:
        print("No evaluation entries found in log_history.")
        return 1

    # Pair each eval with the nearest preceding training loss, so train and
    # eval loss can be compared at roughly the same point in training --
    # that divergence is the actual overfitting signal.
    print(f"{'step':>7} {'epoch':>6} {'train_loss':>11} {'eval_loss':>10} "
          f"{'precision':>10} {'recall':>8} {'eval_f1':>9}")
    print("-" * 68)

    rows = []
    for e in evals:
        step = e.get("step", 0)
        prior = [t for t in trains if t.get("step", 0) <= step]
        tl = prior[-1]["loss"] if prior else float("nan")
        rows.append({
            "step": step,
            "epoch": round(e.get("epoch", 0), 4),
            "train_loss": round(tl, 4),
            "eval_loss": round(e.get("eval_loss", 0), 4),
            "precision": round(e.get("eval_precision", 0), 4),
            "recall": round(e.get("eval_recall", 0), 4),
            "eval_f1": round(e.get("eval_f1", 0), 4),
        })
        r = rows[-1]
        print(f"{r['step']:>7} {r['epoch']:>6} {r['train_loss']:>11} "
              f"{r['eval_loss']:>10} {r['precision']:>10} {r['recall']:>8} "
              f"{r['eval_f1']:>9}")

    best = max(rows, key=lambda r: r["eval_f1"])
    last = rows[-1]
    print("\n" + "=" * 68)
    print(f"Best eval_f1 : {best['eval_f1']} at step {best['step']} "
          f"(epoch {best['epoch']})")
    print(f"Final eval_f1: {last['eval_f1']} at step {last['step']}")
    print(f"best_model_checkpoint: {state.get('best_model_checkpoint')}")

    # The two signals that actually distinguish converged from overfitted.
    if best["step"] < last["step"]:
        print(f"\nNOTE: eval_f1 peaked at step {best['step']} and did not "
              f"improve after. Training past the peak is not automatically "
              f"overfitting, but check whether eval_loss rose below.")
    rising = last["eval_loss"] > best["eval_loss"]
    print(f"eval_loss at best step: {best['eval_loss']} -> final: "
          f"{last['eval_loss']}  ({'ROSE' if rising else 'did not rise'})")

    out = os.path.join("data", "training_curve.csv")
    os.makedirs("data", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCurve written to {out} (for plotting in the paper)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
