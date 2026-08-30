"""
Per-entity evaluation of a trained NER model.

train_bert.py reports one aggregate F1 across all five entity types. That
number is dominated by the easy categories: QUANTITY (mostly digits) and
UNIT (a small closed vocabulary) together make up ~52% of test entities,
while B-PRODUCT is only 2.3% and so barely moves the aggregate.

For HalalGuard the entity that matters is ING -- it is what feeds the
classifier. This script breaks performance down per entity type so that
number can be reported directly.

No retraining needed: it loads the saved model and runs inference over the
held-out split.

Usage (from the HalalGuard folder):
    python tools/evaluate_ner.py
    python tools/evaluate_ner.py --split val
    python tools/evaluate_ner.py --model_dir models/bert-ner --batch_size 16
"""

import argparse
import json
import os

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForTokenClassification


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="models/bert-ner")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--out", default="data/per_entity_results.csv")
    args, _ = p.parse_known_args()
    return args


def load_split(data_dir: str, split: str) -> list[dict]:
    path = os.path.join(data_dir, f"{split}.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def predict(model, tokenizer, sentences, device, batch_size, max_length):
    """Run inference and map subword predictions back to word level.

    Only the FIRST subword of each word carries a prediction -- the same
    convention training used (continuation pieces were masked to -100), so
    evaluation must mirror it or the tag sequences will not align.
    """
    id2label = model.config.id2label
    all_true, all_pred = [], []

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]
        tokens = [s["tokens"] for s in batch]

        enc = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)

        logits = model(**enc).logits
        preds = torch.argmax(logits, dim=2).cpu().numpy()

        for i, sent in enumerate(batch):
            word_ids = enc.word_ids(batch_index=i)
            seen = set()
            true_seq, pred_seq = [], []
            for pos, wid in enumerate(word_ids):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                true_seq.append(sent["tags"][wid])
                pred_seq.append(id2label[int(preds[i][pos])])
            # Words dropped by truncation get no prediction; keep the
            # sequences the same length rather than silently misaligning.
            all_true.append(true_seq)
            all_pred.append(pred_seq)

        done = min(start + batch_size, len(sentences))
        if done % (batch_size * 50) == 0 or done == len(sentences):
            print(f"  {done}/{len(sentences)}", end="\r")

    print()
    return all_true, all_pred


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model : {args.model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(device).eval()

    sentences = load_split(args.data_dir, args.split)
    print(f"Split : {args.split} ({len(sentences)} sentences)\n")

    true_seqs, pred_seqs = predict(
        model, tokenizer, sentences, device, args.batch_size, args.max_length
    )

    truncated = sum(1 for s, t in zip(sentences, true_seqs) if len(s["tags"]) != len(t))
    if truncated:
        print(f"NOTE: {truncated} sentences were truncated at max_length="
              f"{args.max_length}; their tail words are excluded from scoring.\n")

    print("=" * 62)
    print(f"AGGREGATE ({args.split})")
    print("=" * 62)
    print(f"  precision {precision_score(true_seqs, pred_seqs):.4f}")
    print(f"  recall    {recall_score(true_seqs, pred_seqs):.4f}")
    print(f"  f1        {f1_score(true_seqs, pred_seqs):.4f}")

    print()
    print("=" * 62)
    print("PER ENTITY TYPE")
    print("=" * 62)
    report = classification_report(true_seqs, pred_seqs, digits=4)
    print(report)

    # Machine-readable copy for the results table.
    rows = classification_report(true_seqs, pred_seqs, output_dict=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("entity,precision,recall,f1,support\n")
        for k, v in rows.items():
            if isinstance(v, dict):
                f.write(f"{k},{v['precision']:.4f},{v['recall']:.4f},"
                        f"{v['f1-score']:.4f},{int(v['support'])}\n")
    print(f"Written to {args.out}")

    ing = rows.get("ING")
    if ing:
        print()
        print("ING is the entity HalalGuard depends on -- it is what feeds")
        print(f"the classifier. ING F1 = {ing['f1-score']:.4f} "
              f"(support {int(ing['support'])}).")


if __name__ == "__main__":
    main()
