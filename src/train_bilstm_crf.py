"""
BiLSTM-CRF baseline for ingredient NER.

The required comparison model. Without it, "BERT performs best" is asserted
rather than demonstrated -- this establishes what a strong pre-transformer
architecture achieves on the same data, so the gain from BERT can be
measured instead of claimed.

Architecture follows Huang et al. (2015) and Lample et al. (2016), both
already cited in the literature review:
    word embeddings -> BiLSTM -> linear -> CRF

Fairness constraints -- these matter more than the model itself:
  * identical train/val/test splits (data/*.jsonl from preprocess.py)
  * identical label vocabulary (data/labels.txt)
  * identical metric (seqeval, per entity type)
so the only difference between the two runs is the architecture.

The vocabulary is built from TRAINING data only. Building it over the full
dataset would leak test-set vocabulary into training and inflate the
baseline.

Usage (from the HalalGuard folder):
    python src/train_bilstm_crf.py
    python src/train_bilstm_crf.py --epochs 15 --hidden_dim 512
"""

import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

PAD, UNK = "<PAD>", "<UNK>"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="models/bilstm-crf")
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_freq", type=int, default=2)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=None)
    args, _ = p.parse_known_args()
    return args


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_vocab(sentences, min_freq):
    """Word vocabulary from TRAINING data only -- see module docstring."""
    counts = {}
    for s in sentences:
        for tok in s["tokens"]:
            t = tok.lower()
            counts[t] = counts.get(t, 0) + 1
    itos = [PAD, UNK] + sorted(w for w, c in counts.items() if c >= min_freq)
    return {w: i for i, w in enumerate(itos)}, itos


class NERDataset(Dataset):
    def __init__(self, sentences, word2id, label2id, max_len):
        self.data = []
        for s in sentences:
            ids = [word2id.get(t.lower(), word2id[UNK]) for t in s["tokens"]][:max_len]
            tags = [label2id[t] for t in s["tags"]][:max_len]
            if ids:
                self.data.append((ids, tags))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch):
    """Pad to the longest sequence in the batch and build a mask.

    The CRF requires the first timestep of every sequence to be unmasked,
    which is why sequences are right-padded and never empty.
    """
    maxlen = max(len(x[0]) for x in batch)
    ids, tags, mask = [], [], []
    for seq, tg in batch:
        pad = maxlen - len(seq)
        ids.append(seq + [0] * pad)
        # Padded positions need a valid tag index; the mask excludes them
        # from both the loss and decoding, so the value is irrelevant.
        tags.append(tg + [0] * pad)
        mask.append([1] * len(seq) + [0] * pad)
    return (torch.tensor(ids), torch.tensor(tags),
            torch.tensor(mask, dtype=torch.bool))


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, num_labels, embed_dim, hidden_dim,
                 num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim // 2, num_layers=num_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def _emissions(self, ids):
        x = self.dropout(self.embedding(ids))
        x, _ = self.lstm(x)
        return self.fc(self.dropout(x))

    def loss(self, ids, tags, mask):
        # torchcrf returns log-likelihood; negate for a minimisable loss.
        return -self.crf(self._emissions(ids), tags, mask=mask,
                          reduction="token_mean")

    def decode(self, ids, mask):
        return self.crf.decode(self._emissions(ids), mask=mask)


@torch.no_grad()
def evaluate(model, loader, id2label, device):
    model.eval()
    true_seqs, pred_seqs = [], []
    for ids, tags, mask in loader:
        ids, tags, mask = ids.to(device), tags.to(device), mask.to(device)
        paths = model.decode(ids, mask)
        for i, path in enumerate(paths):
            n = int(mask[i].sum())
            true_seqs.append([id2label[int(t)] for t in tags[i][:n]])
            pred_seqs.append([id2label[p] for p in path[:n]])
    return true_seqs, pred_seqs


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels = [l.strip() for l in
              open(os.path.join(args.data_dir, "labels.txt"), encoding="utf-8")
              if l.strip()]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}

    train = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    val = load_jsonl(os.path.join(args.data_dir, "val.jsonl"))
    test = load_jsonl(os.path.join(args.data_dir, "test.jsonl"))
    if args.max_train_samples:
        train = train[:args.max_train_samples]

    word2id, itos = build_vocab(train, args.min_freq)
    print(f"Device: {device}")
    print(f"Labels: {len(labels)} | Vocab: {len(itos)} "
          f"(min_freq={args.min_freq}, built from train only)")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}\n")

    mk = lambda data, shuffle: DataLoader(
        NERDataset(data, word2id, label2id, args.max_len),
        batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate)
    train_loader, val_loader, test_loader = mk(train, True), mk(val, False), mk(test, False)

    model = BiLSTMCRF(len(itos), len(labels), args.embed_dim, args.hidden_dim,
                      args.num_layers, args.dropout).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {nparams:,} "
          f"(BERT-base is ~110,000,000 -- roughly {110e6/nparams:.0f}x larger)\n")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    best_f1, history = 0.0, []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, nb = 0.0, 0
        for ids, tags, mask in train_loader:
            ids, tags, mask = ids.to(device), tags.to(device), mask.to(device)
            optim.zero_grad()
            loss = model.loss(ids, tags, mask)
            loss.backward()
            # Gradient clipping -- standard for LSTMs, which are prone to
            # exploding gradients on long sequences.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            total += loss.item()
            nb += 1

        t, p = evaluate(model, val_loader, id2label, device)
        f1 = f1_score(t, p)
        history.append({"epoch": epoch, "train_loss": round(total / nb, 4),
                        "val_f1": round(f1, 4),
                        "val_precision": round(precision_score(t, p), 4),
                        "val_recall": round(recall_score(t, p), 4)})
        print(f"epoch {epoch:>3} | train_loss {total/nb:.4f} | val_f1 {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "word2id": word2id,
                        "labels": labels, "args": vars(args)},
                       os.path.join(args.output_dir, "best.pt"))

    mins = (time.time() - start) / 60
    print(f"\nTrained in {mins:.1f} min. Best val_f1 {best_f1:.4f}")

    # Restore the best checkpoint before touching test -- mirrors BERT's
    # load_best_model_at_end so the comparison stays like-for-like.
    ckpt = torch.load(os.path.join(args.output_dir, "best.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    print("\n--- Final evaluation on the held-out TEST set (only touched once) ---")
    t, p = evaluate(model, test_loader, id2label, device)
    print(f"  precision {precision_score(t, p):.4f}")
    print(f"  recall    {recall_score(t, p):.4f}")
    print(f"  f1        {f1_score(t, p):.4f}\n")
    print(classification_report(t, p, digits=4))

    rows = classification_report(t, p, output_dict=True)
    out = os.path.join(args.data_dir, "bilstm_crf_per_entity.csv")
    with open(out, "w", encoding="utf-8") as f:
        f.write("entity,precision,recall,f1,support\n")
        for k, v in rows.items():
            if isinstance(v, dict):
                f.write(f"{k},{v['precision']:.4f},{v['recall']:.4f},"
                        f"{v['f1-score']:.4f},{int(v['support'])}\n")

    with open(os.path.join(args.data_dir, "bilstm_crf_curve.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    ing = rows.get("ING")
    if ing:
        print(f"ING F1 = {ing['f1-score']:.4f}  "
              f"(BERT scored 0.9669 on the same test set)")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
