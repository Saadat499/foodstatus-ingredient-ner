"""
Stage 2: Fine-tune BERT-base for ingredient NER on the FINER dataset.

Sized specifically for a 4GB VRAM GPU (RTX 3050): fp16 mixed precision,
gradient checkpointing, a small per-device batch size compensated by
gradient accumulation, and dynamic (not fixed-length) padding, since
FINER sentences are short (recipe ingredient lines, ~8 tokens average) --
padding everything to one fixed max length would waste memory for no
reason.

Model choice: bert-base-uncased, not multilingual -- FINER is scraped
from Allrecipes.com, English-only. This also directly traces to Devlin
et al. (2019), already cited in the literature review.

Usage:
    python src/train_bert.py

    Smoke-test first on a tiny slice, before committing to a full run:
    python src/train_bert.py --max_train_samples 200 --max_steps 5
"""

import argparse
import json
import os

import numpy as np
from datasets import load_dataset
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="models/bert-ner")
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    # For quick sanity runs before committing to a full multi-hour job.
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=-1)
    # Full training runs ~20-31 hours on this hardware -- almost certainly
    # spans multiple sessions, so resuming from the last checkpoint matters.
    p.add_argument("--resume", action="store_true",
                    help="Resume from the latest checkpoint in output_dir, if one exists.")
    p.add_argument("--no_grad_checkpointing", action="store_true",
                    help="Disable gradient checkpointing -- turn this off (i.e. pass this "
                         "flag) on a GPU with plenty of VRAM (e.g. Colab's T4) to train faster. "
                         "Keep gradient checkpointing ON (default) on the 4GB RTX 3050.")
    # parse_known_args (not parse_args) ignores extra arguments we didn't
    # define -- important in notebook environments like Colab, where the
    # Jupyter kernel injects its own internal arguments (e.g. "-f kernel-
    # ....json") that would otherwise cause a crash if run via %run or by
    # calling main() directly instead of "!python train_bert.py ...".
    args, _unknown = p.parse_known_args()
    return args


def load_label_list(data_dir):
    with open(os.path.join(data_dir, "labels.txt"), encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def tokenize_and_align_labels(examples, tokenizer, label2id, max_length):
    """The one genuinely tricky part of NER fine-tuning: BERT's tokenizer
    splits words into subword pieces (e.g. "xanthan" -> "xan", "##than"),
    but our labels are per-WORD, not per-subword-piece. We have to expand
    each word's label across all of that word's subword pieces -- except
    we only want the loss computed on the FIRST piece of each word, so
    every other piece (continuation pieces, plus special tokens like
    [CLS]/[SEP]) gets label -100, which PyTorch's loss functions are
    built to automatically ignore.
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
    )

    all_labels = []
    for i, tags in enumerate(examples["tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                # Special token ([CLS], [SEP], padding) -- not a real word.
                label_ids.append(-100)
            elif word_id != previous_word_id:
                # First subword piece of a new word -- give it the real label.
                label_ids.append(label2id[tags[word_id]])
            else:
                # A continuation piece of the same word -- ignore in loss.
                label_ids.append(-100)
            previous_word_id = word_id
        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


def build_compute_metrics(id2label):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        # Strip out the -100 positions (special tokens / continuation
        # pieces) before scoring -- seqeval expects clean tag sequences.
        true_predictions = [
            [id2label[p] for p, l in zip(pred_row, label_row) if l != -100]
            for pred_row, label_row in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for p, l in zip(pred_row, label_row) if l != -100]
            for pred_row, label_row in zip(predictions, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    return compute_metrics


def main():
    args = parse_args()

    label_list = load_label_list(args.data_dir)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    print(f"Loaded {len(label_list)} labels: {label_list}")

    data_files = {
        "train": os.path.join(args.data_dir, "train.jsonl"),
        "validation": os.path.join(args.data_dir, "val.jsonl"),
        "test": os.path.join(args.data_dir, "test.jsonl"),
    }
    raw_datasets = load_dataset("json", data_files=data_files)

    if args.max_train_samples:
        raw_datasets["train"] = raw_datasets["train"].select(
            range(min(args.max_train_samples, len(raw_datasets["train"])))
        )
    print(f"Train: {len(raw_datasets['train'])} | "
          f"Val: {len(raw_datasets['validation'])} | "
          f"Test: {len(raw_datasets['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenized_datasets = raw_datasets.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer, label2id, args.max_length),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    # Dynamic padding per-batch (not a fixed max length) -- saves real
    # memory given most FINER sentences are short.
    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        fp16=True,                     # mixed precision -- required to fit 4GB VRAM
        gradient_checkpointing=not args.no_grad_checkpointing,
        # Full training takes ~20-31 hours on this hardware, so checkpoint
        # every ~1 hour of progress (not once per ~10-hour epoch) -- an
        # interruption then costs you at most ~1 hour, not most of a day.
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,            # see disk note below
        # DISK: each checkpoint is ~1.3GB, not the model's 440MB -- the
        # AdamW optimizer state (two fp32 moments per parameter) is roughly
        # twice the model again, and it must be saved to make --resume work.
        # 3 checkpoints ~= 4GB, and the best checkpoint is protected from
        # rotation on top of that, so budget ~5.5GB free before starting.
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(id2label),
    )

    last_checkpoint = None
    if args.resume:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint:
            print(f"Resuming from checkpoint: {last_checkpoint}")
        else:
            print("--resume was set, but no checkpoint was found in "
                  f"{args.output_dir} -- starting fresh instead.")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    print("\n--- Final evaluation on the held-out TEST set (only touched once) ---")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print(test_results)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    main()
