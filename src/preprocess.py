import json
import random

def parse_conll(path):
    with open (path, "r", encoding="utf-8") as file:
        sentences = []
        tokens = []
        tags = []

        for line in file:
            line = line.rstrip("\r\n")

            if line =="":
                sentences.append({"tokens":tokens, "tags":tags})
                tokens = []
                tags = []

            else:
                token, tag = line.split("\t")
                token = token.strip()
                tag = tag.strip()
                tokens.append(token)
                tags.append(tag)

        if tokens:
            sentences.append({"tokens" : tokens, "tags" : tags})

        return sentences

def collect_label_vocab(sentences :list[dict]) -> list[str]:
    labels = set()
    for sent in sentences:
        labels.update(sent["tags"])
    ordered = ["O"] + sorted(labels - {"O"})
    return ordered

def split_and_save(sentences, out_dir, seed = 42, train_frac = 0.8, val_frac = 0.1):
    random.seed(seed)
    shuffled_sentences = sentences[:]
    random.shuffle(shuffled_sentences)

    n = len(shuffled_sentences)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    split = {
        "train" : shuffled_sentences[:n_train],
        "val" : shuffled_sentences[n_train:n_train+n_val],
        "test" : shuffled_sentences[n_train+n_val :]
    }

    counts = {}
    for name, data in split.items():
        path = f"{out_dir}/{name}.jsonl"

        with open(path, "w", encoding="utf-8") as file:
            for sent in data:
                file.write(json.dumps(sent) + "\n")

        counts[name] = len(data)

    return counts

if __name__ == "__main__":
    sentences = parse_conll("data/finer.conll")
    print(f"Parsed {len(sentences)} sentences from finer.conll")

    labels = collect_label_vocab(sentences)
    print(f"Found {len(labels)} labels: {labels}")

    with open("data/labels.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

    counts = split_and_save(sentences, out_dir="data")
    print(f"Split sizes: {counts}")

    with open("data/train.jsonl", encoding="utf-8") as f:
        example = json.loads(f.readline())
    print("\nExample from train.jsonl:")
    print(example)

