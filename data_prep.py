"""
TruthfulQA data preparation:
  1. format_dataset  — download TruthfulQA, split into 2 folds, write pos/neg jsonl
  2. extract_activations — run model, save per-layer last-token hidden states

Usage:
  python data_prep.py --step format
  python data_prep.py --step extract -m Llama3.1-8B-Base -l 13 -b 10
  python data_prep.py --step all -m Llama3.1-8B-Base -l 13 -b 10
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import trange

DATA_DIR = Path(__file__).parent / "data" / "truthfulqa"
TEXTS_DIR = DATA_DIR / "texts"


def format_dataset():
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    splits = ds.train_test_split(test_size=0.5, seed=42)
    for split_idx, split_key in enumerate(["train", "test"]):
        df = splits[split_key].to_pandas()
        pos_pairs, neg_pairs = [], []
        for idx, row in df.iterrows():
            for ans in row.correct_answers:
                pos_pairs.append({"idx": idx, "question": row.question.strip(), "answer": ans})
            for ans in row.incorrect_answers:
                neg_pairs.append({"idx": idx, "question": row.question.strip(), "answer": ans})
        pd.DataFrame(pos_pairs).to_json(TEXTS_DIR / f"pos_{split_idx}.jsonl", orient="records", lines=True)
        pd.DataFrame(neg_pairs).to_json(TEXTS_DIR / f"neg_{split_idx}.jsonl", orient="records", lines=True)
        print(f"Split {split_idx}: {len(pos_pairs)} pos, {len(neg_pairs)} neg pairs")


def load_questions(split_idx: int, data_dir: Path | None = None) -> list[str]:
    texts_dir = (data_dir / "texts") if data_dir else TEXTS_DIR
    df = pd.read_json(texts_dir / f"pos_{split_idx}.jsonl", lines=True, orient="records")
    return df["question"].unique().tolist()


def load_activations(model_name: str, layer_idx: int, split_idx: int,
                     data_dir: Path | None = None):
    act_dir = ((data_dir or DATA_DIR) / "activations" / model_name)
    pos = torch.load(act_dir / f"pos_{split_idx}_activations_layer{layer_idx}.pt", weights_only=True, map_location="cpu")
    neg = torch.load(act_dir / f"neg_{split_idx}_activations_layer{layer_idx}.pt", weights_only=True, map_location="cpu")
    return pos, neg


def extract_activations(model_name: str, layer_idx: int, batch_size: int = 10):
    from lm import HuggingFaceLM

    act_dir = DATA_DIR / "activations" / model_name
    act_dir.mkdir(parents=True, exist_ok=True)

    model = HuggingFaceLM(model_name, device="auto", dtype=torch.float32)
    if layer_idx < 0:
        layer_idx = len(model.model.model.layers) // 2 - 1

    use_base = "Base" in model_name and "Qwen" not in model_name

    for fpath in sorted(TEXTS_DIR.glob("[pn]*_[01].jsonl")):
        out_path = act_dir / f"{fpath.stem}_activations_layer{layer_idx}.pt"
        if out_path.exists():
            print(f"Activations exist: {fpath.stem} layer {layer_idx} — skipping")
            continue

        print(f"Extracting: {fpath.stem} layer {layer_idx}")
        df = pd.read_json(fpath, lines=True, orient="records")
        questions, answers = df["question"].tolist(), df["answer"].tolist()
        num_batches = (len(df) + batch_size - 1) // batch_size
        all_acts = []
        for i in trange(num_batches, desc=fpath.stem):
            s, e = i * batch_size, min((i + 1) * batch_size, len(df))
            bq, ba = questions[s:e], answers[s:e]
            if use_base:
                prompts = [f"Q: {q}\nA: {a}" for q, a in zip(bq, ba)]
                acts = model.extract_prompt_eos_activations(prompts, layer_idx).cpu()
            else:
                msgs = [[{"role": "user", "content": q}, {"role": "assistant", "content": a}] for q, a in zip(bq, ba)]
                acts = model.extract_message_eos_activations(msgs, layer_idx).cpu()
            all_acts.append(acts)

        all_acts = torch.cat(all_acts, dim=0)
        torch.save(df.idx.values, act_dir / f"{fpath.stem}_question_idx.pt")
        torch.save(all_acts, out_path)
        print(f"  Saved {all_acts.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruthfulQA data preparation")
    parser.add_argument("--step", choices=["format", "extract", "all"], default="all")
    parser.add_argument("-m", "--model", type=str, default="Llama3.1-8B-Base")
    parser.add_argument("-l", "--layer_idx", type=int, default=13)
    parser.add_argument("-b", "--batch_size", type=int, default=10)
    args = parser.parse_args()

    if args.step in ("format", "all"):
        format_dataset()
    if args.step in ("extract", "all"):
        extract_activations(args.model, args.layer_idx, args.batch_size)
