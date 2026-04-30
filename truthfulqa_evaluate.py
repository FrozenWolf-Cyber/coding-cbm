"""
Common evaluation script for TruthfulQA experiments.

Works for ALL methods (baselines + CBM) — any script that writes output in the
standard jsonl format:
    {"prompt": ..., "output": ..., "generator": ..., "dataset": "TruthfulQA", ...}

Replicates the exact same metrics as ODESteer's truthfulqa_eval.py:
  - Truthfulness  (allenai truth judge)
  - Informativeness (allenai info judge)
  - True * Info
  - Perplexity (GPT-2-XL)
  - Dist-1, Dist-2, Dist-3

Usage:
  python truthfulqa_evaluate.py -m Llama3.1-8B-Base -l 13 -d
  python truthfulqa_evaluate.py --results_dir ./results/truthfulqa/raw_outputs -d
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import STEER_METHODS, EVAL_COLUMNS
from metrics import TruthfulQAJudge, QualityEvaluator

RESULTS_DIR = Path(__file__).parent / "results" / "truthfulqa"


def _steer_sort_key(value: str):
    order = {v: i for i, v in enumerate(STEER_METHODS)}
    parts = value.split("-", 1)
    return (order.get(parts[0], len(STEER_METHODS)), parts[1] if len(parts) > 1 else "")


def parse_file_info(file_path: Path) -> tuple[str, str]:
    parts = file_path.stem.split("-")[:-2]
    model = "-".join(parts[:4])
    steer = "-".join(parts[4:])
    return model, steer


def evaluate_outputs(
    raw_dir: Path,
    eval_csv_path: Path,
    model: str,
    layer_idx: int,
    seed: int = 42,
    batch_size: int = 10,
    display: bool = False,
):
    """Evaluate all jsonl result files matching the model/layer/seed pattern."""
    eval_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if eval_csv_path.exists():
        eval_df = pd.read_csv(eval_csv_path)
    else:
        eval_df = pd.DataFrame(columns=EVAL_COLUMNS)

    pattern = f"**/{model}-l{layer_idx}-*-TruthfulQA-seed{seed}.jsonl"
    files = sorted(raw_dir.glob(pattern))
    if not files:
        print(f"No files matching {pattern} in {raw_dir}")
        return eval_df

    truth_judge = TruthfulQAJudge(display=display)
    quality_eval = QualityEvaluator(device="auto")

    for fpath in files:
        _, steer_method = parse_file_info(fpath)
        if steer_method in eval_df["Steering Method"].values:
            print(f"Already evaluated: {steer_method} — skipping")
            continue

        print(f"\nEvaluating {model} / {steer_method} ...")
        df = pd.read_json(fpath, orient="records", lines=True)
        prompts, outputs = df.prompt.tolist(), df.output.tolist()

        true_x_info, true_scores, info_scores = truth_judge.batch_evaluate(prompts, outputs, batch_size)
        ppls, d1, d2, d3 = quality_eval.batch_evaluate(outputs, batch_size)

        row = [
            model, steer_method,
            np.nanmean(true_x_info), np.nanmean(true_scores), np.nanmean(info_scores),
            np.nanmean(ppls), np.nanmean(d1), np.nanmean(d2), np.nanmean(d3),
        ]
        eval_df.loc[len(eval_df)] = row

        # Per-sample detailed results
        detail_dir = eval_csv_path.parent.parent / "detailed_eval_results" / model
        detail_dir.mkdir(parents=True, exist_ok=True)
        detail_df = df.copy()
        detail_df["true"] = true_scores
        detail_df["info"] = info_scores
        detail_df["true_info"] = true_x_info
        detail_df["ppl"] = ppls
        detail_df["dist_1"] = d1
        detail_df["dist_2"] = d2
        detail_df["dist_3"] = d3
        detail_df.to_json(
            detail_dir / f"{model}-{steer_method}-TruthfulQA-seed{seed}.jsonl",
            orient="records", lines=True,
        )

        print(f"  True*Info={np.nanmean(true_x_info):.3f}  "
              f"Truth={np.nanmean(true_scores):.3f}  "
              f"Info={np.nanmean(info_scores):.3f}  "
              f"PPL={np.nanmean(ppls):.2f}  "
              f"D1={np.nanmean(d1):.3f}")

        eval_df = eval_df.sort_values("Steering Method", key=lambda x: x.map(_steer_sort_key))
        eval_df.to_csv(eval_csv_path, index=False)

    eval_df = eval_df.sort_values("Steering Method", key=lambda x: x.map(_steer_sort_key))
    eval_df.to_csv(eval_csv_path, index=False)
    print(f"\nResults saved to {eval_csv_path}")
    print(eval_df.to_string(index=False))
    return eval_df


def evaluate_from_jsonl_list(
    jsonl_paths: list[Path],
    eval_csv_path: Path,
    batch_size: int = 10,
    display: bool = False,
    seed: int = 42,
):
    """Evaluate a list of specific jsonl files (for CBM / GRPO post-training eval)."""
    eval_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if eval_csv_path.exists():
        eval_df = pd.read_csv(eval_csv_path)
    else:
        eval_df = pd.DataFrame(columns=EVAL_COLUMNS)

    truth_judge = TruthfulQAJudge(display=display)
    quality_eval = QualityEvaluator(device="auto")

    for fpath in jsonl_paths:
        fpath = Path(fpath)
        if not fpath.exists():
            print(f"File not found: {fpath}")
            continue

        df = pd.read_json(fpath, orient="records", lines=True)
        generator = df["generator"].iloc[0] if "generator" in df.columns else fpath.stem
        model = df.get("model", [generator.split("-")[0]])[0] if "model" in df.columns else "unknown"

        if generator in eval_df["Steering Method"].values:
            print(f"Already evaluated: {generator} — skipping")
            continue

        print(f"\nEvaluating {generator} ...")
        prompts, outputs = df.prompt.tolist(), df.output.tolist()

        true_x_info, true_scores, info_scores = truth_judge.batch_evaluate(prompts, outputs, batch_size)
        ppls, d1, d2, d3 = quality_eval.batch_evaluate(outputs, batch_size)

        eval_df.loc[len(eval_df)] = [
            model, generator,
            np.nanmean(true_x_info), np.nanmean(true_scores), np.nanmean(info_scores),
            np.nanmean(ppls), np.nanmean(d1), np.nanmean(d2), np.nanmean(d3),
        ]
        eval_df.to_csv(eval_csv_path, index=False)

        print(f"  True*Info={np.nanmean(true_x_info):.3f}  "
              f"Truth={np.nanmean(true_scores):.3f}  "
              f"Info={np.nanmean(info_scores):.3f}  "
              f"PPL={np.nanmean(ppls):.2f}")

    eval_df.to_csv(eval_csv_path, index=False)
    print(f"\nResults saved to {eval_csv_path}")
    return eval_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TruthfulQA results")
    parser.add_argument("-m", "--model", type=str, default="Llama3.1-8B-Base")
    parser.add_argument("-l", "--layer_idx", type=int, default=13)
    parser.add_argument("-b", "--batch_size", type=int, default=10)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-d", "--display", action="store_true")
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    raw_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR / "raw_outputs"
    eval_path = (
        RESULTS_DIR / "eval_results" / "stat_results"
        / f"{args.model}-l{args.layer_idx}-TruthfulQA-seed{args.seed}.csv"
    )

    evaluate_outputs(raw_dir, eval_path, args.model, args.layer_idx, args.seed, args.batch_size, args.display)
