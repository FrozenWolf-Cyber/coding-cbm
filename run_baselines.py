"""
Run probing-based baseline steering methods on TruthfulQA.

Replicates the exact same pipeline as ODESteer's truthfulqa_generate.py:
  - seed_everything(seed) before any work
  - 2-fold cross-validation (train on split 0, test on split 1, then swap)
  - Fit steering model on train-split activations
  - Generate answers to test-split questions with steering applied
  - Save outputs in standard jsonl format for truthfulqa_evaluate.py

Usage:
  # Run a single method:
  python run_baselines.py -m Llama3.1-8B-Base -l 13 --steer CAA --T 1.0

  # ODESteer (matches hydra defaults):
  python run_baselines.py -m Llama3.1-8B-Base -l 13 --steer ODESteer --T 5.0

  # Run all baselines:
  python run_baselines.py -m Llama3.1-8B-Base -l 13 --steer all

  # Run + evaluate:
  python run_baselines.py -m Llama3.1-8B-Base -l 13 --steer all --evaluate

  # Multiple seeds:
  python run_baselines.py -m Llama3.1-8B-Base -l 13 --steer ODESteer --T 5.0 --seeds 42 123 456

  # Use activations from the original ODESteer repo:
  python run_baselines.py -m Llama3.1-8B-Base -l 13 --steer ODESteer --T 5.0 \\
      --data-dir /path/to/odesteer/data/truthfulqa
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from lightning import seed_everything
from transformers import GenerationConfig

from config import (
    TRUTHFULQA_SYSTEM_PROMPT, DEFAULT_SEED, STEER_METHODS,
    STEER_DEFAULT_KWARGS, build_steer_name,
)
from data_prep import load_questions, load_activations
from lm import HuggingFaceLM, batch_chat

RESULTS_DIR = Path(__file__).parent / "results" / "truthfulqa"


def run_single_method(
    model_name: str,
    layer_idx: int,
    steer_name: str,
    T: float,
    batch_size: int,
    seed: int,
    steer_model_kwargs: dict | None = None,
    pace_cfg: dict | None = None,
    data_dir: Path | None = None,
):
    # Re-seed at the start of each method so that running multiple methods
    # sequentially (--steer all) gives the same result as running each one
    # independently — matching the original Hydra-based script behaviour.
    seed_everything(seed)

    if steer_model_kwargs is None:
        steer_model_kwargs = STEER_DEFAULT_KWARGS.get(steer_name, {})

    output_dir = RESULTS_DIR / "raw_outputs" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    steer_label = build_steer_name(steer_name, steer_model_kwargs, T)
    if steer_name == "PaCE" and pace_cfg is not None:
        reuse_suffix = "reuseCoeff" if bool(pace_cfg.get("reuse_coeff_across_tokens", False)) else "noReuseCoeff"
        steer_label = f"{steer_label}-{reuse_suffix}"
    filename = f"{model_name}-l{layer_idx}-{steer_label}-TruthfulQA-seed{seed}.jsonl"

    if (output_dir / filename).exists():
        print(f"✓ Output file {filename} already exists. Skipping TruthfulQA generation.")
        print("-" * 120)
        return output_dir / filename

    try:
        all_prompts, all_outputs = [], []
        print(f"→ Running 2-fold cross-validation for {model_name}-{steer_label} on layer {layer_idx}")

        for test_split in [0, 1]:
            train_split = 1 - test_split

            print(f"\n→ Fold {test_split + 1}: Training on split {train_split}, testing on split {test_split}")
            print("→ Loading LLM & Fitting Steer Model ...")

            gen_config = GenerationConfig(
                max_new_tokens=50, do_sample=True, temperature=0.7,
                top_p=0.9, repetition_penalty=1.1, seed=seed,
            )

            model = HuggingFaceLM(
                model_name, steer_name,
                default_generation_config=gen_config,
                steer_model_kwargs=steer_model_kwargs,
                steer_layer_idx=layer_idx,
                device="auto", dtype=torch.float32,
                pace_cfg=pace_cfg if steer_name == "PaCE" else None,
            )

            # Match truthfulqa_generate.py: always load train activations and call
            # fit_steer_model (no-op for NoSteer / PaCE) so RNG and CUDA state match.
            pos_train, neg_train = load_activations(
                model_name, layer_idx, train_split, data_dir=data_dir,
            )
            model.fit_steer_model(pos_train, neg_train)

            print(f"→ Loading test questions from split {test_split} ...")
            questions = load_questions(test_split, data_dir=data_dir)
            messages = [[
                {"role": "system", "content": TRUTHFULQA_SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ] for q in questions]

            print(f"→ Generating {len(questions)} responses with T={T} ...")
            outputs = batch_chat(model, messages, T=T, batch_size=batch_size)
            print(f"→ Generated {len(outputs)} outputs")

            all_prompts.extend(questions)
            all_outputs.extend(outputs)

            del model
            gc.collect()
            torch.cuda.empty_cache()

        print(f"\n→ Saving all outputs to {filename} ...")
        with open(output_dir / filename, "w") as f:
            for prompt, output in zip(all_prompts, all_outputs):
                f.write(json.dumps({
                    "prompt": prompt,
                    "output": output,
                    "generator": f"{model_name}-{steer_label}",
                    "dataset": "TruthfulQA",
                    "T": T,
                }) + "\n")

        print(f"✓ Completed {model_name}-{steer_label} on TruthfulQA")
        print(f"  Total responses generated: {len(all_outputs)}")
        print(f"  Configuration: T={T}")
        print("-" * 120)

    except Exception as e:
        print(f"→ Error: {e}")
        import traceback
        traceback.print_exc()

    return output_dir / filename


def build_pace_cfg(layer_idx: int, args) -> dict:
    max_concepts = -1 if args.pace_max_concepts == -1 else args.pace_max_concepts
    return {
        "index_path": args.pace_index_path,
        "representation_path": args.pace_representation_path,
        "max_concepts": max_concepts,
        "partition_mode": "heuristic",
        "partition_file": None,
        "vector_cache_path": f"./pace_cache/layer{layer_idx}",
        "encode_batch_size": 8,
        "alpha": args.pace_alpha,
        "layer_idx": layer_idx,
        "pace_gpu": args.pace_gpu,
        "pace_token_timing": args.pace_token_timing,
        "reuse_coeff_across_tokens": args.pace_reuse_coeff_across_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Run baseline steering on TruthfulQA")
    parser.add_argument("-m", "--model", type=str, default="Llama3.1-8B-Base")
    parser.add_argument("-l", "--layer_idx", type=int, default=13)
    parser.add_argument("-b", "--batch_size", type=int, default=10)
    parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Run with multiple seeds (overrides --seed)")
    parser.add_argument("--steer", type=str, default="all",
                        help="Steering method name or 'all'. Choices: " + ", ".join(STEER_METHODS))
    parser.add_argument("--T", type=float, default=1.0, help="Steering strength T")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after generation")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to truthfulqa data dir containing texts/ and activations/ "
                             "(default: ./data/truthfulqa). Use this to point to the original "
                             "ODESteer repo's data dir for exact reproducibility.")

    parser.add_argument("--pace_index_path", type=str, default="./pace_data/concept_index.txt")
    parser.add_argument("--pace_representation_path", type=str, default="./pace_data/concept/")
    parser.add_argument("--pace_max_concepts", type=int, default=5000,
                        help="Max concepts to use for PaCE. Set -1 to use all concepts in index.")
    parser.add_argument("--pace_alpha", type=float, default=1.0)
    parser.add_argument("--pace_gpu", action="store_true",
                        help="Run PaCE sparse decomposition on GPU (faster, may slightly change outputs).")
    parser.add_argument("--pace_token_timing", action="store_true",
                        help="Print per (batch, seq) position PaCE timings during generation (verbose).")
    parser.add_argument(
        "--pace_reuse_coeff_across_tokens",
        action="store_true",
        help="Enable PaCE coefficient reuse across tokens (faster; may affect steering behavior).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    seeds = args.seeds if args.seeds else [args.seed]
    methods = STEER_METHODS if args.steer == "all" else [args.steer]

    for seed in seeds:
        for method in methods:
            pace_cfg = build_pace_cfg(args.layer_idx, args) if method == "PaCE" else None
            run_single_method(
                model_name=args.model,
                layer_idx=args.layer_idx,
                steer_name=method,
                T=args.T,
                batch_size=args.batch_size,
                seed=seed,
                pace_cfg=pace_cfg,
                data_dir=data_dir,
            )

        if args.evaluate:
            from truthfulqa_evaluate import evaluate_outputs
            raw_dir = RESULTS_DIR / "raw_outputs"
            eval_path = (
                RESULTS_DIR / "eval_results" / "stat_results"
                / f"{args.model}-l{args.layer_idx}-TruthfulQA-seed{seed}.csv"
            )
            evaluate_outputs(raw_dir, eval_path, args.model, args.layer_idx, seed, args.batch_size, display=True)


if __name__ == "__main__":
    main()
