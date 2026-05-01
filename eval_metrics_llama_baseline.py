from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional

import torch
import wandb
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_metrics import (
    _extract_code_from_output,
    _format_code_generation_prompt,
    _import_lcb,
    print_extracted_code_samples_preview,
    set_seed,
)
from shared_code_prompt import LCB_LLAMA3_INSTRUCT_MODEL_ID


@torch.no_grad()
def _generate_solutions_llama(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    n_samples: int = 1,
    max_new_tokens: int = 2000,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.05,
) -> List[str]:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = enc["input_ids"]
    prompt_len = prompt_ids.shape[1]
    # Use max_length (prompt + budget) only — do not pass max_new_tokens alongside the
    # model's default generation_config.max_length or Transformers warns and ignores max_length.
    total_max_length = prompt_len + max_new_tokens

    do_sample = temperature > 0
    generation_kwargs = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "max_length": total_max_length,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "top_k": top_k if do_sample else None,
        "repetition_penalty": repetition_penalty,
        "num_return_sequences": n_samples,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
    gen_ids = model.generate(**generation_kwargs)

    outputs = []
    for i in range(n_samples):
        completion = gen_ids[i, prompt_len:]
        outputs.append(tokenizer.decode(completion, skip_special_tokens=True).strip())
    return outputs


@torch.no_grad()
def _generate_solutions_llama_batched(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    n_samples: int = 1,
    max_new_tokens: int = 2000,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.05,
) -> List[List[str]]:
    if not prompts:
        return []

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    finally:
        tokenizer.padding_side = original_padding_side

    prompt_width = enc["input_ids"].shape[1]
    total_max_length = prompt_width + max_new_tokens
    do_sample = temperature > 0
    generation_kwargs = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "max_length": total_max_length,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "top_k": top_k if do_sample else None,
        "repetition_penalty": repetition_penalty,
        "num_return_sequences": n_samples,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
    gen_ids = model.generate(**generation_kwargs)

    batch_outputs: List[List[str]] = []
    for prompt_idx in range(len(prompts)):
        row = []
        base = prompt_idx * n_samples
        for sample_idx in range(n_samples):
            completion = gen_ids[base + sample_idx, prompt_width:]
            row.append(tokenizer.decode(completion, skip_special_tokens=True).strip())
        batch_outputs.append(row)
    return batch_outputs


def run_codecontests_evaluation_for_llama_instruct(
    model=None,
    tokenizer=None,
    seed: int = 42,
    model_label: str = "Llama3-8B-Instruct-baseline",
    layer_idx: int = -1,
    run_id=None,
    results_root=None,
    display: bool = True,
    livecodebench_release: str = "release_v6",
    lcb_n_samples: int = 10,
    lcb_temperature: float = 0.2,
    lcb_top_p: float = 0.95,
    lcb_top_k: int = 50,
    lcb_max_new_tokens: int = 2000,
    lcb_repetition_penalty: float = 1.05,
    lcb_prompt_batch_size: int = 4,
    lcb_num_process_evaluate: int = 4,
    lcb_timeout: int = 6,
    print_extracted_code_preview: bool = False,
    extracted_preview_chars: int = 420,
    unload_model_before_grading: bool = True,
) -> dict:
    """Run LCB generation and pass@k grading.

    When ``model`` and ``tokenizer`` are omitted, weights are loaded here and dropped
    before grading so ``ProcessPoolExecutor`` workers are not forked from a process
    that still maps a full Llama checkpoint (avoids multiplicative RAM on Linux).

    Set ``unload_model_before_grading=False`` only if you pass a shared ``model`` you
    still need after this call (not recommended on memory-constrained hosts).
    """
    owns_weights = model is None or tokenizer is None
    if owns_weights:
        if model is not None or tokenizer is not None:
            raise ValueError("Pass both model and tokenizer, or neither.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(LCB_LLAMA3_INSTRUCT_MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            LCB_LLAMA3_INSTRUCT_MODEL_ID,
            torch_dtype=dtype,
        ).to(device)
    else:
        device = next(model.parameters()).device

    model.eval()
    set_seed(seed)

    lcb_repo = Path(__file__).parent / "LiveCodeBench"
    if run_id is None:
        run_id = wandb.run.id if wandb.run is not None else "norun"
    base_root = Path(results_root) if results_root else Path(__file__).parent / "results"

    all_results: dict = {}
    steer_mode = "none"

    print(f"\n{'='*60}")
    print(f" LiveCodeBench  (release={livecodebench_release}, n={lcb_n_samples}, T={lcb_temperature})")
    print(f"{'='*60}")

    load_code_generation_dataset, codegen_metrics, extract_instance_results = _import_lcb()
    benchmark = load_code_generation_dataset(livecodebench_release)
    print(f"  Loaded {len(benchmark)} LCB problems")

    mode_repr = f"{model_label}-{steer_mode}"
    lcb_out_dir = lcb_repo / "output" / mode_repr / str(run_id)
    lcb_out_dir.mkdir(parents=True, exist_ok=True)
    lcb_out_path = lcb_out_dir / f"codegeneration_{lcb_n_samples}_{lcb_temperature}.json"
    lcb_eval_path = lcb_out_dir / f"codegeneration_{lcb_n_samples}_{lcb_temperature}_eval.json"
    lcb_eval_all_path = lcb_out_dir / f"codegeneration_{lcb_n_samples}_{lcb_temperature}_eval_all.json"

    print(f"\n[{steer_mode}] Generating {lcb_n_samples} solutions x {len(benchmark)} LCB problems ...")
    all_outputs: List[List[str]] = []
    all_extracted: List[List[str]] = []
    benchmark_sorted = sorted(benchmark, key=lambda x: x.question_id)
    pending_prompts: List[str] = []
    pending_headings: List[str] = []

    def _flush_batch():
        if not pending_prompts:
            return
        generated = _generate_solutions_llama_batched(
            model,
            tokenizer,
            pending_prompts,
            device,
            n_samples=lcb_n_samples,
            max_new_tokens=lcb_max_new_tokens,
            temperature=lcb_temperature,
            top_p=lcb_top_p,
            top_k=lcb_top_k,
            repetition_penalty=lcb_repetition_penalty,
        )
        for heading, raw_samples in zip(pending_headings, generated):
            extracted = [_extract_code_from_output(s) for s in raw_samples]
            if print_extracted_code_preview:
                print_extracted_code_samples_preview(
                    heading,
                    extracted,
                    preview_chars=extracted_preview_chars,
                )
            all_outputs.append(raw_samples)
            all_extracted.append(extracted)
        pending_prompts.clear()
        pending_headings.clear()

    for problem in tqdm(benchmark_sorted, desc=f"lcb/{steer_mode}", disable=not display):
        prompt = _format_code_generation_prompt(
            tokenizer,
            problem.question_content,
            starter_code=getattr(problem, "starter_code", "") or "",
            language="python",
        )
        pending_prompts.append(prompt)
        desc_flat = problem.question_content.replace("\n", " ").strip()
        desc_short = desc_flat[:260] + ("..." if len(desc_flat) > 260 else "")
        pending_headings.append(
            f"[{steer_mode}] LCB  question_id={problem.question_id}  "
            f"description (start): {desc_short!r}"
        )
        if len(pending_prompts) >= max(1, int(lcb_prompt_batch_size)):
            _flush_batch()
    _flush_batch()

    save_results = [
        problem.insert_output(outputs, codes)
        for problem, outputs, codes in zip(benchmark_sorted, all_outputs, all_extracted)
    ]
    with open(lcb_out_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=4)
    print(f"  Saved LCB outputs -> {lcb_out_path}")

    if unload_model_before_grading and owns_weights:
        del model
        del tokenizer
        model = None  # type: ignore[assignment]
        tokenizer = None  # type: ignore[assignment]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    print(f"[{steer_mode}] Running pass@k grading ({lcb_num_process_evaluate} workers) ...")
    eval_samples = [p.get_evaluation_sample() for p in benchmark_sorted]
    metrics_tuple = codegen_metrics(
        eval_samples,
        all_extracted,
        num_process_evaluate=lcb_num_process_evaluate,
        timeout=lcb_timeout,
    )
    metrics, results_dict, metadatas = metrics_tuple
    graded = extract_instance_results(results_dict)
    save_eval_results = [
        p.insert_output_evaluation(o, c, g, metadata=m)
        for p, o, c, g, m in zip(
            benchmark_sorted, all_outputs, all_extracted, graded, metadatas
        )
    ]
    with open(lcb_eval_all_path, "w", encoding="utf-8") as f:
        json.dump(save_eval_results, f, indent=4)
    with open(lcb_eval_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"  Saved LCB eval -> {lcb_eval_all_path}")

    pass1 = metrics.get("pass@1", float("nan")) if isinstance(metrics, dict) else float("nan")
    pass5 = metrics.get("pass@5", float("nan")) if isinstance(metrics, dict) else float("nan")
    print(f"\n  [none] LCB pass@1 = {pass1:.4f}  |  pass@5 = {pass5:.4f}")
    print(f"  model_repr : {mode_repr}")
    print(f"  eval_all   : {lcb_eval_all_path}")
    log_payload = {
        "lcb/none/pass@1": pass1,
        "lcb/none/pass@5": pass5,
        "lcb/none/n_samples": lcb_n_samples,
        "lcb/none/temperature": lcb_temperature,
        "lcb/none/steer_value": 0.0,
        "lcb/none/steer_topk": 0,
        "lcb/none/release": livecodebench_release,
        "lcb/none/output_path": str(lcb_out_path),
    }
    if wandb.run is not None:
        wandb.log(log_payload)
    all_results["lcb/none"] = {
        "pass@1": pass1,
        "pass@5": pass5,
        "output_path": str(lcb_out_path),
        "eval_all_path": str(lcb_eval_all_path),
    }
    return all_results


def run_llama8b_instruct_baseline(
    seed: int = 42,
    model_label: str = "Llama3-8B-Instruct-baseline",
    run_id: Optional[str] = None,
    results_root=None,
    display: bool = True,
    livecodebench_release: str = "release_v6",
    lcb_n_samples: int = 10,
    lcb_temperature: float = 0.2,
    lcb_top_p: float = 0.95,
    lcb_top_k: int = 50,
    lcb_max_new_tokens: int = 2000,
    lcb_prompt_batch_size: int = 4,
    lcb_num_process_evaluate: int = 4,
    lcb_timeout: int = 6,
    print_extracted_code_preview: bool = False,
    extracted_preview_chars: int = 420,
    unload_model_before_grading: bool = True,
) -> dict:
    return run_codecontests_evaluation_for_llama_instruct(
        model=None,
        tokenizer=None,
        seed=seed,
        model_label=model_label,
        run_id=run_id,
        results_root=results_root,
        display=display,
        livecodebench_release=livecodebench_release,
        lcb_n_samples=lcb_n_samples,
        lcb_temperature=lcb_temperature,
        lcb_top_p=lcb_top_p,
        lcb_top_k=lcb_top_k,
        lcb_max_new_tokens=lcb_max_new_tokens,
        lcb_prompt_batch_size=lcb_prompt_batch_size,
        lcb_num_process_evaluate=lcb_num_process_evaluate,
        lcb_timeout=lcb_timeout,
        print_extracted_code_preview=print_extracted_code_preview,
        extracted_preview_chars=extracted_preview_chars,
        unload_model_before_grading=unload_model_before_grading,
    )


if __name__ == "__main__":
    # Avoid fork-after-CUDA: worker processes would CoW-copy a parent that mapped Llama weights.
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Run pure Llama-3-8B-Instruct baseline on code_contests + LiveCodeBench."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_label", type=str, default="Llama3-8B-Instruct-baseline")
    parser.add_argument("--results_root", type=str, default=None)
    parser.add_argument("--livecodebench_release", type=str, default="release_v6")
    parser.add_argument("--lcb_n_samples", type=int, default=10)
    parser.add_argument("--lcb_temperature", type=float, default=0.2)
    parser.add_argument("--lcb_top_p", type=float, default=0.95)
    parser.add_argument("--lcb_top_k", type=int, default=50)
    parser.add_argument("--lcb_max_new_tokens", type=int, default=2000)
    parser.add_argument("--lcb_prompt_batch_size", type=int, default=4)
    parser.add_argument(
        "--lcb_num_process_evaluate",
        type=int,
        default=4,
        help="Parallel LCB grading workers (each task also spawns short-lived helper processes). "
        "Lower if the host OOMs during eval (try 2 on ~64Gi RAM with the model still in memory elsewhere).",
    )
    parser.add_argument("--lcb_timeout", type=int, default=6)
    parser.add_argument(
        "--keep_model_loaded_for_grading",
        action="store_true",
        help="Skip unloading the LLM before pass@k grading (uses more RAM; only for debugging).",
    )
    parser.add_argument(
        "--print_extracted_code_preview",
        action="store_true",
        help="Print a short excerpt of extracted code per sample (===== between samples).",
    )
    parser.add_argument(
        "--extracted_preview_chars",
        type=int,
        default=420,
        help="Max chars of extracted code to print per sample (with eval_metrics_llama_baseline --print_extracted_code_preview).",
    )
    parser.add_argument("--wandb_project", type=str, default="coding-qa")
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Optional explicit W&B run name. Defaults to llama8b-baseline-{release}-seed{seed}.",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable W&B logging and run offline.",
    )
    args = parser.parse_args()

    run = None
    if not args.disable_wandb:
        run_name = args.wandb_name or f"llama8b-baseline-{args.livecodebench_release}-seed{args.seed}"
        run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )

    try:
        results = run_llama8b_instruct_baseline(
            seed=args.seed,
            model_label=args.model_label,
            run_id=(run.id if run is not None else None),
            results_root=args.results_root,
            livecodebench_release=args.livecodebench_release,
            lcb_n_samples=args.lcb_n_samples,
            lcb_temperature=args.lcb_temperature,
            lcb_top_p=args.lcb_top_p,
            lcb_top_k=args.lcb_top_k,
            lcb_max_new_tokens=args.lcb_max_new_tokens,
            lcb_prompt_batch_size=args.lcb_prompt_batch_size,
            lcb_num_process_evaluate=args.lcb_num_process_evaluate,
            lcb_timeout=args.lcb_timeout,
            print_extracted_code_preview=args.print_extracted_code_preview,
            extracted_preview_chars=args.extracted_preview_chars,
            unload_model_before_grading=not args.keep_model_loaded_for_grading,
        )
        print(json.dumps(results, indent=2))
    finally:
        if run is not None:
            wandb.finish()
