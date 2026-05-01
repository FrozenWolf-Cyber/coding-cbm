"""
Centralized evaluation metrics for CB-LLMs generation.

All eval/test functions used by train scripts and resume scripts live here.
Supports caching for steerability text generation and perplexity text generation.

Default metrics (called by train scripts after training):
  - Perplexity (under 30 tokens + all tokens)
  - Steerability (RoBERTa classifiers or MPNet similarity)
  - Concept accuracy (hard labels or cosine similarity)
  - RM rewards (relevance, grammar, together)
"""
from __future__ import annotations

import gc
import glob
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaConfig,
    LlamaModel,
    RobertaTokenizerFast,
)

from modules import CBL, CBLResidual, Roberta_classifier
from shared_code_prompt import (
    LCB_LLAMA3_INSTRUCT_MODEL_ID,
    format_lcb_llama3_instruct_prompt,
)
from steerability_cache import (
    load_concept_samples,
    save_all_steerability_texts,
    steerability_output_root,
    write_samples_batch,
)
from utils import (
    cos_sim_cubed,
    eos_pooling,
    mean_pooling,
    compute_multilabel_topk_accuracy,
    compute_multilabel_concept_metrics,
)


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def safe_wandb_log(payload):
    """Log to W&B only when a run is initialized."""
    if payload is None:
        return
    run = getattr(wandb, "run", None)
    if run is None:
        return
    try:
        wandb.log(payload)
    except wandb.Error:
        # Keep evaluation running in debug/non-wandb mode.
        pass


_CACHED_LLAMA_VOCAB_WEIGHT = None
CLEANED_TAGS_MAP = pickle.load(open(Path(__file__).parent / "cleaned_tags.pkl", "rb"))


def get_llama_vocab_weight(device):
    global _CACHED_LLAMA_VOCAB_WEIGHT
    if _CACHED_LLAMA_VOCAB_WEIGHT is not None:
        return _CACHED_LLAMA_VOCAB_WEIGHT
    lm_head_model = AutoModelForCausalLM.from_pretrained(
        LCB_LLAMA3_INSTRUCT_MODEL_ID, torch_dtype=torch.bfloat16,
    ).to(device)
    _CACHED_LLAMA_VOCAB_WEIGHT = lm_head_model.get_output_embeddings().weight.detach()
    del lm_head_model
    torch.cuda.empty_cache()
    return _CACHED_LLAMA_VOCAB_WEIGHT


def release_llama_vocab_weight():
    global _CACHED_LLAMA_VOCAB_WEIGHT
    _CACHED_LLAMA_VOCAB_WEIGHT = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def get_intervention_value(dataset: str) -> int:
    return 150


# ═══════════════════════════════════════════════════════════════
# code_contests / LiveCodeBench Evaluation
# ═══════════════════════════════════════════════════════════════

def _format_code_generation_prompt(
    tokenizer,
    problem_description: str,
    starter_code: str = "",
    language: str = "python",
) -> str:
    return format_lcb_llama3_instruct_prompt(
        tokenizer=tokenizer,
        problem_description=problem_description,
        starter_code=starter_code,
        language=language,
    )


# ── Concept steering helpers ─────────────────────────────────────────────────

def _build_groundtruth_intervene(
    cf_tags: Optional[Sequence[str]],
    concept_set: List[str],
    steer_value: float,
) -> Optional[List[float]]:
    """Build an intervention vector directly from ground-truth CF tags."""
    if not cf_tags:
        return None
    concept_index = {c: i for i, c in enumerate(concept_set)}
    v = [0.0] * len(concept_set)
    for tag in cf_tags:
        idx = concept_index.get(tag)
        if idx is not None:
            v[idx] = float(steer_value)
    if all(val == 0.0 for val in v):
        return None
    return v


def _resolve_intervene(
    steer_mode: str,
    text: str,
    cf_tags: Optional[Sequence[str]],
    preLM,
    cbl,
    tokenizer,
    device: torch.device,
    concept_set: List[str],
    steer_value: float,
) -> Optional[List[float]]:
    """Return an intervention vector for *one* problem, or None for unsteered."""
    if steer_mode == "none":
        return None
    if steer_mode == "groundtruth":
        return _build_groundtruth_intervene(cf_tags, concept_set, steer_value)
    raise ValueError(f"Unknown steer_mode: {steer_mode!r}. Choose 'none' or 'groundtruth'.")


# ── Core per-problem generation ───────────────────────────────────────────────

@torch.no_grad()
def _generate_solutions(
    preLM,
    cbl,
    tokenizer,
    prompt: str,
    device: torch.device,
    n_samples: int = 1,
    intervene=None,
    max_new_tokens: int = 2000,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.05,
    llama_vocab_weight=None,
) -> List[str]:
    """Generate n_samples completions for a single prompt string."""
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = enc["input_ids"]
    prompt_len = prompt_ids.shape[1]

    gen_ids, _ = cbl.generate_batch(
        prompt_ids,
        preLM,
        num_samples=n_samples,
        intervene=intervene,
        length=max_new_tokens,
        temp=temperature,
        topk=top_k,
        topp=top_p,
        repetition_penalty=repetition_penalty,
        llama_vocab_weight=llama_vocab_weight,
    )
    outputs = []
    for i in range(n_samples):
        completion = gen_ids[i, prompt_len:]
        outputs.append(tokenizer.decode(completion, skip_special_tokens=True).strip())
    return outputs


@torch.no_grad()
def _generate_solutions_batched(
    preLM,
    cbl,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    n_samples: int = 1,
    intervenes: Optional[List[Optional[List[float]]]] = None,
    keep_other_concepts: bool = False,
    max_new_tokens: int = 2000,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.05,
    llama_vocab_weight=None,
) -> List[List[str]]:
    """Generate completions for a batch of prompts in one GPU pass."""
    if not prompts:
        return []

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    finally:
        tokenizer.padding_side = original_padding_side

    prompt_width = enc["input_ids"].shape[1]
    intervene_tensor = None
    intervene_mask = None
    if intervenes is not None and any(v is not None for v in intervenes):
        concept_dim = cbl.concept_dim
        dense_rows = []
        mask_rows = []
        for v in intervenes:
            if v is None:
                dense_rows.append([0.0] * concept_dim)
                mask_rows.append(False)
            else:
                dense_rows.append([float(x) for x in v])
                mask_rows.append(True)
        intervene_tensor = torch.tensor(dense_rows, dtype=torch.float32, device=device)
        intervene_mask = torch.tensor(mask_rows, dtype=torch.bool, device=device)

    gen_ids, _ = cbl.generate_intervention_batch_parallel(
        enc["input_ids"],
        preLM,
        attention_mask=enc["attention_mask"],
        num_samples=n_samples,
        interventions=intervene_tensor,
        intervention_mask=intervene_mask,
        length=max_new_tokens,
        temp=temperature,
        topk=top_k,
        topp=top_p,
        repetition_penalty=repetition_penalty,
        keep_other_concepts=keep_other_concepts,
        llama_vocab_weight=llama_vocab_weight,
    )

    num_prompts = len(prompts)
    outputs: List[List[str]] = []
    for i in range(num_prompts):
        row_outputs: List[str] = []
        base_idx = i * n_samples
        for s_idx in range(n_samples):
            completion = gen_ids[base_idx + s_idx, prompt_width:]
            row_outputs.append(tokenizer.decode(completion, skip_special_tokens=True).strip())
        outputs.append(row_outputs)
    return outputs



# ── LCB code extraction (mirrors lcb_runner/utils/extraction_utils.py) ────────

def _extract_code_from_output(model_output: str) -> str:
    """Extract code between the last pair of ``` fences, identical to LCB's extract_code."""
    lines = model_output.split("\n")
    fence_lines = [i for i, l in enumerate(lines) if "```" in l]
    if len(fence_lines) < 2:
        return ""
    return "\n".join(lines[fence_lines[-2] + 1 : fence_lines[-1]])


def print_extracted_code_samples_preview(
    heading: str,
    extracted_codes: Sequence[str],
    *,
    preview_chars: int = 420,
    sep_width: int = 60,
) -> None:
    """Log a heading and the start of each extracted code sample, with ``=====`` dividers."""
    sep = "=" * sep_width
    print(f"\n{heading}")
    for j, code in enumerate(extracted_codes):
        if j > 0:
            print(sep)
        body = (code or "").strip()
        if preview_chars > 0 and len(body) > preview_chars:
            body = body[:preview_chars] + "\n  ... [truncated]"
        if not body:
            print(f"  [sample {j + 1}/{len(extracted_codes)}] extracted: (empty)")
        else:
            indented = "\n".join(f"  {ln}" for ln in body.split("\n"))
            print(f"  [sample {j + 1}/{len(extracted_codes)}] extracted (start):\n{indented}")


# ── LCB import helper ─────────────────────────────────────────────────────────

def _import_lcb():
    """Add the local LiveCodeBench repo to sys.path and return the key modules."""
    lcb_path = str((Path(__file__).parent / "LiveCodeBench").resolve())
    if lcb_path not in sys.path:
        sys.path.insert(0, lcb_path)
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
    from lcb_runner.evaluation.pass_k_utils import extract_instance_results
    return load_code_generation_dataset, codegen_metrics, extract_instance_results


# ── Main entry point ──────────────────────────────────────────────────────────

def run_codecontests_evaluation_for_cbm(
    preLM,
    cbl,
    tokenizer,
    concept_set: List[str],
    # code_contests test split (HF Dataset) — for internal metrics
    test_dataset=None,
    seed: int = 42,
    batch_size: int = 4,
    model_label: str = "CBM-Llama3-code_contests",
    layer_idx: int = -1,
    run_id=None,
    # Generation params (code_contests test set, 1 sample per problem)
    max_new_tokens: int = 2000,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.05,
    results_root=None,
    llama_vocab_weight=None,
    display: bool = True,
    # ── Steering ──────────────────────────────────────────────────────────────
    # Pass a list to run multiple modes in one call, each logged separately.
    # Valid values: "none" (unsteered baseline), "groundtruth" (CF-tag steering).
    steer_modes: Optional[List[str]] = None,
    steer_value: Optional[float] = None,
    keep_other_concepts: bool = False,
    # ── LiveCodeBench ─────────────────────────────────────────────────────────
    livecodebench_release: str = "release_v6",
    # LCB generation uses n=10, temp=0.2 to match the leaderboard exactly.
    lcb_n_samples: int = 10,
    lcb_temperature: float = 0.2,
    lcb_top_p: float = 0.95,
    lcb_top_k: int = 50,
    lcb_max_new_tokens: int = 2000,
    lcb_repetition_penalty: float = 1.05,
    lcb_num_process_evaluate: int = 4,
    lcb_timeout: int = 6,
    livecodebench_split: str = "test",
    print_extracted_code_preview: bool = False,
    extracted_preview_chars: int = 420,
) -> dict:
    """Generate and evaluate code for both code_contests test set and LiveCodeBench.

    Runs each steer_mode independently and logs metrics to wandb under separate keys.

    Steering modes
    --------------
    "none"       : plain CBM forward pass — the apples-to-apples baseline vs LLaMA-3-8B.
    "groundtruth": use CF tags from the dataset to select concept(s) to steer.

    LiveCodeBench comparability
    ---------------------------
    Results are saved to  output/{model_label}-{steer_mode}/codegeneration_{n}_{temp}.json
    (a path that compute_scores.py and the LCB leaderboard tooling can directly consume).
    Use  --n 10 --temperature 0.2  to match all published baselines.
    """
    import json
    from pathlib import Path

    preLM.eval()
    cbl.eval()
    set_seed(seed)

    if steer_value is None:
        steer_value = float(get_intervention_value("code_contests"))

    if steer_modes is None:
        steer_modes = ["none"]

    lcb_repo = Path(__file__).parent / "LiveCodeBench"

    if run_id is None:
        run_id = wandb.run.id if wandb.run is not None else "norun"

    device = next(preLM.parameters()).device
    base_root = Path(results_root) if results_root else Path(__file__).parent / "results"

    all_results: dict = {}

    # ═════════════════════════════════════════════════════════════════════════
    # 1. code_contests internal test set  (1 sample per problem, pass@1)
    # ═════════════════════════════════════════════════════════════════════════
    if test_dataset is not None:
        print(f"\n{'='*60}")
        print(f" code_contests test set  ({len(test_dataset)} problems)")
        print(f"{'='*60}")
        concept_index = {c: idx for idx, c in enumerate(concept_set)}

        for steer_mode in steer_modes:
            mode_label = f"{model_label}-{steer_mode}"
            cc_dir = base_root / "code_contests" / mode_label
            cc_dir.mkdir(parents=True, exist_ok=True)
            out_path = cc_dir / f"l{layer_idx}-seed{seed}-{run_id}.jsonl"

            print(f"\n[{steer_mode}] Generating solutions for code_contests test set ...")
            rows = []
            concept_pred_rows = []
            concept_target_rows = []
            prompt_batch_size = max(1, int(batch_size))
            pending_prompts: List[str] = []
            pending_intervenes: List[Optional[List[float]]] = []
            pending_meta_rows: List[dict] = []

            def _flush_cc_batch():
                if not pending_prompts:
                    return
                generated = _generate_solutions_batched(
                    preLM,
                    cbl,
                    tokenizer,
                    pending_prompts,
                    device,
                    n_samples=1,
                    intervenes=pending_intervenes,
                    keep_other_concepts=keep_other_concepts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    llama_vocab_weight=llama_vocab_weight,
                )
                for meta, outputs_for_prompt in zip(pending_meta_rows, generated):
                    solution = outputs_for_prompt[0]
                    extracted = _extract_code_from_output(solution)
                    rows.append(
                        {
                            **meta,
                            "raw_output": solution,
                            "extracted_code": extracted,
                        }
                    )
                    if print_extracted_code_preview:
                        desc = meta.get("description_preview") or ""
                        pname = meta.get("problem_name", "")
                        print_extracted_code_samples_preview(
                            f"[{steer_mode}] code_contests  problem={pname!r}  "
                            f"description (start): {desc!r}",
                            [extracted],
                            preview_chars=extracted_preview_chars,
                        )
                pending_prompts.clear()
                pending_intervenes.clear()
                pending_meta_rows.clear()

            for i in tqdm(range(len(test_dataset)), desc=f"cc/{steer_mode}", disable=not display):
                problem = test_dataset[i]
                description = problem["description"].strip()
                if not description:
                    continue

                cf_tags = problem["cf_tags"]
                intervene = _resolve_intervene(
                    steer_mode, description, cf_tags, preLM, cbl, tokenizer,
                    device, concept_set, steer_value,
                )
                prompt = _format_code_generation_prompt(
                    tokenizer,
                    description,
                    language="python",
                )
                eval_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
                with torch.no_grad():
                    eval_features = preLM(
                        input_ids=eval_enc["input_ids"],
                        attention_mask=eval_enc["attention_mask"],
                    ).last_hidden_state
                    eval_llama_logits = (
                        F.linear(eval_features, llama_vocab_weight)
                        if llama_vocab_weight is not None
                        else None
                    )
                    eval_concepts, _, _, _ = cbl(eval_features.float(), llama_logits=eval_llama_logits)
                    pooled_eval_concepts = eos_pooling(eval_concepts, eval_enc["attention_mask"]).squeeze(0).detach().cpu()

                target_multihot = torch.zeros(len(concept_set), dtype=torch.float32)
                for tag in cf_tags:
                    idx = concept_index.get(tag)
                    if idx is not None:
                        target_multihot[idx] = 1.0
                if bool((target_multihot > 0).any()):
                    concept_pred_rows.append(pooled_eval_concepts)
                    concept_target_rows.append(target_multihot)

                pending_prompts.append(prompt)
                pending_intervenes.append(intervene)
                pending_meta_rows.append({
                    "problem_name": problem.get("name", f"problem_{i}"),
                    "description_preview": description[:300],
                    "cf_tags": cf_tags,
                    "cf_rating": problem.get("cf_rating", -1),
                    "steer_mode": steer_mode,
                    "steer_value": steer_value if steer_mode == "groundtruth" else 0.0,
                    "steer_topk": 0,
                    "layer_idx": layer_idx,
                    "seed": seed,
                    "run_id": run_id,
                })
                if len(pending_prompts) >= prompt_batch_size:
                    _flush_cc_batch()
            _flush_cc_batch()

            with open(out_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            print(f"  Saved {len(rows)} solutions → {out_path}")

            concept_acc_metrics = {}
            if concept_pred_rows:
                pred_tensor = torch.stack(concept_pred_rows, dim=0)
                target_tensor = torch.stack(concept_target_rows, dim=0)
                topk_metrics = compute_multilabel_concept_metrics(
                    prediction_scores=pred_tensor,
                    target_scores=target_tensor,
                    topk=(1, 5, 10),
                )
                concept_acc_metrics = {
                    f"cc/{steer_mode}/concept_tag_top1_acc": topk_metrics["top1_acc"],
                    f"cc/{steer_mode}/concept_tag_top5_acc": topk_metrics["top5_acc"],
                    f"cc/{steer_mode}/concept_tag_top10_acc": topk_metrics["top10_acc"],
                    f"cc/{steer_mode}/concept_tag_top1_iou": topk_metrics["top1_iou"],
                    f"cc/{steer_mode}/concept_tag_top5_iou": topk_metrics["top5_iou"],
                    f"cc/{steer_mode}/concept_tag_top10_iou": topk_metrics["top10_iou"],
                    f"cc/{steer_mode}/concept_tag_cosine_raw": topk_metrics["cosine_raw"],
                    f"cc/{steer_mode}/concept_tag_cosine_cubed": topk_metrics["cosine_cubed"],
                }
                print(
                    "  Concept-tag metrics: "
                    f"top1={topk_metrics['top1_acc']:.4f}, "
                    f"top5={topk_metrics['top5_acc']:.4f}, "
                    f"top10={topk_metrics['top10_acc']:.4f}, "
                    f"iou@1={topk_metrics['top1_iou']:.4f}, "
                    f"iou@5={topk_metrics['top5_iou']:.4f}, "
                    f"iou@10={topk_metrics['top10_iou']:.4f}, "
                    f"cos={topk_metrics['cosine_raw']:.4f}, "
                    f"cos_cubed={topk_metrics['cosine_cubed']:.4f}"
                )

            log_payload = {
                f"cc/{steer_mode}/solutions_written": len(rows),
                f"cc/{steer_mode}/output_path": str(out_path),
            }
            log_payload.update(concept_acc_metrics)
            if wandb.run is not None:
                safe_wandb_log(log_payload)
            all_results[f"cc/{steer_mode}"] = {"output_path": str(out_path), **concept_acc_metrics}

    # ═════════════════════════════════════════════════════════════════════════
    # 2. LiveCodeBench  (n_samples per problem, full pass@k grading)
    # ═════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f" LiveCodeBench  (release={livecodebench_release}, n={lcb_n_samples}, T={lcb_temperature})")
    print(f"{'='*60}")
    
    
    load_code_generation_dataset, codegen_metrics, extract_instance_results = _import_lcb()

    benchmark = load_code_generation_dataset(livecodebench_release)
    print(f"  Loaded {len(benchmark)} LCB problems")
    for steer_mode in steer_modes:
        # LCB canonical output path mirrors what main.py would produce.
        # model_repr = "{model_label}-{steer_mode}"  → leaderboard row label
        mode_repr = f"{model_label}-{steer_mode}"
        lcb_out_dir = Path(lcb_repo) / "output" / mode_repr / str(run_id)
        lcb_out_dir.mkdir(parents=True, exist_ok=True)
        lcb_out_path = lcb_out_dir / f"codegeneration_{lcb_n_samples}_{lcb_temperature}.json"
        lcb_eval_path = lcb_out_dir / f"codegeneration_{lcb_n_samples}_{lcb_temperature}_eval.json"
        lcb_eval_all_path = lcb_out_dir / f"codegeneration_{lcb_n_samples}_{lcb_temperature}_eval_all.json"
        # ── Generation ──
        print(f"\n[{steer_mode}] Generating {lcb_n_samples} solutions × {len(benchmark)} LCB problems ...")
        all_outputs: List[List[str]] = []
        all_extracted: List[List[str]] = []
        benchmark_sorted = sorted(benchmark, key=lambda x: x.question_id)
        prompt_batch_size = max(1, int(batch_size))
        pending_lcb_prompts: List[str] = []
        pending_lcb_intervenes: List[Optional[List[float]]] = []
        pending_lcb_headings: List[str] = []

        def _flush_lcb_batch():
            if not pending_lcb_prompts:
                return
            generated = _generate_solutions_batched(
                preLM,
                cbl,
                tokenizer,
                pending_lcb_prompts,
                device,
                n_samples=lcb_n_samples,
                intervenes=pending_lcb_intervenes,
                keep_other_concepts=keep_other_concepts,
                max_new_tokens=lcb_max_new_tokens,
                temperature=lcb_temperature,
                top_p=lcb_top_p,
                top_k=lcb_top_k,
                repetition_penalty=lcb_repetition_penalty,
                llama_vocab_weight=llama_vocab_weight,
            )
            for heading, raw_samples in zip(pending_lcb_headings, generated):
                extracted = [_extract_code_from_output(s) for s in raw_samples]
                if print_extracted_code_preview:
                    print_extracted_code_samples_preview(
                        heading,
                        extracted,
                        preview_chars=extracted_preview_chars,
                    )
                all_outputs.append(raw_samples)
                all_extracted.append(extracted)
            pending_lcb_prompts.clear()
            pending_lcb_intervenes.clear()
            pending_lcb_headings.clear()

        for problem in tqdm(benchmark_sorted, desc=f"lcb/{steer_mode}", disable=not display):
            text_for_steer = problem.question_content  # problem statement text
            problem_id = str(problem.question_id)
            mapped_cf_tags = CLEANED_TAGS_MAP[problem_id]["tags"]
            intervene = _resolve_intervene(
                steer_mode, text_for_steer, mapped_cf_tags, preLM, cbl, tokenizer,
                device, concept_set, steer_value,
            )
            prompt = _format_code_generation_prompt(
                tokenizer,
                problem.question_content,
                starter_code=getattr(problem, "starter_code", "") or "",
                language="python",
            )
            pending_lcb_prompts.append(prompt)
            pending_lcb_intervenes.append(intervene)
            desc_flat = problem.question_content.replace("\n", " ").strip()
            desc_short = desc_flat[:260] + ("..." if len(desc_flat) > 260 else "")
            pending_lcb_headings.append(
                f"[{steer_mode}] LCB  question_id={problem.question_id}  "
                f"description (start): {desc_short!r}"
            )
            if len(pending_lcb_prompts) >= prompt_batch_size:
                _flush_lcb_batch()
        _flush_lcb_batch()
        # Save in LCB canonical JSON format
        save_results = [
            problem.insert_output(outputs, codes)
            for problem, outputs, codes in zip(benchmark_sorted, all_outputs, all_extracted)
        ]
        with open(lcb_out_path, "w") as f:
            json.dump(save_results, f, indent=4)
        print(f"  Saved LCB outputs → {lcb_out_path}")
        # ── Grading (pass@k via LCB's codegen_metrics) ──
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
        with open(lcb_eval_all_path, "w") as f:
            json.dump(save_eval_results, f, indent=4)
        with open(lcb_eval_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"  Saved LCB eval → {lcb_eval_all_path}")
        # ── Log & report ──
        pass1 = metrics.get("pass@1", float("nan")) if isinstance(metrics, dict) else float("nan")
        pass5 = metrics.get("pass@5", float("nan")) if isinstance(metrics, dict) else float("nan")
        print(f"\n  [{steer_mode}] LCB pass@1 = {pass1:.4f}  |  pass@5 = {pass5:.4f}")
        print(f"  model_repr : {mode_repr}")
        print(f"  eval_all   : {lcb_eval_all_path}")
        log_payload = {
            f"lcb/{steer_mode}/pass@1": pass1,
            f"lcb/{steer_mode}/pass@5": pass5,
            f"lcb/{steer_mode}/n_samples": lcb_n_samples,
            f"lcb/{steer_mode}/temperature": lcb_temperature,
            f"lcb/{steer_mode}/steer_value": steer_value if steer_mode == "groundtruth" else 0.0,
            f"lcb/{steer_mode}/steer_topk": 0,
            f"lcb/{steer_mode}/release": livecodebench_release,
            f"lcb/{steer_mode}/output_path": str(lcb_out_path),
        }
        if wandb.run is not None:
            safe_wandb_log(log_payload)
        all_results[f"lcb/{steer_mode}"] = {
            "pass@1": pass1, "pass@5": pass5,
            "output_path": str(lcb_out_path),
            "eval_all_path": str(lcb_eval_all_path),
        }



    return all_results





# ═══════════════════════════════════════════════════════════════
# Checkpoint Discovery
# ═══════════════════════════════════════════════════════════════

def infer_run_layout(run_id, dataset, run_config):
    d_name = dataset.replace("/", "_")
    cbm_prefix = f"./from_pretained_llama3_lora_cbm_{run_id}/{d_name}/"
    grpo_prefix = f"./from_pretained_llama3_lora_grpo_{run_id}/{d_name}/"

    cbm_exists = os.path.isdir(cbm_prefix)
    grpo_exists = os.path.isdir(grpo_prefix)

    if cbm_exists and not grpo_exists:
        return "cbm", cbm_prefix
    if grpo_exists and not cbm_exists:
        return "grpo", grpo_prefix
    if "grpo_epochs" in run_config and "pretrained_run_id" in run_config:
        return "grpo", grpo_prefix
    if "discrimination_loss" in run_config:
        return "cbm", cbm_prefix
    if cbm_exists:
        return "cbm", cbm_prefix
    if grpo_exists:
        return "grpo", grpo_prefix
    return None, None


def parse_epoch_from_path(path, marker):
    basename = os.path.basename(path)
    try:
        return int(basename.replace(marker, "").replace(".pt", ""))
    except Exception:
        return None


def find_eval_checkpoint(prefix, run_type, dataset):
    if not os.path.isdir(prefix):
        return None, None, None, None

    # Prefer explicit best checkpoints if present.
    peft_best = os.path.join(prefix, "llama3_best")
    cbl_best = os.path.join(prefix, "cbl_best.pt")
    if os.path.isdir(peft_best) and os.path.isfile(cbl_best):
        return peft_best, cbl_best, -1, False

    cbl_best_files = sorted(glob.glob(os.path.join(prefix, "cbl_epoch_*.pt")))
    cbl_low_files = sorted(glob.glob(os.path.join(prefix, "cbl_low_score_epoch_*.pt")))

    best_epoch = None
    is_low_score = False

    if cbl_best_files:
        epochs = [parse_epoch_from_path(f, "cbl_epoch_") for f in cbl_best_files]
        epochs = [e for e in epochs if e is not None]
        if epochs:
            best_epoch = max(epochs)
            is_low_score = False

    if best_epoch is None and cbl_low_files:
        low_epochs = [parse_epoch_from_path(f, "cbl_low_score_epoch_") for f in cbl_low_files]
        low_epochs = [e for e in low_epochs if e is not None]
        if low_epochs:
            best_epoch = max(low_epochs)
            is_low_score = True

    if best_epoch is None:
        return None, None, None, None

    if is_low_score:
        peft_path = os.path.join(prefix, f"llama3_low_score_epoch_{best_epoch}")
        cbl_path = os.path.join(prefix, f"cbl_low_score_epoch_{best_epoch}.pt")
    else:
        peft_path = os.path.join(prefix, f"llama3_epoch_{best_epoch}")
        cbl_path = os.path.join(prefix, f"cbl_epoch_{best_epoch}.pt")

    if not os.path.isdir(peft_path):
        return None, None, None, None
    if not os.path.isfile(cbl_path):
        return None, None, None, None

    return peft_path, cbl_path, best_epoch, is_low_score


# ═══════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════

def load_model_and_cbl(
    peft_path, cbl_path, config, concept_set, tokenizer,
    discrimination_loss, residual_dim, device,
):
    preLM = LlamaModel.from_pretrained(
        LCB_LLAMA3_INSTRUCT_MODEL_ID, torch_dtype=torch.bfloat16,
    ).to(device)
    preLM.load_adapter(peft_path)
    preLM.eval()

    if discrimination_loss > 0:
        cbl = CBL(config, len(concept_set), tokenizer).to(device)
    else:
        cbl = CBLResidual(config, len(concept_set), residual_dim, tokenizer).to(device)

    state_dict = torch.load(cbl_path, map_location=device)
    try:
        cbl.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: strict load_state_dict failed for {cbl_path}: {e}")
        incompatible = cbl.load_state_dict(state_dict, strict=False)
        print(
            f"Falling back to strict=False: "
            f"missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
    cbl.eval()
    return preLM, cbl


# ═══════════════════════════════════════════════════════════════
# Steerability Text Generation (with disk caching)
# ═══════════════════════════════════════════════════════════════

def generate_steerability_texts(
    preLM,
    cbl,
    tokenizer,
    concept_set,
    dataset,
    device,
    samples_per_concept,
    print_k=3,
    llama_vocab_weight=None,
    keep_other_concepts=False,
    steerability_cache_dir=None,
    steerability_cache_seed=42,
    interventions_per_batch=1,
):
    """
    Generate steered texts for each concept with caching.

    Returns list-of-lists: ``decoded_texts_by_concept[concept_idx][sample_idx]``.
    """
    intervention_value = get_intervention_value(dataset)
    input_ids = torch.tensor([tokenizer.encode("")]).to(device)
    special_tokens_mask = torch.tensor([128000, 128001]).to(device)
    num_concepts = len(concept_set)
    chunk_size = 25
    cseed = steerability_cache_seed

    all_slots: list[list[str | None]] = []
    for concept_idx in range(num_concepts):
        cname = concept_set[concept_idx]
        slots = load_concept_samples(
            steerability_cache_dir, cseed, concept_idx, cname, samples_per_concept,
        )
        all_slots.append(slots)

    with torch.no_grad():
        if interventions_per_batch <= 1:
            for concept_idx in tqdm(range(num_concepts), desc="Steerability generation"):
                v = [0] * num_concepts
                v[concept_idx] = intervention_value
                cname = concept_set[concept_idx]
                slots = all_slots[concept_idx]
                pos = 0
                while pos < samples_per_concept:
                    if slots[pos] is not None:
                        pos += 1
                        continue
                    end = pos
                    while end < samples_per_concept and slots[end] is None:
                        end += 1
                    gen_pos = pos
                    while gen_pos < end:
                        current_batch = min(chunk_size, end - gen_pos)
                        text_ids_batch, _ = cbl.generate_batch(
                            input_ids, preLM,
                            num_samples=current_batch,
                            intervene=v, length=50,
                            keep_other_concepts=keep_other_concepts,
                            llama_vocab_weight=llama_vocab_weight,
                        )
                        pending_writes: list[tuple] = []
                        for b in range(current_batch):
                            sample_idx = gen_pos + b
                            decoded = tokenizer.decode(
                                text_ids_batch[b][~torch.isin(text_ids_batch[b], special_tokens_mask)]
                            )
                            slots[sample_idx] = decoded
                            if steerability_cache_dir:
                                pending_writes.append(
                                    (concept_idx, cname, cseed, sample_idx, decoded)
                                )
                        if pending_writes:
                            write_samples_batch(steerability_cache_dir, pending_writes)
                        gen_pos += current_batch
                    pos = end
        else:
            for group_start in tqdm(
                range(0, num_concepts, interventions_per_batch),
                desc=f"Steerability generation (x{interventions_per_batch} concepts/batch)",
            ):
                group_end = min(group_start + interventions_per_batch, num_concepts)
                group_indices = list(range(group_start, group_end))

                needs_gen: list[int] = []
                missing_indices: dict[int, list[int]] = {}
                for ci in group_indices:
                    missing = [i for i in range(samples_per_concept) if all_slots[ci][i] is None]
                    if missing:
                        needs_gen.append(ci)
                        missing_indices[ci] = missing

                if not needs_gen:
                    continue

                interventions = []
                for ci in needs_gen:
                    v = [0] * num_concepts
                    v[ci] = intervention_value
                    interventions.append(v)

                max_missing = max(len(missing_indices[ci]) for ci in needs_gen)
                gen_offset = 0
                while gen_offset < max_missing:
                    current_chunk = min(chunk_size, max_missing - gen_offset)
                    text_ids_batch, _ = cbl.generate_multi_concept_batch(
                        input_ids, preLM,
                        interventions=interventions,
                        samples_per_intervention=current_chunk,
                        length=50,
                        keep_other_concepts=keep_other_concepts,
                        llama_vocab_weight=llama_vocab_weight,
                    )
                    pending_writes = []
                    for g, ci in enumerate(needs_gen):
                        row_start = g * current_chunk
                        mi = missing_indices[ci]
                        cname = concept_set[ci]
                        for b in range(current_chunk):
                            abs_idx = gen_offset + b
                            if abs_idx >= len(mi):
                                continue
                            sample_idx = mi[abs_idx]
                            decoded = tokenizer.decode(
                                text_ids_batch[row_start + b][
                                    ~torch.isin(text_ids_batch[row_start + b], special_tokens_mask)
                                ]
                            )
                            all_slots[ci][sample_idx] = decoded
                            if steerability_cache_dir:
                                pending_writes.append(
                                    (ci, cname, cseed, sample_idx, decoded)
                                )
                    if pending_writes:
                        write_samples_batch(steerability_cache_dir, pending_writes)
                    gen_offset += current_chunk

    should_log_samples_to_wandb = wandb.run is not None

    all_texts: list[list[str]] = []
    for concept_idx in range(num_concepts):
        cname = concept_set[concept_idx]
        concept_texts = [all_slots[concept_idx][k] for k in range(samples_per_concept)]
        if should_log_samples_to_wandb:
            for idx, t in enumerate(concept_texts):
                safe_wandb_log({f"steerability_sample_{cname}_{idx + 1}": t})
        if print_k > 0:
            print(f"Concept '{cname}' sample preview:")
            for k in range(min(print_k, len(concept_texts))):
                print(f"  [{k+1}] {concept_texts[k]}")
        all_texts.append(concept_texts)

    return all_texts


# ═══════════════════════════════════════════════════════════════
# Steerability Evaluation: RoBERTa Classifiers
# ═══════════════════════════════════════════════════════════════

def score_steerability_roberta(
    decoded_texts_by_concept, roberta_tokenizer, classifier, concept_set, device,
):
    """Score steerability texts with a single RoBERTa classifier. Returns accuracy dict."""
    pred, ref = [], []
    acc = evaluate.load("accuracy")
    with torch.no_grad():
        for concept_idx, concept_texts in enumerate(
            tqdm(decoded_texts_by_concept, desc="Steerability scoring")
        ):
            if not concept_texts:
                continue
            roberta_enc = roberta_tokenizer(
                concept_texts, return_tensors="pt", truncation=True,
                max_length=512, padding=True,
            ).to(device)
            roberta_input = {
                "input_ids": roberta_enc["input_ids"],
                "attention_mask": roberta_enc["attention_mask"],
            }
            logits = classifier(roberta_input)
            pred.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
            ref.extend([concept_idx] * len(concept_texts))
    acc.add_batch(predictions=np.array(pred), references=np.array(ref))
    return acc.compute()


# Backward-compatible alias used by resume_steerability_test
run_steerability_test_from_texts = score_steerability_roberta


def run_steerability_roberta(
    decoded_texts_by_concept,
    concept_set,
    dataset,
    device,
    classifier_weight_suffixes=("_seed42", "_seed123", "_seed456"),
):
    """Run steerability eval with multiple RoBERTa classifiers. Returns dict of accuracies."""
    roberta_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    d_name = dataset.replace("/", "_")
    classifier_paths = [f"{d_name}_classifier.pt"]
    for suffix in classifier_weight_suffixes:
        classifier_paths.append(f"{d_name}_classifier{suffix}.pt")

    results = {}
    for clf_idx, classifier_path in enumerate(classifier_paths):
        if not os.path.exists(classifier_path):
            print(f"Warning: Classifier not found at {classifier_path}, skipping...")
            continue
        print(f"Testing steerability with classifier: {classifier_path}")
        classifier = Roberta_classifier(len(concept_set)).to(device)
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.eval()
        try:
            classifier = torch.compile(classifier)
        except Exception:
            pass

        acc = score_steerability_roberta(
            decoded_texts_by_concept, roberta_tokenizer, classifier, concept_set, device,
        )
        log_key = "steerability_test_accuracy" if clf_idx == 0 else f"steerability_test_accuracy_{clf_idx}"
        print(f"  {log_key}: {acc}")
        safe_wandb_log({log_key: acc})
        results[log_key] = acc

        del classifier
        torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════
# Steerability Evaluation: MPNet Similarity
# ═══════════════════════════════════════════════════════════════

def run_steerability_mpnet(
    decoded_texts_by_concept, concept_set, intervention_value, max_length, device,
):
    """Run steerability eval using MPNet sentence similarity. Returns dict of metrics."""
    tokenizer_sim = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    sim_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(device)
    sim_model.eval()

    encoded_c = tokenizer_sim(concept_set, padding=True, truncation=True, max_length=max_length)
    encoded_c = {k: torch.tensor(v).to(device) for k, v in encoded_c.items()}
    concept_features = sim_model(
        input_ids=encoded_c["input_ids"], attention_mask=encoded_c["attention_mask"],
    )
    concept_features = mean_pooling(concept_features.last_hidden_state, encoded_c["attention_mask"])
    concept_features = F.normalize(concept_features, p=2, dim=1)

    cos_sim_cubed_values: list[float] = []
    softmax_values: list[float] = []
    top1_correct = top3_correct = top5_correct = top10_correct = top20_correct = 0
    total_evals = 0
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for j in tqdm(range(len(concept_set)), desc="Steerability MPNet scoring"):
            decoded_texts = decoded_texts_by_concept[j]
            if not decoded_texts:
                continue

            v = [0] * len(concept_set)
            v[j] = intervention_value

            generated_c = tokenizer_sim(
                decoded_texts, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            generated_c = {k: v_t.to(device) for k, v_t in generated_c.items()}
            generated_features = sim_model(
                input_ids=generated_c["input_ids"],
                attention_mask=generated_c["attention_mask"],
            )
            generated_features = mean_pooling(
                generated_features.last_hidden_state, generated_c["attention_mask"],
            )
            generated_features = F.normalize(generated_features, p=2, dim=1)

            sims = generated_features @ concept_features.T
            v_tensor = torch.tensor(v).to(device).unsqueeze(0).expand(sims.size(0), -1)

            cos_vals = cos_sim_cubed(sims, v_tensor.float(), reduce=False)
            cos_sim_cubed_values.extend(cos_vals.detach().cpu().tolist())

            targets = torch.full((sims.size(0),), j, dtype=torch.long, device=device)
            ce_vals = ce_loss_fn(sims, targets)
            softmax_values.extend(ce_vals.detach().cpu().tolist())

            sorted_indices = torch.argsort(sims, dim=1, descending=True)
            top1_correct += (sorted_indices[:, 0] == j).sum().item()
            top3_correct += (sorted_indices[:, :3] == j).any(dim=1).sum().item()
            top5_correct += (sorted_indices[:, :5] == j).any(dim=1).sum().item()
            top10_correct += (sorted_indices[:, :10] == j).any(dim=1).sum().item()
            top20_correct += (sorted_indices[:, :20] == j).any(dim=1).sum().item()
            total_evals += sims.size(0)

    del sim_model
    torch.cuda.empty_cache()

    metrics = {
        "steerability_cos_sim_cubed": (
            sum(cos_sim_cubed_values) / len(cos_sim_cubed_values)
            if cos_sim_cubed_values else float("nan")
        ),
        "steerability_softmax": (
            sum(softmax_values) / len(softmax_values)
            if softmax_values else float("nan")
        ),
        "steerability_top1_acc": top1_correct / total_evals if total_evals else 0.0,
        "steerability_top3_acc": top3_correct / total_evals if total_evals else 0.0,
        "steerability_top5_acc": top5_correct / total_evals if total_evals else 0.0,
        "steerability_top10_acc": top10_correct / total_evals if total_evals else 0.0,
        "steerability_top20_acc": top20_correct / total_evals if total_evals else 0.0,
    }
    safe_wandb_log(metrics)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


# ═══════════════════════════════════════════════════════════════
# Steerability Evaluation: llama.cpp Judge (annotate_llamacpp style)
# ═══════════════════════════════════════════════════════════════

def _llamacpp_build_raw_prompt(text: str, concepts: list[str]) -> str:
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    nl = "\n"
    system_text = (
        "You are a strict classifier for coding tasks. "
        "Given generated code or a coding solution, output ONLY a single line containing exactly one algorithm "
        "or concept name copied verbatim from OPTIONS. "
        "No explanation, no preamble, no bullets."
    )
    assistant_prefill = "<think>\n\n</think>\n\n"
    opts_block = "\n".join(f"- {c}" for c in concepts)
    user_text = (
        "From OPTIONS, pick the single algorithm or programming concept that best matches the approach, "
        "technique, or topic reflected in GENERATED_TEXT (code, pseudocode, or solution text).\n\n"
        f"OPTIONS:\n{opts_block}\n\n"
        f"GENERATED_TEXT:\n{text}\n\n"
        "Answer (one label verbatim from OPTIONS, nothing else):"
    )
    return (
        f"{im_start}system{nl}{system_text}{im_end}{nl}"
        f"{im_start}user{nl}{user_text}{im_end}{nl}"
        f"{im_start}assistant{nl}{assistant_prefill}"
    )


def _llamacpp_parse_output(output: str, concepts: list[str]) -> str:
    first_line = next((ln.strip() for ln in str(output).splitlines() if ln.strip()), "")
    if not first_line:
        return concepts[0]

    parts = [p.strip() for p in re.split(r"[,;]", first_line) if p.strip()]
    if not parts:
        parts = [first_line]

    for p in parts:
        if p in concepts:
            return p
        m = next((c for c in concepts if c.lower() == p.lower()), None)
        if m is not None:
            return m
        for c in concepts:
            if p.lower() in c.lower() or c.lower() in p.lower():
                return c
    return concepts[0]


def run_steerability_llamacpp_judge(
    decoded_texts_by_concept,
    concept_set,
    model_repo_id="unsloth/Qwen3.5-27B-GGUF",
    model_filename="Qwen3.5-27B-Q8_0.gguf",
    n_ctx=2048,
    max_tokens=64,
    repeat_penalty=1.15,
    temperature=0.1,
):
    """Judge steerability by classifying generated text to closest concept with llama.cpp."""
    try:
        from llama_cpp import Llama
    except Exception as e:
        print(f"[WARN] llama_cpp import failed: {e}")
        print('Attempting install: CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python')
        try:
            env = os.environ.copy()
            env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "llama-cpp-python"],
                env=env,
            )
            from llama_cpp import Llama
            print("Successfully installed/imported llama_cpp.")
        except Exception as install_err:
            print(
                "[WARN] Failed to install/import llama_cpp after retry; "
                f"skipping llama.cpp steerability eval: {install_err}"
            )
            return {}

    print(
        f"Loading llama.cpp judge | repo={model_repo_id} file={model_filename} "
        f"n_ctx={n_ctx} max_tokens={max_tokens} temp={temperature}"
    )
    llm = Llama.from_pretrained(
        repo_id=model_repo_id,
        filename=model_filename,
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        verbose=False,
    )

    total = 0
    correct = 0
    raw_outputs = []

    for target_idx, texts in enumerate(tqdm(decoded_texts_by_concept, desc="Steerability llama.cpp judging")):
        for sample_idx, text in enumerate(texts):
            prompt = _llamacpp_build_raw_prompt(text, concept_set)
            try:
                out = llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=40,
                    repeat_penalty=repeat_penalty,
                    stop=["<|im_end|>", "<|im_start|>", "\n\n"],
                )
                raw = out["choices"][0]["text"] if out and out.get("choices") else ""
            except Exception as e:
                print(f"[WARN] llama.cpp judge failed at concept={target_idx} sample={sample_idx}: {e}")
                raw = ""

            pred_label = _llamacpp_parse_output(raw, concept_set)
            pred_idx = concept_set.index(pred_label) if pred_label in concept_set else 0
            is_correct = int(pred_idx == target_idx)
            correct += is_correct
            total += 1

            raw_outputs.append(
                {
                    "target_idx": int(target_idx),
                    "target_label": concept_set[target_idx],
                    "sample_idx": int(sample_idx),
                    "pred_label": pred_label,
                    "raw_output": raw,
                }
            )
            safe_wandb_log(
                {
                    f"steerability_llamacpp_pred_{target_idx}_{sample_idx}": pred_label,
                    f"steerability_llamacpp_correct_{target_idx}_{sample_idx}": is_correct,
                }
            )

    acc = (correct / total) if total > 0 else 0.0
    metrics = {
        "steerability_llamacpp_judge_accuracy": float(acc),
        "steerability_llamacpp_judge_total": int(total),
    }
    print(f"  steerability_llamacpp_judge_accuracy: {acc:.4f} ({correct}/{total})")
    safe_wandb_log(metrics)

    return {"metrics": metrics, "raw_outputs": raw_outputs}


# ═══════════════════════════════════════════════════════════════
# Concept Accuracy: Hard Labels (train_combined.py style)
# ═══════════════════════════════════════════════════════════════

def run_concept_accuracy_labels(preLM, cbl, test_loader, concept_set, encoded_test_dataset, device):
    """Concept prediction accuracy using argmax (hard labels). Returns accuracy dict."""
    print("eval concepts...")
    metric = evaluate.load("accuracy")
    concept_predictions = []
    for batch in tqdm(test_loader, total=len(test_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            features = preLM(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            ).last_hidden_state
            concepts, _, _, _ = cbl(features.float())
        concept_predictions.append(eos_pooling(concepts, batch["attention_mask"]))
    concept_predictions = torch.cat(concept_predictions, dim=0).detach().cpu()
    pred = np.argmax(concept_predictions.numpy(), axis=-1)
    metric.add_batch(predictions=pred, references=encoded_test_dataset["label"])
    acc = metric.compute()
    print(f"Concept prediction accuracy: {acc}")
    safe_wandb_log({"concept_prediction_accuracy": acc})
    return acc


# ═══════════════════════════════════════════════════════════════
# Concept Accuracy: Cosine Similarity (train_combined_finegrained.py style)
# ═══════════════════════════════════════════════════════════════

def run_concept_accuracy_cosine(
    preLM,
    cbl,
    test_loader,
    concept_set,
    label_prefix,
    device,
    test_similarity_np=None,
    llama_vocab_weight=None,
):
    """Concept prediction evaluation using cosine similarity to target concept vectors.

    - Default behavior (backwards compatible): load targets from ``label_prefix/concept_labels_test.npy``.
    - If ``test_similarity_np`` is provided, use it directly (e.g., one-hot class concepts) and skip disk loading.

    Args:
        test_similarity_np: Optional array-like of shape (N, C).
        llama_vocab_weight: Optional tensor (vocab_size, hidden_dim). If provided, compute llama logits from
            backbone hidden states and pass them into ``cbl(..., llama_logits=...)`` (for --add_llama_logits).
    """
    print("eval concepts (cosine similarity)...")

    concept_predictions = []
    for batch, _ in tqdm(test_loader, total=len(test_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            features = preLM(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            ).last_hidden_state
            llama_logits = F.linear(features, llama_vocab_weight) if llama_vocab_weight is not None else None
            if llama_logits is not None:
                concepts, _, _, _ = cbl(features.float(), llama_logits=llama_logits)
            else:
                concepts, _, _, _ = cbl(features.float())
        pooled_concepts = eos_pooling(concepts, batch["attention_mask"])
        concept_predictions.append(pooled_concepts.detach().cpu())
    concept_predictions = torch.cat(concept_predictions, dim=0)

    if test_similarity_np is None:
        test_sim_path = os.path.join(label_prefix, "concept_labels_test.npy")
        if not os.path.exists(test_sim_path):
            print(f"[WARN] {test_sim_path} not found. Skipping cosine concept evaluation.")
            return {}
        test_similarity_np = np.load(test_sim_path)

    test_similarity = torch.tensor(np.asarray(test_similarity_np), dtype=torch.float32)

    if test_similarity.shape != concept_predictions.shape:
        print(
            f"[WARN] Shape mismatch: predictions {tuple(concept_predictions.shape)} "
            f"vs labels {tuple(test_similarity.shape)}."
        )
        return {}

    test_cos_sim = cos_sim_cubed(concept_predictions, test_similarity)
    test_cos_loss = -test_cos_sim.item()

    pred_norm = F.normalize(concept_predictions, p=2, dim=-1)
    label_norm = F.normalize(test_similarity, p=2, dim=-1)
    test_cos_raw = (pred_norm * label_norm).sum(dim=-1).mean().item()

    print(f"Test concept cosine similarity (cos_sim_cubed): {test_cos_sim.item():.4f}")
    print(f"Test concept cosine loss: {test_cos_loss:.4f}")
    print(f"Test concept cosine similarity (raw): {test_cos_raw:.4f}")

    topk_list = [1, 3, 5, 10, 20]
    topk_metrics = compute_multilabel_topk_accuracy(
        prediction_scores=concept_predictions,
        target_scores=test_similarity,
        topk=topk_list,
    )
    topk_iou_sums = {k: 0.0 for k in topk_list}
    total = concept_predictions.size(0)
    pred_sorted = torch.argsort(concept_predictions, dim=-1, descending=True)

    for i in range(total):
        row = pred_sorted[i]
        for k in topk_list:
            k_clipped = min(k, row.size(0))
            gt_topk = torch.topk(test_similarity[i], k=k_clipped, dim=-1).indices.tolist()
            pred_topk = row[:k_clipped].tolist()
            gt_set, pred_set = set(gt_topk), set(pred_topk)
            inter = len(gt_set & pred_set)
            union = len(gt_set | pred_set)
            if union > 0:
                topk_iou_sums[k] += inter / union

    topk_acc = {f"test_concept_top{k}_acc": topk_metrics[f"top{k}_acc"] for k in topk_list}
    topk_iou = {f"test_concept_top{k}_iou": topk_iou_sums[k] / total for k in topk_list}

    for k in topk_list:
        print(f"Test concept Top-{k} Acc: {topk_acc[f'test_concept_top{k}_acc']:.4f}")
        print(f"Test concept Top-{k} IoU: {topk_iou[f'test_concept_top{k}_iou']:.4f}")

    metrics = {
        "test_concept_cosine_similarity": float(test_cos_sim.item()),
        "test_concept_cosine_loss": float(test_cos_loss),
        "test_concept_cosine_raw": float(test_cos_raw),
        **topk_acc,
        **topk_iou,
    }
    safe_wandb_log(metrics)
    return metrics


# ═══════════════════════════════════════════════════════════════
# Weight Analysis
# ═══════════════════════════════════════════════════════════════

def run_weight_analysis(cbl, concept_set, tokenizer):
    """Print and log top tokens per concept neuron and sparsity."""
    print("Top tokens for each concept neuron:")
    w = cbl.fc.weight.data[:, : len(concept_set)].T
    for i in tqdm(range(len(concept_set))):
        top_values, top_ids = torch.topk(w[i], k=10)
        print(f"Neuron: {concept_set[i]}")
        print("Top 10 tokens with highest weight:")
        for j in range(10):
            print(
                f"Neuron: {concept_set[i]} "
                f"[{round(float(top_values.detach().cpu()[j]), 3)}] "
                f"{tokenizer.decode(top_ids[j])}"
            )

    sparsity = (w > 1e-6).count_nonzero() / w.numel()
    print(f"Sparsity of concept weight matrix: {sparsity}")
    safe_wandb_log({"concept_weight_sparsity": sparsity})


# ═══════════════════════════════════════════════════════════════
# Perplexity (split: generate texts, then compute metric)
# ═══════════════════════════════════════════════════════════════

def _perplexity_cache_path(cache_dir, seed):
    return os.path.join(cache_dir, f"perplexity_texts_seed{seed}.pkl")


def generate_perplexity_texts(
    cbl, preLM, tokenizer, seed, device,
    n_samples=100, cache_dir=None, run_name=None,
    llama_vocab_weight=None,
):
    """Generate free (un-intervened) texts for perplexity. Supports caching."""
    set_seed(seed)
    pred: list[str] = []
    cached = False

    if cache_dir:
        cache_path = _perplexity_cache_path(cache_dir, seed)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                pred = pickle.load(f)
            if len(pred) >= n_samples:
                pred = pred[:n_samples]
                cached = True
                print(f"Loaded {len(pred)} cached perplexity texts from {cache_path}")

    if not cached:
        input_ids = torch.tensor([tokenizer.encode("")]).to(device)
        for _ in tqdm(range(n_samples), desc="Generating perplexity texts"):
            with torch.no_grad():
                text_ids, _ = cbl.generate(input_ids, preLM, llama_vocab_weight=llama_vocab_weight)
                pred.append(tokenizer.decode(text_ids[0], skip_special_tokens=True))

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            with open(_perplexity_cache_path(cache_dir, seed), "wb") as f:
                pickle.dump(pred, f)
            print(f"Saved perplexity texts to cache")

    if run_name:
        os.makedirs("perplexity_text", exist_ok=True)
        with open(f"perplexity_text/{run_name}_generated_texts_{seed}.pkl", "wb") as f:
            pickle.dump(pred, f)

    print("Some generated texts:")
    for i in range(min(5, len(pred))):
        print(pred[i])

    return pred


def compute_perplexity(texts: list[str]) -> dict:
    """Compute perplexity (under-30 tokens and all tokens) from pre-generated texts.

    This function loads a fresh LLM via the ``evaluate`` library, so the training
    model should be freed from GPU beforehand.
    """
    results = {}

    c = 0
    perplexity_metric = evaluate.load("perplexity", module_type="metric")
    for p in texts:
        if len(p.split()) > 30:
            continue
        c += 1
        perplexity_metric.add_batch(predictions=[p])

    if c > 0:
        ppl_short = perplexity_metric.compute(
            model_id=LCB_LLAMA3_INSTRUCT_MODEL_ID, max_length=100,
        )["mean_perplexity"]
        print(f"Perplexity (under 30 tokens): {ppl_short}")
        safe_wandb_log({"perplexity_under_30_tokens": ppl_short})
        results["perplexity_under_30_tokens"] = ppl_short
    else:
        print("No generated texts under 30 tokens to compute perplexity.")
        safe_wandb_log({"perplexity_under_30_tokens": None})

    perplexity_all = evaluate.load("perplexity", module_type="metric")
    for p in texts:
        perplexity_all.add_batch(predictions=[p])
    ppl_all = perplexity_all.compute(
        model_id=LCB_LLAMA3_INSTRUCT_MODEL_ID, max_length=100,
    )["mean_perplexity"]
    print(f"Perplexity (all tokens): {ppl_all}")
    safe_wandb_log({"perplexity_all_tokens": ppl_all})
    results["perplexity_all_tokens"] = ppl_all

    return results


# ═══════════════════════════════════════════════════════════════
# RM (Reward Model) Metrics
# ═══════════════════════════════════════════════════════════════

RM_USER_RELEVANCE = "Write a text about the concept: {concept_name}"
RM_USER_GRAMMAR = "Write a grammatically correct and fluent paragraph."
RM_USER_TOGETHER = "Write a grammatically correct and fluent text about the concept: {concept_name}"
RM_LOGIT_CLIP_MIN = -100.0
RM_LOGIT_CLIP_MAX = 100.0


def load_reward_model(rm_model_name: str, rm_device: torch.device):
    """Load a Skywork-style sequence-classification RM."""
    print(f"Loading reward model: {rm_model_name} ...")
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_model_name)
    _kwargs = dict(torch_dtype=torch.bfloat16, num_labels=1)
    try:
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            rm_model_name, attn_implementation="flash_attention_2", **_kwargs,
        )
        print("  Loaded RM with flash_attention_2.")
    except Exception as fa2_err:
        print(f"  flash_attention_2 unavailable ({fa2_err}), falling back to eager attention.")
        rm_model = AutoModelForSequenceClassification.from_pretrained(rm_model_name, **_kwargs)
    rm_model.eval()
    for p in rm_model.parameters():
        p.requires_grad = False
    rm_model.to(rm_device)
    print(f"  RM device: {rm_device}")
    return rm_model, rm_tokenizer


def _make_rm_formatted(rm_tokenizer, user_turn: str, response_text: str, max_text_len: int) -> str:
    conv = [
        {"role": "user", "content": user_turn},
        {"role": "assistant", "content": response_text[:max_text_len]},
    ]
    formatted = rm_tokenizer.apply_chat_template(conv, tokenize=False)
    if rm_tokenizer.bos_token and formatted.startswith(rm_tokenizer.bos_token):
        formatted = formatted[len(rm_tokenizer.bos_token):]
    return formatted


def _raw_logits_for_texts(
    rm_model, rm_tokenizer, texts, user_turn: str,
    device: torch.device, rm_batch_size: int, max_text_len: int,
):
    if not texts:
        return []
    formatted = [_make_rm_formatted(rm_tokenizer, user_turn, t, max_text_len) for t in texts]
    chunk = rm_batch_size if rm_batch_size > 0 else len(formatted)
    all_scores: list[float] = []
    for start in range(0, len(formatted), chunk):
        chunk_list = formatted[start : start + chunk]
        tokenized = rm_tokenizer(
            chunk_list, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(device)
        with torch.no_grad():
            logits = rm_model(**tokenized).logits
        clipped = logits[:, 0].float().clamp(RM_LOGIT_CLIP_MIN, RM_LOGIT_CLIP_MAX)
        all_scores.extend(clipped.detach().cpu().tolist())
        del tokenized, logits
    return all_scores


def run_rm_metrics(
    decoded_texts_by_concept,
    concept_set,
    rm_model,
    rm_tokenizer,
    rm_device,
    rm_batch_size=0,
    rm_max_text_len=500,
):
    """Score steerability texts with RM (relevance, grammar, together).

    Returns dict with global means/stds and per-concept breakdown.
    """
    all_rel, all_gram, all_tog = [], [], []
    per_concept: dict = {}

    for concept_idx, concept_name in enumerate(concept_set):
        texts = (
            decoded_texts_by_concept[concept_idx]
            if concept_idx < len(decoded_texts_by_concept) else []
        )
        if not texts:
            per_concept[concept_name] = {
                "n": 0,
                "rm_relevance_mean": float("nan"),
                "rm_grammar_mean": float("nan"),
                "rm_together_mean": float("nan"),
            }
            continue

        u_rel = RM_USER_RELEVANCE.format(concept_name=concept_name)
        u_tog = RM_USER_TOGETHER.format(concept_name=concept_name)

        rel = _raw_logits_for_texts(
            rm_model, rm_tokenizer, texts, u_rel, rm_device, rm_batch_size, rm_max_text_len,
        )
        gram = _raw_logits_for_texts(
            rm_model, rm_tokenizer, texts, RM_USER_GRAMMAR, rm_device, rm_batch_size, rm_max_text_len,
        )
        tog = _raw_logits_for_texts(
            rm_model, rm_tokenizer, texts, u_tog, rm_device, rm_batch_size, rm_max_text_len,
        )

        all_rel.extend(rel)
        all_gram.extend(gram)
        all_tog.extend(tog)

        per_concept[concept_name] = {
            "n": len(texts),
            "rm_relevance_mean": float(np.mean(rel)) if rel else float("nan"),
            "rm_grammar_mean": float(np.mean(gram)) if gram else float("nan"),
            "rm_together_mean": float(np.mean(tog)) if tog else float("nan"),
        }

        for b, (t, r, g, o) in enumerate(zip(texts, rel, gram, tog)):
            safe_wandb_log({
                f"rm_sample_{concept_name}_{b + 1}": t,
                f"rm_relevance_logit_{concept_name}_{b + 1}": r,
                f"rm_grammar_logit_{concept_name}_{b + 1}": g,
                f"rm_together_logit_{concept_name}_{b + 1}": o,
            })

    def _ms(xs):
        if not xs:
            return float("nan"), 0.0
        a = np.array(xs, dtype=np.float64)
        return float(a.mean()), float(a.std()) if a.size > 1 else 0.0

    r_m, r_s = _ms(all_rel)
    g_m, g_s = _ms(all_gram)
    t_m, t_s = _ms(all_tog)

    global_metrics = {
        "rm_relevance_mean": r_m, "rm_relevance_std": r_s,
        "rm_grammar_mean": g_m, "rm_grammar_std": g_s,
        "rm_together_mean": t_m, "rm_together_std": t_s,
        "rm_total_n": len(all_rel),
    }
    safe_wandb_log(global_metrics)
    print(
        f"  rm_relevance_mean={r_m:.4f} rm_grammar_mean={g_m:.4f} "
        f"rm_together_mean={t_m:.4f} (n={len(all_rel)})"
    )

    return {**global_metrics, "per_concept": per_concept}
