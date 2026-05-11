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
import importlib
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaModel,
)

from modules import CBL, CBLResidual
from shared_code_prompt import (
    LCB_LLAMA3_INSTRUCT_MODEL_ID,
    format_lcb_llama3_instruct_prompt,
)
from utils import (
    cos_sim_cubed,
    eos_pooling,
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


def _format_host_memory_stats() -> str:
    """Process RSS plus system-wide used/available RAM (Linux-friendly via psutil)."""
    try:
        import psutil

        rss_b = psutil.Process().memory_info().rss
        vm = psutil.virtual_memory()
        rss_g = rss_b / (1024**3)
        used_g = vm.used / (1024**3)
        avail_g = vm.available / (1024**3)
        total_g = vm.total / (1024**3)
        pct = getattr(vm, "percent", (used_g / total_g * 100.0) if total_g else 0.0)
        return (
            f"proc_RSS={rss_g:.2f}GiB  "
            f"sys_used={used_g:.1f}/{total_g:.1f}GiB ({pct:.1f}%)  "
            f"sys_avail={avail_g:.1f}GiB"
        )
    except Exception:
        return ""


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


def _fmt_seconds(sec: float) -> str:
    return f"{float(sec):.2f}s"


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


# Default steering magnitude when callers pass `steer_value=None` (matches
# `train_combined_finegrained.py --intervention_value` default).
DEFAULT_INTERVENTION_VALUE = 150


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
        outputs.append(
            tokenizer.decode(
                completion,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
        )
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
            row_outputs.append(
                tokenizer.decode(
                    completion,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
            )
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


def print_solution_question_and_extracted_code(
    *,
    heading: str,
    question: str,
    extracted_codes: Sequence[str],
    question_max_chars: int = 0,
    code_max_chars: int = 0,
    sep_width: int = 72,
) -> None:
    """Print full (or capped) problem statement and extracted code for CLI inspection."""
    sep = "=" * sep_width
    print(f"\n{sep}\n{heading}\n{sep}", flush=True)
    q = (question or "").strip()
    if question_max_chars > 0 and len(q) > question_max_chars:
        q = q[:question_max_chars] + "\n... [question truncated]"
    print("--- QUESTION ---", flush=True)
    print(q if q else "(empty)", flush=True)
    print("--- EXTRACTED CODE ---", flush=True)
    for j, code in enumerate(extracted_codes):
        body = (code or "").strip()
        if code_max_chars > 0 and len(body) > code_max_chars:
            body = body[:code_max_chars] + "\n... [code truncated]"
        if len(extracted_codes) > 1:
            print(f"[sample {j + 1}/{len(extracted_codes)}]", flush=True)
        print(body if body else "(empty)", flush=True)


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


def _memory_checkpoint(msg: str, *, log_host_ram: bool = False) -> None:
    """Log a progress line and encourage freeing cached allocator memory."""
    extra = ""
    if log_host_ram:
        hm = _format_host_memory_stats()
        if hm:
            extra = f"  |  {hm}"
    print(f"[eval-mem] {msg}{extra}", flush=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Split entry points (code_contests test set vs LiveCodeBench benchmark) ──
def run_codecontests_testset_evaluation_for_cbm(
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
    steer_modes: Optional[List[str]] = None,
    steer_value: Optional[float] = None,
    keep_other_concepts: bool = False,
    # Preview/debug controls
    print_extracted_code_preview: bool = False,
    extracted_preview_chars: int = 420,
    # If set to a one-element list [ds], use ds for code_contests then set holder[0]=None
    # so the HF test split can be freed before LiveCodeBench loads.
    test_dataset_holder: Optional[List[Any]] = None,
    # When True, append process RSS + system used/available to every [eval-mem] line.
    eval_log_host_memory: bool = False,
) -> dict:
    """Run only code_contests internal test-set generation + concept-tag metrics."""
    import json

    preLM.eval()
    cbl.eval()
    set_seed(seed)
    eval_start_t = time.perf_counter()

    if run_id is None:
        run_id = wandb.run.id if wandb.run is not None else "norun"

    if steer_value is None:
        steer_value = float(DEFAULT_INTERVENTION_VALUE)

    if steer_modes is None:
        steer_modes = ["none"]

    base_root = Path(results_root) if results_root else Path(__file__).parent / "results"
    all_results: dict = {}

    def _eval_ck(msg: str) -> None:
        _memory_checkpoint(msg, log_host_ram=eval_log_host_memory)

    def _eval_mem_line(msg: str) -> None:
        """[eval-mem] log line without forcing gc.collect (use between tight loops)."""
        extra = ""
        if eval_log_host_memory:
            hm = _format_host_memory_stats()
            if hm:
                extra = f"  |  {hm}"
        print(f"[eval-mem] {msg}{extra}", flush=True)

    if test_dataset_holder is not None and len(test_dataset_holder) > 0:
        _cc_td = test_dataset_holder[0]
    else:
        _cc_td = test_dataset

    _eval_ck("run_codecontests_testset_evaluation_for_cbm: start (before code_contests test set)")

    if _cc_td is None:
        return {}

    print(f"\n{'='*60}")
    print(f" code_contests test set  ({len(_cc_td)} problems)")
    print(f"{'='*60}")
    concept_index = {c: idx for idx, c in enumerate(concept_set)}

    for steer_mode in steer_modes:
        mode_label = f"{model_label}-{steer_mode}"
        cc_dir = base_root / "code_contests" / mode_label
        cc_dir.mkdir(parents=True, exist_ok=True)
        out_path = cc_dir / f"l{layer_idx}-seed{seed}-{run_id}.jsonl"
        cc_total_prompts = sum(1 for idx in range(len(_cc_td)) if str(_cc_td[idx].get("description", "")).strip())

        print(f"\n[{steer_mode}] Generating solutions for code_contests test set ...", flush=True)
        cc_generation_start_t = time.perf_counter()
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
            flush_batch_size = len(pending_prompts)
            flush_start_t = time.perf_counter()
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
            flush_elapsed = time.perf_counter() - flush_start_t
            cc_done = len(rows)
            cc_left = max(0, cc_total_prompts - cc_done)
            print(
                f"[eval-timing] code_contests/{steer_mode}: flush_generation="
                f"{_fmt_seconds(flush_elapsed)} | batch={flush_batch_size} | done={cc_done}/{cc_total_prompts} | left={cc_left}",
                flush=True,
            )
            del generated

        device = next(preLM.parameters()).device

        for i in tqdm(range(len(_cc_td)), desc=f"cc/{steer_mode}", disable=not display):
            problem = _cc_td[i]
            description = problem["description"].strip()
            if not description:
                continue

            cf_tags = problem["cf_tags"]
            intervene = _resolve_intervene(
                steer_mode,
                description,
                cf_tags,
                preLM,
                cbl,
                tokenizer,
                device,
                concept_set,
                steer_value,
            )
            prompt = _format_code_generation_prompt(tokenizer, description, language="python")
            eval_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                if getattr(cbl, "cbl_layer_idx", -1) == -1:
                    eval_features = preLM(
                        input_ids=eval_enc["input_ids"],
                        attention_mask=eval_enc["attention_mask"],
                    ).last_hidden_state
                    eval_llama_logits = (
                        F.linear(eval_features, llama_vocab_weight) if llama_vocab_weight is not None else None
                    )
                    eval_concepts, _, _, _ = cbl(eval_features.float(), llama_logits=eval_llama_logits)
                else:
                    eval_concepts, _, _, _, _, _ = cbl.forward_full(
                        preLM,
                        eval_enc["input_ids"],
                        eval_enc["attention_mask"],
                        llama_vocab_weight=llama_vocab_weight,
                    )
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
            pending_meta_rows.append(
                {
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
                }
            )
            if len(pending_prompts) >= prompt_batch_size:
                _flush_cc_batch()

        _flush_cc_batch()
        cc_generation_elapsed = time.perf_counter() - cc_generation_start_t

        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        print(f"  Saved {len(rows)} solutions → {out_path}", flush=True)
        print(f"[eval-timing] code_contests/{steer_mode}: generation={_fmt_seconds(cc_generation_elapsed)}", flush=True)
        _eval_mem_line(f"code_contests/{steer_mode}: jsonl written; computing concept-tag metrics ...")

        cc_testing_start_t = time.perf_counter()
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
                f"cos_cubed={topk_metrics['cosine_cubed']:.4f}",
            )

        cc_testing_elapsed = time.perf_counter() - cc_testing_start_t
        print(f"[eval-timing] code_contests/{steer_mode}: testing={_fmt_seconds(cc_testing_elapsed)}", flush=True)

        log_payload = {
            f"cc/{steer_mode}/solutions_written": len(rows),
            f"cc/{steer_mode}/output_path": str(out_path),
        }
        log_payload.update(concept_acc_metrics)
        if wandb.run is not None:
            safe_wandb_log(log_payload)

        # Per-mode generations payload kept in memory so the caller can run downstream
        # evals (perplexity, llama.cpp judge, RM) AFTER preLM/cbl are freed from GPU.
        generations_payload = {
            "output_path": str(out_path),
            "raw_outputs": [r["raw_output"] for r in rows],
            "extracted_codes": [r["extracted_code"] for r in rows],
            "cf_tags_per_problem": [list(r["cf_tags"]) for r in rows],
            "problem_names": [r["problem_name"] for r in rows],
            "concept_metrics": concept_acc_metrics,
        }
        all_results[f"cc/{steer_mode}"] = {
            "output_path": str(out_path),
            **concept_acc_metrics,
            "generations": generations_payload,
        }
        del rows, concept_pred_rows, concept_target_rows

    print(f"[eval-timing] code_contests_all_total={_fmt_seconds(time.perf_counter() - eval_start_t)}", flush=True)
    del concept_index
    _eval_ck("run_codecontests_testset_evaluation_for_cbm: done (concept_index dropped)")

    if test_dataset_holder is not None:
        test_dataset_holder[0] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_results


def run_livecodebench_benchmark_generation_for_cbm(
    preLM,
    cbl,
    tokenizer,
    concept_set: List[str],
    seed: int = 42,
    batch_size: int = 4,
    model_label: str = "CBM-Llama3-code_contests",
    layer_idx: int = -1,
    run_id=None,
    results_root=None,  # kept for API symmetry; not used for LCB outputs
    llama_vocab_weight=None,
    display: bool = True,
    # ── Steering ──────────────────────────────────────────────────────────────
    steer_modes: Optional[List[str]] = None,
    steer_value: Optional[float] = None,
    keep_other_concepts: bool = False,
    # ── LiveCodeBench ─────────────────────────────────────────────────────────
    livecodebench_release: str = "release_v6",
    lcb_n_samples: int = 10,
    lcb_temperature: float = 0.2,
    lcb_top_p: float = 0.95,
    lcb_top_k: int = 50,
    lcb_max_new_tokens: int = 2000,
    lcb_repetition_penalty: float = 1.05,
    lcb_max_retries: int = 0,  # unused placeholder; kept to avoid breaking older experiments
    lcb_prompt_batch_size: Optional[int] = None,  # unused; caller controls prompt batch size via batch_size
    print_extracted_code_preview: bool = False,
    extracted_preview_chars: int = 420,
    eval_log_host_memory: bool = False,
) -> dict:
    """Run only LiveCodeBench benchmark generation + write eval lock JSON."""
    import json

    preLM.eval()
    cbl.eval()
    set_seed(seed)
    eval_start_t = time.perf_counter()

    if run_id is None:
        run_id = wandb.run.id if wandb.run is not None else "norun"

    if steer_value is None:
        steer_value = float(DEFAULT_INTERVENTION_VALUE)
    if steer_modes is None:
        steer_modes = ["none"]

    lcb_repo = Path(__file__).parent / "LiveCodeBench"
    all_results: dict = {}

    def _eval_ck(msg: str) -> None:
        _memory_checkpoint(msg, log_host_ram=eval_log_host_memory)

    def _eval_mem_line(msg: str) -> None:
        extra = ""
        if eval_log_host_memory:
            hm = _format_host_memory_stats()
            if hm:
                extra = f"  |  {hm}"
        print(f"[eval-mem] {msg}{extra}", flush=True)

    _eval_ck("run_livecodebench_benchmark_generation_for_cbm: before _import_lcb")
    load_code_generation_dataset, codegen_metrics, extract_instance_results = _import_lcb()
    _eval_ck("run_livecodebench_benchmark_generation_for_cbm: _import_lcb done; calling load_code_generation_dataset")

    lcb_dataset_start_t = time.perf_counter()
    benchmark = load_code_generation_dataset(livecodebench_release)
    lcb_dataset_elapsed = time.perf_counter() - lcb_dataset_start_t
    print(f"  Loaded {len(benchmark)} LCB problems", flush=True)
    print(f"[eval-timing] livecodebench: dataset_loading={_fmt_seconds(lcb_dataset_elapsed)}", flush=True)
    _eval_ck("run_livecodebench_benchmark_generation_for_cbm: benchmark loaded")

    device = next(preLM.parameters()).device

    for steer_mode in steer_modes:
        mode_repr = f"{model_label}-{steer_mode}"
        lcb_out_dir = Path(lcb_repo) / "output" / mode_repr / str(run_id)
        lcb_out_dir.mkdir(parents=True, exist_ok=True)
        lcb_out_path = lcb_out_dir / f"codegeneration_{lcb_n_samples}_{lcb_temperature}.json"
        lcb_eval_path = lcb_out_dir / f"codegeneration_{lcb_n_samples}_{lcb_temperature}_eval.json"
        lcb_eval_all_path = lcb_out_dir / f"codegeneration_{lcb_n_samples}_{lcb_temperature}_eval_all.json"

        print(f"\n[{steer_mode}] Generating {lcb_n_samples} solutions × {len(benchmark)} LCB problems ...")
        lcb_generation_start_t = time.perf_counter()

        all_outputs: List[List[str]] = []
        all_extracted: List[List[str]] = []
        benchmark_sorted = sorted(benchmark, key=lambda x: x.question_id)

        lcb_total_prompts = len(benchmark_sorted)
        prompt_batch_size = max(1, int(batch_size))
        pending_lcb_prompts: List[str] = []
        pending_lcb_intervenes: List[Optional[List[float]]] = []
        pending_lcb_headings: List[str] = []

        def _flush_lcb_batch():
            if not pending_lcb_prompts:
                return
            flush_batch_size = len(pending_lcb_prompts)
            flush_start_t = time.perf_counter()
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
            flush_elapsed = time.perf_counter() - flush_start_t
            lcb_done = len(all_outputs)
            lcb_left = max(0, lcb_total_prompts - lcb_done)
            print(
                f"[eval-timing] livecodebench/{steer_mode}: flush_generation="
                f"{_fmt_seconds(flush_elapsed)} | batch={flush_batch_size} | done={lcb_done}/{lcb_total_prompts} | left={lcb_left}",
                flush=True,
            )
            del generated

        for problem in tqdm(benchmark_sorted, desc=f"lcb/{steer_mode}", disable=not display):
            text_for_steer = problem.question_content
            problem_id = str(problem.question_id)
            mapped_cf_tags = CLEANED_TAGS_MAP.get(problem_id, {}).get("tags", [])
            if problem_id not in CLEANED_TAGS_MAP and steer_mode == "groundtruth":
                print(
                    f"[warn] Missing CLEANED_TAGS_MAP entry for LCB question_id={problem_id}; using empty tags (no steering for this sample).",
                    flush=True,
                )
            intervene = _resolve_intervene(
                steer_mode,
                text_for_steer,
                mapped_cf_tags,
                preLM,
                cbl,
                tokenizer,
                device,
                concept_set,
                steer_value,
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
        lcb_generation_elapsed = time.perf_counter() - lcb_generation_start_t
        _eval_mem_line(
            f"LCB steer_mode={steer_mode!r}: generation loop finished ({len(all_outputs)} prompt batches)"
        )
        print(f"[eval-timing] livecodebench/{steer_mode}: generation={_fmt_seconds(lcb_generation_elapsed)}", flush=True)
        _eval_ck(f"LCB/{steer_mode}: before building save_results JSON")

        save_results = [
            problem.insert_output(outputs, codes)
            for problem, outputs, codes in zip(benchmark_sorted, all_outputs, all_extracted)
        ]
        with open(lcb_out_path, "w") as f:
            json.dump(save_results, f, indent=4)
        print(f"  Saved LCB outputs → {lcb_out_path}", flush=True)

        lock_path = lcb_out_path.with_name(lcb_out_path.name.replace(".json", "_eval.lock.json"))
        lock_payload = {
            "status": "pending_eval",
            "created_at_unix": time.time(),
            "run_id": str(run_id),
            "steer_mode": steer_mode,
            "model_repr": mode_repr,
            "livecodebench_release": livecodebench_release,
            "lcb_output_path": str(lcb_out_path),
            "lcb_eval_path": str(lcb_eval_path),
            "lcb_eval_all_path": str(lcb_eval_all_path),
        }
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump(lock_payload, f, indent=2)
        print(f"  Wrote LCB eval lock → {lock_path}", flush=True)

        # Generation-only mode: only write outputs + locks. Actual grading happens in eval_lcb_from_locks.py.
        log_payload = {
            f"lcb/{steer_mode}/generation_only": 1,
            f"lcb/{steer_mode}/n_samples": lcb_n_samples,
            f"lcb/{steer_mode}/temperature": lcb_temperature,
            f"lcb/{steer_mode}/steer_value": steer_value if steer_mode == "groundtruth" else 0.0,
            f"lcb/{steer_mode}/steer_topk": 0,
            f"lcb/{steer_mode}/release": livecodebench_release,
            f"lcb/{steer_mode}/output_path": str(lcb_out_path),
            f"lcb/{steer_mode}/eval_lock_path": str(lock_path),
        }
        if wandb.run is not None:
            safe_wandb_log(log_payload)
        all_results[f"lcb/{steer_mode}"] = {
            "generation_only": True,
            "output_path": str(lcb_out_path),
            "eval_lock_path": str(lock_path),
        }

        # Release large lists before next steer_mode.
        del save_results
        del all_outputs, all_extracted, benchmark_sorted
        del pending_lcb_prompts, pending_lcb_intervenes, pending_lcb_headings
        try:
            del _flush_lcb_batch
        except Exception:
            pass

    print(
        f"[eval-timing] all_code_evaluations_total={_fmt_seconds(time.perf_counter() - eval_start_t)}",
        flush=True,
    )

    del benchmark, load_code_generation_dataset, codegen_metrics, extract_instance_results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _eval_ck("run_livecodebench_benchmark_generation_for_cbm: done")

    return all_results





def evaluate_saved_livecodebench_generation(
    lcb_output_path: str,
    *,
    livecodebench_release: str,
    lcb_num_process_evaluate: int = 4,
    lcb_timeout: int = 6,
    lcb_eval_path: Optional[str] = None,
    lcb_eval_all_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a previously saved LCB generation JSON and write eval artifacts."""
    output_path = Path(lcb_output_path)
    if not output_path.is_file():
        raise FileNotFoundError(f"LCB generation JSON not found: {output_path}")

    if lcb_eval_path is None:
        lcb_eval_path = str(output_path.with_name(output_path.name.replace(".json", "_eval.json")))
    if lcb_eval_all_path is None:
        lcb_eval_all_path = str(output_path.with_name(output_path.name.replace(".json", "_eval_all.json")))

    load_code_generation_dataset, codegen_metrics, extract_instance_results = _import_lcb()
    benchmark = load_code_generation_dataset(livecodebench_release)
    benchmark_sorted = sorted(benchmark, key=lambda x: x.question_id)

    with open(output_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    if len(saved) != len(benchmark_sorted):
        raise ValueError(
            f"Saved generations count ({len(saved)}) != benchmark count ({len(benchmark_sorted)})."
        )

    all_outputs: List[List[str]] = []
    all_extracted: List[List[str]] = []
    for row in saved:
        outputs = row.get("output_list") or row.get("outputs") or []
        if not isinstance(outputs, list):
            outputs = []
        codes = row.get("code_list") or row.get("codes")
        if isinstance(codes, list):
            extracted = [str(x) for x in codes]
        else:
            extracted = [_extract_code_from_output(str(x)) for x in outputs]
        all_outputs.append([str(x) for x in outputs])
        all_extracted.append(extracted)

    eval_samples = [p.get_evaluation_sample() for p in benchmark_sorted]
    metrics, results_dict, metadatas = codegen_metrics(
        eval_samples,
        all_extracted,
        num_process_evaluate=int(lcb_num_process_evaluate),
        timeout=int(lcb_timeout),
    )
    graded = extract_instance_results(results_dict)
    save_eval_results = [
        p.insert_output_evaluation(o, c, g, metadata=m)
        for p, o, c, g, m in zip(benchmark_sorted, all_outputs, all_extracted, graded, metadatas)
    ]

    with open(lcb_eval_all_path, "w", encoding="utf-8") as f:
        json.dump(save_eval_results, f, indent=4)
    with open(lcb_eval_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    return {
        "pass@1": metrics.get("pass@1", float("nan")) if isinstance(metrics, dict) else float("nan"),
        "pass@5": metrics.get("pass@5", float("nan")) if isinstance(metrics, dict) else float("nan"),
        "lcb_eval_path": str(lcb_eval_path),
        "lcb_eval_all_path": str(lcb_eval_all_path),
    }





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

def _read_cbl_meta(cbl_path: str) -> dict:
    """Read the sidecar ``<cbl_path>.meta.json`` written by training.

    Returns ``{"cbl_layer_idx": int, "use_residual": bool}``. If the sidecar
    does not exist (legacy last-layer-mode checkpoint), returns the defaults
    ``{"cbl_layer_idx": -1, "use_residual": True}`` which exactly matches the
    pre-intermediate-mode behavior.
    """
    meta_path = cbl_path + ".meta.json"
    if not os.path.isfile(meta_path):
        return {"cbl_layer_idx": -1, "use_residual": True}
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return {
        "cbl_layer_idx": int(meta.get("cbl_layer_idx", -1)),
        "use_residual": bool(meta.get("use_residual", True)),
    }


def load_model_and_cbl(
    peft_path, cbl_path, config, concept_set, tokenizer,
    discrimination_loss, residual_dim, device,
):
    """Load preLM + CBL using sidecar mode metadata.

    Reads ``<cbl_path>.meta.json`` to recover ``cbl_layer_idx`` and
    ``use_residual`` so the constructed module's shape matches the checkpoint
    (the ``proj`` layer is allocated iff intermediate mode). Loading is then
    performed with ``strict=True`` so any future drift fails loudly instead of
    silently dropping ``proj.*`` and routing eval through the never-trained
    ``cbl.fc`` head.

    When the checkpoint is in intermediate mode (``cbl_layer_idx >= 0``) the
    ``llama_vocab_weight`` is required for any forward through ``forward_full``
    or the bottleneck generation paths; we eagerly load it here and return it
    as the third tuple element.

    Returns ``(preLM, cbl, llama_vocab_weight)``. ``llama_vocab_weight`` is
    ``None`` for last-layer-mode checkpoints (callers can still pass their own
    if ``--add_llama_logits`` was used at train time).
    """
    preLM = LlamaModel.from_pretrained(
        LCB_LLAMA3_INSTRUCT_MODEL_ID, torch_dtype=torch.bfloat16,
    ).to(device)
    preLM.load_adapter(peft_path)
    preLM.eval()

    meta = _read_cbl_meta(cbl_path)
    cbl_layer_idx = meta["cbl_layer_idx"]
    use_residual = meta["use_residual"]
    print(
        f"[load_model_and_cbl] {cbl_path}: "
        f"cbl_layer_idx={cbl_layer_idx}, use_residual={use_residual} "
        f"(from {'sidecar' if os.path.isfile(cbl_path + '.meta.json') else 'defaults'})"
    )

    if discrimination_loss > 0:
        cbl = CBL(
            config, len(concept_set), tokenizer,
            cbl_layer_idx=cbl_layer_idx, use_residual=use_residual,
        ).to(device)
    else:
        cbl = CBLResidual(
            config, len(concept_set), residual_dim, tokenizer,
            cbl_layer_idx=cbl_layer_idx, use_residual=use_residual,
        ).to(device)

    state_dict = torch.load(cbl_path, map_location=device)
    cbl.load_state_dict(state_dict, strict=True)
    cbl.eval()

    llama_vocab_weight = None
    if cbl_layer_idx >= 0:
        llama_vocab_weight = get_llama_vocab_weight(device)

    return preLM, cbl, llama_vocab_weight


# ═══════════════════════════════════════════════════════════════
# Per-solution llama.cpp Judge (multi-label vs ground-truth cf_tags)
# ═══════════════════════════════════════════════════════════════

def _llamacpp_build_raw_prompt(text: str, concepts: list[str]) -> str:
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    nl = "\n"
    system_text = (
        "You are a strict multi-label classifier for coding tasks. "
        "Given generated code or a coding solution, output ONLY a single line containing a comma-separated "
        "list of algorithm or concept names copied verbatim from OPTIONS. "
        "If no option applies, output exactly the word: none. "
        "No explanation, no preamble, no bullets."
    )
    assistant_prefill = "<think>\n\n</think>\n\n"
    opts_block = "\n".join(f"- {c}" for c in concepts)
    user_text = (
        "From OPTIONS, list ALL algorithm or programming concept labels that apply to the approach, "
        "technique, or topic reflected in GENERATED_TEXT (code, pseudocode, or solution text). "
        "Return a comma-separated list using labels copied verbatim from OPTIONS. "
        "If nothing applies, return exactly: none.\n\n"
        f"OPTIONS:\n{opts_block}\n\n"
        f"GENERATED_TEXT:\n{text}\n\n"
        "Answer (comma-separated subset of OPTIONS, or 'none', nothing else):"
    )
    return (
        f"{im_start}system{nl}{system_text}{im_end}{nl}"
        f"{im_start}user{nl}{user_text}{im_end}{nl}"
        f"{im_start}assistant{nl}{assistant_prefill}"
    )


def _llamacpp_parse_output(output: str, concepts: list[str]) -> list[str]:
    """Return the subset of `concepts` predicted by the judge, in order of first occurrence.

    The judge is prompted to return a comma-separated list of labels copied verbatim from
    OPTIONS, or the word ``none``. We accept some sloppiness: case-insensitive matching and
    fuzzy substring fallback, mirroring the previous single-label behaviour but applied to
    every comma/semicolon-separated piece. Empty input or ``none`` yields an empty list.
    """
    first_line = next((ln.strip() for ln in str(output).splitlines() if ln.strip()), "")
    if not first_line or first_line.strip().lower() == "none":
        return []

    parts = [p.strip() for p in re.split(r"[,;]", first_line) if p.strip()]
    if not parts:
        parts = [first_line]

    seen: set[str] = set()
    matched: list[str] = []

    def _add(label: str) -> None:
        if label not in seen:
            seen.add(label)
            matched.append(label)

    for p in parts:
        if p.lower() == "none":
            continue
        if p in concepts:
            _add(p)
            continue
        exact_ci = next((c for c in concepts if c.lower() == p.lower()), None)
        if exact_ci is not None:
            _add(exact_ci)
            continue
        fuzzy = next(
            (c for c in concepts if p.lower() in c.lower() or c.lower() in p.lower()),
            None,
        )
        if fuzzy is not None:
            _add(fuzzy)
    return matched


def _multi_label_set_metrics(pred: Sequence[str], gold: Sequence[str]) -> Dict[str, float]:
    """Per-problem precision / recall / F1 / IoU on label sets."""
    pred_set = {p for p in pred if p}
    gold_set = {g for g in gold if g}
    if not gold_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0}
    if not gold_set:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "iou": 0.0}
    if not pred_set:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
    inter = len(pred_set & gold_set)
    union = len(pred_set | gold_set)
    precision = inter / len(pred_set)
    recall = inter / len(gold_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = inter / union if union > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def run_llamacpp_judge_per_solution(
    generations_by_mode: Dict[str, dict],
    concept_set: List[str],
    *,
    model_repo_id: str = "unsloth/Qwen3.5-27B-GGUF",
    model_filename: str = "Qwen3.5-27B-Q8_0.gguf",
    cache_dir: Optional[str] = None,
    n_ctx: int = 2048,
    max_tokens: int = 128,
    repeat_penalty: float = 1.15,
    temperature: float = 0.1,
    use_extracted_code: bool = True,
    audit_jsonl_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Multi-label llama.cpp judge over per-solution generations from code_contests.

    Each ``generations_by_mode[steer_mode]`` payload comes from
    :func:`run_codecontests_testset_evaluation_for_cbm` and provides parallel lists of
    ``raw_outputs`` / ``extracted_codes`` / ``cf_tags_per_problem`` / ``problem_names``.
    For every problem we ask the judge to list ALL applicable concepts and score the
    prediction set against ``cf_tags`` with precision / recall / F1 / IoU.

    Loading / unloading the GGUF model uses the same pattern as the previous
    steerability judge: prefer local cache, fall back to download, ``del`` and ``gc``
    once done. The actual ``llm(prompt, ...)`` call is unchanged.
    """
    if not generations_by_mode:
        return {}

    try:
        Llama = importlib.import_module("llama_cpp").Llama
    except Exception as import_err:
        print(f"[WARN] llama_cpp not available (install llama-cpp-python): {import_err}")
        return {}

    print(
        f"Loading llama.cpp judge | repo={model_repo_id} file={model_filename} "
        f"n_ctx={n_ctx} max_tokens={max_tokens} temp={temperature}"
    )
    base_kwargs = {
        "repo_id": model_repo_id,
        "filename": model_filename,
        "n_gpu_layers": -1,
        "n_ctx": n_ctx,
        "verbose": False,
    }
    if cache_dir:
        base_kwargs["cache_dir"] = cache_dir
    try:
        llm = Llama.from_pretrained(local_files_only=True, **base_kwargs)
        print(f"[cache] llama.cpp local-only load succeeded (cache_dir={cache_dir})")
    except Exception as local_err:
        print(f"[cache] llama.cpp local miss, downloading model: {local_err}")
        llm = Llama.from_pretrained(**base_kwargs)

    results_by_mode: Dict[str, Any] = {}
    try:
        for steer_mode, payload in generations_by_mode.items():
            if not payload:
                continue
            raw_outputs = payload.get("raw_outputs") or []
            extracted_codes = payload.get("extracted_codes") or []
            cf_tags_per_problem = payload.get("cf_tags_per_problem") or []
            problem_names = payload.get("problem_names") or [
                f"problem_{i}" for i in range(len(raw_outputs))
            ]
            n = len(raw_outputs)
            if n == 0:
                continue

            judge_rows: List[Dict[str, Any]] = []
            agg_p, agg_r, agg_f1, agg_iou = [], [], [], []

            for i in tqdm(
                range(n),
                desc=f"llama.cpp judge cc/{steer_mode}",
            ):
                if use_extracted_code and i < len(extracted_codes) and extracted_codes[i]:
                    text = extracted_codes[i]
                else:
                    text = raw_outputs[i] if i < len(raw_outputs) else ""
                gold = list(cf_tags_per_problem[i]) if i < len(cf_tags_per_problem) else []

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
                    print(
                        f"[WARN] llama.cpp judge failed at cc/{steer_mode} idx={i} "
                        f"problem={problem_names[i] if i < len(problem_names) else '?'}: {e}"
                    )
                    raw = ""

                pred_labels = _llamacpp_parse_output(raw, concept_set)
                row_metrics = _multi_label_set_metrics(pred_labels, gold)
                agg_p.append(row_metrics["precision"])
                agg_r.append(row_metrics["recall"])
                agg_f1.append(row_metrics["f1"])
                agg_iou.append(row_metrics["iou"])

                judge_rows.append(
                    {
                        "steer_mode": steer_mode,
                        "problem_idx": i,
                        "problem_name": problem_names[i] if i < len(problem_names) else f"problem_{i}",
                        "gold_cf_tags": gold,
                        "pred_labels": pred_labels,
                        "raw_output": raw,
                        **row_metrics,
                    }
                )

            mode_metrics = {
                f"cc/{steer_mode}/llamacpp_judge_precision": float(np.mean(agg_p)),
                f"cc/{steer_mode}/llamacpp_judge_recall": float(np.mean(agg_r)),
                f"cc/{steer_mode}/llamacpp_judge_f1": float(np.mean(agg_f1)),
                f"cc/{steer_mode}/llamacpp_judge_iou": float(np.mean(agg_iou)),
                f"cc/{steer_mode}/llamacpp_judge_total": int(len(agg_p)),
            }
            print(
                f"  cc/{steer_mode}: llamacpp_judge "
                f"precision={mode_metrics[f'cc/{steer_mode}/llamacpp_judge_precision']:.4f} "
                f"recall={mode_metrics[f'cc/{steer_mode}/llamacpp_judge_recall']:.4f} "
                f"f1={mode_metrics[f'cc/{steer_mode}/llamacpp_judge_f1']:.4f} "
                f"iou={mode_metrics[f'cc/{steer_mode}/llamacpp_judge_iou']:.4f} "
                f"(n={mode_metrics[f'cc/{steer_mode}/llamacpp_judge_total']})"
            )
            safe_wandb_log(mode_metrics)

            audit_path: Optional[str] = None
            output_path = payload.get("output_path")
            if output_path:
                p = Path(output_path)
                audit_path = str(p.with_name(p.stem + ".judge.jsonl"))
            elif audit_jsonl_root:
                Path(audit_jsonl_root).mkdir(parents=True, exist_ok=True)
                audit_path = str(Path(audit_jsonl_root) / f"{steer_mode}.judge.jsonl")
            if audit_path:
                with open(audit_path, "w", encoding="utf-8") as f:
                    for row in judge_rows:
                        f.write(json.dumps(row) + "\n")
                print(f"  Saved judge audit → {audit_path}")

            results_by_mode[steer_mode] = {
                "metrics": mode_metrics,
                "audit_path": audit_path,
                "rows": judge_rows,
            }
    finally:
        try:
            del llm
        except NameError:
            pass
        gc.collect()

    return results_by_mode


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
            if getattr(cbl, "cbl_layer_idx", -1) == -1:
                features = preLM(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                ).last_hidden_state
                llama_logits = F.linear(features, llama_vocab_weight) if llama_vocab_weight is not None else None
                if llama_logits is not None:
                    concepts, _, _, _ = cbl(features.float(), llama_logits=llama_logits)
                else:
                    concepts, _, _, _ = cbl(features.float())
            else:
                concepts, _, _, _, _, _ = cbl.forward_full(
                    preLM,
                    batch["input_ids"],
                    batch["attention_mask"],
                    llama_vocab_weight=llama_vocab_weight,
                )
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
    """Print and log top tokens per concept neuron and sparsity.

    Last-layer mode: top *vocab tokens* per concept (rows of ``cbl.fc``).
    Intermediate mode: ``cbl.fc`` is unused for the model output; the analogous
    artifact is ``cbl.proj`` (Linear(C+U, hidden_size)). Reporting top *hidden-state
    dimensions* there isn't decodable as tokens, so we just log sparsity in that
    case (still on the concept slice of the projection weight).
    """
    if getattr(cbl, "cbl_layer_idx", -1) == -1:
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
                    f"{tokenizer.decode(top_ids[j], clean_up_tokenization_spaces=False)}"
                )
        sparsity = (w > 1e-6).count_nonzero() / w.numel()
    else:
        print("[weight-analysis] cbl_layer_idx >= 0: skipping top-token listing "
              "(proj outputs hidden dims, not vocab); reporting sparsity only.")
        w = cbl.proj.weight.data[:, : len(concept_set)].T
        sparsity = (w > 1e-6).count_nonzero() / w.numel()
    print(f"Sparsity of concept weight matrix: {sparsity}")
    safe_wandb_log({"concept_weight_sparsity": sparsity})


# ═══════════════════════════════════════════════════════════════
# Perplexity (computed from pre-generated texts; no generator helper).
# ═══════════════════════════════════════════════════════════════

def compute_perplexity(texts: list[str]) -> dict:
    """Compute perplexity (under-30 tokens and all tokens) from pre-generated texts.

    This function loads a fresh LLM via the ``evaluate`` library, so the training
    model should be freed from GPU beforehand.
    """
    results = {}

    short_texts = [p for p in texts if len(p.split()) <= 30]

    def _compute_ppl(predictions: list[str]):
        if len(predictions) == 0:
            return float("nan")

        model_id = LCB_LLAMA3_INSTRUCT_MODEL_ID
        max_length = 100
        ppl_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ppl_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        if ppl_tokenizer.pad_token is None:
            ppl_tokenizer.pad_token = ppl_tokenizer.eos_token

        ppl_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=(torch.bfloat16 if ppl_device.type == "cuda" else torch.float32),
        ).to(ppl_device)
        ppl_model.eval()

        sample_ppls = []
        with torch.no_grad():
            for text in tqdm(predictions, desc="Perplexity scoring", leave=False):
                encoded = ppl_tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                input_ids = encoded["input_ids"].to(ppl_device)
                attention_mask = encoded["attention_mask"].to(ppl_device)
                if input_ids.shape[1] < 2:
                    continue

                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                out = ppl_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                sample_ppls.append(float(torch.exp(out.loss.float()).cpu().item()))

        if len(sample_ppls) == 0:
            mean_ppl = float("nan")
        else:
            mean_ppl = float(np.mean(sample_ppls))

        del ppl_model, ppl_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return mean_ppl

    if short_texts:
        ppl_short = _compute_ppl(short_texts)
        print(f"Perplexity (under 30 tokens): {ppl_short}")
        safe_wandb_log({"perplexity_under_30_tokens": ppl_short})
        results["perplexity_under_30_tokens"] = ppl_short
    else:
        print("No generated texts under 30 tokens to compute perplexity.")
        safe_wandb_log({"perplexity_under_30_tokens": None})

    ppl_all = _compute_ppl(texts)
    print(f"Perplexity (all tokens): {ppl_all}")
    safe_wandb_log({"perplexity_all_tokens": ppl_all})
    results["perplexity_all_tokens"] = ppl_all

    return results


# ═══════════════════════════════════════════════════════════════
# RM (Reward Model) Metrics
# ═══════════════════════════════════════════════════════════════

RM_USER_RELEVANCE_MULTI = (
    "Write code that reflects the algorithm(s) or concept(s) named: {concepts}. "
    "It may also reflect other concepts."
)
RM_USER_GRAMMAR = (
    "Write code that is syntactically valid (must parse/compile). "
    "It does not need to be optimal or efficient."
)
RM_USER_TOGETHER_MULTI = (
    "Write code that reflects the algorithm(s) or concept(s) named: {concepts} "
    "and is syntactically valid. It may also reflect other concepts."
)
RM_LOGIT_CLIP_MIN = -100.0
RM_LOGIT_CLIP_MAX = 100.0


def _format_concepts_for_rm(concepts: Sequence[str]) -> str:
    """Comma-join cf_tags for the RM user-turn templates."""
    return ", ".join(str(c).strip() for c in concepts if str(c).strip())


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


def _rm_score_grouped(
    rm_model,
    rm_tokenizer,
    rm_device,
    rm_batch_size: int,
    rm_max_text_len: int,
    texts: Sequence[str],
    user_turns: Sequence[str],
) -> List[float]:
    """Score (text, user_turn) pairs by grouping rows with identical user_turn into batches.

    The actual RM forward (``_raw_logits_for_texts``) is unchanged — we just minimize
    duplicate template formatting work and amortize the batch axis across rows that
    share the same user prompt (e.g. RM_USER_GRAMMAR is constant across all rows).
    """
    assert len(texts) == len(user_turns)
    out: List[float] = [float("nan")] * len(texts)
    if not texts:
        return out
    groups: Dict[str, List[int]] = {}
    for i, turn in enumerate(user_turns):
        groups.setdefault(turn, []).append(i)
    for turn, idxs in groups.items():
        sub_texts = [texts[i] for i in idxs]
        scores = _raw_logits_for_texts(
            rm_model, rm_tokenizer, sub_texts, turn,
            rm_device, rm_batch_size, rm_max_text_len,
        )
        for i, s in zip(idxs, scores):
            out[i] = float(s)
    return out


def run_rm_metrics_per_solution(
    generations_by_mode: Dict[str, dict],
    concept_set: List[str],
    rm_model,
    rm_tokenizer,
    rm_device,
    *,
    rm_batch_size: int = 0,
    rm_max_text_len: int = 500,
    use_extracted_code: bool = False,
    audit_jsonl_root: Optional[str] = None,
) -> Dict[str, Any]:
    """RM scoring (relevance / grammar / together) over per-solution test-set generations.

    Each ``generations_by_mode[steer_mode]`` payload comes from
    :func:`run_codecontests_testset_evaluation_for_cbm`. For each problem the user-turn
    for relevance / together is built from that problem's ``cf_tags`` joined by commas
    via :data:`RM_USER_RELEVANCE_MULTI` / :data:`RM_USER_TOGETHER_MULTI`.
    Grammar uses the concept-agnostic :data:`RM_USER_GRAMMAR`.

    The reward-model forward + tokenization helpers (``_make_rm_formatted``,
    ``_raw_logits_for_texts``) and ``load_reward_model`` are reused unchanged.
    """
    if not generations_by_mode:
        return {}

    results_by_mode: Dict[str, Any] = {}

    for steer_mode, payload in generations_by_mode.items():
        if not payload:
            continue
        raw_outputs = payload.get("raw_outputs") or []
        extracted_codes = payload.get("extracted_codes") or []
        cf_tags_per_problem = payload.get("cf_tags_per_problem") or []
        problem_names = payload.get("problem_names") or [
            f"problem_{i}" for i in range(len(raw_outputs))
        ]

        keep: List[int] = []
        for i, tags in enumerate(cf_tags_per_problem):
            if tags and any(str(t).strip() for t in tags):
                keep.append(i)
        if not keep:
            print(f"  cc/{steer_mode}: no rows with cf_tags; skipping RM scoring.")
            continue

        if use_extracted_code:
            texts = [
                (extracted_codes[i] if i < len(extracted_codes) and extracted_codes[i]
                 else (raw_outputs[i] if i < len(raw_outputs) else ""))
                for i in keep
            ]
        else:
            texts = [raw_outputs[i] if i < len(raw_outputs) else "" for i in keep]

        rel_turns = [
            RM_USER_RELEVANCE_MULTI.format(concepts=_format_concepts_for_rm(cf_tags_per_problem[i]))
            for i in keep
        ]
        tog_turns = [
            RM_USER_TOGETHER_MULTI.format(concepts=_format_concepts_for_rm(cf_tags_per_problem[i]))
            for i in keep
        ]
        gram_turns = [RM_USER_GRAMMAR] * len(keep)

        print(f"  cc/{steer_mode}: scoring RM on {len(keep)} solutions ...", flush=True)
        rel = _rm_score_grouped(
            rm_model, rm_tokenizer, rm_device, rm_batch_size, rm_max_text_len, texts, rel_turns,
        )
        gram = _rm_score_grouped(
            rm_model, rm_tokenizer, rm_device, rm_batch_size, rm_max_text_len, texts, gram_turns,
        )
        tog = _rm_score_grouped(
            rm_model, rm_tokenizer, rm_device, rm_batch_size, rm_max_text_len, texts, tog_turns,
        )

        def _ms(xs: Sequence[float]):
            arr = np.array([x for x in xs if not (isinstance(x, float) and np.isnan(x))], dtype=np.float64)
            if arr.size == 0:
                return float("nan"), 0.0
            return float(arr.mean()), float(arr.std()) if arr.size > 1 else 0.0

        r_m, r_s = _ms(rel)
        g_m, g_s = _ms(gram)
        t_m, t_s = _ms(tog)

        mode_metrics = {
            f"cc/{steer_mode}/rm_relevance_mean": r_m,
            f"cc/{steer_mode}/rm_relevance_std": r_s,
            f"cc/{steer_mode}/rm_grammar_mean": g_m,
            f"cc/{steer_mode}/rm_grammar_std": g_s,
            f"cc/{steer_mode}/rm_together_mean": t_m,
            f"cc/{steer_mode}/rm_together_std": t_s,
            f"cc/{steer_mode}/rm_total_n": int(len(keep)),
        }
        safe_wandb_log(mode_metrics)
        print(
            f"  cc/{steer_mode}: rm_relevance_mean={r_m:.4f} rm_grammar_mean={g_m:.4f} "
            f"rm_together_mean={t_m:.4f} (n={len(keep)})"
        )

        per_problem: List[Dict[str, Any]] = []
        for k, i in enumerate(keep):
            per_problem.append(
                {
                    "steer_mode": steer_mode,
                    "problem_idx": int(i),
                    "problem_name": problem_names[i] if i < len(problem_names) else f"problem_{i}",
                    "cf_tags": list(cf_tags_per_problem[i]),
                    "rm_relevance_logit": rel[k],
                    "rm_grammar_logit": gram[k],
                    "rm_together_logit": tog[k],
                }
            )

        audit_path: Optional[str] = None
        output_path = payload.get("output_path")
        if output_path:
            p = Path(output_path)
            audit_path = str(p.with_name(p.stem + ".rm.jsonl"))
        elif audit_jsonl_root:
            Path(audit_jsonl_root).mkdir(parents=True, exist_ok=True)
            audit_path = str(Path(audit_jsonl_root) / f"{steer_mode}.rm.jsonl")
        if audit_path:
            with open(audit_path, "w", encoding="utf-8") as f:
                for row in per_problem:
                    f.write(json.dumps(row) + "\n")
            print(f"  Saved RM audit → {audit_path}")

        results_by_mode[steer_mode] = {
            "metrics": mode_metrics,
            "audit_path": audit_path,
            "rows": per_problem,
        }

    return results_by_mode
