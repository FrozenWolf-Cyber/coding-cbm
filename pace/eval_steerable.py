"""Steerer-agnostic evaluation cascade for PaCE-CBM and the vector / transform
/ ODE steerers (CAA / ITI / RepE / LinAcT / MiMiC / ODESteer / …), plus the no-steer baseline.

Reuses every helper from ``eval_metrics`` it can; the only new piece is
``_generate_with_steerer_batched`` which calls
``LlamaForCausalLM.generate(...)`` with a hook attached at layer ``L``.

Per-solution downstream evals (perplexity, llama.cpp judge, RM scoring) are
imported and called *unchanged* — see ``eval_metrics.compute_perplexity``,
``run_llamacpp_judge_per_solution``, ``run_rm_metrics_per_solution``.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

import wandb

# Reused helpers from the existing eval pipeline.
from eval_metrics import (
    CLEANED_TAGS_MAP,
    _extract_code_from_output,
    _fmt_seconds,
    _format_code_generation_prompt,
    _format_host_memory_stats,
    _import_lcb,
    _memory_checkpoint,
    print_extracted_code_samples_preview,
    print_solution_question_and_extracted_code,
    safe_wandb_log,
    set_seed,
)
from utils import compute_multilabel_concept_metrics, eos_pooling

from .hook_steerer import HookSteerer, NoSteer, PaCECBMSteerer
from .intervention import (
    cf_tags_to_multihot,
    configure_steerer,
    expand_payload_for_n_samples,
)


def _eval_debug(enabled: bool, stage: str, message: str = "", **info: Any) -> None:
    """Structured stdout for ``--eval_debug`` (shapes, timings, stages)."""
    if not enabled:
        return
    parts = [f"[eval-debug] {stage}"]
    if message:
        parts.append(message)
    for k, v in info.items():
        parts.append(f"{k}={v}")
    print(" | ".join(parts), flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Concept-accuracy (PaCE-CBM only — vector steerers don't expose c_sparse)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_concept_accuracy_pace(
    *,
    pace_cbm,
    llm,
    test_loader,
    cf_concept_set: List[str],
    cf_offset: int,
    cf_size: int,
    device: torch.device,
    test_similarity_np,
    eval_debug: bool = False,
) -> Dict[str, float]:
    """EOS-pooled c_sparse[CF block] vs ground-truth multi-hot, top-k metrics."""
    print("eval concepts (PaCE-CBM cosine similarity)...")
    pace_cbm.eval()
    llm.eval()

    steerer = PaCECBMSteerer(pace_cbm)
    pred_chunks: list[Tensor] = []
    steerer.attach(llm)
    t_concept = time.perf_counter()
    _eval_debug(
        eval_debug,
        "concept_acc/start",
        n_batches=len(test_loader),
        device=str(device),
    )
    with steerer:
        steerer.configure_for_batch(cf_multihot=None)  # no intervention
        for bi, (batch, _) in enumerate(tqdm(test_loader, total=len(test_loader), desc="concept-acc")):
            batch = {k: v.to(device) for k, v in batch.items()}
            if eval_debug and bi == 0:
                _eval_debug(
                    eval_debug,
                    "concept_acc/first_batch",
                    input_ids=tuple(batch["input_ids"].shape),
                    attention_mask=tuple(batch["attention_mask"].shape),
                )
            t_step = time.perf_counter()
            llm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            if eval_debug and bi == 0:
                _eval_debug(
                    eval_debug,
                    "concept_acc/first_forward",
                    forward_s=_fmt_seconds(time.perf_counter() - t_step),
                    c_sparse=tuple(steerer.last_c_sparse.shape),
                )
            c_sparse = steerer.last_c_sparse  # (B, T, C_total)
            c_sparse_cf = c_sparse[:, :, cf_offset:cf_offset + cf_size]  # (B, T, C_cf)
            pooled = eos_pooling(c_sparse_cf, batch["attention_mask"]).detach().cpu()  # (B, C_cf)
            pred_chunks.append(pooled)

    pred = torch.cat(pred_chunks, dim=0)  # (N_test, C_cf)
    target = torch.tensor(test_similarity_np, dtype=torch.float32)  # (N_test, C_cf)
    if target.shape != pred.shape:
        print(
            f"[WARN] concept-acc shape mismatch: pred={tuple(pred.shape)} "
            f"target={tuple(target.shape)}; skipping."
        )
        return {}
    metrics = compute_multilabel_concept_metrics(
        prediction_scores=pred, target_scores=target, topk=(1, 5, 10),
    )
    payload = {
        "test_concept_top1_acc": metrics["top1_acc"],
        "test_concept_top5_acc": metrics["top5_acc"],
        "test_concept_top10_acc": metrics["top10_acc"],
        "test_concept_top1_iou": metrics["top1_iou"],
        "test_concept_top5_iou": metrics["top5_iou"],
        "test_concept_top10_iou": metrics["top10_iou"],
        "test_concept_cosine_raw": metrics["cosine_raw"],
        "test_concept_cosine_cubed": metrics["cosine_cubed"],
    }
    print(
        f"PaCE concept-acc: top1={payload['test_concept_top1_acc']:.4f} "
        f"top5={payload['test_concept_top5_acc']:.4f} "
        f"top10={payload['test_concept_top10_acc']:.4f} "
        f"cos={payload['test_concept_cosine_raw']:.4f}"
    )
    _eval_debug(
        eval_debug,
        "concept_acc/done",
        total_s=_fmt_seconds(time.perf_counter() - t_concept),
        pred_shape=tuple(pred.shape),
    )
    safe_wandb_log(payload)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Steerer-agnostic batched generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _generate_with_steerer_batched(
    *,
    steerer: HookSteerer,
    llm,
    tokenizer,
    prompts: List[str],
    cf_tags_per_prompt: Optional[Sequence[Sequence[str]]],
    cf_concepts: Sequence[str],
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    alpha: float,
    zero_other_concepts: bool,
    device: torch.device,
    use_cache: bool = True,
    eval_debug: bool = False,
) -> List[List[str]]:
    """Tokenize, repeat-interleave for n_samples, configure steerer, run
    ``llm.generate(...)`` with the hook attached, decode.

    ``use_cache``: passed to ``generate`` (default True).  Set False for slower
    generation that recomputes attention without incremental KV cache—sometimes
    used when debugging steering + cache interactions.
    """
    if not prompts:
        return []

    t_all = time.perf_counter()
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    tokenizer.padding_side = original_padding_side

    input_ids = enc["input_ids"]  # (B_prompt, T_pad) left-padded to max prompt len in batch
    attention_mask = enc["attention_mask"]  # (B_prompt, T_pad)
    prompt_width = input_ids.shape[1]  # T_pad — slice completions from here onward
    _eval_debug(
        eval_debug,
        "generate/tokenize",
        n_prompts=len(prompts),
        input_ids=tuple(input_ids.shape),
        attention_mask=tuple(attention_mask.shape),
        prompt_width=prompt_width,
        n_samples=n_samples,
        steerer=type(steerer).__name__,
        use_cache=use_cache,
    )
    if hasattr(steerer, "steer_vecs"):
        _eval_debug(
            eval_debug,
            "generate/vec_steerer",
            steer_vecs=tuple(steerer.steer_vecs.shape),
            method_name=getattr(steerer, "method_name", "?"),
            layer_idx=getattr(steerer, "layer_idx", "?"),
        )

    if n_samples > 1:
        # Expand batch for num_return_sequences: (B_prompt, T) -> (B_prompt*n_samples, T)
        input_ids = input_ids.repeat_interleave(n_samples, dim=0)
        attention_mask = attention_mask.repeat_interleave(n_samples, dim=0)
        _eval_debug(
            eval_debug,
            "generate/after_repeat_interleave",
            input_ids=tuple(input_ids.shape),
            attention_mask=tuple(attention_mask.shape),
        )

    # Stage payload at the original B granularity, then expand for n_samples.
    if not isinstance(steerer, NoSteer):
        cf_multihot = None
        if cf_tags_per_prompt is not None:
            cf_multihot = cf_tags_to_multihot(cf_tags_per_prompt, cf_concepts).to(device)
        _eval_debug(
            eval_debug,
            "generate/before_configure",
            alpha=alpha,
            zero_other_concepts=zero_other_concepts,
            cf_multihot=tuple(cf_multihot.shape) if cf_multihot is not None else None,
        )
        configure_steerer(
            steerer,
            cf_multihot=cf_multihot,
            cf_concepts=cf_concepts,
            alpha=alpha,
            zero_other_concepts=zero_other_concepts,
            device=device,
        )
        expand_payload_for_n_samples(steerer, n_samples)
        payload = getattr(steerer, "_payload", None)
        if eval_debug and isinstance(payload, dict):
            av = payload.get("add_vec")
            _eval_debug(
                eval_debug,
                "generate/after_configure",
                payload_keys=list(payload.keys()),
                add_vec_shape=tuple(av.shape) if isinstance(av, Tensor) else None,
            )

    steerer.attach(llm)
    with steerer:
        t_gen = time.perf_counter()
        # Use ``max_length = prompt + new-token budget`` only. Passing
        # ``max_new_tokens`` alongside the model default ``generation_config.max_length``
        # makes Transformers warn that both are set.
        total_max_length = int(input_ids.shape[1]) + int(max_new_tokens)
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=total_max_length,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=use_cache,
        )
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = eos_id
        gen_ids = llm.generate(**gen_kwargs)  # (B_total, T_pad + T_new); B_total = B_prompt * n_samples
        _eval_debug(
            eval_debug,
            "generate/llm.generate",
            elapsed_s=_fmt_seconds(time.perf_counter() - t_gen),
            gen_ids=tuple(gen_ids.shape),
            total_max_length=total_max_length,
            new_token_budget=max_new_tokens,
            temperature=temperature,
        )

    num_prompts = len(prompts)
    outputs: List[List[str]] = []
    for i in range(num_prompts):
        rows: List[str] = []
        base = i * n_samples
        for s in range(n_samples):
            completion = gen_ids[base + s, prompt_width:]  # (T_new,)
            rows.append(
                tokenizer.decode(
                    completion,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
            )
        outputs.append(rows)
    _eval_debug(
        eval_debug,
        "generate/done",
        total_wall_s=_fmt_seconds(time.perf_counter() - t_all),
        num_prompts=num_prompts,
    )
    return outputs


@torch.no_grad()
def _eos_pool_concepts_with_steerer(
    *,
    steerer: PaCECBMSteerer,
    llm,
    tokenizer,
    text: str,
    device: torch.device,
    cf_offset: int,
    cf_size: int,
    max_length: int = 2048,
) -> Tensor:
    """Forward ``text`` through ``llm`` with PaCE hook (no intervention) and
    return EOS-pooled c_sparse[CF block] ⇒ ``(C_cf,)`` tensor on CPU.

    Used for per-problem concept-tag accuracy reporting in the eval cascade.
    """
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length,
    ).to(device)  # input_ids: (1, T)
    steerer.attach(llm)
    with steerer:
        steerer.configure_for_batch(cf_multihot=None)
        llm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        c_sparse = steerer.last_c_sparse  # (1, T, C_total)
    c_sparse_cf = c_sparse[:, :, cf_offset:cf_offset + cf_size]  # (1, T, C_cf)
    pooled = eos_pooling(c_sparse_cf, enc["attention_mask"]).squeeze(0).detach().cpu()  # (C_cf,)
    return pooled


# ─────────────────────────────────────────────────────────────────────────────
# code_contests test-set generation + concept metrics
# ─────────────────────────────────────────────────────────────────────────────

def run_codecontests_testset_eval_steerable(
    *,
    steerer_factory,
    llm,
    tokenizer,
    concept_set: List[str],
    test_dataset_holder: Optional[List[Any]] = None,
    test_dataset=None,
    seed: int = 42,
    batch_size: int = 4,
    model_label: str = "PaCE-Steerable",
    layer_idx: int = -1,
    run_id=None,
    max_new_tokens: int = 2000,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.05,
    results_root=None,
    display: bool = True,
    steer_modes: Optional[List[str]] = None,
    steer_value: float = 1.0,
    zero_other_concepts: bool = False,
    print_extracted_code_preview: bool = False,
    extracted_preview_chars: int = 420,
    print_each_solution: bool = False,
    each_solution_question_chars: int = 0,
    each_solution_code_chars: int = 0,
    eval_log_host_memory: bool = False,
    pace_steerer_for_concept_metrics: Optional[PaCECBMSteerer] = None,
    cf_offset: int = 0,
    cf_size: Optional[int] = None,
    generate_use_cache: bool = True,
    eval_debug: bool = False,
) -> dict:
    """Mirror of ``eval_metrics.run_codecontests_testset_evaluation_for_cbm``
    but parameterised on ``steerer_factory`` (a callable returning a fresh
    ``HookSteerer`` per steer_mode) so any method can be evaluated.

    ``pace_steerer_for_concept_metrics``: if provided, used to compute the
    EOS-pooled concept-tag prediction per problem (no intervention) so that
    the same cc/{mode}/concept_tag_* metrics are reported as in the existing
    eval. If None, those metrics are skipped (vector steerers don't have a
    concept space).
    """
    llm.eval()
    set_seed(seed)
    eval_start_t = time.perf_counter()

    if run_id is None:
        run_id = wandb.run.id if wandb.run is not None else "norun"
    if steer_modes is None:
        steer_modes = ["none"]

    base_root = Path(results_root) if results_root else Path(__file__).resolve().parent.parent / "results_pace"
    all_results: dict = {}
    cf_size = cf_size if cf_size is not None else len(concept_set)
    device = next(llm.parameters()).device
    concept_index = {c: i for i, c in enumerate(concept_set)}

    def _eval_ck(msg: str) -> None:
        _memory_checkpoint(msg, log_host_ram=eval_log_host_memory)

    def _eval_mem_line(msg: str) -> None:
        extra = ""
        if eval_log_host_memory:
            hm = _format_host_memory_stats()
            if hm:
                extra = f"  |  {hm}"
        print(f"[eval-mem] {msg}{extra}", flush=True)

    cc_td = (
        test_dataset_holder[0]
        if test_dataset_holder is not None and len(test_dataset_holder) > 0
        else test_dataset
    )
    _eval_ck("run_codecontests_testset_eval_steerable: start")
    if cc_td is None:
        return {}

    _eval_debug(
        eval_debug,
        "code_contests/init",
        n_rows=len(cc_td),
        steer_modes=steer_modes,
        batch_size=batch_size,
        model_label=model_label,
        layer_idx=layer_idx,
        generate_use_cache=generate_use_cache,
        device=str(device),
    )
    print(f"\n{'='*60}\n code_contests test set  ({len(cc_td)} problems)\n{'='*60}")

    cc_total_prompts = sum(
        1 for idx in range(len(cc_td)) if str(cc_td[idx].get("description", "")).strip()
    )

    for steer_mode in steer_modes:
        mode_label = f"{model_label}-{steer_mode}"
        cc_dir = base_root / "code_contests" / mode_label
        cc_dir.mkdir(parents=True, exist_ok=True)
        out_path = cc_dir / f"l{layer_idx}-seed{seed}-{run_id}.jsonl"

        steerer = steerer_factory(steer_mode)
        active_alpha = steer_value if steer_mode != "none" else 0.0
        _eval_debug(
            eval_debug,
            "code_contests/steer_mode",
            steer_mode=steer_mode,
            steerer_cls=type(steerer).__name__,
            active_alpha=active_alpha,
            intervene_phase=getattr(steerer, "intervene_phase", None),
        )

        print(f"\n[{steer_mode}] Generating code_contests solutions ...", flush=True)
        gen_start_t = time.perf_counter()
        rows: List[dict] = []
        concept_pred_rows: List[Tensor] = []
        concept_target_rows: List[Tensor] = []

        prompt_batch_size = max(1, int(batch_size))
        pending_prompts: List[str] = []
        pending_cf_tags: List[List[str]] = []
        pending_meta: List[dict] = []
        pending_descriptions: List[str] = []

        def _flush():
            if not pending_prompts:
                return
            flush_size = len(pending_prompts)
            t0 = time.perf_counter()
            generated = _generate_with_steerer_batched(
                steerer=steerer,
                llm=llm,
                tokenizer=tokenizer,
                prompts=pending_prompts,
                cf_tags_per_prompt=pending_cf_tags if steer_mode != "none" else None,
                cf_concepts=concept_set,
                n_samples=1,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                alpha=active_alpha,
                zero_other_concepts=zero_other_concepts,
                device=device,
                use_cache=generate_use_cache,
                eval_debug=eval_debug,
            )
            for meta, outs, desc_full in zip(pending_meta, generated, pending_descriptions):
                solution = outs[0]
                extracted = _extract_code_from_output(solution)
                rows.append({**meta, "raw_output": solution, "extracted_code": extracted})
                if print_each_solution:
                    print_solution_question_and_extracted_code(
                        heading=(
                            f"[{steer_mode}] code_contests | "
                            f"problem={meta.get('problem_name', '?')!r}"
                        ),
                        question=desc_full,
                        extracted_codes=[extracted],
                        question_max_chars=each_solution_question_chars,
                        code_max_chars=each_solution_code_chars,
                    )
                if print_extracted_code_preview:
                    print_extracted_code_samples_preview(
                        f"[{steer_mode}] code_contests problem={meta.get('problem_name','?')!r} "
                        f"description (start): {meta.get('description_preview','')!r}",
                        [extracted],
                        preview_chars=extracted_preview_chars,
                    )
            pending_prompts.clear()
            pending_cf_tags.clear()
            pending_meta.clear()
            pending_descriptions.clear()
            done = len(rows)
            left = max(0, cc_total_prompts - done)
            print(
                f"[eval-timing] code_contests/{steer_mode}: flush_generation="
                f"{_fmt_seconds(time.perf_counter() - t0)} | batch={flush_size} | done={done}/{cc_total_prompts} | left={left}",
                flush=True,
            )
            _eval_debug(
                eval_debug,
                "code_contests/flush",
                steer_mode=steer_mode,
                flush_batch=flush_size,
                flush_wall_s=_fmt_seconds(time.perf_counter() - t0),
                n_outputs=len(generated),
            )

        for i in tqdm(range(len(cc_td)), desc=f"cc/{steer_mode}", disable=not display):
            problem = cc_td[i]
            description = (problem.get("description") or "").strip()
            if not description:
                continue
            cf_tags = list(problem.get("cf_tags") or [])
            prompt = _format_code_generation_prompt(tokenizer, description, language="python")

            if pace_steerer_for_concept_metrics is not None:
                pooled = _eos_pool_concepts_with_steerer(
                    steerer=pace_steerer_for_concept_metrics,
                    llm=llm,
                    tokenizer=tokenizer,
                    text=prompt,
                    device=device,
                    cf_offset=cf_offset,
                    cf_size=cf_size,
                )
                target = torch.zeros(len(concept_set), dtype=torch.float32)
                for tag in cf_tags:
                    j = concept_index.get(tag)
                    if j is not None:
                        target[j] = 1.0
                if (target > 0).any():
                    concept_pred_rows.append(pooled)
                    concept_target_rows.append(target)

            pending_prompts.append(prompt)
            pending_cf_tags.append(cf_tags)
            pending_descriptions.append(description)
            pending_meta.append({
                "problem_name": problem.get("name", f"problem_{i}"),
                "description_preview": description[:300],
                "cf_tags": cf_tags,
                "cf_rating": problem.get("cf_rating", -1),
                "steer_mode": steer_mode,
                "steer_value": active_alpha,
                "layer_idx": layer_idx,
                "seed": seed,
                "run_id": run_id,
            })
            if len(pending_prompts) >= prompt_batch_size:
                _flush()

        _flush()
        gen_elapsed = time.perf_counter() - gen_start_t

        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  Saved {len(rows)} solutions → {out_path}", flush=True)
        print(
            f"[eval-timing] code_contests/{steer_mode}: generation={_fmt_seconds(gen_elapsed)}",
            flush=True,
        )
        _eval_mem_line(f"code_contests/{steer_mode}: jsonl written; computing concept-tag metrics...")

        concept_acc_metrics: Dict[str, float] = {}
        if concept_pred_rows:
            pred_t = torch.stack(concept_pred_rows, dim=0)
            tgt_t = torch.stack(concept_target_rows, dim=0)
            topk = compute_multilabel_concept_metrics(
                prediction_scores=pred_t, target_scores=tgt_t, topk=(1, 5, 10),
            )
            concept_acc_metrics = {
                f"cc/{steer_mode}/concept_tag_top1_acc": topk["top1_acc"],
                f"cc/{steer_mode}/concept_tag_top5_acc": topk["top5_acc"],
                f"cc/{steer_mode}/concept_tag_top10_acc": topk["top10_acc"],
                f"cc/{steer_mode}/concept_tag_top1_iou": topk["top1_iou"],
                f"cc/{steer_mode}/concept_tag_top5_iou": topk["top5_iou"],
                f"cc/{steer_mode}/concept_tag_top10_iou": topk["top10_iou"],
                f"cc/{steer_mode}/concept_tag_cosine_raw": topk["cosine_raw"],
                f"cc/{steer_mode}/concept_tag_cosine_cubed": topk["cosine_cubed"],
            }
            print(
                f"  Concept-tag metrics: top1={topk['top1_acc']:.4f} "
                f"top5={topk['top5_acc']:.4f} cos={topk['cosine_raw']:.4f}"
            )

        log_payload = {
            f"cc/{steer_mode}/solutions_written": len(rows),
            f"cc/{steer_mode}/output_path": str(out_path),
            **concept_acc_metrics,
        }
        safe_wandb_log(log_payload)

        all_results[f"cc/{steer_mode}"] = {
            "output_path": str(out_path),
            **concept_acc_metrics,
            "generations": {
                "output_path": str(out_path),
                "raw_outputs": [r["raw_output"] for r in rows],
                "extracted_codes": [r["extracted_code"] for r in rows],
                "cf_tags_per_problem": [list(r["cf_tags"]) for r in rows],
                "problem_names": [r["problem_name"] for r in rows],
                "concept_metrics": concept_acc_metrics,
            },
        }
        del rows, concept_pred_rows, concept_target_rows

    print(f"[eval-timing] code_contests_all_total={_fmt_seconds(time.perf_counter() - eval_start_t)}", flush=True)
    _eval_debug(
        eval_debug,
        "code_contests/all_total",
        wall_s=_fmt_seconds(time.perf_counter() - eval_start_t),
    )
    if test_dataset_holder is not None:
        test_dataset_holder[0] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _eval_ck("run_codecontests_testset_eval_steerable: done")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# LiveCodeBench generation (writes outputs JSON + an eval lock JSON)
# ─────────────────────────────────────────────────────────────────────────────

def run_lcb_eval_steerable(
    *,
    steerer_factory,
    llm,
    tokenizer,
    concept_set: List[str],
    seed: int = 42,
    batch_size: int = 4,
    model_label: str = "PaCE-Steerable",
    layer_idx: int = -1,
    run_id=None,
    display: bool = True,
    steer_modes: Optional[List[str]] = None,
    steer_value: float = 1.0,
    zero_other_concepts: bool = False,
    livecodebench_release: str = "release_v6",
    lcb_n_samples: int = 10,
    lcb_temperature: float = 0.2,
    lcb_top_p: float = 0.95,
    lcb_top_k: int = 50,
    lcb_max_new_tokens: int = 2000,
    lcb_repetition_penalty: float = 1.05,
    print_extracted_code_preview: bool = False,
    extracted_preview_chars: int = 420,
    print_each_solution: bool = False,
    each_solution_question_chars: int = 0,
    each_solution_code_chars: int = 0,
    eval_log_host_memory: bool = False,
    generate_use_cache: bool = True,
    eval_debug: bool = False,
) -> dict:
    """Steerer-agnostic LCB benchmark generation. Mirrors
    ``eval_metrics.run_livecodebench_benchmark_generation_for_cbm`` exactly in
    output layout (JSON + eval lock).  Grading happens later via
    ``eval_metrics.evaluate_saved_livecodebench_generation`` from the lock.
    """
    llm.eval()
    set_seed(seed)
    eval_start_t = time.perf_counter()
    if run_id is None:
        run_id = wandb.run.id if wandb.run is not None else "norun"
    if steer_modes is None:
        steer_modes = ["none"]

    lcb_repo = Path(__file__).resolve().parent.parent / "LiveCodeBench"
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

    _eval_ck("run_lcb_eval_steerable: before _import_lcb")
    load_code_generation_dataset, codegen_metrics, extract_instance_results = _import_lcb()
    _eval_ck("run_lcb_eval_steerable: dataset import done")

    t0 = time.perf_counter()
    benchmark = load_code_generation_dataset(livecodebench_release)
    print(f"  Loaded {len(benchmark)} LCB problems", flush=True)
    print(f"[eval-timing] livecodebench: dataset_loading={_fmt_seconds(time.perf_counter() - t0)}", flush=True)

    device = next(llm.parameters()).device
    _eval_debug(
        eval_debug,
        "lcb/init",
        steer_modes=steer_modes,
        batch_size=batch_size,
        lcb_n_samples=lcb_n_samples,
        n_problems=len(benchmark),
        model_label=model_label,
        layer_idx=layer_idx,
        generate_use_cache=generate_use_cache,
        device=str(device),
    )

    for steer_mode in steer_modes:
        mode_repr = f"{model_label}-{steer_mode}"
        lcb_out_dir = lcb_repo / "output" / mode_repr / str(run_id)
        lcb_out_dir.mkdir(parents=True, exist_ok=True)
        lcb_out_path = lcb_out_dir / f"codegeneration_{lcb_n_samples}_{lcb_temperature}.json"
        lcb_eval_path = lcb_out_dir / f"codegeneration_{lcb_n_samples}_{lcb_temperature}_eval.json"
        lcb_eval_all_path = lcb_out_dir / f"codegeneration_{lcb_n_samples}_{lcb_temperature}_eval_all.json"

        steerer = steerer_factory(steer_mode)
        active_alpha = steer_value if steer_mode != "none" else 0.0
        _eval_debug(
            eval_debug,
            "lcb/steer_mode",
            steer_mode=steer_mode,
            steerer_cls=type(steerer).__name__,
            active_alpha=active_alpha,
            n_problems=len(benchmark),
            intervene_phase=getattr(steerer, "intervene_phase", None),
        )

        print(f"\n[{steer_mode}] Generating {lcb_n_samples} × {len(benchmark)} LCB problems ...")
        gen_start_t = time.perf_counter()
        all_outputs: List[List[str]] = []
        all_extracted: List[List[str]] = []
        benchmark_sorted = sorted(benchmark, key=lambda x: x.question_id)

        prompt_batch_size = max(1, int(batch_size))
        pending_prompts: List[str] = []
        pending_cf_tags: List[List[str]] = []
        pending_headings: List[str] = []
        pending_lcb_questions: List[str] = []
        pending_lcb_ids: List[str] = []

        def _flush():
            if not pending_prompts:
                return
            flush_size = len(pending_prompts)
            t = time.perf_counter()
            generated = _generate_with_steerer_batched(
                steerer=steerer,
                llm=llm,
                tokenizer=tokenizer,
                prompts=pending_prompts,
                cf_tags_per_prompt=pending_cf_tags if steer_mode != "none" else None,
                cf_concepts=concept_set,
                n_samples=lcb_n_samples,
                max_new_tokens=lcb_max_new_tokens,
                temperature=lcb_temperature,
                top_p=lcb_top_p,
                top_k=lcb_top_k,
                repetition_penalty=lcb_repetition_penalty,
                alpha=active_alpha,
                zero_other_concepts=zero_other_concepts,
                device=device,
                use_cache=generate_use_cache,
                eval_debug=eval_debug,
            )
            for heading, raw_samples, q_body, q_id in zip(
                pending_headings, generated, pending_lcb_questions, pending_lcb_ids,
            ):
                extracted = [_extract_code_from_output(s) for s in raw_samples]
                if print_each_solution:
                    print_solution_question_and_extracted_code(
                        heading=f"[{steer_mode}] LCB | question_id={q_id}",
                        question=q_body,
                        extracted_codes=extracted,
                        question_max_chars=each_solution_question_chars,
                        code_max_chars=each_solution_code_chars,
                    )
                if print_extracted_code_preview:
                    print_extracted_code_samples_preview(
                        heading, extracted, preview_chars=extracted_preview_chars,
                    )
                all_outputs.append(raw_samples)
                all_extracted.append(extracted)
            pending_prompts.clear()
            pending_cf_tags.clear()
            pending_headings.clear()
            pending_lcb_questions.clear()
            pending_lcb_ids.clear()
            done = len(all_outputs)
            left = max(0, len(benchmark_sorted) - done)
            print(
                f"[eval-timing] livecodebench/{steer_mode}: flush_generation="
                f"{_fmt_seconds(time.perf_counter() - t)} | batch={flush_size} | done={done}/{len(benchmark_sorted)} | left={left}",
                flush=True,
            )
            _eval_debug(
                eval_debug,
                "lcb/flush",
                steer_mode=steer_mode,
                flush_batch=flush_size,
                flush_wall_s=_fmt_seconds(time.perf_counter() - t),
                n_problem_batches=len(generated),
                n_samples_each=lcb_n_samples,
            )

        for problem in tqdm(benchmark_sorted, desc=f"lcb/{steer_mode}", disable=not display):
            problem_id = str(problem.question_id)
            cf_tags = list(CLEANED_TAGS_MAP.get(problem_id, {}).get("tags", []))
            if problem_id not in CLEANED_TAGS_MAP and steer_mode != "none":
                print(
                    f"[warn] Missing CLEANED_TAGS_MAP entry for LCB question_id={problem_id}; using empty tags.",
                    flush=True,
                )
            prompt = _format_code_generation_prompt(
                tokenizer,
                problem.question_content,
                starter_code=getattr(problem, "starter_code", "") or "",
                language="python",
            )
            pending_prompts.append(prompt)
            pending_cf_tags.append(cf_tags)
            pending_lcb_questions.append(problem.question_content or "")
            pending_lcb_ids.append(problem_id)
            desc_flat = problem.question_content.replace("\n", " ").strip()
            desc_short = desc_flat[:260] + ("..." if len(desc_flat) > 260 else "")
            pending_headings.append(
                f"[{steer_mode}] LCB question_id={problem.question_id}  "
                f"description (start): {desc_short!r}"
            )
            if len(pending_prompts) >= prompt_batch_size:
                _flush()
        _flush()

        gen_elapsed = time.perf_counter() - gen_start_t
        _eval_mem_line(f"LCB steer_mode={steer_mode!r}: generation finished ({len(all_outputs)} prompt batches)")
        print(f"[eval-timing] livecodebench/{steer_mode}: generation={_fmt_seconds(gen_elapsed)}", flush=True)
        _eval_ck(f"LCB/{steer_mode}: before save_results JSON")

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

        log_payload = {
            f"lcb/{steer_mode}/generation_only": 1,
            f"lcb/{steer_mode}/n_samples": lcb_n_samples,
            f"lcb/{steer_mode}/temperature": lcb_temperature,
            f"lcb/{steer_mode}/steer_value": active_alpha,
            f"lcb/{steer_mode}/release": livecodebench_release,
            f"lcb/{steer_mode}/output_path": str(lcb_out_path),
            f"lcb/{steer_mode}/eval_lock_path": str(lock_path),
        }
        safe_wandb_log(log_payload)
        all_results[f"lcb/{steer_mode}"] = {
            "generation_only": True,
            "output_path": str(lcb_out_path),
            "eval_lock_path": str(lock_path),
        }

        del save_results, all_outputs, all_extracted, benchmark_sorted

    print(f"[eval-timing] all_code_evaluations_total={_fmt_seconds(time.perf_counter() - eval_start_t)}", flush=True)
    _eval_debug(
        eval_debug,
        "lcb/all_total",
        wall_s=_fmt_seconds(time.perf_counter() - eval_start_t),
    )
    del benchmark, load_code_generation_dataset, codegen_metrics, extract_instance_results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _eval_ck("run_lcb_eval_steerable: done")
    return all_results
