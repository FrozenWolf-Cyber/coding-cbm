"""PaCE-CBM trainer (Architecture 1 — learned dense decomposition).

Mirrors ``train_combined_finegrained.py``'s CLI surface for shared knobs
(``--batch_size``, ``--max_length``, ``--num_epochs``, ``--seed``, ``--lcb_*``,
``--rm_*``, ``--code_*``, ``--print_extracted_code_preview``, ``--hf_cache_root``,
``--debug``, ``--debug_0_step``) so existing job templates can swap binaries.

PaCE-specific knobs
-------------------
- ``--layer_idx``           : single Llama layer the hook attaches to.
- ``--bottleneck_k``        : k for ``W_A: (C, k)`` / ``W_B: (k, C)``.
- ``--dictionary_mode``     : ``hybrid`` | ``cf_only`` | ``pace_only``.
- ``--max_pace_concepts``   : 0 = use all entries from concept_index.txt.
- ``--sparsity_weight``     : L1 on c_sparse.
- ``--word_loss_weight``    : CE on intervened-forward vocab logits.
- ``--concept_loss_weight`` : EOS-pooled concept loss vs CF multi-hot.
- ``--concept_loss_type``   : ``cosine_cubed`` | ``ce``.
- ``--identity_weight``     : optional MSE(z_ctrl_no_intervene, z_in).
- ``--intervention_alpha``  : value to write into selected concept entries.
- ``--steer_modes``         : eval steer modes (``none,groundtruth``).
- ``--init_tau_percentile`` : percentile for τ init from data; 0 disables.

Existing CBM code (``train_combined_finegrained.py``, ``modules.py``,
``eval_metrics.py``, ``utils.py``, ``config.py``) is imported, never modified.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import DownloadConfig, load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM
import wandb

from config import CODEFORCES_CONCEPT_SET, CODEFORCES_CONCEPT_SET_LOOKUP
from eval_metrics import (
    _format_host_memory_stats,
    compute_perplexity,
    load_reward_model,
    run_llamacpp_judge_per_solution,
    run_rm_metrics_per_solution,
    set_seed,
)
from shared_code_prompt import LCB_LLAMA3_INSTRUCT_MODEL_ID

from pace.activations import collect_token_activations_at_layer
from pace.data import (
    build_loaders_param,
    build_multihot,
    filter_codecontests,
    resolve_cache_subdir,
)
from pace.dictionary import build_dictionary, load_dictionary
from pace.eval_steerable import (
    run_codecontests_testset_eval_steerable,
    run_concept_accuracy_pace,
    run_lcb_eval_steerable,
)
from pace.hook_steerer import PaCECBMSteerer
from pace.loops import train_one_epoch, validate_one_epoch
from pace.pace_cbm import PaCECBM


def _hf_load_dataset_cache_first(dataset_name: str, cache_dir: str):
    print(f"[cache] loading {dataset_name} (cache_dir={cache_dir})")
    return load_dataset(
        dataset_name,
        cache_dir=cache_dir,
        download_config=DownloadConfig(local_files_only=False),
    )


def _hf_from_pretrained_cache_first(loader_fn, model_id: str, cache_dir: str, **kwargs):
    return loader_fn(model_id, cache_dir=cache_dir, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser()
    # ── Shared with existing trainer ──────────────────────────────────────────
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_valid_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--epoch_multiplier", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_0_step", action="store_true")
    parser.add_argument("--skip_loss_mask", action="store_true")
    parser.add_argument("--hf_cache_root", type=str, default="./.hf_cache")

    # ── PaCE-CBM specific ─────────────────────────────────────────────────────
    parser.add_argument("--layer_idx", type=int, default=16,
                        help="Llama decoder layer the PaCE hook attaches to (0-31 for Llama-3-8B).")
    parser.add_argument("--bottleneck_k", type=int, default=64)
    parser.add_argument("--dictionary_mode", type=str, default="hybrid",
                        choices=["hybrid", "cf_only", "pace_only"])
    parser.add_argument("--max_pace_concepts", type=int, default=5000,
                        help="Max PaCE English concepts (0 = use all from concept_index.txt; "
                             "memory grows ~163 MB per 10K cols at H=4096 fp32).")
    parser.add_argument("--sparsity_weight", type=float, default=1e-3)
    parser.add_argument("--word_loss_weight", type=float, default=1.0)
    parser.add_argument("--concept_loss_weight", type=float, default=1.0)
    parser.add_argument("--concept_loss_type", type=str, default="ce", choices=["cosine_cubed", "ce"])
    parser.add_argument("--identity_weight", type=float, default=0.0)
    parser.add_argument("--intervention_alpha", type=float, default=150.0)
    parser.add_argument("--init_tau_percentile", type=float, default=0.0,
                        help="Set τ[i] to per-concept percentile of |h[:, i]| over a token sample. "
                             "0 disables (init τ = 0).")
    parser.add_argument("--init_tau_max_batches", type=int, default=8)
    parser.add_argument("--init_tau_max_tokens_per_seq", type=int, default=64)
    parser.add_argument("--cf_probe_max_batches", type=int, default=0)
    parser.add_argument("--steer_modes", type=str, default="none,groundtruth",
                        help="Comma-separated; supports 'none' and 'groundtruth' (boost ground-truth CF tags).")
    parser.add_argument("--zero_other_concepts", action="store_true")
    parser.add_argument(
        "--eval_intervene_phase",
        type=str,
        default="decode_only",
        choices=["all", "decode_only"],
        help=(
            "When to apply steering during eval generation. 'decode_only' "
            "(recommended) skips the prefill forward so the prompt is "
            "interpreted by the unmodified model and only newly generated "
            "tokens are steered. 'all' steers every position (including the "
            "prompt) and is what training uses internally. Note: under "
            "'decode_only', the very first sampled token is drawn from "
            "logits computed during prefill and is therefore not steered; "
            "every subsequent token is."
        ),
    )
    parser.add_argument(
        "--eval_generate_no_kv_cache",
        action="store_true",
        help=(
            "Post-training eval generation passes use_cache=False into HF generate "
            "(recomputes attention; very slow). Optional ablation / debug."
        ),
    )
    parser.add_argument(
        "--eval_debug",
        action="store_true",
        help="Verbose post-training eval logs (shapes, timings); see pace.eval_steerable._eval_debug.",
    )
    parser.add_argument("--compute_dtype", type=str, default="float32",
                        choices=["float32", "bfloat16", "float16"])

    # ── Code-eval / LCB / RM (mirror existing) ────────────────────────────────
    parser.add_argument("--code_results_root", type=str, default="")
    parser.add_argument("--code_max_new_tokens", type=int, default=512)
    parser.add_argument("--code_temperature", type=float, default=0.7)
    parser.add_argument("--code_top_p", type=float, default=0.9)
    parser.add_argument("--code_top_k", type=int, default=100)
    parser.add_argument("--code_repetition_penalty", type=float, default=1.05)
    parser.add_argument("--lcb_prompt_batch_size", type=int, default=1)
    parser.add_argument("--livecodebench_release", type=str, default="release_v6")
    parser.add_argument("--lcb_n_samples", type=int, default=10)
    parser.add_argument("--lcb_temperature", type=float, default=0.2)
    parser.add_argument("--lcb_top_p", type=float, default=0.95)
    parser.add_argument("--lcb_max_new_tokens", type=int, default=2000)
    parser.add_argument("--print_extracted_code_preview", action="store_true")
    parser.add_argument("--extracted_preview_chars", type=int, default=420)
    parser.add_argument("--eval_log_host_memory", action="store_true")
    parser.add_argument("--rm_model_name", type=str, default="Skywork/Skywork-Reward-V2-Llama-3.1-8B")
    parser.add_argument("--rm_batch_size", type=int, default=0)
    parser.add_argument("--rm_max_text_len", type=int, default=500)
    parser.add_argument("--skip_rm", action="store_true")
    parser.add_argument("--skip_llamacpp_steer_eval", action="store_true")
    parser.add_argument("--llamacpp_eval_model_repo_id", type=str, default="unsloth/Qwen3.5-27B-GGUF")
    parser.add_argument("--llamacpp_eval_model_filename", type=str, default="Qwen3.5-27B-Q8_0.gguf")
    parser.add_argument("--llamacpp_eval_n_ctx", type=int, default=2048)
    parser.add_argument("--llamacpp_eval_max_tokens", type=int, default=128)
    parser.add_argument("--llamacpp_eval_repeat_penalty", type=float, default=1.15)
    parser.add_argument("--llamacpp_eval_temperature", type=float, default=0.1)

    return parser.parse_args()


def _save_pace_cbm_with_meta(pace_cbm: PaCECBM, dict_meta: dict, ckpt_path: str) -> None:
    """Persist W_A/W_B/τ + sidecar referencing the dictionary cache.

    The dictionary itself is *not* saved here (it lives in the cache referenced
    by ``dict_meta['dictionary_path']``); reload via ``load_dictionary`` when
    reconstructing the module.
    """
    torch.save(pace_cbm.state_dict_no_dict(), ckpt_path)
    sidecar = {
        "layer_idx": int(pace_cbm.layer_idx),
        "bottleneck_k": int(pace_cbm.k),
        "cf_offset": int(pace_cbm.cf_offset),
        "cf_size": int(pace_cbm.cf_size),
        "C": int(pace_cbm.C),
        "H": int(pace_cbm.H),
        "compute_dtype": str(pace_cbm.compute_dtype).split(".")[-1],
        "dictionary_path": dict_meta["dictionary_path"],
        "dictionary_meta_path": dict_meta["meta_path"],
        "dictionary_mode": dict_meta["mode"],
    }
    Path(ckpt_path + ".meta.json").write_text(json.dumps(sidecar, indent=2))


def _resolve_compute_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def main():
    mp.set_start_method("spawn", force=False)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_mode = bool(args.debug or args.debug_0_step)

    hf_cache_root = str(Path(args.hf_cache_root).expanduser())
    Path(hf_cache_root).mkdir(parents=True, exist_ok=True)
    dataset_cache_dir = resolve_cache_subdir(hf_cache_root, "datasets")
    model_cache_dir = resolve_cache_subdir(hf_cache_root, "models")
    llamacpp_cache_dir = resolve_cache_subdir(hf_cache_root, "llamacpp")

    use_wandb = not debug_mode
    if use_wandb:
        wandb.init(project="coding-qa", name=f"pace-cbm-l{args.layer_idx}-seed{args.seed}", config=vars(args))
        run_name = wandb.run.id
    else:
        run_name = f"debug-{int(time.time())}"

    def wandb_log(payload):
        if use_wandb:
            wandb.log(payload)

    print("loading code_contests dataset...")
    raw_dataset = _hf_load_dataset_cache_first("deepmind/code_contests", dataset_cache_dir)

    train_dataset, valid_dataset, test_dataset = filter_codecontests(
        raw_dataset,
        cf_concept_lookup=CODEFORCES_CONCEPT_SET_LOOKUP,
        max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples,
        max_test_samples=args.max_test_samples,
    )
    if debug_mode:
        train_dataset = train_dataset.select(range(min(64, len(train_dataset))))
        test_dataset = test_dataset.select(range(min(32, len(test_dataset))))

    concept_set = list(CODEFORCES_CONCEPT_SET)
    train_similarity = build_multihot(train_dataset, concept_set)
    val_similarity = build_multihot(valid_dataset, concept_set)
    test_similarity = build_multihot(test_dataset, concept_set)
    print(
        f"split sizes | train={len(train_dataset)} valid={len(valid_dataset)} "
        f"test={len(test_dataset)} | C_cf={len(concept_set)}"
    )

    tokenizer = _hf_from_pretrained_cache_first(
        AutoTokenizer.from_pretrained, LCB_LLAMA3_INSTRUCT_MODEL_ID, model_cache_dir,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    train_loader = build_loaders_param(train_dataset, train_similarity, "train", tokenizer, args)
    valid_loader = build_loaders_param(valid_dataset, val_similarity, "valid", tokenizer, args)
    test_dummy = np.zeros((len(test_dataset), len(concept_set)), dtype=np.float32)
    test_loader = build_loaders_param(test_dataset, test_dummy, "test", tokenizer, args)

    print("preparing backbone (frozen LlamaForCausalLM)...")
    llm = _hf_from_pretrained_cache_first(
        LlamaForCausalLM.from_pretrained,
        LCB_LLAMA3_INSTRUCT_MODEL_ID,
        model_cache_dir,
        torch_dtype=torch.bfloat16,
    ).to(device)
    for p in llm.parameters():
        p.requires_grad = False
    llm.eval()

    cf_cache_key_extras = {
        "model_id": LCB_LLAMA3_INSTRUCT_MODEL_ID,
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "n_train_rows": int(len(train_dataset)),
        "max_train_samples": int(args.max_train_samples),
        "skip_loss_mask": bool(args.skip_loss_mask),
        "cf_probe_max_batches": int(args.cf_probe_max_batches),
        "seed": int(args.seed),
    }
    print(f"building dictionary (mode={args.dictionary_mode}, layer={args.layer_idx}, max_pace={args.max_pace_concepts})...")
    D, dict_meta = build_dictionary(
        preLM=llm,
        tokenizer=tokenizer,
        layer_idx=args.layer_idx,
        device=device,
        cf_concept_set=concept_set,
        cf_train_dataloader=train_loader,
        cf_cache_key_extras=cf_cache_key_extras,
        mode=args.dictionary_mode,
        max_pace_concepts=args.max_pace_concepts,
        cf_probe_max_batches=args.cf_probe_max_batches,
    )
    print(
        f"dictionary built: D shape {tuple(D.shape)} | "
        f"cf_offset={dict_meta['cf_offset']} cf_size={dict_meta['cf_size']} pace_size={dict_meta['pace_size']} | "
        f"cached at {dict_meta['dictionary_path']}"
    )
    hm_stats = _format_host_memory_stats()
    if hm_stats:
        print(f"[post-dictionary] {hm_stats}", flush=True)

    compute_dtype = _resolve_compute_dtype(args.compute_dtype)
    pace_cbm = PaCECBM(
        D=D,
        k=args.bottleneck_k,
        layer_idx=args.layer_idx,
        cf_offset=dict_meta["cf_offset"],
        cf_size=dict_meta["cf_size"],
        compute_dtype=compute_dtype,
    ).to(device)
    del D
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.init_tau_percentile > 0:
        print(
            f"initialising τ from data percentile ({args.init_tau_percentile}%) "
            f"over a token sample ..."
        )
        token_acts = collect_token_activations_at_layer(
            preLM=llm,
            dataloader=train_loader,
            layer_idx=args.layer_idx,
            device=device,
            max_tokens_per_seq=args.init_tau_max_tokens_per_seq,
            max_batches=args.init_tau_max_batches,
            desc="τ-init",
            dtype=torch.float32,
        )
        # h_samples = token_acts @ D ; we redo on CPU + bf16 to fit memory.
        with torch.no_grad():
            chunk = max(1, 4096)
            n = token_acts.size(0)
            D_cpu = pace_cbm.D.detach().to(device="cpu", dtype=torch.float32)
            h_samples = torch.empty((n, pace_cbm.C), dtype=torch.float32)
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                h_samples[s:e] = token_acts[s:e].float() @ D_cpu
            del D_cpu
        pace_cbm.init_tau_from_h_samples(h_samples, percentile=args.init_tau_percentile)
        del token_acts, h_samples
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    trainable_params = sum(p.numel() for p in pace_cbm.parameters() if p.requires_grad)
    print(f"PaCE-CBM trainable params: {trainable_params:,}")
    wandb_log({"trainable_parameters": trainable_params})

    optimizer = torch.optim.Adam(pace_cbm.parameters(), lr=args.lr)
    # Training: intervene at every position (no KV cache, single forward, the
    # gradient signal needs the steered prompt context).
    steerer = PaCECBMSteerer(pace_cbm, intervene_phase="all")

    prefix = Path(f"./from_pretained_llama3_pace_cbm_{run_name}/code_contests_l{args.layer_idx}/")
    prefix.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint prefix: {prefix}")

    epochs = 2 if debug_mode else args.num_epochs * args.epoch_multiplier
    # ``max_steps_per_epoch``: None = no limit, 0 = true 0-step debug, N = N steps.
    if debug_mode:
        max_steps_per_epoch: Optional[int] = 0 if args.debug_0_step else 2
    else:
        max_steps_per_epoch = None
    best_loss = float("inf")
    best_epoch = -1

    for e in range(epochs):
        train_metrics = train_one_epoch(
            pace_cbm=pace_cbm,
            llm=llm,
            steerer=steerer,
            train_loader=train_loader,
            optimizer=optimizer,
            args=args,
            device=device,
            epoch=e,
            max_steps_per_epoch=max_steps_per_epoch,
            log_fn=wandb_log,
        )
        print(f"Epoch {e+1} train averages: {train_metrics}")
        wandb_log({**train_metrics, "epoch": e + 1})

        val_metrics = validate_one_epoch(
            pace_cbm=pace_cbm,
            llm=llm,
            steerer=steerer,
            valid_loader=valid_loader,
            args=args,
            device=device,
            epoch=e,
            cf_offset=pace_cbm.cf_offset,
            cf_size=pace_cbm.cf_size,
        )
        print(
            f"Epoch {e+1} val: top1={val_metrics['valid_concept_top1_acc']:.4f} "
            f"top5={val_metrics['valid_concept_top5_acc']:.4f} "
            f"top10={val_metrics['valid_concept_top10_acc']:.4f} "
            f"valid_loss={val_metrics['valid_loss']:.6f}"
        )
        wandb_log({**val_metrics, "epoch": e + 1})

        ckpt_path = str(prefix / f"pace_cbm_epoch_{e+1}.pt")
        _save_pace_cbm_with_meta(pace_cbm, dict_meta, ckpt_path)

        if val_metrics["valid_loss"] < best_loss:
            best_loss = val_metrics["valid_loss"]
            best_epoch = e + 1
            best_path = str(prefix / "pace_cbm_best.pt")
            _save_pace_cbm_with_meta(pace_cbm, dict_meta, best_path)
            print(f"  new best epoch {best_epoch} (valid_loss={best_loss:.6f})")
            wandb_log({"best_epoch": best_epoch, "best_valid_loss": best_loss})

    if epochs == 0:
        # Persist initial checkpoint so eval can load a uniform path.
        _save_pace_cbm_with_meta(pace_cbm, dict_meta, str(prefix / "pace_cbm_epoch_0.pt"))
        _save_pace_cbm_with_meta(pace_cbm, dict_meta, str(prefix / "pace_cbm_best.pt"))
        best_epoch = 0

    print("=" * 60)
    print(" Post-training evaluation cascade")
    print("=" * 60)

    # Concept-tag accuracy on the test split (un-intervened forward).
    run_concept_accuracy_pace(
        pace_cbm=pace_cbm,
        llm=llm,
        test_loader=test_loader,
        cf_concept_set=concept_set,
        cf_offset=pace_cbm.cf_offset,
        cf_size=pace_cbm.cf_size,
        device=device,
        test_similarity_np=test_similarity,
        eval_debug=bool(args.eval_debug),
    )

    steer_modes = [m.strip() for m in args.steer_modes.split(",") if m.strip()]
    # Eval generation: by default skip prefill so only newly generated tokens
    # are steered (prompt context stays clean). Pass intervene_phase="all" to
    # restore the legacy "steer everything" behaviour.
    eval_intervene_phase = args.eval_intervene_phase
    generate_use_cache = not args.eval_generate_no_kv_cache
    pace_factory = lambda mode: PaCECBMSteerer(  # noqa: E731
        pace_cbm, intervene_phase=eval_intervene_phase,
    )

    cc_results = run_codecontests_testset_eval_steerable(
        steerer_factory=pace_factory,
        llm=llm,
        tokenizer=tokenizer,
        concept_set=concept_set,
        test_dataset_holder=[test_dataset],
        seed=args.seed,
        batch_size=args.lcb_prompt_batch_size,
        model_label=f"PaCE-CBM-l{args.layer_idx}",
        layer_idx=args.layer_idx,
        run_id=run_name,
        max_new_tokens=args.code_max_new_tokens,
        temperature=args.code_temperature,
        top_p=args.code_top_p,
        top_k=args.code_top_k,
        repetition_penalty=args.code_repetition_penalty,
        results_root=(args.code_results_root or None),
        display=not debug_mode,
        steer_modes=steer_modes,
        steer_value=args.intervention_alpha,
        zero_other_concepts=args.zero_other_concepts,
        print_extracted_code_preview=args.print_extracted_code_preview,
        extracted_preview_chars=args.extracted_preview_chars,
        eval_log_host_memory=bool(debug_mode or args.eval_log_host_memory),
        # Concept-metric steerer must use ``"all"`` because it runs a single
        # forward over the *prompt* (not generation) and must fire the hook
        # to stash ``last_c_sparse`` even when ``T == T_prompt > 1``.
        pace_steerer_for_concept_metrics=PaCECBMSteerer(pace_cbm, intervene_phase="all"),
        cf_offset=pace_cbm.cf_offset,
        cf_size=pace_cbm.cf_size,
        generate_use_cache=generate_use_cache,
        eval_debug=bool(args.eval_debug),
    )

    cc_generations_by_mode = {}
    for mode in steer_modes:
        entry = cc_results.get(f"cc/{mode}")
        if isinstance(entry, dict) and isinstance(entry.get("generations"), dict):
            cc_generations_by_mode[mode] = entry["generations"]

    run_lcb_eval_steerable(
        steerer_factory=pace_factory,
        llm=llm,
        tokenizer=tokenizer,
        concept_set=concept_set,
        seed=args.seed,
        batch_size=args.lcb_prompt_batch_size,
        model_label=f"PaCE-CBM-l{args.layer_idx}",
        layer_idx=args.layer_idx,
        run_id=run_name,
        display=not debug_mode,
        steer_modes=steer_modes,
        steer_value=args.intervention_alpha,
        zero_other_concepts=args.zero_other_concepts,
        livecodebench_release=args.livecodebench_release,
        lcb_n_samples=args.lcb_n_samples,
        lcb_temperature=args.lcb_temperature,
        lcb_top_p=args.lcb_top_p,
        lcb_max_new_tokens=args.lcb_max_new_tokens,
        print_extracted_code_preview=args.print_extracted_code_preview,
        extracted_preview_chars=args.extracted_preview_chars,
        eval_log_host_memory=bool(debug_mode or args.eval_log_host_memory),
        generate_use_cache=generate_use_cache,
        eval_debug=bool(args.eval_debug),
    )

    # Free LM + PaCE-CBM before loading downstream evaluators.
    del llm, pace_cbm, steerer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Per-solution downstream evaluations (perplexity, llama.cpp judge, RM)
    # — all reused unchanged from eval_metrics.
    for mode, payload in cc_generations_by_mode.items():
        texts = payload.get("raw_outputs") or []
        if not texts:
            print(f"cc/{mode}: no raw outputs; skipping perplexity.")
            continue
        print(f"\n[cc/{mode}] Computing perplexity over {len(texts)} test-set generations ...", flush=True)
        ppl = compute_perplexity(texts)
        wandb_log({f"cc/{mode}/{k}": v for k, v in ppl.items()})

    if cc_generations_by_mode and not args.skip_llamacpp_steer_eval:
        run_llamacpp_judge_per_solution(
            generations_by_mode=cc_generations_by_mode,
            concept_set=concept_set,
            model_repo_id=args.llamacpp_eval_model_repo_id,
            model_filename=args.llamacpp_eval_model_filename,
            cache_dir=llamacpp_cache_dir,
            n_ctx=args.llamacpp_eval_n_ctx,
            max_tokens=args.llamacpp_eval_max_tokens,
            repeat_penalty=args.llamacpp_eval_repeat_penalty,
            temperature=args.llamacpp_eval_temperature,
        )

    if cc_generations_by_mode and not args.skip_rm:
        rm_model, rm_tok = load_reward_model(args.rm_model_name, device)
        run_rm_metrics_per_solution(
            generations_by_mode=cc_generations_by_mode,
            concept_set=concept_set,
            rm_model=rm_model,
            rm_tokenizer=rm_tok,
            rm_device=device,
            rm_batch_size=args.rm_batch_size,
            rm_max_text_len=args.rm_max_text_len,
        )
        del rm_model, rm_tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
