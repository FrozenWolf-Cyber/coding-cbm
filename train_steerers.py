"""Fit per-CF-tag vector / transform steerers from ``steer/`` and persist them.

For each ``--method`` (CAA / ITI / RepE / LinAcT / MiMiC) and each CF tag in
``CODEFORCES_CONCEPT_SET``:

  1. Collect EOS-pooled hidden states at ``--layer_idx`` over the train split.
  2. ``pos_X = acts[label_c == 1]``, ``neg_X = acts[label_c == 0]``.
  3. ``steerer = get_steer_model(method).fit(pos_X, neg_X)``.
  4. Save to ``steer_ckpts/{method}/layer{L}/{tag}.pt``.

This is purely "fit + persist" — no training loop, no LM forward beyond the
single activation-collection pass that's reused across every tag and method
(via ``pace.activations.collect_activations_at_layer``'s on-disk cache).

Vector steerers (``CAA``/``ITI``/``RepE``) get stacked into a single
``(C_cf, H)`` tensor for fast batched ``VecAddSteerer`` runs at eval time;
transform steerers (``LinAcT``/``MiMiC``) are pickled per tag.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from config import CODEFORCES_CONCEPT_SET, CODEFORCES_CONCEPT_SET_LOOKUP
from eval_metrics import set_seed
from shared_code_prompt import LCB_LLAMA3_INSTRUCT_MODEL_ID, configure_code_eval_tokenizer
from steer import get_steer_model

from pace.activations import (
    collect_activations_at_layer,
    per_concept_pos_neg,
)
from pace.data import (
    build_loaders_param,
    build_multihot,
    filter_codecontests,
    resolve_cache_subdir,
)
from train_pace_cbm import _hf_load_dataset_cache_first  # cache-first loader

VECTOR_METHODS = {"CAA", "ITI", "RepE"}
TRANSFORM_METHODS = {"LinAcT", "MiMiC"}
SUPPORTED_METHODS = VECTOR_METHODS | TRANSFORM_METHODS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer_idx", type=int, default=16)
    parser.add_argument("--methods", type=str, default="CAA,ITI,RepE,LinAcT,MiMiC")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_valid_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--probe_max_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_cache_root", type=str, default="./.hf_cache")
    parser.add_argument("--out_dir", type=str, default="./steer_ckpts")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_0_step", action="store_true")
    parser.add_argument("--skip_loss_mask", action="store_true")
    return parser.parse_args()


def _save_vector_pack(
    *,
    method: str,
    layer_idx: int,
    out_root: Path,
    cf_concepts: List[str],
    fitted_per_tag: dict,
) -> Path:
    """Stack per-tag steer vectors into ``(C_cf, H)`` and persist alongside meta.

    ``fitted_per_tag``: dict ``{tag: VecSteer}``. Tags missing from the dict
    are emitted as zero rows (preserves CF-set ordering for VecAddSteerer).
    """
    method_dir = out_root / method / f"layer{layer_idx}"
    method_dir.mkdir(parents=True, exist_ok=True)
    H = next(iter(fitted_per_tag.values())).steer_vec.numel()
    stacked = torch.zeros(len(cf_concepts), H, dtype=torch.float32)
    for i, tag in enumerate(cf_concepts):
        steerer = fitted_per_tag.get(tag)
        if steerer is None:
            continue
        stacked[i] = steerer.steer_vec.detach().cpu().float()
    pack_path = method_dir / "vec_pack.pt"
    torch.save(stacked, pack_path)
    meta = {
        "method": method,
        "layer_idx": int(layer_idx),
        "cf_concepts": list(cf_concepts),
        "shape": list(stacked.shape),
        "fitted_count": int(sum(1 for _ in fitted_per_tag.values())),
    }
    (method_dir / "vec_pack.meta.json").write_text(json.dumps(meta, indent=2))
    return pack_path


def _save_transform_per_tag(
    *,
    method: str,
    layer_idx: int,
    out_root: Path,
    tag: str,
    steerer,
) -> Path:
    method_dir = out_root / method / f"layer{layer_idx}"
    method_dir.mkdir(parents=True, exist_ok=True)
    path = method_dir / f"{tag}.pkl"
    with open(path, "wb") as f:
        pickle.dump(steerer, f)
    return path


def main():
    mp.set_start_method("spawn", force=False)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    invalid = [m for m in methods if m not in SUPPORTED_METHODS]
    if invalid:
        raise ValueError(
            f"Unsupported method(s): {invalid}. Supported: {sorted(SUPPORTED_METHODS)}"
        )

    hf_cache_root = str(Path(args.hf_cache_root).expanduser())
    Path(hf_cache_root).mkdir(parents=True, exist_ok=True)
    dataset_cache_dir = resolve_cache_subdir(hf_cache_root, "datasets")
    model_cache_dir = resolve_cache_subdir(hf_cache_root, "models")

    print("loading code_contests...")
    raw_dataset = _hf_load_dataset_cache_first("deepmind/code_contests", dataset_cache_dir)
    train_dataset, _, _ = filter_codecontests(
        raw_dataset,
        cf_concept_lookup=CODEFORCES_CONCEPT_SET_LOOKUP,
        max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples,
        max_test_samples=args.max_test_samples,
    )

    concept_set = list(CODEFORCES_CONCEPT_SET)
    train_similarity = build_multihot(train_dataset, concept_set)
    print(f"train rows kept: {len(train_dataset)} | C_cf={len(concept_set)}")

    tokenizer = AutoTokenizer.from_pretrained(
        LCB_LLAMA3_INSTRUCT_MODEL_ID, cache_dir=model_cache_dir, use_fast=False,
    )
    configure_code_eval_tokenizer(tokenizer)

    train_loader = build_loaders_param(
        train_dataset, train_similarity, "train", tokenizer, args, shuffle=False,
    )

    print(f"loading frozen LlamaForCausalLM @ layer={args.layer_idx} ...")
    llm = LlamaForCausalLM.from_pretrained(
        LCB_LLAMA3_INSTRUCT_MODEL_ID,
        cache_dir=model_cache_dir,
        torch_dtype=torch.bfloat16,
    ).to(device)
    for p in llm.parameters():
        p.requires_grad = False
    llm.eval()

    cache_dir = Path(args.hf_cache_root).expanduser() / "pace_cache" / "steer_acts"
    probe_extras = {
        "model_id": LCB_LLAMA3_INSTRUCT_MODEL_ID,
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "n_train_rows": int(len(train_dataset)),
        "max_train_samples": int(args.max_train_samples),
        "skip_loss_mask": bool(args.skip_loss_mask),
        "probe_max_batches": int(args.probe_max_batches),
        "seed": int(args.seed),
    }
    probe = collect_activations_at_layer(
        preLM=llm,
        dataloader=train_loader,
        layer_idx=args.layer_idx,
        device=device,
        cache_dir=str(cache_dir),
        cache_tag=f"steer_train_layer{args.layer_idx}",
        cache_key_extras=probe_extras,
        desc=f"steer-probe layer={args.layer_idx}",
        max_batches=args.probe_max_batches,
        return_labels=True,
    )
    acts: torch.Tensor = probe["acts"]
    labels: torch.Tensor = probe["labels"]
    if labels is None:
        raise RuntimeError("Activation probe returned no labels — train_loader sim missing?")
    print(f"probe acts shape: {tuple(acts.shape)} | labels shape: {tuple(labels.shape)}")

    # We can drop the LM now — the rest is pure tensor work on CPU.
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_root = Path(args.out_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    for method in methods:
        print(f"\n{'='*40}\n fitting {method} per CF tag \n{'='*40}")
        fitted_vec: dict = {}
        fitted_count = 0
        skipped_count = 0
        for c, tag in enumerate(concept_set):
            pos_X, neg_X = per_concept_pos_neg(acts=acts, labels=labels, concept_idx=c)
            if pos_X.size(0) == 0 or neg_X.size(0) == 0:
                skipped_count += 1
                continue
            steerer = get_steer_model(method)
            steerer.fit(pos_X, neg_X)
            if method in VECTOR_METHODS:
                fitted_vec[tag] = steerer
            else:
                _save_transform_per_tag(
                    method=method,
                    layer_idx=args.layer_idx,
                    out_root=out_root,
                    tag=tag,
                    steerer=steerer,
                )
            fitted_count += 1
            print(f"  {method}/{tag}: pos={pos_X.size(0)} neg={neg_X.size(0)} ✓", flush=True)

        if method in VECTOR_METHODS:
            pack_path = _save_vector_pack(
                method=method,
                layer_idx=args.layer_idx,
                out_root=out_root,
                cf_concepts=concept_set,
                fitted_per_tag=fitted_vec,
            )
            print(f"  saved vec pack → {pack_path}")
        summary[method] = {
            "fitted": fitted_count,
            "skipped": skipped_count,
            "out_dir": str(out_root / method / f"layer{args.layer_idx}"),
        }

    summary_path = out_root / f"summary_layer{args.layer_idx}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary → {summary_path}")


if __name__ == "__main__":
    main()
