"""Single entrypoint that evaluates the steerability of any chosen method
(``none`` baseline, PaCE-CBM checkpoint, vector steerers, transform steerers)
through the unified eval cascade.

Each method is evaluated **on its own** — this is *not* a composition CLI.
Pass the methods you want to compare with ``--methods`` and the script will
dispatch a fresh ``HookSteerer`` per method, run the full cascade
(concept-tag metrics + code_contests + LCB + perplexity + RM + llama.cpp
judge), and persist generation lock files for offline LCB grading.

Examples
--------
- Baseline only:
    python eval_steerable_cli.py --methods none --layer_idx 16

- PaCE-CBM only:
    python eval_steerable_cli.py --methods pace_cbm --layer_idx 16 \
        --pace_ckpt ./from_pretained_llama3_pace_cbm_<run>/code_contests_l16/pace_cbm_best.pt

- Vector steerers from a fitted ckpt directory:
    python eval_steerable_cli.py --methods CAA,ITI,RepE --layer_idx 16 \
        --vec_pack_root ./steer_ckpts

- Transform steerers:
    python eval_steerable_cli.py --methods LinAcT,MiMiC --layer_idx 16 \
        --transform_root ./steer_ckpts
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import wandb

from config import CODEFORCES_CONCEPT_SET, CODEFORCES_CONCEPT_SET_LOOKUP
from eval_metrics import (
    compute_perplexity,
    load_reward_model,
    run_llamacpp_judge_per_solution,
    run_rm_metrics_per_solution,
    set_seed,
)
from shared_code_prompt import LCB_LLAMA3_INSTRUCT_MODEL_ID, configure_code_eval_tokenizer

from pace.data import (
    build_loaders_param,
    build_multihot,
    filter_codecontests,
    resolve_cache_subdir,
)
from pace.dictionary import load_dictionary
from pace.eval_steerable import (
    run_codecontests_testset_eval_steerable,
    run_concept_accuracy_pace,
    run_lcb_eval_steerable,
)
from pace.hook_steerer import (
    HookSteerer,
    NoSteer,
    PaCECBMSteerer,
    TransformSteerer,
    VecAddSteerer,
)
from pace.pace_cbm import PaCECBM
from train_pace_cbm import _hf_load_dataset_cache_first

VECTOR_METHODS = {"CAA", "ITI", "RepE"}
TRANSFORM_METHODS = {"LinAcT", "MiMiC"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, default="none",
                        help="Comma-separated: none, pace_cbm, CAA, ITI, RepE, LinAcT, MiMiC.")
    parser.add_argument("--layer_idx", type=int, default=16)
    parser.add_argument("--pace_ckpt", type=str, default="",
                        help="Path to pace_cbm_*.pt; required if 'pace_cbm' in --methods.")
    parser.add_argument("--vec_pack_root", type=str, default="./steer_ckpts",
                        help="Root containing {METHOD}/layer{L}/vec_pack.pt (CAA / ITI / RepE).")
    parser.add_argument("--transform_root", type=str, default="./steer_ckpts",
                        help="Root containing {METHOD}/layer{L}/{tag}.pkl (LinAcT / MiMiC).")
    parser.add_argument("--steer_value", type=float, default=1.0,
                        help="Magnitude scale per steerer (CAA/ITI/RepE α; PaCE-CBM intervene_value).")
    parser.add_argument("--zero_other_concepts", action="store_true")
    parser.add_argument("--steer_modes", type=str, default="none,groundtruth")
    parser.add_argument(
        "--intervene_phase",
        type=str,
        default="decode_only",
        choices=["all", "decode_only"],
        help=(
            "When the hook applies the steering vector during generation. "
            "'decode_only' (default) skips the prefill so the prompt is "
            "interpreted unmodified and only newly generated tokens are "
            "steered. 'all' steers every position (including the prompt)."
        ),
    )
    parser.add_argument(
        "--eval_generate_no_kv_cache",
        action="store_true",
        help="HF generate(use_cache=False) for code_contests + LCB (slow; ablation).",
    )
    parser.add_argument(
        "--eval_debug",
        action="store_true",
        help=(
            "Verbose eval-only logs: tensor shapes, per-stage timings, steerer "
            "class (VecAdd / PaCE / etc.). Does not affect training."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_valid_samples", type=int, default=0)
    parser.add_argument("--hf_cache_root", type=str, default="./.hf_cache")
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
    parser.add_argument("--skip_lcb", action="store_true")
    parser.add_argument("--skip_llamacpp_steer_eval", action="store_true")
    parser.add_argument("--llamacpp_eval_model_repo_id", type=str, default="unsloth/Qwen3.5-27B-GGUF")
    parser.add_argument("--llamacpp_eval_model_filename", type=str, default="Qwen3.5-27B-Q8_0.gguf")
    parser.add_argument("--llamacpp_eval_n_ctx", type=int, default=2048)
    parser.add_argument("--llamacpp_eval_max_tokens", type=int, default=128)
    parser.add_argument("--llamacpp_eval_repeat_penalty", type=float, default=1.15)
    parser.add_argument("--llamacpp_eval_temperature", type=float, default=0.1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip_loss_mask", action="store_true")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Loaders for each steerer method
# ─────────────────────────────────────────────────────────────────────────────

def _load_pace_cbm(ckpt_path: str, *, device: torch.device) -> PaCECBM:
    if not ckpt_path:
        raise ValueError("--pace_ckpt is required when 'pace_cbm' is in --methods.")
    sidecar_path = Path(ckpt_path + ".meta.json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"sidecar meta missing: {sidecar_path}; run train_pace_cbm.py first."
        )
    sidecar = json.loads(sidecar_path.read_text())
    D, _ = load_dictionary(
        dictionary_path=sidecar["dictionary_path"],
        meta_path=sidecar["dictionary_meta_path"],
    )
    compute_dtype = {
        "float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16,
    }[sidecar.get("compute_dtype", "float32")]
    pace_cbm = PaCECBM(
        D=D,
        k=sidecar["bottleneck_k"],
        layer_idx=sidecar["layer_idx"],
        cf_offset=sidecar["cf_offset"],
        cf_size=sidecar["cf_size"],
        compute_dtype=compute_dtype,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    pace_cbm.load_state_dict(state, strict=True)
    pace_cbm.eval()
    return pace_cbm


def _load_vec_pack(method: str, *, layer_idx: int, vec_pack_root: str) -> torch.Tensor:
    pack_path = Path(vec_pack_root) / method / f"layer{layer_idx}" / "vec_pack.pt"
    if not pack_path.is_file():
        raise FileNotFoundError(
            f"{method} vec pack missing at {pack_path}; run train_steerers.py first."
        )
    return torch.load(pack_path, map_location="cpu")


def _load_transform_per_tag(method: str, *, layer_idx: int, transform_root: str, cf_concepts: List[str]) -> List:
    method_dir = Path(transform_root) / method / f"layer{layer_idx}"
    if not method_dir.is_dir():
        raise FileNotFoundError(
            f"{method} transform dir missing at {method_dir}; run train_steerers.py first."
        )
    out = []
    for tag in cf_concepts:
        path = method_dir / f"{tag}.pkl"
        if not path.is_file():
            out.append(None)
            continue
        with open(path, "rb") as f:
            out.append(pickle.load(f))
    if all(s is None for s in out):
        raise RuntimeError(
            f"No transform steerers loaded for {method} from {method_dir}."
        )
    return out


def _build_steerer_factory(
    method: str,
    *,
    layer_idx: int,
    args,
    cf_concepts: List[str],
    pace_cbm: Optional[PaCECBM],
):
    """Return a callable ``steer_mode -> HookSteerer`` for the unified eval API.

    For non-``none`` modes, returns the method's hook; ``"none"`` always falls
    back to ``NoSteer`` (so a single method+mode pair doubles as the baseline).
    Every concrete steerer is constructed with the chosen
    ``args.intervene_phase`` so the hook applies only at the right phase
    (default ``decode_only`` ⇒ skip prefill, steer only newly generated tokens).
    """
    phase = args.intervene_phase
    if method == "none":
        return lambda mode: NoSteer(layer_idx=layer_idx)
    if method == "pace_cbm":
        if pace_cbm is None:
            raise RuntimeError("pace_cbm method selected but PaCE-CBM module not loaded.")
        def _factory(mode: str) -> HookSteerer:
            if mode == "none":
                return NoSteer(layer_idx=layer_idx)
            return PaCECBMSteerer(pace_cbm, intervene_phase=phase)
        return _factory
    if method in VECTOR_METHODS:
        pack = _load_vec_pack(method, layer_idx=layer_idx, vec_pack_root=args.vec_pack_root)
        def _factory(mode: str) -> HookSteerer:
            if mode == "none":
                return NoSteer(layer_idx=layer_idx)
            return VecAddSteerer(
                pack, layer_idx=layer_idx, method_name=method, intervene_phase=phase,
            )
        return _factory
    if method in TRANSFORM_METHODS:
        steerers = _load_transform_per_tag(
            method, layer_idx=layer_idx,
            transform_root=args.transform_root, cf_concepts=cf_concepts,
        )
        def _factory(mode: str) -> HookSteerer:
            if mode == "none":
                return NoSteer(layer_idx=layer_idx)
            return TransformSteerer(
                steerers, layer_idx=layer_idx, method_name=method,
                alpha=args.steer_value, intervene_phase=phase,
            )
        return _factory
    raise ValueError(f"Unknown method: {method}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-method eval cascade
# ─────────────────────────────────────────────────────────────────────────────

def _run_method(
    method: str,
    *,
    args,
    llm,
    tokenizer,
    concept_set: List[str],
    test_dataset_holder: List,
    test_similarity: np.ndarray,
    test_loader,
    device: torch.device,
    pace_cbm: Optional[PaCECBM],
    debug_mode: bool,
    run_id: str,
    llamacpp_cache_dir: str,
    eval_debug: bool,
):
    print(f"\n{'#'*72}\n# evaluating method: {method}\n{'#'*72}")

    generate_use_cache = not args.eval_generate_no_kv_cache

    factory = _build_steerer_factory(
        method, layer_idx=args.layer_idx, args=args,
        cf_concepts=concept_set, pace_cbm=pace_cbm,
    )
    if eval_debug:
        sm = [m.strip() for m in args.steer_modes.split(",") if m.strip()]
        if method == "none":
            probe_mode = "none"
        else:
            probe_mode = next((m for m in sm if m != "none"), sm[0] if sm else "groundtruth")
        probe = factory(probe_mode)
        print(
            f"[eval-debug] cli/method_setup | method={method} | probe_mode={probe_mode!r} | "
            f"intervene_phase={args.intervene_phase!r} | generate_use_cache={generate_use_cache} | "
            f"probe_steerer={type(probe).__name__}",
            flush=True,
        )
        del probe

    if method == "pace_cbm" and pace_cbm is not None:
        run_concept_accuracy_pace(
            pace_cbm=pace_cbm,
            llm=llm,
            test_loader=test_loader,
            cf_concept_set=concept_set,
            cf_offset=pace_cbm.cf_offset,
            cf_size=pace_cbm.cf_size,
            device=device,
            test_similarity_np=test_similarity,
            eval_debug=eval_debug,
        )

    # Concept-metric steerer always uses ``"all"``: it runs a single forward
    # over the prompt (no autoregressive generation), so the hook must fire
    # to stash ``last_c_sparse`` regardless of T.
    pace_for_metrics = (
        PaCECBMSteerer(pace_cbm, intervene_phase="all")
        if (pace_cbm is not None and method == "pace_cbm")
        else None
    )
    cf_offset = pace_cbm.cf_offset if pace_cbm is not None else 0
    cf_size = pace_cbm.cf_size if pace_cbm is not None else len(concept_set)

    steer_modes = [m.strip() for m in args.steer_modes.split(",") if m.strip()]
    if method == "none":
        steer_modes = ["none"]

    cc_results = run_codecontests_testset_eval_steerable(
        steerer_factory=factory,
        llm=llm,
        tokenizer=tokenizer,
        concept_set=concept_set,
        test_dataset_holder=test_dataset_holder,
        seed=args.seed,
        batch_size=args.lcb_prompt_batch_size,
        model_label=f"{method}-l{args.layer_idx}",
        layer_idx=args.layer_idx,
        run_id=run_id,
        max_new_tokens=args.code_max_new_tokens,
        temperature=args.code_temperature,
        top_p=args.code_top_p,
        top_k=args.code_top_k,
        repetition_penalty=args.code_repetition_penalty,
        results_root=(args.code_results_root or None),
        display=not debug_mode,
        steer_modes=steer_modes,
        steer_value=args.steer_value,
        zero_other_concepts=args.zero_other_concepts,
        print_extracted_code_preview=args.print_extracted_code_preview,
        extracted_preview_chars=args.extracted_preview_chars,
        eval_log_host_memory=bool(debug_mode or args.eval_log_host_memory),
        pace_steerer_for_concept_metrics=pace_for_metrics,
        cf_offset=cf_offset,
        cf_size=cf_size,
        generate_use_cache=generate_use_cache,
        eval_debug=eval_debug,
    )

    cc_generations_by_mode: Dict[str, dict] = {}
    for mode in steer_modes:
        entry = cc_results.get(f"cc/{mode}")
        if isinstance(entry, dict) and isinstance(entry.get("generations"), dict):
            cc_generations_by_mode[mode] = entry["generations"]

    if not args.skip_lcb:
        run_lcb_eval_steerable(
            steerer_factory=factory,
            llm=llm,
            tokenizer=tokenizer,
            concept_set=concept_set,
            seed=args.seed,
            batch_size=args.lcb_prompt_batch_size,
            model_label=f"{method}-l{args.layer_idx}",
            layer_idx=args.layer_idx,
            run_id=run_id,
            display=not debug_mode,
            steer_modes=steer_modes,
            steer_value=args.steer_value,
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
            eval_debug=eval_debug,
        )

    for mode, payload in cc_generations_by_mode.items():
        texts = payload.get("raw_outputs") or []
        if not texts:
            print(f"{method}/{mode}: no raw outputs; skipping perplexity.")
            continue
        ppl = compute_perplexity(texts)
        if wandb.run is not None:
            wandb.log({f"{method}/cc/{mode}/{k}": v for k, v in ppl.items()})

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


def main():
    mp.set_start_method("spawn", force=False)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_mode = bool(args.debug)
    eval_debug = bool(args.eval_debug)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods:
        raise ValueError("--methods must be non-empty.")

    hf_cache_root = str(Path(args.hf_cache_root).expanduser())
    Path(hf_cache_root).mkdir(parents=True, exist_ok=True)
    dataset_cache_dir = resolve_cache_subdir(hf_cache_root, "datasets")
    model_cache_dir = resolve_cache_subdir(hf_cache_root, "models")
    llamacpp_cache_dir = resolve_cache_subdir(hf_cache_root, "llamacpp")

    use_wandb = not debug_mode
    if use_wandb:
        wandb.init(
            project="coding-qa",
            name=f"steerable-eval-l{args.layer_idx}-seed{args.seed}",
            config=vars(args),
        )
        run_id = wandb.run.id
    else:
        run_id = f"debug-{int(time.time())}"

    print("loading code_contests test split for evaluation ...")
    t_ds = time.perf_counter()
    raw_dataset = _hf_load_dataset_cache_first("deepmind/code_contests", dataset_cache_dir)
    if eval_debug:
        print(f"[eval-debug] cli/dataset | load_wall_s={time.perf_counter() - t_ds:.3f}s", flush=True)
    _, _, test_dataset = filter_codecontests(
        raw_dataset,
        cf_concept_lookup=CODEFORCES_CONCEPT_SET_LOOKUP,
        max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples,
        max_test_samples=args.max_test_samples,
    )
    if debug_mode:
        test_dataset = test_dataset.select(range(min(8, len(test_dataset))))

    concept_set = list(CODEFORCES_CONCEPT_SET)
    test_similarity = build_multihot(test_dataset, concept_set)

    tokenizer = AutoTokenizer.from_pretrained(
        LCB_LLAMA3_INSTRUCT_MODEL_ID, cache_dir=model_cache_dir, use_fast=False,
    )
    configure_code_eval_tokenizer(tokenizer)
    test_dummy = np.zeros((len(test_dataset), len(concept_set)), dtype=np.float32)
    test_loader = build_loaders_param(test_dataset, test_dummy, "test", tokenizer, args)

    print("loading frozen LlamaForCausalLM ...")
    t_lm = time.perf_counter()
    llm = LlamaForCausalLM.from_pretrained(
        LCB_LLAMA3_INSTRUCT_MODEL_ID,
        cache_dir=model_cache_dir,
        torch_dtype=torch.bfloat16,
    ).to(device)
    if eval_debug:
        p = next(llm.parameters())
        print(
            f"[eval-debug] cli/llm_loaded | wall_s={time.perf_counter() - t_lm:.3f}s | "
            f"device={p.device} | dtype={p.dtype}",
            flush=True,
        )
    for p in llm.parameters():
        p.requires_grad = False
    llm.eval()

    pace_cbm = _load_pace_cbm(args.pace_ckpt, device=device) if "pace_cbm" in methods else None

    test_dataset_holder = [test_dataset]
    for method in methods:
        _run_method(
            method,
            args=args,
            llm=llm,
            tokenizer=tokenizer,
            concept_set=concept_set,
            test_dataset_holder=test_dataset_holder,
            test_similarity=test_similarity,
            test_loader=test_loader,
            device=device,
            pace_cbm=pace_cbm,
            debug_mode=debug_mode,
            run_id=str(run_id),
            llamacpp_cache_dir=llamacpp_cache_dir,
            eval_debug=eval_debug,
        )
        if test_dataset_holder[0] is None:
            test_dataset_holder[0] = test_dataset


if __name__ == "__main__":
    main()
