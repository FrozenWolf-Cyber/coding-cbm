import argparse
import gc
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import evaluate
from tqdm.auto import tqdm
from datasets import load_dataset, DownloadConfig

from transformers import LlamaConfig, LlamaModel, AutoTokenizer, RobertaTokenizerFast, AutoModel, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from modules import CBLResidual, CBL, Roberta_classifier
from utils import (
    elastic_net_penalty,
    mean_pooling,
    eos_pooling,
    cos_sim_cubed,
    build_intervened_concepts_from_similarity,
    compute_multilabel_concept_metrics,
)
from eval_metrics import (
    _format_host_memory_stats,
    set_seed,
    run_concept_accuracy_cosine,
    run_weight_analysis,
    compute_perplexity,
    load_reward_model,
    run_rm_metrics_per_solution,
    run_llamacpp_judge_per_solution,
    run_codecontests_testset_evaluation_for_cbm,
    run_livecodebench_benchmark_generation_for_cbm,
)
from shared_code_prompt import (
    LCB_LLAMA3_INSTRUCT_MODEL_ID,
    build_lcb_user_prompt,
    format_lcb_llama3_instruct_prompt,
)
from config import CODEFORCES_CONCEPT_SET, CODEFORCES_CONCEPT_SET_LOOKUP
import wandb


parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This script only runs the HuggingFace deepmind/code_contests pipeline (no dataset switch).
DATASET = "code_contests"


def _select_random_python_solution(solutions_obj, row_idx: int, seed: int) -> str:
    """Pick one random Python solution (PY2=1 or PY3=3) for LM target text."""
    if not isinstance(solutions_obj, dict):
        return ""
    if "language" not in solutions_obj or "solution" not in solutions_obj:
        return ""
    languages = solutions_obj["language"] or []
    texts = solutions_obj["solution"] or []
    if not isinstance(languages, list) or not isinstance(texts, list):
        return ""

    py_candidates = [
        sol for lang, sol in zip(languages, texts)
        if lang in (1, 3) and isinstance(sol, str) and sol.strip()
    ]
    if not py_candidates:
        return ""

    rng = np.random.default_rng(seed + int(row_idx))
    pick = int(rng.integers(low=0, high=len(py_candidates)))
    return py_candidates[pick]


def _tokenize_supervised_row(row, row_idx: int, tokenizer, args):
    """One row of former batched _tok_train / _tok_valid (identical logic)."""
    desc = (row.get("description") or "").strip()
    sols = row["solutions"]
    user_body = build_lcb_user_prompt(
        problem_description=desc,
        starter_code="",
        language="python",
    )
    solution = _select_random_python_solution(sols, row_idx=row_idx, seed=args.seed)
    assistant_body = f"```python\n{solution}\n```" if solution else ""
    prompt_only = format_lcb_llama3_instruct_prompt(
        tokenizer=tokenizer,
        problem_description=desc,
        starter_code="",
        language="python",
    )
    prompt_ids = tokenizer(
        prompt_only,
        truncation=True,
        max_length=args.max_length,
    )["input_ids"]
    assistant_start = min(len(prompt_ids), args.max_length)
    messages = [
        {"role": "system", "content": "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."},
        {"role": "user", "content": user_body},
        {"role": "assistant", "content": assistant_body},
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
    )
    enc = tokenizer(
        formatted,
        truncation=True,
        max_length=args.max_length,
    )
    attn_mask = enc["attention_mask"]
    lm_mask = []
    for i in range(len(attn_mask) - 1):
        label_pos = i + 1
        use_label = int(attn_mask[label_pos] == 1 and label_pos >= assistant_start)
        lm_mask.append(use_label)
    return {
        "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        "loss_mask": torch.tensor(lm_mask, dtype=torch.long),
    }


def _tokenize_eval_row(row, tokenizer, args):
    """One row of former batched _tok_eval."""
    desc = (row.get("description") or "").strip()
    prompt = format_lcb_llama3_instruct_prompt(
        tokenizer=tokenizer,
        problem_description=desc,
        starter_code="",
        language="python",
    )
    enc = tokenizer(
        prompt,
        truncation=True,
        max_length=args.max_length,
    )
    return {
        "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
    }


parser.add_argument(
    "--max_train_samples",
    type=int,
    default=0,
    help="Optional: truncate code_contests train split to first N rows (0/<=0 disables).",
)
parser.add_argument(
    "--max_valid_samples",
    type=int,
    default=0,
    help="Optional: truncate code_contests valid split to first N rows (0/<=0 disables).",
)
parser.add_argument(
    "--max_test_samples",
    type=int,
    default=0,
    help="Optional: truncate code_contests test split to first N rows (0/<=0 disables).",
)
parser.add_argument(
    "--num_concepts",
    type=int,
    default=40,
    help="Number of top CF tags to use as concept set (by frequency in training split).",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=5,
    help="Number of training epochs (overrides config_finegrained.epoch for code_contests).",
)

parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epoch_multiplier", type=int, default=1, help="Epoch multiplier to increase total training steps (for debugging).")
parser.add_argument("--max_length", type=int, default=1024)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--discrimination_loss", type=float, default=1.0)
parser.add_argument("--neg_entropy_loss", type=float, default=1.0)
parser.add_argument("--concept_loss", type=float, default=1.0)
parser.add_argument("--word_loss", type=float, default=1.0)
parser.add_argument("--elastic_net_alpha", type=float, default=1.0)
parser.add_argument("--residual_dim", type=int, default=768)
parser.add_argument("--orthogonal_loss_weight", type=float, default=0)
parser.add_argument("--residual_penalty_weight", type=float, default=0)
parser.add_argument(
    "--debug",
    "--DEBUG",
    action="store_true",
    help=(
        "Debug mode: small train/test subset, 2 epochs and few train steps per epoch, "
        "disable wandb; evaluation still runs."
    ),
)
parser.add_argument(
    "--debug_0_step",
    "--DEBUG_0_STEP",
    action="store_true",
    help=(
        "Debug mode variant: small train/test subset, 2 epochs but 0 train steps per epoch "
        "(skips the optimization loop); disable wandb; evaluation still runs."
    ),
)
parser.add_argument("--intervention_gen_loss", type=float, default=0.0)
parser.add_argument("--no_detach_intervention", action='store_true', help="If set, do not detach unsup during intervention generation loss computation.")
parser.add_argument(
    "--cbl_layer_idx",
    type=int,
    default=-1,
    help=(
        "If -1 (default), keep the existing flow (CBL applied to the last hidden state, vocab via cbl.fc). "
        "If L >= 0, insert the CBL bottleneck between Llama layers L and L+1: "
        "concepts/unsup are read from h_L, projected back to hidden size via a new proj head, "
        "and (optionally) added back as a residual h_L_out = h_L_proj + h_L; the rest of Llama + the original lm_head produce vocab logits."
    ),
)
parser.add_argument(
    "--reconstruction_loss_weight",
    type=float,
    default=0.0,
    help=(
        "(Intermediate mode only) Weight on MSE(h_L_proj, h_L). "
        "When > 0 the bottleneck residual is dropped (h_L_out = h_L_proj) so the reconstruction loss "
        "drives the bottleneck to preserve h_L; mutually exclusive with --add_llama_logits in intermediate mode."
    ),
)
parser.add_argument(
    "--intervention_keep_other_concepts",
    action="store_true",
    help="If set, intervention overwrites only the selected concept(s) and keeps all other concept activations as-is (instead of setting them to 0).",
)
parser.add_argument(
    "--skip_loss_mask",
    action="store_true",
    help="If set, do not apply assistant-only loss_mask to concept_loss (word_loss masking is unchanged).",
)


parser.add_argument("--concept_loss_type", type=str, default="ce", help="Type of concept loss to use: 'cosine_cubed' or 'ce'.")

# Label sources
parser.add_argument("--labeling", type=str, default="direct", choices=["direct"], help="Concept label source. 'direct' uses CF tags from the dataset.")

parser.add_argument(
    "--add_llama_logits",
    action="store_true",
    help=(
        "If set, add the original Llama vocab projection logits (from the backbone hidden states) to the CBL/CBLResidual logits. "
        "This keeps CBL unchanged (no extra parameters) and acts like a residual-on-logits."
    ),
)
parser.add_argument("--rm_model_name", type=str, default="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                    help="HF id for sequence-classification reward model.")
parser.add_argument("--rm_batch_size", type=int, default=0, help="0 = score all texts per chunk in one forward.")
parser.add_argument("--rm_max_text_len", type=int, default=500)
parser.add_argument(
    "--skip_rm",
    action="store_true",
    help="Skip RM (Skywork) per-solution scoring on code_contests test-set generations.",
)
parser.add_argument(
    "--skip_llamacpp_steer_eval",
    action="store_true",
    help="Skip llama.cpp multi-label judge on code_contests test-set generations.",
)
parser.add_argument(
    "--llamacpp_eval_model_repo_id",
    type=str,
    default="unsloth/Qwen3.5-27B-GGUF",
    help="HF repo id for llama.cpp per-solution judge.",
)
parser.add_argument(
    "--llamacpp_eval_model_filename",
    type=str,
    default="Qwen3.5-27B-Q8_0.gguf",
    help="GGUF filename for llama.cpp per-solution judge.",
)
parser.add_argument(
    "--llamacpp_eval_n_ctx",
    type=int,
    default=2048,
    help="Context size for llama.cpp per-solution judge.",
)
parser.add_argument(
    "--llamacpp_eval_max_tokens",
    type=int,
    default=128,
    help="Max tokens for llama.cpp judge output (multi-label list).",
)
parser.add_argument(
    "--llamacpp_eval_repeat_penalty",
    type=float,
    default=1.15,
    help="Repeat penalty for llama.cpp per-solution judge.",
)
parser.add_argument(
    "--llamacpp_eval_temperature",
    type=float,
    default=0.1,
    help="Temperature for llama.cpp per-solution judge.",
)
parser.add_argument(
    "--code_results_root",
    type=str,
    default="",
    help="Optional root directory for code generation raw outputs and eval results.",
)
parser.add_argument(
    "--code_max_new_tokens",
    type=int,
    default=512,
    help="Max new tokens for final code generation.",
)
parser.add_argument(
    "--code_temperature",
    type=float,
    default=0.7,
    help="Sampling temperature for final code generation.",
)
parser.add_argument(
    "--code_top_p",
    type=float,
    default=0.9,
    help="Top-p for final code generation.",
)
parser.add_argument(
    "--code_top_k",
    type=int,
    default=100,
    help="Top-k for final code generation.",
)
parser.add_argument(
    "--code_repetition_penalty",
    type=float,
    default=1.05,
    help="Repetition penalty for final code generation.",
)
# ── LiveCodeBench args ────────────────────────────────────────────────────────
# See: https://github.com/livecodebench/livecodebench
# Place the repo at ./LiveCodeBench (next to this file) to enable LCB eval.
parser.add_argument(
    "--livecodebench_release",
    type=str,
    default="release_v6",
    help="LCB dataset release version tag (default: release_v6 for reproducible benchmarking).",
)
# Steering modes — pass a comma-separated list, e.g. "none,groundtruth" to run both.
# Valid values: "none" (unsteered baseline), "groundtruth" (CF-tag steering).
parser.add_argument(
    "--lcb_steer_modes",
    type=str,
    default="none,groundtruth",
    help="Comma-separated list of steering modes to evaluate: none,groundtruth.",
)
parser.add_argument(
    "--intervention_value",
    type=int,
    default=150,
    help="Magnitude applied to steered concepts (intervention-gen loss validation/train and LCB/code_contests eval).",
)
# LCB generation params — defaults match the leaderboard so numbers are comparable.
parser.add_argument(
    "--lcb_n_samples",
    type=int,
    default=10,
    help="Solutions per LCB problem (default 10, same as leaderboard).",
)
parser.add_argument(
    "--lcb_temperature",
    type=float,
    default=0.2,
    help="Sampling temperature for LCB eval (default 0.2, same as leaderboard).",
)
parser.add_argument(
    "--lcb_top_p",
    type=float,
    default=0.95,
    help="Top-p for LCB eval (default 0.95, same as leaderboard).",
)
parser.add_argument(
    "--lcb_max_new_tokens",
    type=int,
    default=2000,
    help="Max new tokens for LCB generation (default 2000, same as leaderboard).",
)
parser.add_argument(
    "--lcb_prompt_batch_size",
    type=int,
    default=1,
    help=(
        "How many LCB prompts to generate in each GPU batched forward pass "
        "(also used for batched code_contests generation; passed as batch_size to run_codecontests_evaluation_for_cbm)."
    ),
)
parser.add_argument(
    "--eval_log_host_memory",
    action="store_true",
    help=(
        "Append process RSS + system-used/available RAM to each [eval-mem] line (psutil). "
        "Enabled automatically with --debug; pass this flag to enable without debug."
    ),
)
parser.add_argument(
    "--print_extracted_code_preview",
    action="store_true",
    help=(
        "During final code contests + LCB evaluation, print a short excerpt of extracted code per "
        "sample for each problem (separated by ===== between samples)."
    ),
)
parser.add_argument(
    "--extracted_preview_chars",
    type=int,
    default=420,
    help="Max characters of extracted code to print per sample (with --print_extracted_code_preview).",
)
parser.add_argument(
    "--hf_cache_root",
    type=str,
    default="./.hf_cache",
    help="Local HuggingFace cache root used for datasets/models and llama.cpp GGUF files.",
)


class LazyTokenizedClassificationDataset(torch.utils.data.Dataset):
    """HF row dataset + supervision; tokenizes in __getitem__ to avoid materializing all token ids.

    `mode` is either ``"train"`` / ``"valid"`` (supervised chat + loss_mask) or ``"test"`` (prompt-only,
    no reference solution — same distinction as former _tok_train/_tok_valid vs _tok_eval).
    """

    def __init__(self, raw_dataset, s, mode: str, tokenizer, args):
        self.raw_dataset = raw_dataset
        self.s = s
        self.mode = mode
        self.tokenizer = tokenizer
        self.args = args

    def __getitem__(self, idx):
        row = self.raw_dataset[int(idx)]
        if self.mode == "test":
            t = _tokenize_eval_row(row, self.tokenizer, self.args)
        else:
            # train and valid share supervised formatting (historically identical _tok_train / _tok_valid).
            t = _tokenize_supervised_row(row, int(idx), self.tokenizer, self.args)
        y = torch.tensor(self.s[int(idx)], dtype=torch.float32)
        return t, y

    def __len__(self):
        return len(self.raw_dataset)


def _dynamic_padding_collate(batch):
    batch_text, batch_sim = zip(*batch)
    pad_id = tokenizer.pad_token_id
    max_len = max(int(x["input_ids"].shape[0]) for x in batch_text)

    input_ids = []
    attention_mask = []
    loss_mask = []
    has_loss_mask = all("loss_mask" in x for x in batch_text)

    for x in batch_text:
        cur_len = int(x["input_ids"].shape[0])
        pad_len = max_len - cur_len

        if pad_len > 0:
            ids = F.pad(x["input_ids"], (0, pad_len), value=pad_id)
            attn = F.pad(x["attention_mask"], (0, pad_len), value=0)
        else:
            ids = x["input_ids"]
            attn = x["attention_mask"]

        input_ids.append(ids)
        attention_mask.append(attn)

        if has_loss_mask:
            # loss_mask is aligned to shifted labels and has length seq_len - 1
            lm_target_len = max_len - 1
            cur_lm_len = int(x["loss_mask"].shape[0])
            lm_pad_len = lm_target_len - cur_lm_len
            if lm_pad_len > 0:
                lm = F.pad(x["loss_mask"], (0, lm_pad_len), value=0)
            else:
                lm = x["loss_mask"][:lm_target_len]
            loss_mask.append(lm)

    out_text = {
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
    }
    if has_loss_mask:
        out_text["loss_mask"] = torch.stack(loss_mask, dim=0)

    out_sim = torch.stack(batch_sim, dim=0)
    return out_text, out_sim


def build_loaders(raw_hf_dataset, s, mode, tokenizer, args):
    dataset = LazyTokenizedClassificationDataset(raw_hf_dataset, s, mode, tokenizer, args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True if mode == "train" else False,
        collate_fn=_dynamic_padding_collate,
    )
    return dataloader


def _resolve_cache_subdir(root_dir: str, name: str) -> str:
    path = Path(root_dir).expanduser() / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _hf_load_dataset_cache_first(dataset_name: str, cache_dir: str):
    try:
        print(f"[cache] loading dataset local-only from {cache_dir}: {dataset_name}")
        return load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            download_config=DownloadConfig(local_files_only=True),
        )
    except Exception as local_err:
        print(f"[cache] local dataset miss; downloading {dataset_name}: {local_err}")
        return load_dataset(dataset_name, cache_dir=cache_dir)


def _hf_from_pretrained_cache_first(loader_fn, model_id: str, cache_dir: str, **kwargs):
    local_kwargs = dict(kwargs)
    local_kwargs["cache_dir"] = cache_dir
    local_kwargs["local_files_only"] = True
    try:
        print(f"[cache] local-only load: {model_id} (cache_dir={cache_dir})")
        return loader_fn(model_id, **local_kwargs)
    except Exception as local_err:
        print(f"[cache] local miss; downloading {model_id}: {local_err}")
        remote_kwargs = dict(kwargs)
        remote_kwargs["cache_dir"] = cache_dir
        return loader_fn(model_id, **remote_kwargs)


def _zero_cbl_concept_and_unsup_branches_if_requested(cbl_model, enabled: bool) -> None:
    """Zero concept + unsupervised/residual branch params when logits-add mode is on.

    In last-layer mode this makes ``vocabs ≈ llama_logits`` at init.
    In intermediate mode (``cbl_layer_idx >= 0``) we *also* zero ``proj`` so that
    ``h_L_proj = 0`` ⇒ ``h_L_out = h_L`` (when use_residual=True) ⇒ the rest of
    Llama runs unchanged ⇒ vocabs equal the unmodified base model at step 0.
    """
    if not enabled:
        return
    with torch.no_grad():
        if isinstance(cbl_model, CBL):
            cbl_model.cbl.weight.zero_()
            cbl_model.cbl.bias.zero_()
            cbl_model.unsup.weight.zero_()
            cbl_model.unsup.bias.zero_()
            extra = ""
            if cbl_model.cbl_layer_idx >= 0 and cbl_model.proj is not None:
                cbl_model.proj.weight.zero_()
                cbl_model.proj.bias.zero_()
                extra = "+proj"
            print(f"[init] --add_llama_logits enabled: zeroed CBL concept+unsup{extra} branches")
        elif isinstance(cbl_model, CBLResidual):
            cbl_model.cbl.weight.zero_()
            cbl_model.cbl.bias.zero_()
            cbl_model.cbl_residual.weight.zero_()
            cbl_model.cbl_residual.bias.zero_()
            extra = ""
            if cbl_model.cbl_layer_idx >= 0 and cbl_model.proj is not None:
                cbl_model.proj.weight.zero_()
                cbl_model.proj.bias.zero_()
                extra = "+proj"
            print(f"[init] --add_llama_logits enabled: zeroed CBLResidual concept+residual{extra} branches")


def _save_cbl_with_meta(cbl_model, cbl_path: str) -> None:
    """Save the CBL state_dict AND a sidecar ``<cbl_path>.meta.json``.

    The sidecar persists ``cbl_layer_idx`` and ``use_residual`` so that eval-time
    loaders (eval_metrics.load_model_and_cbl, LiveCodeBench cbm_runner) can
    reconstruct the *exact* module shape (proj layer present iff intermediate
    mode) before calling ``load_state_dict(strict=True)``. Without this sidecar
    the loader would default to ``cbl_layer_idx=-1``, drop the saved ``proj.*``
    keys silently, and route eval through the never-trained ``cbl.fc`` head.
    """
    torch.save(cbl_model.state_dict(), cbl_path)
    meta = {
        "cbl_layer_idx": int(getattr(cbl_model, "cbl_layer_idx", -1)),
        "use_residual": bool(getattr(cbl_model, "use_residual", True)),
    }
    with open(cbl_path + ".meta.json", "w") as f:
        json.dump(meta, f)


if __name__ == "__main__":
    # Use spawn so LCB grading workers do not fork a CUDA parent process.
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    set_seed(args.seed)
    debug_mode = bool(args.debug or args.debug_0_step)

    # ── Bottleneck-mode sanity checks ─────────────────────────────────────────
    # A1 reconstruction (drop residual when recon weight > 0) is incompatible
    # with --add_llama_logits zero-init (which assumes the residual is on).
    if args.cbl_layer_idx >= 0 and args.reconstruction_loss_weight > 0 and args.add_llama_logits:
        raise ValueError(
            "--reconstruction_loss_weight > 0 (drops the bottleneck residual) is "
            "mutually exclusive with --add_llama_logits (which relies on the residual "
            "for identity-at-init). Disable one of them in intermediate mode."
        )
    # In intermediate mode without recon, the residual is on; otherwise the
    # bottleneck fully replaces h_L and the reconstruction loss must teach it
    # to preserve h_L.
    cbl_use_residual = not (args.cbl_layer_idx >= 0 and args.reconstruction_loss_weight > 0)
    hf_cache_root = str(Path(args.hf_cache_root).expanduser())
    Path(hf_cache_root).mkdir(parents=True, exist_ok=True)
    dataset_cache_dir = _resolve_cache_subdir(hf_cache_root, "datasets")
    model_cache_dir = _resolve_cache_subdir(hf_cache_root, "models")
    llamacpp_cache_dir = _resolve_cache_subdir(hf_cache_root, "llamacpp")

    use_wandb = not debug_mode
    if use_wandb:
        wandb.init(
            project="coding-qa",
            name=f"finegrained-{DATASET}-seed{args.seed}",
            config=vars(args),
        )
        run_name = wandb.run.id
    else:
        run_name = f"debug-{int(time.time())}"
        if args.debug_0_step:
            print("Debug mode enabled: disabling wandb logging and setting training to 2 epochs / 0 steps per epoch.")
        else:
            print("Debug mode enabled: disabling wandb logging and limiting training to 2 epochs / 2 steps per epoch.")

    def wandb_log(payload):
        if use_wandb:
            wandb.log(payload)

    # ─────────────────────────────────────────────────────────────
    # code_contests data loading (deepmind/code_contests via HuggingFace)
    # ─────────────────────────────────────────────────────────────
    data_loading_start = time.time()
    print("loading code_contests dataset from HuggingFace cache-first...")
    raw_dataset = _hf_load_dataset_cache_first("deepmind/code_contests", dataset_cache_dir)
    train_dataset_raw = raw_dataset["train"]
    valid_dataset_raw = raw_dataset["valid"]
    test_dataset_raw  = raw_dataset["test"]

    def _has_valid_cf_tag(example):
        tags = example["cf_tags"]
        return any(tag in CODEFORCES_CONCEPT_SET_LOOKUP for tag in tags)

    def _has_python_solution(example):
        solutions = example["solutions"]
        if not isinstance(solutions, dict):
            return False
        languages = solutions["language"]
        texts = solutions["solution"] or []
        return any(lang in (1, 3) and isinstance(sol, str) and sol.strip() for lang, sol in zip(languages, texts))

    train_dataset_for_length_stats = None

    if args.max_train_samples > 0:
        train_dataset_raw = train_dataset_raw.select(range(min(args.max_train_samples, len(train_dataset_raw))))
    if args.max_valid_samples > 0:
        valid_dataset_raw = valid_dataset_raw.select(range(min(args.max_valid_samples, len(valid_dataset_raw))))
    if args.max_test_samples > 0:
        test_dataset_raw = test_dataset_raw.select(range(min(args.max_test_samples, len(test_dataset_raw))))

    # DEBUG: small subset
    if debug_mode:
        train_dataset_raw = train_dataset_raw.select(range(min(64, len(train_dataset_raw))))
        test_dataset_raw  = test_dataset_raw.select(range(min(32, len(test_dataset_raw))))

    # Keep only rows with at least one allowed CF tag.
    filter_start = time.time()
    train_dataset_raw = train_dataset_raw.filter(_has_valid_cf_tag)
    valid_dataset_raw = valid_dataset_raw.filter(_has_valid_cf_tag)
    test_dataset_raw = test_dataset_raw.filter(_has_valid_cf_tag)
    # For training LM targets, drop rows without Python reference solutions.
    train_dataset_raw = train_dataset_raw.filter(_has_python_solution)
    filter_elapsed = time.time() - filter_start

    print(
        f"filtered dataset lengths | train: {len(train_dataset_raw)}, "
        f"valid: {len(valid_dataset_raw)}, test: {len(test_dataset_raw)}"
    )

    # ── Use hard static concept set from shared config ────────────────────
    concept_set = list(CODEFORCES_CONCEPT_SET)
    if not concept_set:
        raise ValueError("CODEFORCES_CONCEPT_SET is empty in config.py.")
    print(f"concept set ({len(concept_set)}): {concept_set[:10]} ...")

    concept_set_idx = {tag: i for i, tag in enumerate(concept_set)}

    # ── Build multi-hot supervision vectors ──────────────────────
    def _build_multihot(dataset):
        """Build (N, C) multi-hot supervision from CF tags.

        Rows with no tags in concept_set receive a uniform prior (1/C)
        so the cosine loss doesn't blow up on zero vectors.
        """
        n = len(dataset)
        sim = np.zeros((n, len(concept_set)), dtype=np.float32)
        for i in range(n):
            tags = dataset[i]["cf_tags"] or []
            for tag in tags:
                if tag in concept_set_idx:
                    sim[i, concept_set_idx[tag]] = 1.0
            if sim[i].sum() == 0:
                raise ValueError(f"Row {i} has no valid CF tags in concept_set. Check dataset filtering and concept_set for {DATASET}.")
        return sim

    train_similarity = _build_multihot(train_dataset_raw)
    test_similarity_for_eval = _build_multihot(test_dataset_raw)
    val_similarity = _build_multihot(valid_dataset_raw)

    print(f"train_similarity shape: {train_similarity.shape}")
    print(f"val_similarity shape: {val_similarity.shape}")
    print(f"test_similarity_for_eval shape: {test_similarity_for_eval.shape}")

    # Use train_dataset_raw / test_dataset_raw as the HF Dataset objects going forward
    train_dataset = train_dataset_raw
    valid_dataset = valid_dataset_raw
    test_dataset  = test_dataset_raw

    print("lazy tokenization (runs during DataLoader sampling; no upfront token-id cache).")

    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    config = _hf_from_pretrained_cache_first(
        LlamaConfig.from_pretrained,
        LCB_LLAMA3_INSTRUCT_MODEL_ID,
        model_cache_dir,
    )
    tokenizer = _hf_from_pretrained_cache_first(
        AutoTokenizer.from_pretrained,
        LCB_LLAMA3_INSTRUCT_MODEL_ID,
        model_cache_dir,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    tokenization_elapsed = 0.0

    # Raw HF row splits are tokenized lazily in DataLoader workers; release the umbrella dataset handle only.
    del raw_dataset
    del train_dataset_raw, valid_dataset_raw, test_dataset_raw
    gc.collect()

    # concept_set already built above from CF tags.
    print("concept len: ", len(concept_set))
    hm_stats = _format_host_memory_stats()
    if hm_stats:
        print(f"[post-data-prep] {hm_stats}", flush=True)

    d_name = DATASET.replace('/', '_')
    label_prefix = "./"   # unused for code_contests (supervision built directly from CF tags)
    # val_similarity already set above (None unless you want to add valid split eval)

    # Require exact alignment between concept-label rows and HF row datasets (lazy tokenization).
    assert int(np.asarray(train_similarity).shape[0]) == len(train_dataset), (
        f"train: concept-label rows ({int(np.asarray(train_similarity).shape[0])}) != dataset rows ({len(train_dataset)})"
    )
    assert int(np.asarray(val_similarity).shape[0]) == len(valid_dataset), (
        f"valid: concept-label rows ({int(np.asarray(val_similarity).shape[0])}) != dataset rows ({len(valid_dataset)})"
    )

    # Basic shape sanity checks.
    if train_similarity.ndim != 2 or train_similarity.shape[1] != len(concept_set):
        raise ValueError(
            f"Unexpected train_similarity shape {train_similarity.shape}; expected (N, {len(concept_set)}). "
            f"Check concept vectors / labels and concept_set for {DATASET}."
        )

    # NOTE: FEVER label-based concept masking is not applied for code_contests.
    # concept_set labels come directly from CF tags (no class-based masking needed).

    print("creating loader...")
    loader_start = time.time()
    train_loader = build_loaders(train_dataset, train_similarity, "train", tokenizer, args)
    valid_loader = build_loaders(valid_dataset, val_similarity, "valid", tokenizer, args)

    # test_loader is used for post-training analyses.
    # Supervision is multi-hot from CF tags (already built as test_similarity_for_eval).
    test_dummy_sim = np.zeros((len(test_dataset), len(concept_set)), dtype=np.float32)
    test_loader = build_loaders(test_dataset, test_dummy_sim, "test", tokenizer, args)
    loader_elapsed = time.time() - loader_start
    data_loading_elapsed = time.time() - data_loading_start
    print(
        "data loading timings (sec) | "
        f"filter: {filter_elapsed:.2f}, "
        f"tokenize: {tokenization_elapsed:.2f}, "
        f"dataloader: {loader_elapsed:.2f}, "
        f"total: {data_loading_elapsed:.2f}"
    )

    print("preparing backbone")
    preLM = _hf_from_pretrained_cache_first(
        LlamaModel.from_pretrained,
        LCB_LLAMA3_INSTRUCT_MODEL_ID,
        model_cache_dir,
        torch_dtype=torch.bfloat16,
    ).to(device)
    preLM = get_peft_model(preLM, lora_config)
    del lora_config
    preLM.print_trainable_parameters()
    lora_layers = filter(lambda p: p.requires_grad, preLM.parameters())
    opt_prelm = torch.optim.Adam(lora_layers, lr=5e-5)

    llama_vocab_weight = None
    # Always load the lm_head weights when running in intermediate mode (we use them
    # to project h_final → vocabs); otherwise only when --add_llama_logits is set.
    if args.add_llama_logits or args.cbl_layer_idx >= 0:
        # IMPORTANT: For Llama-3, lm_head weights are not necessarily tied to input embeddings.
        # We therefore grab the *output* projection (lm_head) weights from a CausalLM head.
        # This does not add parameters to CBL; it's just an external tensor used in forward.
        lm_head_model = _hf_from_pretrained_cache_first(
            AutoModelForCausalLM.from_pretrained,
            LCB_LLAMA3_INSTRUCT_MODEL_ID,
            model_cache_dir,
            torch_dtype=torch.bfloat16,
        ).to(device)
        llama_vocab_weight = lm_head_model.get_output_embeddings().weight.detach()
        del lm_head_model

    if args.discrimination_loss > 0:
        cbl = CBL(
            config, len(concept_set), tokenizer,
            cbl_layer_idx=args.cbl_layer_idx, use_residual=cbl_use_residual,
        ).to(device)
    else:
        cbl = CBLResidual(
            config, len(concept_set), args.residual_dim, tokenizer,
            cbl_layer_idx=args.cbl_layer_idx, use_residual=cbl_use_residual,
        ).to(device)
    _zero_cbl_concept_and_unsup_branches_if_requested(cbl, args.add_llama_logits)
    opt_cbl = torch.optim.Adam(cbl.parameters(), lr=5e-5)
    print("preparing classifier")
    total_params = sum(p.numel() for p in preLM.parameters())
    trainable_params = sum(p.numel() for p in preLM.parameters() if p.requires_grad)
    cbl_params = sum(p.numel() for p in cbl.parameters())
    trainable_params += cbl_params
    total_params += cbl_params
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params} = {trainable_params/total_params:.4f} of total")
    wandb_log({"trainable_parameters": trainable_params, "trainable_ratio": trainable_params/total_params})

    classifier = torch.nn.Linear(args.residual_dim, len(concept_set)).to(device)

    if args.discrimination_loss > 0:
        opt_classifier = torch.optim.Adam(classifier.parameters(), lr=1e-3)


    intervention_value = args.intervention_value

    print("start training...")
    best_loss = float('inf')
    d_name = DATASET.replace('/', '_')
    prefix = "./"
    prefix += "./from_pretained_llama3_lora_cbm_" + run_name
    prefix += "/"
    prefix += d_name
    prefix += "/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    model_name = "llama3"
    cbl_name = "cbl"



    start = time.time()
    best_epoch = -1
    epochs = 2 if debug_mode else args.num_epochs * args.epoch_multiplier
    debug_max_steps_per_epoch = 0 if args.debug_0_step else 2
    for e in range(epochs):
        print("Epoch ", e+1, ":")
        preLM.train()
        cbl.train()
        classifier.train()
        training_losses = {
            "concept_loss": [],
            "word_loss": [],
            "neg_entropy_loss": [],
            "reg_loss": [],
            "orthogonal_loss": [],
            "residual_penalty_loss": [],
            "intervention_gen_loss": [],
            "reconstruction_loss": [],
        }

    
        for i, (batch, batch_sim) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # If requested, skip the optimization loop entirely (true 0-step debug).
            if debug_mode and debug_max_steps_per_epoch == 0:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_sim = batch_sim.to(device)

            word_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["input_ids"][:, 1:])
            if "loss_mask" in batch:
                ignore = torch.full_like(word_label, -100)
                word_label = torch.where(batch["loss_mask"] > 0, word_label, ignore)
            if debug_mode:
                print(
                    f"[debug][train][epoch {e+1} step {i+1}] pre-preLM "
                    f"input_ids={tuple(batch['input_ids'].shape)} "
                    f"attention_mask={tuple(batch['attention_mask'].shape)} "
                    f"batch_sim={tuple(batch_sim.shape)} "
                    f"word_label={tuple(word_label.shape)}"
                )
                if "loss_mask" in batch:
                    print(
                        f"[debug][train][epoch {e+1} step {i+1}] pre-preLM "
                        f"loss_mask={tuple(batch['loss_mask'].shape)}"
                    )
            if args.cbl_layer_idx == -1:
                features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                llama_logits = F.linear(features, llama_vocab_weight) if llama_vocab_weight is not None else None
                concepts, unsup, vocabs, matched_unsup = cbl(features.float(), llama_logits=llama_logits)
                h_L = h_L_proj = None
            else:
                concepts, unsup, vocabs, matched_unsup, h_L, h_L_proj = cbl.forward_full(
                    preLM,
                    batch["input_ids"],
                    batch["attention_mask"],
                    llama_vocab_weight=llama_vocab_weight,
                )
                llama_logits = None
                features = None
            # print("concepts shape in training loop:", concepts.shape)
            # print("elastic_net_alphaunsup shape in training loop:", unsup.shape)
            # print("vocabs shape in training loop:", vocabs.shape)
        
            mask = (batch["attention_mask"][:, :-1] != 0).reshape(-1) # (B * (seq_len - 1))
            if (not args.skip_loss_mask) and ("loss_mask" in batch):
                # By default, supervise concept loss only on assistant tokens.
                # --skip_loss_mask restores previous behavior (all non-pad tokens).
                mask = mask & (batch["loss_mask"] > 0).reshape(-1)
            c_slice = concepts[:, :-1, :].contiguous().view(-1, concepts.shape[-1]) # (B * (seq_len - 1), C)
            batch_sim_slice = batch_sim.unsqueeze(1).expand(-1, concepts.shape[1] - 1, -1).contiguous().view(-1, batch_sim.shape[-1])
        
            valid_c = c_slice[mask]          # (N_valid, C)
            valid_sim = batch_sim_slice[mask]  # (N_valid, C)

            if valid_c.shape[0] == 0:
                concept_loss = torch.zeros((), device=device)
            elif args.concept_loss_type == "cosine_cubed":
                # Cosine-similarity-based concept loss against soft ACS labels
                concept_loss = -cos_sim_cubed(valid_c, valid_sim)
            elif args.concept_loss_type == "ce":
                # Cross-entropy concept loss using hard labels from ACS top concept
                hard_targets = torch.argmax(valid_sim, dim=-1)  # (N_valid,)
                concept_loss = torch.nn.CrossEntropyLoss()(valid_c, hard_targets)
            else:
                raise ValueError(f"Unknown concept_loss_type: {args.concept_loss_type}")
            word_loss = torch.nn.CrossEntropyLoss()(vocabs[:, :-1, :].reshape(-1, config.vocab_size), word_label.reshape(-1))
            loss = args.concept_loss * concept_loss + word_loss*args.word_loss
            # In intermediate mode the concept→output mapping lives on `proj` (Linear(C+U, H));
            # apply the same elastic-net penalty pattern to its concept slice.
            if args.cbl_layer_idx == -1:
                reg = elastic_net_penalty(cbl.fc.weight[:, :len(concept_set)])
            else:
                reg = elastic_net_penalty(cbl.proj.weight[:, :len(concept_set)])
        
            if matched_unsup is not None:
                orthogonal_loss = torch.cosine_similarity(concepts, matched_unsup, dim=-1).mean().abs() ## TODO: check shape
                loss += args.orthogonal_loss_weight * orthogonal_loss
                training_losses["orthogonal_loss"].append(orthogonal_loss.detach().cpu().numpy())
        
            if args.residual_penalty_weight > 0 and args.cbl_layer_idx == -1:
                residual_contrib = cbl.compute_residual_contrib(unsup)
                residual_penalty = torch.mean(torch.abs(residual_contrib)) ## TODO: check logic
                loss += args.residual_penalty_weight * residual_penalty
                training_losses["residual_penalty_loss"].append(residual_penalty.detach().cpu().numpy())
            elif args.residual_penalty_weight > 0 and args.cbl_layer_idx >= 0 and i == 0 and e == 0:
                print("[warn] --residual_penalty_weight > 0 is ignored in intermediate mode (cbl_layer_idx >= 0).")

            # Reconstruction loss (intermediate mode only): MSE(h_L_proj, h_L) with the same
            # assistant-only loss_mask as concept/word losses. Recon is per-position
            # (not shifted), so the shifted loss_mask must be left-padded with 0 to
            # convert "target at i+1 is assistant" → "current pos p is assistant".
            if args.cbl_layer_idx >= 0 and args.reconstruction_loss_weight > 0:
                recon_target = h_L.detach().to(h_L_proj.dtype)
                recon_diff = (h_L_proj - recon_target) ** 2
                recon_mask = batch["attention_mask"].bool()
                if (not args.skip_loss_mask) and ("loss_mask" in batch):
                    lm_full = F.pad(batch["loss_mask"], (1, 0), value=0).bool()
                    recon_mask = recon_mask & lm_full
                recon_mask_f = recon_mask.unsqueeze(-1).to(recon_diff.dtype)
                recon_denom = recon_mask_f.sum().clamp_min(1.0) * recon_diff.shape[-1]
                reconstruction_loss = (recon_diff * recon_mask_f).sum() / recon_denom
                loss += args.reconstruction_loss_weight * reconstruction_loss
                training_losses.setdefault("reconstruction_loss", []).append(
                    reconstruction_loss.detach().cpu().numpy()
                )

            if args.intervention_gen_loss > 0:
                ### concepts shapes: (B, seq_len, concept_dim)
                intervened_concept = build_intervened_concepts_from_similarity(
                    concepts=concepts,
                    batch_sim=batch_sim,
                    intervention_value=intervention_value,
                    keep_other_concepts=args.intervention_keep_other_concepts,
                )
                
                # print("intervened_concept shape: ", intervened_concept.shape, intervened_concept.max(), intervened_concept.min())
                if args.cbl_layer_idx == -1:
                    llama_logits_for_intervene = None
                    if llama_logits is not None:
                        llama_logits_for_intervene = llama_logits if args.no_detach_intervention else llama_logits.detach()
                    if args.no_detach_intervention:
                        vocab = cbl.intervene(unsup, intervened_concept.detach(), llama_logits=llama_logits_for_intervene)
                    else:
                        vocab = cbl.intervene(unsup.detach(), intervened_concept.detach(), llama_logits=llama_logits_for_intervene)
                else:
                    # Tail-only re-forward through layers[L+1:] + norm + lm_head.
                    unsup_for_intervene = unsup if args.no_detach_intervention else unsup.detach()
                    vocab = cbl.intervene_full(
                        preLM,
                        h_L.detach(),
                        batch["attention_mask"],
                        intervened_concept.detach(),
                        unsup_for_intervene,
                        llama_vocab_weight,
                    )
                intervention_gen_loss = torch.nn.CrossEntropyLoss()(vocab[:, :-1, :].reshape(-1, config.vocab_size), word_label.reshape(-1))
                loss += args.intervention_gen_loss * intervention_gen_loss
                training_losses["intervention_gen_loss"].append(intervention_gen_loss.detach().cpu().numpy())
            
            loss += args.elastic_net_alpha * reg
        
        
        
            opt_prelm.zero_grad()
            opt_cbl.zero_grad()
            loss.backward()
            opt_prelm.step()
            opt_cbl.step()

            if args.discrimination_loss > 0:
                classification = classifier(mean_pooling(unsup.detach(), batch["attention_mask"]))

                # Probe loss: train the classifier to predict finegrained concept similarities from unsup.
                # This keeps the probe consistent with the concept supervision and avoids class labels.
                if args.concept_loss_type == "cosine_cubed":
                    discrimination_loss = -cos_sim_cubed(classification, batch_sim)
                elif args.concept_loss_type == "ce":
                    hard_targets = torch.argmax(batch_sim, dim=-1)
                    discrimination_loss = torch.nn.CrossEntropyLoss()(classification, hard_targets)
                else:
                    raise ValueError(f"Unknown concept_loss_type: {args.concept_loss_type}")
                opt_classifier.zero_grad()
                (args.discrimination_loss * discrimination_loss).backward(inputs=list(classifier.parameters()))
                opt_classifier.step()

            if args.neg_entropy_loss > 0:
                if args.cbl_layer_idx == -1:
                    _, unsup, _, _ = cbl(features.detach().float())
                else:
                    # Intermediate mode: re-derive unsup from cached h_L without rerunning preLM.
                    unsup = cbl._unsup_branch(h_L.detach().float())
                classification = classifier(mean_pooling(unsup, batch["attention_mask"]))
                p = F.softmax(classification, dim=-1)
                neg_entropy_loss = torch.sum(p * torch.log(p), dim=-1).mean()
                opt_cbl.zero_grad()
                (args.neg_entropy_loss * neg_entropy_loss).backward(inputs=list(cbl.unsup.parameters()))
                opt_cbl.step()
                training_losses["neg_entropy_loss"].append(neg_entropy_loss.detach().cpu().numpy())


            training_losses["concept_loss"].append(concept_loss.detach().cpu().numpy())
            training_losses["word_loss"].append(word_loss.detach().cpu().numpy())
        
            training_losses["reg_loss"].append(reg.detach().cpu().numpy())
        
            log = {}
            for key in training_losses.keys():
                if len(training_losses[key]) > 0:
                    print(f"{key}: {training_losses[key][-1]}", end=" ")
                    log[key] = training_losses[key][-1]
            # print(" | batch ", i+1, " / ", len(train_loader), end="\r")
        
        
            log["epoch"] = e + 1
            log["batch"] = i + 1
            wandb_log(log)

            # Explicitly release per-step tensors to curb CUDA fragmentation/OOM in long runs.
            del (
                features,
                concepts,
                unsup,
                vocabs,
                matched_unsup,
                c_slice,
                batch_sim_slice,
                valid_c,
                valid_sim,
                loss,
                reg,
                concept_loss,
                word_loss,
                word_label,
                batch,
                batch_sim,
                h_L,
                h_L_proj,
            )
            if "reconstruction_loss" in locals():
                del reconstruction_loss
            if "classification" in locals():
                del classification
            if "discrimination_loss" in locals():
                del discrimination_loss
            if "neg_entropy_loss" in locals():
                del neg_entropy_loss
            if "residual_contrib" in locals():
                del residual_contrib
            if "residual_penalty" in locals():
                del residual_penalty
            if "intervened_concept" in locals():
                del intervened_concept
            if "vocab" in locals():
                del vocab
            if "intervention_gen_loss" in locals():
                del intervention_gen_loss
            if "orthogonal_loss" in locals():
                del orthogonal_loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
            if debug_mode and (i + 1) >= debug_max_steps_per_epoch:
                break
        
        
        avg_metrics = {}
        for key in training_losses.keys():
            if len(training_losses[key]) > 0:
                avg_metrics[key] = sum(training_losses[key]) / len(training_losses[key])
        print("Epoch ", e + 1, " training losses: ", avg_metrics)
        wandb_log({f"avg_{k}": avg_metrics[k] for k in avg_metrics.keys()})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Validation: concept-tag metrics + validation loss by epoch.
        preLM.eval()
        cbl.eval()
        val_preds = []
        val_targets = []
        val_losses = {
            "concept_loss": [],
            "word_loss": [],
            "reg_loss": [],
            "orthogonal_loss": [],
            "residual_penalty_loss": [],
            "intervention_gen_loss": [],
            "reconstruction_loss": [],
            "total_loss": [],
        }
        with torch.no_grad():
            for batch, batch_sim in tqdm(valid_loader, total=len(valid_loader), desc=f"valid/epoch_{e+1}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_sim = batch_sim.to(device)
                if args.cbl_layer_idx == -1:
                    val_features = preLM(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).last_hidden_state
                    val_llama_logits = F.linear(val_features, llama_vocab_weight) if llama_vocab_weight is not None else None
                    val_concepts, val_unsup, val_vocabs, val_matched_unsup = cbl(
                        val_features.float(), llama_logits=val_llama_logits
                    )
                    val_h_L = val_h_L_proj = None
                else:
                    val_concepts, val_unsup, val_vocabs, val_matched_unsup, val_h_L, val_h_L_proj = (
                        cbl.forward_full(
                            preLM,
                            batch["input_ids"],
                            batch["attention_mask"],
                            llama_vocab_weight=llama_vocab_weight,
                        )
                    )
                    val_features = None
                    val_llama_logits = None
                pooled_val_concepts = eos_pooling(val_concepts, batch["attention_mask"])
                val_preds.append(pooled_val_concepts.detach().cpu())
                val_targets.append(batch_sim.detach().cpu())

                val_word_label = torch.where(
                    batch["attention_mask"][:, :-1] == 0,
                    -100,
                    batch["input_ids"][:, 1:],
                )
                if "loss_mask" in batch:
                    val_ignore = torch.full_like(val_word_label, -100)
                    val_word_label = torch.where(batch["loss_mask"] > 0, val_word_label, val_ignore)
                val_mask = (batch["attention_mask"][:, :-1] != 0).reshape(-1)
                if (not args.skip_loss_mask) and ("loss_mask" in batch):
                    val_mask = val_mask & (batch["loss_mask"] > 0).reshape(-1)
                val_c_slice = val_concepts[:, :-1, :].contiguous().view(-1, val_concepts.shape[-1])
                val_batch_sim_slice = batch_sim.unsqueeze(1).expand(
                    -1, val_concepts.shape[1] - 1, -1
                ).contiguous().view(-1, batch_sim.shape[-1])
                val_valid_c = val_c_slice[val_mask]
                val_valid_sim = val_batch_sim_slice[val_mask]

                if val_valid_c.shape[0] == 0:
                    val_concept_loss = torch.zeros((), device=device)
                elif args.concept_loss_type == "cosine_cubed":
                    val_concept_loss = -cos_sim_cubed(val_valid_c, val_valid_sim)
                elif args.concept_loss_type == "ce":
                    val_hard_targets = torch.argmax(val_valid_sim, dim=-1)
                    val_concept_loss = torch.nn.CrossEntropyLoss()(val_valid_c, val_hard_targets)
                else:
                    raise ValueError(f"Unknown concept_loss_type: {args.concept_loss_type}")

                val_word_loss = torch.nn.CrossEntropyLoss()(
                    val_vocabs[:, :-1, :].reshape(-1, config.vocab_size),
                    val_word_label.reshape(-1),
                )
                if args.cbl_layer_idx == -1:
                    val_reg = elastic_net_penalty(cbl.fc.weight[:, :len(concept_set)])
                else:
                    val_reg = elastic_net_penalty(cbl.proj.weight[:, :len(concept_set)])

                val_orthogonal_loss = torch.zeros((), device=device)
                if val_matched_unsup is not None:
                    val_orthogonal_loss = torch.cosine_similarity(
                        val_concepts, val_matched_unsup, dim=-1
                    ).mean().abs()

                val_residual_penalty = torch.zeros((), device=device)
                if args.residual_penalty_weight > 0 and args.cbl_layer_idx == -1:
                    val_residual_contrib = cbl.compute_residual_contrib(val_unsup)
                    val_residual_penalty = torch.mean(torch.abs(val_residual_contrib))

                val_reconstruction_loss = torch.zeros((), device=device)
                if args.cbl_layer_idx >= 0 and args.reconstruction_loss_weight > 0:
                    val_recon_target = val_h_L.detach().to(val_h_L_proj.dtype)
                    val_recon_diff = (val_h_L_proj - val_recon_target) ** 2
                    val_recon_mask = batch["attention_mask"].bool()
                    if (not args.skip_loss_mask) and ("loss_mask" in batch):
                        # See training-loop comment: pad LEFT to realign shifted loss_mask
                        # to current-position (p ↔ loss_mask[p-1]).
                        val_lm_full = F.pad(batch["loss_mask"], (1, 0), value=0).bool()
                        val_recon_mask = val_recon_mask & val_lm_full
                    val_recon_mask_f = val_recon_mask.unsqueeze(-1).to(val_recon_diff.dtype)
                    val_recon_denom = val_recon_mask_f.sum().clamp_min(1.0) * val_recon_diff.shape[-1]
                    val_reconstruction_loss = (val_recon_diff * val_recon_mask_f).sum() / val_recon_denom

                val_intervention_gen_loss = torch.zeros((), device=device)
                if args.intervention_gen_loss > 0:
                    val_intervened_concept = build_intervened_concepts_from_similarity(
                        concepts=val_concepts,
                        batch_sim=batch_sim,
                        intervention_value=intervention_value,
                        keep_other_concepts=args.intervention_keep_other_concepts,
                    )
                    if args.cbl_layer_idx == -1:
                        val_intervene_vocab = cbl.intervene(
                            val_unsup.detach(),
                            val_intervened_concept.detach(),
                            llama_logits=val_llama_logits.detach() if val_llama_logits is not None else None,
                        )
                    else:
                        val_intervene_vocab = cbl.intervene_full(
                            preLM,
                            val_h_L.detach(),
                            batch["attention_mask"],
                            val_intervened_concept.detach(),
                            val_unsup.detach(),
                            llama_vocab_weight,
                        )
                    val_intervention_gen_loss = torch.nn.CrossEntropyLoss()(
                        val_intervene_vocab[:, :-1, :].reshape(-1, config.vocab_size),
                        val_word_label.reshape(-1),
                    )

                val_total_loss = (
                    args.concept_loss * val_concept_loss
                    + args.word_loss * val_word_loss
                    + args.elastic_net_alpha * val_reg
                    + args.orthogonal_loss_weight * val_orthogonal_loss
                    + args.residual_penalty_weight * val_residual_penalty
                    + args.intervention_gen_loss * val_intervention_gen_loss
                    + args.reconstruction_loss_weight * val_reconstruction_loss
                )

                val_losses["concept_loss"].append(float(val_concept_loss.detach().cpu().item()))
                val_losses["word_loss"].append(float(val_word_loss.detach().cpu().item()))
                val_losses["reg_loss"].append(float(val_reg.detach().cpu().item()))
                val_losses["orthogonal_loss"].append(float(val_orthogonal_loss.detach().cpu().item()))
                val_losses["residual_penalty_loss"].append(float(val_residual_penalty.detach().cpu().item()))
                val_losses["intervention_gen_loss"].append(float(val_intervention_gen_loss.detach().cpu().item()))
                val_losses["reconstruction_loss"].append(float(val_reconstruction_loss.detach().cpu().item()))
                val_losses["total_loss"].append(float(val_total_loss.detach().cpu().item()))

                del (
                    val_features,
                    val_concepts,
                    val_unsup,
                    val_vocabs,
                    val_matched_unsup,
                    val_c_slice,
                    val_batch_sim_slice,
                    val_valid_c,
                    val_valid_sim,
                    val_total_loss,
                    val_concept_loss,
                    val_word_loss,
                    val_reg,
                    val_orthogonal_loss,
                    val_residual_penalty,
                    val_intervention_gen_loss,
                    val_reconstruction_loss,
                    val_h_L,
                    val_h_L_proj,
                    val_word_label,
                    batch,
                    batch_sim,
                )
                if "val_llama_logits" in locals():
                    del val_llama_logits
                if "val_residual_contrib" in locals():
                    del val_residual_contrib
                if "val_intervened_concept" in locals():
                    del val_intervened_concept
                if "val_intervene_vocab" in locals():
                    del val_intervene_vocab
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            val_pred_tensor = torch.cat(val_preds, dim=0)
            val_target_tensor = torch.cat(val_targets, dim=0)
            val_topk = compute_multilabel_concept_metrics(
                prediction_scores=val_pred_tensor,
                target_scores=val_target_tensor,
                topk=(1, 5, 10),
            )
            val_log = {
                "valid_concept_top1_acc": val_topk["top1_acc"],
                "valid_concept_top5_acc": val_topk["top5_acc"],
                "valid_concept_top10_acc": val_topk["top10_acc"],
                "valid_concept_top1_iou": val_topk["top1_iou"],
                "valid_concept_top5_iou": val_topk["top5_iou"],
                "valid_concept_top10_iou": val_topk["top10_iou"],
                "valid_concept_cosine_raw": val_topk["cosine_raw"],
                "valid_concept_cosine_cubed": val_topk["cosine_cubed"],
                "valid_loss": (
                    sum(val_losses["total_loss"]) / len(val_losses["total_loss"])
                    if len(val_losses["total_loss"]) > 0 else float("inf")
                ),
                "epoch": e + 1,
            }
            for loss_key in ("concept_loss", "word_loss", "reg_loss", "orthogonal_loss", "residual_penalty_loss", "intervention_gen_loss", "reconstruction_loss"):
                if len(val_losses[loss_key]) > 0:
                    val_log[f"valid_{loss_key}"] = sum(val_losses[loss_key]) / len(val_losses[loss_key])
            print(
                f"Epoch {e + 1} validation concept metrics: "
                f"top1={val_topk['top1_acc']:.4f}, "
                f"top5={val_topk['top5_acc']:.4f}, "
                f"top10={val_topk['top10_acc']:.4f}, "
                f"iou@1={val_topk['top1_iou']:.4f}, "
                f"iou@5={val_topk['top5_iou']:.4f}, "
                f"iou@10={val_topk['top10_iou']:.4f}, "
                f"cos={val_topk['cosine_raw']:.4f}, "
                f"cos_cubed={val_topk['cosine_cubed']:.4f}, "
                f"valid_loss={val_log['valid_loss']:.6f}"
            )
            wandb_log(val_log)
            avg_metrics.update(val_log)
            del val_preds, val_targets, val_pred_tensor, val_target_tensor

        # Track and save best checkpoint by total averaged training objective.
        # (No validation loop exists in this script yet, so "best" is train-loss based.)
        avg_total_loss = (
            args.concept_loss * float(avg_metrics.get("concept_loss", 0.0))
            + args.word_loss * float(avg_metrics.get("word_loss", 0.0))
            + args.elastic_net_alpha * float(avg_metrics.get("reg_loss", 0.0))
            + args.orthogonal_loss_weight * float(avg_metrics.get("orthogonal_loss", 0.0))
            + args.residual_penalty_weight * float(avg_metrics.get("residual_penalty_loss", 0.0))
            + args.intervention_gen_loss * float(avg_metrics.get("intervention_gen_loss", 0.0))
            + args.reconstruction_loss_weight * float(avg_metrics.get("reconstruction_loss", 0.0))
        )
        wandb_log({"avg_total_loss": avg_total_loss})

        print("save model")
        preLM.save_pretrained(prefix + model_name + "_epoch_" + str(e + 1))
        _save_cbl_with_meta(cbl, prefix + cbl_name + "_epoch_" + str(e + 1) + ".pt")

        score_for_best = float(avg_metrics.get("valid_loss", float("inf")))
        if score_for_best < best_loss:
            best_loss = score_for_best
            best_epoch = e + 1
            preLM.save_pretrained(prefix + model_name + "_best")
            _save_cbl_with_meta(cbl, prefix + cbl_name + "_best.pt")
            print(f"New best checkpoint at epoch {best_epoch} (valid_loss={best_loss:.6f})")
            wandb_log({"best_epoch": best_epoch, "best_valid_loss": best_loss})

    if epochs == 0:
        # For raw-model benchmarking with 0 training epochs, persist an epoch_0 checkpoint
        # so downstream evaluation can load a local adapter path consistently.
        best_epoch = 0
        print("No training epochs requested; saving initial adapter and CBL as epoch_0 and best.")
        preLM.save_pretrained(prefix + model_name + "_epoch_0")
        _save_cbl_with_meta(cbl, prefix + cbl_name + "_epoch_0.pt")
        preLM.save_pretrained(prefix + model_name + "_best")
        _save_cbl_with_meta(cbl, prefix + cbl_name + "_best.pt")
        wandb_log({"best_epoch": best_epoch})

    end = time.time()
    print("time of training CBM:", (end - start) / 3600, "hours")

    ## delete training objects and free GPU before evaluation
    if llama_vocab_weight is not None:
        del llama_vocab_weight
        llama_vocab_weight = None
    del preLM, cbl, classifier, opt_prelm, opt_cbl

    if args.discrimination_loss > 0:
        del opt_classifier
    gc.collect()
    torch.cuda.empty_cache()

    ## lOAD BEST MODEL AND
    if best_epoch == -1:
        best_epoch = epochs
    preLM = _hf_from_pretrained_cache_first(
        LlamaModel.from_pretrained,
        LCB_LLAMA3_INSTRUCT_MODEL_ID,
        model_cache_dir,
        torch_dtype=torch.bfloat16,
    ).to(device)
    best_peft_path = prefix + model_name + "_best"
    if os.path.isdir(best_peft_path):
        peft_path = best_peft_path
    else:
        peft_path = prefix + model_name + "_epoch_" + str(best_epoch)
    preLM.load_adapter(peft_path)
    preLM.eval()

    llama_vocab_weight = None
    if args.add_llama_logits or args.cbl_layer_idx >= 0:
        from eval_metrics import get_llama_vocab_weight
        llama_vocab_weight = get_llama_vocab_weight(device)

    if args.discrimination_loss > 0:
        cbl = CBL(
            config, len(concept_set), tokenizer,
            cbl_layer_idx=args.cbl_layer_idx, use_residual=cbl_use_residual,
        ).to(device)
    else:
        cbl = CBLResidual(
            config, len(concept_set), args.residual_dim, tokenizer,
            cbl_layer_idx=args.cbl_layer_idx, use_residual=cbl_use_residual,
        ).to(device)
    best_cbl_path = prefix + cbl_name + "_best.pt"
    if os.path.isfile(best_cbl_path):
        cbl_state_path = best_cbl_path
    else:
        cbl_state_path = prefix + cbl_name + "_epoch_" + str(best_epoch) + ".pt"
    cbl.load_state_dict(torch.load(cbl_state_path, map_location=device))
    _zero_cbl_concept_and_unsup_branches_if_requested(cbl, args.add_llama_logits)
    cbl.eval()

    # ── Configure evaluation ──
    set_seed(args.seed)

    # ── Concept accuracy ──
    # Pass test multi-hot labels directly (built from CF tags above).
    run_concept_accuracy_cosine(
        preLM,
        cbl,
        test_loader,
        concept_set,
        label_prefix,
        device,
        test_similarity_np=test_similarity_for_eval,
        llama_vocab_weight=llama_vocab_weight,
    )

    # ── Weight analysis ──
    run_weight_analysis(cbl, concept_set, tokenizer)

    # ── Final generation: code_contests + LiveCodeBench outputs only (no grading) ──
    cc_generations_by_mode: dict = {}
    try:
        print(
            "[pre-code-eval] Dropping training loaders / HF train & valid splits / similarity matrices "
            "and duplicate dataset handles (test split held in a single ref for code_contests) ...",
            flush=True,
        )
        if train_dataset_for_length_stats is not None:
            del train_dataset_for_length_stats
            train_dataset_for_length_stats = None
        del train_loader, valid_loader, test_loader
        del train_dataset, valid_dataset
        del train_similarity, val_similarity, test_similarity_for_eval, test_dummy_sim
        # train/valid row datasets released; test split kept for code_contests eval below.
        _code_eval_test_holder = [test_dataset]
        del test_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        eval_log_host_memory = bool(debug_mode or args.eval_log_host_memory)
        if eval_log_host_memory:
            hm = _format_host_memory_stats()
            if hm:
                print(f"[pre-code-eval] {hm}", flush=True)
        print("[pre-code-eval] gc/cuda done; running code_contests test-set eval + LiveCodeBench separately ...", flush=True)

        lcb_steer_modes = [m.strip() for m in args.lcb_steer_modes.split(",") if m.strip()]

        steer_value = args.intervention_value
        print(f"Running code generation eval (steer_modes={lcb_steer_modes}) ...", flush=True)

        print("[pre-code-eval] Running code_contests test-set generation + metrics ...", flush=True)
        cc_results = run_codecontests_testset_evaluation_for_cbm(
            preLM=preLM,
            cbl=cbl,
            tokenizer=tokenizer,
            concept_set=concept_set,
            test_dataset_holder=_code_eval_test_holder,
            batch_size=args.lcb_prompt_batch_size,
            seed=args.seed,
            model_label=f"CBM-Llama3-{DATASET}",
            layer_idx=best_epoch,
            run_id=run_name,
            # code_contests generation params
            max_new_tokens=args.code_max_new_tokens,
            temperature=args.code_temperature,
            top_p=args.code_top_p,
            top_k=args.code_top_k,
            repetition_penalty=args.code_repetition_penalty,
            results_root=(args.code_results_root or None),
            llama_vocab_weight=llama_vocab_weight,
            display=not debug_mode,
            # Steering
            steer_modes=lcb_steer_modes,
            steer_value=steer_value,
            keep_other_concepts=args.intervention_keep_other_concepts,
            # Preview
            print_extracted_code_preview=args.print_extracted_code_preview,
            extracted_preview_chars=args.extracted_preview_chars,
            eval_log_host_memory=eval_log_host_memory,
        )

        # Per-mode payloads from the cc test-set run (raw_outputs, extracted_codes,
        # cf_tags_per_problem, problem_names) — consumed by the post-release evals
        # below once preLM/cbl are off the GPU.
        cc_generations_by_mode = {}
        if isinstance(cc_results, dict):
            for _mode in lcb_steer_modes:
                _entry = cc_results.get(f"cc/{_mode}")
                if isinstance(_entry, dict) and isinstance(_entry.get("generations"), dict):
                    cc_generations_by_mode[_mode] = _entry["generations"]

        # Ensure we can free HF test split refs even if the eval crashes.
        if isinstance(_code_eval_test_holder, list) and _code_eval_test_holder:
            _code_eval_test_holder[0] = None
        gc.collect()
        torch.cuda.empty_cache()

        # ── (2) LiveCodeBench benchmark ───────────────────────────────────────

        print("[pre-code-eval] Running LiveCodeBench benchmark generation + eval locks ...", flush=True)
        run_livecodebench_benchmark_generation_for_cbm(
            preLM=preLM,
            cbl=cbl,
            tokenizer=tokenizer,
            concept_set=concept_set,
            seed=args.seed,
            batch_size=args.lcb_prompt_batch_size,
            model_label=f"CBM-Llama3-{DATASET}",
            layer_idx=best_epoch,
            run_id=run_name,
            llama_vocab_weight=llama_vocab_weight,
            display=not debug_mode,
            # Steering
            steer_modes=lcb_steer_modes,
            steer_value=steer_value,
            keep_other_concepts=args.intervention_keep_other_concepts,
            # LiveCodeBench generation
            livecodebench_release=args.livecodebench_release,
            lcb_n_samples=args.lcb_n_samples,
            lcb_temperature=args.lcb_temperature,
            lcb_top_p=args.lcb_top_p,
            lcb_max_new_tokens=args.lcb_max_new_tokens,
            print_extracted_code_preview=args.print_extracted_code_preview,
            extracted_preview_chars=args.extracted_preview_chars,
            eval_log_host_memory=eval_log_host_memory,
        )
    except Exception as code_eval_err:
        import traceback
        print(f"Code generation evaluation failed (non-fatal):\n{traceback.format_exc()}")

    # ── Free model from GPU ──
    del preLM, cbl
    if llama_vocab_weight is not None:
        from eval_metrics import release_llama_vocab_weight
        release_llama_vocab_weight()
        llama_vocab_weight = None
    gc.collect()
    torch.cuda.empty_cache()

    # ── Per-solution evals on code_contests test-set generations ────────────
    # These run AFTER preLM/cbl are freed, so RM (~8B) and the perplexity LM (~8B)
    # have GPU room. Each mode (e.g. "none" before steering, "groundtruth" after
    # steering) is scored independently; metrics are keyed under cc/{steer_mode}/.
    if not cc_generations_by_mode:
        print("No code_contests test-set generations available; skipping per-solution evals.")
    else:
        # Perplexity per steer_mode (uses raw_outputs verbatim).
        for _mode, _payload in cc_generations_by_mode.items():
            _texts = _payload.get("raw_outputs") or []
            if not _texts:
                print(f"cc/{_mode}: no raw outputs; skipping perplexity.")
                continue
            try:
                print(f"\n[cc/{_mode}] Computing perplexity over {len(_texts)} test-set generations ...", flush=True)
                _ppl = compute_perplexity(_texts)
                wandb_log({f"cc/{_mode}/{k}": v for k, v in _ppl.items()})
            except Exception as ppl_err:
                print(f"cc/{_mode}: perplexity failed (non-fatal): {ppl_err}")

        # llama.cpp multi-label judge per solution, scored against ground-truth cf_tags.
        if not args.skip_llamacpp_steer_eval:
            try:
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
            except Exception as llama_eval_err:
                print(f"llama.cpp per-solution judge failed (non-fatal): {llama_eval_err}")
        else:
            print("Skipping llama.cpp per-solution judge.")

        # RM scoring per solution (relevance / grammar / together) with multi-concept prompts.
        if not args.skip_rm:
            try:
                rm_model, rm_tokenizer_rm = load_reward_model(args.rm_model_name, device)
                run_rm_metrics_per_solution(
                    generations_by_mode=cc_generations_by_mode,
                    concept_set=concept_set,
                    rm_model=rm_model,
                    rm_tokenizer=rm_tokenizer_rm,
                    rm_device=device,
                    rm_batch_size=args.rm_batch_size,
                    rm_max_text_len=args.rm_max_text_len,
                )
                del rm_model, rm_tokenizer_rm
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as rm_err:
                print(f"RM per-solution scoring failed (non-fatal): {rm_err}")
        else:
            print("Skipping RM per-solution scoring.")