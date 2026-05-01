import argparse
import multiprocessing as mp
import os
import time

import torch
import torch.nn.functional as F
import numpy as np
import evaluate
from tqdm.auto import tqdm
from datasets import load_dataset

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
from steerability_cache import save_all_steerability_texts, steerability_output_root
from eval_metrics import (
    set_seed,
    get_intervention_value,
    generate_steerability_texts,
    run_steerability_mpnet,
    run_concept_accuracy_cosine,
    run_weight_analysis,
    generate_perplexity_texts,
    compute_perplexity,
    load_reward_model,
    run_rm_metrics,
    run_steerability_llamacpp_judge,
    run_codecontests_evaluation_for_cbm,
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
    default=3,
    help="Number of training epochs (overrides config_finegrained.epoch for code_contests).",
)

parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epoch_multiplier", type=int, default=1, help="Epoch multiplier to increase total training steps (for debugging).")
parser.add_argument("--max_length", type=int, default=1024)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--samples_per_concept",
    type=int,
    default=50,
    help="Steerability evaluation: samples per concept. Default 50.",
)

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
parser.add_argument("--intervention_gen_loss", type=float, default=0.0)
parser.add_argument("--no_detach_intervention", action='store_true', help="If set, do not detach unsup during intervention generation loss computation.")
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
    "--mpnet_eval",
    action="store_true",
    help="Run MPNet-based steerability evaluation.",
)
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
parser.add_argument("--skip_rm", action="store_true", help="Skip RM reward evaluation after training.")
parser.add_argument(
    "--skip_llamacpp_steer_eval",
    action="store_true",
    help="Skip llama.cpp judge-based steerability evaluation.",
)
parser.add_argument(
    "--llamacpp_eval_model_repo_id",
    type=str,
    default="unsloth/Qwen3.5-27B-GGUF",
    help="HF repo id for llama.cpp steerability judge.",
)
parser.add_argument(
    "--llamacpp_eval_model_filename",
    type=str,
    default="Qwen3.5-27B-Q8_0.gguf",
    help="GGUF filename for llama.cpp steerability judge.",
)
parser.add_argument(
    "--llamacpp_eval_n_ctx",
    type=int,
    default=2048,
    help="Context size for llama.cpp steerability judge.",
)
parser.add_argument(
    "--llamacpp_eval_max_tokens",
    type=int,
    default=64,
    help="Max tokens for llama.cpp judge output.",
)
parser.add_argument(
    "--llamacpp_eval_repeat_penalty",
    type=float,
    default=1.15,
    help="Repeat penalty for llama.cpp steerability judge.",
)
parser.add_argument(
    "--llamacpp_eval_temperature",
    type=float,
    default=0.1,
    help="Temperature for llama.cpp steerability judge.",
)
parser.add_argument(
    "--skip_code_final_test",
    action="store_true",
    help="Skip final code generation + evaluation on code_contests test set.",
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
# Intervention value for LCB steering is taken from get_intervention_value(DATASET).
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
    "--lcb_num_process_evaluate",
    type=int,
    default=2,
    help=(
        "Parallel workers for LCB pass@k grading. "
        "Lower this on RAM-constrained hosts (for example, 2 on ~64Gi RAM)."
    ),
)
parser.add_argument(
    "--lcb_timeout",
    type=int,
    default=6,
    help="Per-test-case timeout (seconds) for LCB grading.",
)
parser.add_argument(
    "--lcb_recursion_limit",
    type=int,
    default=12000,
    help=(
        "Recursion limit injected into sandboxed LiveCodeBench code execution. "
        "Kept below extreme values to reduce C-stack segfault risk."
    ),
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


class ClassificationDataset(torch.utils.data.Dataset):
    """Thin wrapper around a HF Dataset + numpy supervision array.

    train_combined_finegrained.py previously assumed `encoded_text` was a dict of lists.
    Here `encoded_text` is a `datasets.Dataset`, so we index per-row.
    """

    def __init__(self, encoded_dataset, s):
        self.encoded_dataset = encoded_dataset
        self.s = s

    def __getitem__(self, idx):
        row = self.encoded_dataset[int(idx)]
        t = {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
        }
        if "loss_mask" in row:
            t["loss_mask"] = torch.tensor(row["loss_mask"], dtype=torch.long)
        y = torch.tensor(self.s[int(idx)], dtype=torch.float32)
        return t, y

    def __len__(self):
        return len(self.encoded_dataset)


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


def build_loaders(encoded_dataset, s, mode):
    dataset = ClassificationDataset(encoded_dataset, s)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True if mode == "train" else False,
        collate_fn=_dynamic_padding_collate,
    )
    return dataloader



if __name__ == "__main__":
    # Use spawn so LCB grading workers do not fork a CUDA parent process.
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()
    os.environ.setdefault("LCB_RECURSION_LIMIT", str(args.lcb_recursion_limit))
    set_seed(args.seed)
    debug_mode = args.debug

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
        print("Debug mode enabled: disabling wandb logging and limiting training to 2 epochs / 2 steps per epoch.")

    def wandb_log(payload):
        if use_wandb:
            wandb.log(payload)

    # ─────────────────────────────────────────────────────────────
    # code_contests data loading (deepmind/code_contests via HuggingFace)
    # ─────────────────────────────────────────────────────────────
    data_loading_start = time.time()
    print("loading code_contests dataset from HuggingFace...")
    raw_dataset = load_dataset("deepmind/code_contests")
    train_dataset_raw = raw_dataset["train"]
    valid_dataset_raw = raw_dataset["valid"]
    test_dataset_raw  = raw_dataset["test"]

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
    concept_set_for_similarity = concept_set
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

    print("tokenizing...")

    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    config = LlamaConfig.from_pretrained(LCB_LLAMA3_INSTRUCT_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(LCB_LLAMA3_INSTRUCT_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    def _select_random_python_solution(solutions_obj, row_idx: int) -> str:
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

        rng = np.random.default_rng(args.seed + int(row_idx))
        pick = int(rng.integers(low=0, high=len(py_candidates)))
        return py_candidates[pick]

    train_token_lengths = []

    def _tok_train(batch, indices):
        descriptions = batch["description"]
        solutions_all = batch["solutions"]
        formatted = []
        assistant_starts = []
        for desc, sols, row_idx in zip(descriptions, solutions_all, indices):
            desc = (desc or "").strip()
            user_body = build_lcb_user_prompt(
                problem_description=desc,
                starter_code="",
                language="python",
            )
            solution = _select_random_python_solution(sols, row_idx=row_idx)
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
            assistant_starts.append(min(len(prompt_ids), args.max_length))
            messages = [
                {"role": "system", "content": "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."},
                {"role": "user", "content": user_body},
                {"role": "assistant", "content": assistant_body},
            ]
            formatted.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=False,
                )
            )
        enc = tokenizer(
            formatted,
            truncation=True,
            max_length=args.max_length,
        )
        train_token_lengths.extend(len(ids) for ids in enc["input_ids"])
        loss_masks = []
        for attn_mask, assistant_start in zip(enc["attention_mask"], assistant_starts):
            # Shifted LM labels predict token i+1 from token i; supervise only assistant tokens.
            lm_mask = []
            for i in range(len(attn_mask) - 1):
                label_pos = i + 1
                use_label = int(attn_mask[label_pos] == 1 and label_pos >= assistant_start)
                lm_mask.append(use_label)
            loss_masks.append(lm_mask)
        enc["loss_mask"] = loss_masks
        return enc

    def _tok_valid(batch, indices):
        descriptions = batch["description"]
        solutions_all = batch["solutions"]
        formatted = []
        assistant_starts = []
        for desc, sols, row_idx in zip(descriptions, solutions_all, indices):
            desc = (desc or "").strip()
            user_body = build_lcb_user_prompt(
                problem_description=desc,
                starter_code="",
                language="python",
            )
            solution = _select_random_python_solution(sols, row_idx=row_idx)
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
            assistant_starts.append(min(len(prompt_ids), args.max_length))
            messages = [
                {"role": "system", "content": "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."},
                {"role": "user", "content": user_body},
                {"role": "assistant", "content": assistant_body},
            ]
            formatted.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=False,
                )
            )
        enc = tokenizer(
            formatted,
            truncation=True,
            max_length=args.max_length,
        )
        loss_masks = []
        for attn_mask, assistant_start in zip(enc["attention_mask"], assistant_starts):
            lm_mask = []
            for i in range(len(attn_mask) - 1):
                label_pos = i + 1
                use_label = int(attn_mask[label_pos] == 1 and label_pos >= assistant_start)
                lm_mask.append(use_label)
            loss_masks.append(lm_mask)
        enc["loss_mask"] = loss_masks
        return enc

    def _tok_eval(batch):
        descriptions = batch["description"]
        prompts = [
            format_lcb_llama3_instruct_prompt(
                tokenizer=tokenizer,
                problem_description=(desc or "").strip(),
                starter_code="",
                language="python",
            )
            for desc in descriptions
        ]
        return tokenizer(
            prompts,
            truncation=True,
            max_length=args.max_length,
        )

    tokenization_start = time.time()
    encoded_train_dataset = train_dataset.map(_tok_train, batched=True, with_indices=True, batch_size=1024)
    encoded_valid_dataset = valid_dataset.map(_tok_valid, batched=True, with_indices=True, batch_size=1024)
    encoded_test_dataset = test_dataset.map(_tok_eval, batched=True, batch_size=1024)
    if len(train_token_lengths) > 0:
        train_len_arr = np.asarray(train_token_lengths, dtype=np.int32)
        print(
            "train token length stats | "
            f"min: {int(train_len_arr.min())}, "
            f"mean: {float(train_len_arr.mean()):.2f}, "
            f"median: {float(np.median(train_len_arr)):.2f}, "
            f"max: {int(train_len_arr.max())}"
        )
    tokenization_elapsed = time.time() - tokenization_start

    # Keep only tensors (no label column in code_contests encoded datasets)
    keep_cols_train = {"input_ids", "attention_mask", "loss_mask"}
    keep_cols_valid = {"input_ids", "attention_mask", "loss_mask"}
    keep_cols_eval = {"input_ids", "attention_mask"}
    encoded_train_dataset = encoded_train_dataset.remove_columns([c for c in encoded_train_dataset.column_names if c not in keep_cols_train])
    encoded_valid_dataset = encoded_valid_dataset.remove_columns([c for c in encoded_valid_dataset.column_names if c not in keep_cols_valid])
    encoded_test_dataset = encoded_test_dataset.remove_columns([c for c in encoded_test_dataset.column_names if c not in keep_cols_eval])

    # concept_set already built above from CF tags; concept_set_for_similarity == concept_set
    print("concept len: ", len(concept_set))

    d_name = DATASET.replace('/', '_')
    label_prefix = "./"   # unused for code_contests (supervision built directly from CF tags)
    # val_similarity already set above (None unless you want to add valid split eval)

    # Require exact alignment between concept-label rows and tokenized dataset rows.
    assert int(np.asarray(train_similarity).shape[0]) == len(encoded_train_dataset), (
        f"train: concept-label rows ({int(np.asarray(train_similarity).shape[0])}) != tokenized dataset rows ({len(encoded_train_dataset)})"
    )
    assert int(np.asarray(val_similarity).shape[0]) == len(encoded_valid_dataset), (
        f"valid: concept-label rows ({int(np.asarray(val_similarity).shape[0])}) != tokenized dataset rows ({len(encoded_valid_dataset)})"
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
    train_loader = build_loaders(encoded_train_dataset, train_similarity, mode="train")
    valid_loader = build_loaders(encoded_valid_dataset, val_similarity, mode="valid")

    # test_loader is used for post-training analyses.
    # Supervision is multi-hot from CF tags (already built as test_similarity_for_eval).
    test_dummy_sim = np.zeros((len(encoded_test_dataset), len(concept_set)), dtype=np.float32)
    test_loader = build_loaders(encoded_test_dataset, test_dummy_sim, mode="test")
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
    preLM = LlamaModel.from_pretrained(LCB_LLAMA3_INSTRUCT_MODEL_ID, torch_dtype=torch.bfloat16).to(device)
    preLM = get_peft_model(preLM, lora_config)
    preLM.print_trainable_parameters()
    lora_layers = filter(lambda p: p.requires_grad, preLM.parameters())
    opt_prelm = torch.optim.Adam(lora_layers, lr=5e-5)

    llama_vocab_weight = None
    if args.add_llama_logits:
        # IMPORTANT: For Llama-3, lm_head weights are not necessarily tied to input embeddings.
        # We therefore grab the *output* projection (lm_head) weights from a CausalLM head.
        # This does not add parameters to CBL; it's just an external tensor used in forward.
        lm_head_model = AutoModelForCausalLM.from_pretrained(
            LCB_LLAMA3_INSTRUCT_MODEL_ID,
            torch_dtype=torch.bfloat16,
        ).to(device)
        llama_vocab_weight = lm_head_model.get_output_embeddings().weight.detach()
        del lm_head_model
    
    if args.discrimination_loss > 0:
        cbl = CBL(config, len(concept_set), tokenizer).to(device)
    else:
        cbl = CBLResidual(config, len(concept_set), args.residual_dim, tokenizer).to(device)
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


    intervention_value = get_intervention_value(DATASET)


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
    debug_max_steps_per_epoch = 2
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
        }

        
        for i, (batch, batch_sim) in tqdm(enumerate(train_loader), total=len(train_loader)):
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
            features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            llama_logits = F.linear(features, llama_vocab_weight) if llama_vocab_weight is not None else None
            concepts, unsup, vocabs, matched_unsup = cbl(features.float(), llama_logits=llama_logits)
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
            reg = elastic_net_penalty(cbl.fc.weight[:, :len(concept_set)])
            
            if matched_unsup is not None:
                orthogonal_loss = torch.cosine_similarity(concepts, matched_unsup, dim=-1).mean().abs() ## TODO: check shape
                loss += args.orthogonal_loss_weight * orthogonal_loss
                training_losses["orthogonal_loss"].append(orthogonal_loss.detach().cpu().numpy())
            
            if args.residual_penalty_weight > 0:
                residual_contrib = cbl.compute_residual_contrib(unsup)
                residual_penalty = torch.mean(torch.abs(residual_contrib)) ## TODO: check logic
                loss += args.residual_penalty_weight * residual_penalty
                training_losses["residual_penalty_loss"].append(residual_penalty.detach().cpu().numpy())
                
            if args.intervention_gen_loss > 0:
                ### concepts shapes: (B, seq_len, concept_dim)
                intervention_value = get_intervention_value(DATASET)

                intervened_concept = build_intervened_concepts_from_similarity(
                    concepts=concepts,
                    batch_sim=batch_sim,
                    intervention_value=intervention_value,
                    keep_other_concepts=args.intervention_keep_other_concepts,
                )
                    
                # print("intervened_concept shape: ", intervened_concept.shape, intervened_concept.max(), intervened_concept.min())
                llama_logits_for_intervene = None
                if llama_logits is not None:
                    llama_logits_for_intervene = llama_logits if args.no_detach_intervention else llama_logits.detach()

                if args.no_detach_intervention:
                    vocab = cbl.intervene(unsup, intervened_concept.detach(), llama_logits=llama_logits_for_intervene)
                else:
                    vocab = cbl.intervene(unsup.detach(), intervened_concept.detach(), llama_logits=llama_logits_for_intervene)
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
                _, unsup, _, _ = cbl(features.detach().float())
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
            
            if debug_mode and (i + 1) >= debug_max_steps_per_epoch:
                break
            
            
        avg_metrics = {}
        for key in training_losses.keys():
            if len(training_losses[key]) > 0:
                avg_metrics[key] = sum(training_losses[key]) / len(training_losses[key])
        print("Epoch ", e + 1, " training losses: ", avg_metrics)
        wandb_log({f"avg_{k}": avg_metrics[k] for k in avg_metrics.keys()})

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
            "total_loss": [],
        }
        with torch.no_grad():
            for batch, batch_sim in tqdm(valid_loader, total=len(valid_loader), desc=f"valid/epoch_{e+1}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_sim = batch_sim.to(device)
                val_features = preLM(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).last_hidden_state
                val_llama_logits = F.linear(val_features, llama_vocab_weight) if llama_vocab_weight is not None else None
                val_concepts, val_unsup, val_vocabs, val_matched_unsup = cbl(
                    val_features.float(), llama_logits=val_llama_logits
                )
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
                val_reg = elastic_net_penalty(cbl.fc.weight[:, :len(concept_set)])

                val_orthogonal_loss = torch.zeros((), device=device)
                if val_matched_unsup is not None:
                    val_orthogonal_loss = torch.cosine_similarity(
                        val_concepts, val_matched_unsup, dim=-1
                    ).mean().abs()

                val_residual_penalty = torch.zeros((), device=device)
                if args.residual_penalty_weight > 0:
                    val_residual_contrib = cbl.compute_residual_contrib(val_unsup)
                    val_residual_penalty = torch.mean(torch.abs(val_residual_contrib))

                val_intervention_gen_loss = torch.zeros((), device=device)
                if args.intervention_gen_loss > 0:
                    val_intervened_concept = build_intervened_concepts_from_similarity(
                        concepts=val_concepts,
                        batch_sim=batch_sim,
                        intervention_value=intervention_value,
                        keep_other_concepts=args.intervention_keep_other_concepts,
                    )
                    val_intervene_vocab = cbl.intervene(
                        val_unsup.detach(),
                        val_intervened_concept.detach(),
                        llama_logits=val_llama_logits.detach() if val_llama_logits is not None else None,
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
                )

                val_losses["concept_loss"].append(float(val_concept_loss.detach().cpu().item()))
                val_losses["word_loss"].append(float(val_word_loss.detach().cpu().item()))
                val_losses["reg_loss"].append(float(val_reg.detach().cpu().item()))
                val_losses["orthogonal_loss"].append(float(val_orthogonal_loss.detach().cpu().item()))
                val_losses["residual_penalty_loss"].append(float(val_residual_penalty.detach().cpu().item()))
                val_losses["intervention_gen_loss"].append(float(val_intervention_gen_loss.detach().cpu().item()))
                val_losses["total_loss"].append(float(val_total_loss.detach().cpu().item()))

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
            for loss_key in ("concept_loss", "word_loss", "reg_loss", "orthogonal_loss", "residual_penalty_loss", "intervention_gen_loss"):
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

        # Track and save best checkpoint by total averaged training objective.
        # (No validation loop exists in this script yet, so "best" is train-loss based.)
        avg_total_loss = (
            args.concept_loss * float(avg_metrics.get("concept_loss", 0.0))
            + args.word_loss * float(avg_metrics.get("word_loss", 0.0))
            + args.elastic_net_alpha * float(avg_metrics.get("reg_loss", 0.0))
            + args.orthogonal_loss_weight * float(avg_metrics.get("orthogonal_loss", 0.0))
            + args.residual_penalty_weight * float(avg_metrics.get("residual_penalty_loss", 0.0))
            + args.intervention_gen_loss * float(avg_metrics.get("intervention_gen_loss", 0.0))
        )
        wandb_log({"avg_total_loss": avg_total_loss})

        print("save model")
        preLM.save_pretrained(prefix + model_name + "_epoch_" + str(e + 1))
        torch.save(cbl.state_dict(), prefix + cbl_name + "_epoch_" + str(e + 1) + ".pt")

        score_for_best = float(avg_metrics.get("valid_loss", float("inf")))
        if score_for_best < best_loss:
            best_loss = score_for_best
            best_epoch = e + 1
            preLM.save_pretrained(prefix + model_name + "_best")
            torch.save(cbl.state_dict(), prefix + cbl_name + "_best.pt")
            print(f"New best checkpoint at epoch {best_epoch} (valid_loss={best_loss:.6f})")
            wandb_log({"best_epoch": best_epoch, "best_valid_loss": best_loss})

    end = time.time()
    print("time of training CBM:", (end - start) / 3600, "hours")
    
    ## delete training objects and free GPU before evaluation
    import gc
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
    preLM = LlamaModel.from_pretrained(LCB_LLAMA3_INSTRUCT_MODEL_ID, torch_dtype=torch.bfloat16).to(device)
    best_peft_path = prefix + model_name + "_best"
    if os.path.isdir(best_peft_path):
        peft_path = best_peft_path
    else:
        peft_path = prefix + model_name + "_epoch_" + str(best_epoch)
    preLM.load_adapter(peft_path)
    preLM.eval()

    llama_vocab_weight = None
    if args.add_llama_logits:
        from eval_metrics import get_llama_vocab_weight
        llama_vocab_weight = get_llama_vocab_weight(device)

    if args.discrimination_loss > 0:
        cbl = CBL(config, len(concept_set), tokenizer).to(device)
    else:
        cbl = CBLResidual(config, len(concept_set), args.residual_dim, tokenizer).to(device)
    best_cbl_path = prefix + cbl_name + "_best.pt"
    if os.path.isfile(best_cbl_path):
        cbl_state_path = best_cbl_path
    else:
        cbl_state_path = prefix + cbl_name + "_epoch_" + str(best_epoch) + ".pt"
    cbl.load_state_dict(torch.load(cbl_state_path, map_location=device))
    cbl.eval()

    # ── Configure evaluation ──
    intervention_value = get_intervention_value(DATASET)
    num_steerability_samples = (
        max(1, args.samples_per_concept)
        if args.samples_per_concept is not None
        else max(1, 100 // len(concept_set))
    )
    steer_root = steerability_output_root(os.path.normpath(prefix.rstrip("/")), best_epoch, False)
    print(f"Steerability sample cache: {steer_root}")

    # ── Generate steerability texts (cached) ──
    set_seed(args.seed)
    decoded_texts_by_concept = generate_steerability_texts(
        preLM, cbl, tokenizer, concept_set, DATASET, device,
        samples_per_concept=num_steerability_samples,
        llama_vocab_weight=llama_vocab_weight,
        keep_other_concepts=args.intervention_keep_other_concepts,
        steerability_cache_dir=steer_root,
        steerability_cache_seed=args.seed,
        interventions_per_batch=50,
    )

    # ── Generate perplexity texts (cached) ──
    ppl_texts = generate_perplexity_texts(
        cbl, preLM, tokenizer, args.seed, device,
        cache_dir=prefix, run_name=run_name,
        llama_vocab_weight=llama_vocab_weight,
    )

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

    # ── Final test: code_contests test set + LiveCodeBench (unsteered & steered) ──
    if not args.skip_code_final_test:
        try:
            lcb_steer_modes = [m.strip() for m in args.lcb_steer_modes.split(",") if m.strip()]
            print(f"Running code generation evaluation  (steer_modes={lcb_steer_modes}) ...")
            run_codecontests_evaluation_for_cbm(
                preLM=preLM,
                cbl=cbl,
                tokenizer=tokenizer,
                concept_set=concept_set,
                test_dataset=test_dataset,
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
                steer_value=get_intervention_value(DATASET),
                keep_other_concepts=args.intervention_keep_other_concepts,
                # LiveCodeBench
                livecodebench_release=args.livecodebench_release,
                lcb_n_samples=args.lcb_n_samples,
                lcb_temperature=args.lcb_temperature,
                lcb_top_p=args.lcb_top_p,
                lcb_max_new_tokens=args.lcb_max_new_tokens,
                lcb_num_process_evaluate=args.lcb_num_process_evaluate,
                lcb_timeout=args.lcb_timeout,
                print_extracted_code_preview=args.print_extracted_code_preview,
                extracted_preview_chars=args.extracted_preview_chars,
            )
        except Exception as code_eval_err:
            import traceback
            print(f"Code generation evaluation failed (non-fatal):\n{traceback.format_exc()}")
    else:
        print("Skipping final code generation testing.")

    # ── Free model from GPU ──
    del preLM, cbl
    if llama_vocab_weight is not None:
        from eval_metrics import release_llama_vocab_weight
        release_llama_vocab_weight()
        llama_vocab_weight = None
    gc.collect()
    torch.cuda.empty_cache()

    # ── Steerability scoring (MPNet similarity) ──
    if args.mpnet_eval:
        run_steerability_mpnet(
            decoded_texts_by_concept, concept_set_for_similarity,
            intervention_value, args.max_length, device,
        )
    else:
        print("Skipping MPNet steerability evaluation.")

    # ── Steerability scoring (llama.cpp judge) ──
    if not args.skip_llamacpp_steer_eval:
        try:
            run_steerability_llamacpp_judge(
                decoded_texts_by_concept=decoded_texts_by_concept,
                concept_set=concept_set,
                model_repo_id=args.llamacpp_eval_model_repo_id,
                model_filename=args.llamacpp_eval_model_filename,
                n_ctx=args.llamacpp_eval_n_ctx,
                max_tokens=args.llamacpp_eval_max_tokens,
                repeat_penalty=args.llamacpp_eval_repeat_penalty,
                temperature=args.llamacpp_eval_temperature,
            )
        except Exception as llama_eval_err:
            print(f"llama.cpp steerability evaluation failed (non-fatal): {llama_eval_err}")
    else:
        print("Skipping llama.cpp steerability evaluation.")

    # ── Perplexity computation (evaluate library loads its own LLM) ──
    compute_perplexity(ppl_texts)

    # ── RM reward scoring (optional) ──
    if not args.skip_rm:
        try:
            rm_model, rm_tokenizer_rm = load_reward_model(args.rm_model_name, device)
            run_rm_metrics(
                decoded_texts_by_concept, concept_set,
                rm_model, rm_tokenizer_rm, device,
                rm_batch_size=args.rm_batch_size,
                rm_max_text_len=args.rm_max_text_len,
            )
            del rm_model, rm_tokenizer_rm
            torch.cuda.empty_cache()
        except Exception as rm_err:
            print(f"RM evaluation failed (non-fatal): {rm_err}")

    # ── Save steerability text cache ──
    save_all_steerability_texts(steer_root, args.seed, concept_set, decoded_texts_by_concept)