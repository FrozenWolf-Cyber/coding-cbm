import argparse
import json
import os
import re
import pickle
import urllib.request
import numpy as np
from tqdm import tqdm
from llama_cpp import Llama


def parse_args():
    p = argparse.ArgumentParser(description="FEVER concept annotation with llama.cpp")
    p.add_argument("--restart", action="store_true")
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for FEVER train samples. Default: no cap.",
    )
    p.add_argument("--n-ctx", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=128,
                   help="128 is plenty for a single label line.")
    p.add_argument("--repeat-penalty", type=float, default=1.15)
    p.add_argument("--temperature", type=float, default=0.1)
    return p.parse_args()


ARGS = parse_args()

MODEL_REPO_ID = "unsloth/Qwen3.5-27B-GGUF"
MODEL_FILENAME = "Qwen3.5-27B-Q8_0.gguf"
CHECKPOINT_PATH = "fever_progress_llamacpp.pkl"
MAX_SAMPLES = ARGS.max_samples

if ARGS.restart and os.path.isfile(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print(f"Removed checkpoint: {CHECKPOINT_PATH}")

print(
    f"Loading model | n_ctx={ARGS.n_ctx} max_tokens={ARGS.max_tokens} "
    f"repeat_penalty={ARGS.repeat_penalty} temperature={ARGS.temperature}"
)

llm = Llama.from_pretrained(
    repo_id=MODEL_REPO_ID,
    filename=MODEL_FILENAME,
    n_gpu_layers=-1,
    n_ctx=ARGS.n_ctx,
    verbose=False,
)

# Qwen3 chat template tokens - manually formatted so we control the prefill
IM_START = "<|im_start|>"
IM_END   = "<|im_end|>"
NL = "\n"

SYSTEM_TEXT = (
    "You are a strict multi-label classifier. "
    "Output ONLY a single line of comma-separated labels copied verbatim from OPTIONS. "
    "No explanation, no preamble, no bullets."
)

# KEY FIX: prefill the assistant turn with a closed <think></think> block.
# This forces Qwen3 past its thinking phase at the token level.
# The model sees thinking already done and goes straight to answering.
ASSISTANT_PREFILL = "<think>\n\n</think>\n\n"


def build_raw_prompt(claim: str, concepts: list) -> str:
    opts_block = "\n".join(f"- {c}" for c in concepts)
    user_text = (
        f"Select ALL applicable labels from OPTIONS for the CLAIM.\n\n"
        f"OPTIONS:\n{opts_block}\n\n"
        f"CLAIM:\n{claim}\n\n"
        f"Answer (comma-separated labels verbatim from OPTIONS, nothing else):"
    )
    return (
        f"{IM_START}system{NL}{SYSTEM_TEXT}{IM_END}{NL}"
        f"{IM_START}user{NL}{user_text}{IM_END}{NL}"
        f"{IM_START}assistant{NL}{ASSISTANT_PREFILL}"
    )


# =========================
# FEVER concept sets
# =========================
FEVER_CONCEPTS_ALL = [
    "claim directly supported by verifiable documented evidence",
    "claim with explicit attribution to a named source or study",
    "claim asserting certainty on a contested or ambiguous question",
    "claim reflecting a widespread popular myth or misconception",
    "claim that contradicts established scientific or historical consensus",
    "claim generalized from anecdotal or single-case evidence",
    "claim presented as fact but lacking sufficient evidential basis",
    "claim under genuine empirical uncertainty with appropriate hedging",
]

SUPPORTS_CONCEPTS = FEVER_CONCEPTS_ALL[:3]
REFUTES_CONCEPTS  = [FEVER_CONCEPTS_ALL[i] for i in [3, 4, 5, 2]]
NEI_CONCEPTS      = [FEVER_CONCEPTS_ALL[i] for i in [5, 6, 7, 2]]


def normalize_label(label):
    if isinstance(label, (int, np.integer)):
        return ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"][min(int(label), 2)]
    s = str(label).strip().upper()
    return s if s in ("SUPPORTS", "REFUTES") else "NOT ENOUGH INFO"


def get_concepts(label):
    n = normalize_label(label)
    if n == "SUPPORTS": return SUPPORTS_CONCEPTS
    if n == "REFUTES":  return REFUTES_CONCEPTS
    return NEI_CONCEPTS


# =========================
# Model call — raw completion with prefill
# We use llm() directly (create_completion) so the prefilled assistant
# turn is passed as literal tokens, not interpreted by chat templates.
# =========================
def call_model(claim: str, concepts: list) -> str:
    prompt = build_raw_prompt(claim, concepts)
    out = llm(
        prompt,
        max_tokens=ARGS.max_tokens,
        temperature=ARGS.temperature,
        top_p=0.9,
        top_k=40,
        repeat_penalty=ARGS.repeat_penalty,
        stop=["<|im_end|>", "<|im_start|>", "\n\n"],
    )
    text = out["choices"][0]["text"]
    return text.strip() if text else ""


# =========================
# Parser — first non-empty line only
# =========================
def parse_output(output: str, concepts: list) -> list:
    first_line = next(
        (ln.strip() for ln in output.splitlines() if ln.strip()), ""
    )
    if not first_line:
        return [concepts[0]]

    parts = [p.strip() for p in first_line.split(",") if p.strip()]
    labels = []
    for p in parts:
        if p in concepts:
            if p not in labels: labels.append(p)
            continue
        m = next((c for c in concepts if c.lower() == p.lower()), None)
        if m:
            if m not in labels: labels.append(m)
            continue
        for c in concepts:
            if p.lower() in c.lower() or c.lower() in p.lower():
                if c not in labels: labels.append(c)
                break

    return labels if labels else [concepts[0]]


def to_vector(labels: list, concepts: list) -> np.ndarray:
    vec = np.zeros(len(FEVER_CONCEPTS_ALL), dtype=np.float32)
    selected = [c for c in labels if c in concepts] or [concepts[0]]
    for c in selected:
        try:
            vec[FEVER_CONCEPTS_ALL.index(c)] = 1.0
        except ValueError:
            pass
    s = float(vec.sum())
    if s > 0:
        vec /= s
    return vec


def log_sample(claim, raw_output, idx, total):
    print("\n============")
    print(f"sample_index={idx}  |  {idx+1} done, {total-idx-1} remaining")
    print("[claim]")
    print(claim)
    print("------------")
    print("[raw_model_output]")
    print(repr(raw_output))
    print("============\n", flush=True)


# =========================
# Dataset runner
# =========================
def run_dataset(dataset_name, dataset_url, local_path, checkpoint_path,
                output_prefix, max_samples=None):

    if ARGS.restart and os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)

    if not os.path.exists(local_path):
        print(f"Downloading {dataset_name} ...")
        urllib.request.urlretrieve(dataset_url, local_path)

    with open(local_path) as f:
        dataset = [json.loads(line) for line in f]
    if max_samples:
        dataset = dataset[:max_samples]

    all_vectors, all_claims, all_labels = [], [], []
    all_prompts, all_outputs_raw, all_parse_errors = [], [], []

    resumed_from = None
    if os.path.isfile(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        all_vectors      = ckpt.get("all_vectors", [])
        all_claims       = ckpt.get("all_claims", [])
        all_labels       = ckpt.get("all_labels", [])
        all_prompts      = ckpt.get("all_prompts", [])
        all_outputs_raw  = ckpt.get("all_outputs_raw", ckpt.get("all_outputs", []))
        all_parse_errors = ckpt.get("all_parse_errors", [])
        resumed_from = "checkpoint"
    else:
        # Fallback: if checkpoint is missing, recover progress from previous outputs
        # so reruns continue from the last completed index (e.g., 20k cap run).
        raw_json_path = f"{output_prefix}_raw_outputs_llamacpp.json"
        claims_npy_path = f"{output_prefix}_claims_llamacpp.npy"
        vectors_npy_path = f"{output_prefix}_concept_vectors_llamacpp.npy"

        if os.path.isfile(raw_json_path):
            with open(raw_json_path, "r") as f:
                prev = json.load(f)
            all_claims = list(prev.get("claims", []))
            all_labels = list(prev.get("fever_labels", []))
            all_prompts = list(prev.get("prompts", []))
            all_outputs_raw = list(prev.get("outputs_raw", prev.get("all_outputs", [])))
            all_parse_errors = list(prev.get("parse_errors", []))
            resumed_from = "raw_outputs_json"
        elif os.path.isfile(claims_npy_path):
            all_claims = np.load(claims_npy_path, allow_pickle=True).tolist()
            all_labels = ["NOT ENOUGH INFO"] * len(all_claims)
            all_prompts = [""] * len(all_claims)
            all_outputs_raw = [""] * len(all_claims)
            all_parse_errors = [""] * len(all_claims)
            resumed_from = "claims_npy"

        if os.path.isfile(vectors_npy_path):
            prev_vecs = np.load(vectors_npy_path, allow_pickle=True)
            all_vectors = [v for v in prev_vecs]
            if resumed_from is None:
                resumed_from = "vectors_npy"

    if resumed_from is not None:
        # Align lengths defensively in case files were written from different runs.
        lengths = [
            len(all_claims),
            len(all_labels),
            len(all_prompts),
            len(all_outputs_raw),
            len(all_parse_errors),
        ]
        if all_vectors:
            lengths.append(len(all_vectors))
        n = min(lengths) if lengths else 0

        all_claims = all_claims[:n]
        all_labels = all_labels[:n]
        all_prompts = all_prompts[:n]
        all_outputs_raw = all_outputs_raw[:n]
        all_parse_errors = all_parse_errors[:n]
        all_vectors = all_vectors[:n] if all_vectors else []
        print(f"[{dataset_name}] Resuming from {resumed_from}: {n} already done.")
    else:
        print(f"[{dataset_name}] Starting fresh.")

    start_idx = len(all_claims)
    if start_idx > len(dataset):
        print(
            f"[{dataset_name}] Saved progress ({start_idx}) exceeds dataset size ({len(dataset)}). "
            f"Clamping to dataset size."
        )
        start_idx = len(dataset)

    for i, ex in enumerate(
        tqdm(dataset[start_idx:], initial=start_idx, total=len(dataset), desc=dataset_name)
    ):
        global_idx = start_idx + i
        claim    = ex["claim"]
        label    = normalize_label(ex.get("label", "NOT ENOUGH INFO"))
        concepts = get_concepts(label)
        prompt   = build_raw_prompt(claim, concepts)
        parse_error = ""

        try:
            raw_output = call_model(claim, concepts)
        except Exception as e:
            print(f"[{dataset_name}][model-error] {type(e).__name__}: {str(e)[:200]}")
            raw_output = ""

        log_sample(claim, raw_output, global_idx, len(dataset))

        try:
            labels = parse_output(raw_output, concepts)
            vec    = to_vector(labels, concepts)
        except Exception as e:
            parse_error = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"[{dataset_name}][parse-error] {parse_error}")
            labels = [concepts[0]]
            vec    = to_vector(labels, concepts)

        all_vectors.append(vec)
        all_claims.append(claim)
        all_labels.append(label)
        all_prompts.append(prompt)
        all_outputs_raw.append(raw_output)
        all_parse_errors.append(parse_error)

        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                "all_vectors": all_vectors,
                "all_claims": all_claims,
                "all_labels": all_labels,
                "all_prompts": all_prompts,
                "all_outputs_raw": all_outputs_raw,
                "all_parse_errors": all_parse_errors,
            }, f)

    if all_vectors:
        np.save(f"{output_prefix}_concept_vectors_llamacpp.npy", np.stack(all_vectors))
        np.save(f"{output_prefix}_claims_llamacpp.npy", np.array(all_claims))

    with open(f"{output_prefix}_raw_outputs_llamacpp.json", "w") as f:
        json.dump({
            "claims": all_claims,
            "fever_labels": all_labels,
            "prompts": all_prompts,
            "outputs_raw": all_outputs_raw,
            "parse_errors": all_parse_errors,
        }, f, ensure_ascii=False)

    print(f"[{dataset_name}] DONE.")


# =========================
# Run
# =========================
run_dataset(
    dataset_name="FEVER_TRAIN",
    dataset_url="https://fever.ai/download/fever/train.jsonl",
    local_path="fever_train.jsonl",
    checkpoint_path="fever_progress_llamacpp.pkl",
    output_prefix="fever",
    max_samples=MAX_SAMPLES,
)

run_dataset(
    dataset_name="FEVER_PAPER_TEST",
    dataset_url="https://fever.ai/download/fever/paper_test.jsonl",
    local_path="fever_paper_test.jsonl",
    checkpoint_path="fever_paper_test_progress_llamacpp.pkl",
    output_prefix="fever_paper_test",
    max_samples=None,
)