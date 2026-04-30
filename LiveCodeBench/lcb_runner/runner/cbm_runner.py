"""CBM (Concept Bottleneck Model) runner for LiveCodeBench.

Wraps a trained preLM + CBL checkpoint so it satisfies the BaseRunner interface
and can be driven by lcb_runner/runner/main.py exactly like any other model.

Usage
-----
From the livecodebench repo root:

    python -m lcb_runner.runner.main \
        --model CBM-LLaMA3-8B-unsteered \
        --cbm_project_root /path/to/cbm_codebase \
        --cbm_peft_path    /path/to/peft_checkpoint_epoch_3 \
        --cbm_cbl_path     /path/to/cbl_epoch_3.pt \
        --cbm_concept_set_json /path/to/concept_set.json \
        --scenario codegeneration \
        --n 10 \
        --temperature 0.2 \
        --evaluate

Steering modes (set via --cbm_steer_mode):
    "none"  – unsteered baseline
    "probe" – one forward pass → amplify top-k activated concepts
    "mpnet" – MPNet embedding similarity → amplify top-k nearest concept names
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

from lcb_runner.runner.base_runner import BaseRunner
from lcb_runner.lm_styles import LanguageModel


# ── helpers (lazy imports so non-CBM runs don't pay the cost) ─────────────────

def _load_cbm_modules(project_root: str):
    """Add CBM project root to sys.path and return the key module classes."""
    root = str(Path(project_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from modules import CBL, CBLResidual
    from utils import eos_pooling, mean_pooling
    return CBL, CBLResidual, eos_pooling, mean_pooling


# ── MPNet concept feature cache (module-level, shared across instances) ────────

_MPNET_TOK = None
_MPNET_MODEL = None
_MPNET_CONCEPT_FEATS: Optional[torch.Tensor] = None
_MPNET_CONCEPT_KEY: Optional[tuple] = None


def _get_mpnet_concept_feats(concept_set: List[str], device: torch.device):
    global _MPNET_TOK, _MPNET_MODEL, _MPNET_CONCEPT_FEATS, _MPNET_CONCEPT_KEY
    from transformers import AutoTokenizer, AutoModel

    key = tuple(concept_set)
    if _MPNET_TOK is None:
        print("[CBMRunner/MPNet] Loading sentence-transformers/all-mpnet-base-v2 ...")
        _MPNET_TOK = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        _MPNET_MODEL = (
            AutoModel.from_pretrained(
                "sentence-transformers/all-mpnet-base-v2", torch_dtype=torch.float32
            )
            .to(device)
            .eval()
        )

    if _MPNET_CONCEPT_FEATS is None or _MPNET_CONCEPT_KEY != key:
        enc = _MPNET_TOK(
            concept_set, padding=True, truncation=True, max_length=32, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = _MPNET_MODEL(**enc)
        # Mean-pool token embeddings
        mask = enc["attention_mask"].unsqueeze(-1).float()
        feats = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        _MPNET_CONCEPT_FEATS = F.normalize(feats, p=2, dim=1)  # (C, D)
        _MPNET_CONCEPT_KEY = key

    return _MPNET_TOK, _MPNET_MODEL, _MPNET_CONCEPT_FEATS


class CBMRunner(BaseRunner):
    """LiveCodeBench runner backed by a trained Concept Bottleneck Model.

    The runner overrides ``run_batch`` (not ``_run_single``) because cbl.generate_batch
    already handles multiple samples in one call, avoiding the per-sample overhead of
    BaseRunner's sequential loop.
    """

    def __init__(self, args, model: LanguageModel):
        super().__init__(args, model)

        from transformers import LlamaConfig, LlamaModel, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Load CBM code ──────────────────────────────────────────────────
        CBL, CBLResidual, self._eos_pooling, self._mean_pooling = _load_cbm_modules(
            args.cbm_project_root
        )

        # ── Load concept set ───────────────────────────────────────────────
        with open(args.cbm_concept_set_json, "r") as f:
            self.concept_set: List[str] = json.load(f)
        C = len(self.concept_set)

        # ── Tokenizer ─────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", padding_side="left", use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── preLM (LLaMA-3 backbone + LoRA adapter) ───────────────────────
        print(f"[CBMRunner] Loading preLM from {args.cbm_peft_path} ...")
        self.preLM = LlamaModel.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16
        ).to(self.device)
        self.preLM.load_adapter(args.cbm_peft_path)
        self.preLM.eval()

        # ── CBL / CBLResidual ──────────────────────────────────────────────
        config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        residual_dim = getattr(args, "cbm_residual_dim", 768)
        use_cbl = getattr(args, "cbm_use_cbl", False)

        if use_cbl:
            self.cbl = CBL(config, C, self.tokenizer).to(self.device)
        else:
            self.cbl = CBLResidual(config, C, residual_dim, self.tokenizer).to(self.device)

        print(f"[CBMRunner] Loading CBL weights from {args.cbm_cbl_path} ...")
        self.cbl.load_state_dict(
            torch.load(args.cbm_cbl_path, map_location=self.device)
        )
        self.cbl.eval()

        # ── llama_vocab_weight (optional, for add_llama_logits mode) ──────
        self.llama_vocab_weight = None
        if getattr(args, "cbm_add_llama_logits", False):
            from transformers import AutoModelForCausalLM
            print("[CBMRunner] Loading LLaMA vocab weight for logit residual ...")
            lm_head = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16
            ).to(self.device)
            self.llama_vocab_weight = lm_head.get_output_embeddings().weight.detach()
            del lm_head
            torch.cuda.empty_cache()

        # ── Steering config ────────────────────────────────────────────────
        self.steer_mode: str = getattr(args, "cbm_steer_mode", "none")
        self.steer_value: float = getattr(args, "cbm_steer_value", 100.0)
        self.steer_topk: int = getattr(args, "cbm_steer_topk", 2)

        print(
            f"[CBMRunner] Ready. concepts={C}, steer_mode={self.steer_mode}, "
            f"steer_value={self.steer_value}, steer_topk={self.steer_topk}"
        )

    # ── Intervention vector builders ──────────────────────────────────────────

    def _build_intervene(self, problem_text: str) -> Optional[List[float]]:
        """Build an intervention vector for one problem, or return None (unsteered)."""
        if self.steer_mode == "none":
            return None
        if self.steer_mode == "probe":
            return self._probe_intervene(problem_text)
        if self.steer_mode == "mpnet":
            return self._mpnet_intervene(problem_text)
        raise ValueError(
            f"Unknown cbm_steer_mode={self.steer_mode!r}. "
            "Choose 'none', 'probe', or 'mpnet'."
        )

    def _probe_intervene(self, text: str) -> List[float]:
        """Forward text through preLM + CBL, amplify top-k active concept dims."""
        C = len(self.concept_set)
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=False
        ).to(self.device)
        with torch.no_grad():
            hidden = self.preLM(**enc).last_hidden_state        # (1, L, D)
            concepts, _, _, _ = self.cbl(hidden.float())         # (1, L, C)
        pooled = self._eos_pooling(concepts, enc["attention_mask"])[0]  # (C,)
        topk = pooled.topk(min(self.steer_topk, C)).indices.tolist()
        v = [0.0] * C
        for idx in topk:
            v[idx] = self.steer_value
        return v

    def _mpnet_intervene(self, text: str) -> List[float]:
        """Embed problem text with MPNet, steer toward top-k nearest concept names."""
        C = len(self.concept_set)
        mpnet_tok, mpnet_model, concept_feats = _get_mpnet_concept_feats(
            self.concept_set, self.device
        )
        enc = mpnet_tok(text, return_tensors="pt", truncation=True, max_length=512).to(
            self.device
        )
        with torch.no_grad():
            out = mpnet_model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        q = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        q = F.normalize(q, p=2, dim=1)  # (1, D)
        sims = (q @ concept_feats.T)[0]  # (C,)
        topk = sims.topk(min(self.steer_topk, C)).indices.tolist()
        v = [0.0] * C
        for idx in topk:
            v[idx] = self.steer_value
        return v

    # ── BaseRunner interface ───────────────────────────────────────────────────

    def _run_single(self, prompt: str) -> List[str]:
        """Satisfy the abstract interface; see run_batch for actual logic."""
        raise NotImplementedError("CBMRunner overrides run_batch directly.")

    def run_batch(self, prompts: List[str]) -> List[List[str]]:
        """Generate args.n completions for each prompt, one problem at a time."""
        from tqdm import tqdm

        all_outputs: List[List[str]] = []
        for prompt in tqdm(prompts, desc=f"CBM/{self.steer_mode}"):
            # Heuristic: use the first 1024 chars of the decoded prompt as steering text
            # (the full chat-template prompt is already tokenizer-formatted)
            steer_text = prompt[:1024]
            intervene = self._build_intervene(steer_text)

            enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_ids = enc["input_ids"]
            prompt_len = prompt_ids.shape[1]

            with torch.no_grad():
                gen_ids, _ = self.cbl.generate_batch(
                    prompt_ids,
                    self.preLM,
                    num_samples=self.args.n,
                    intervene=intervene,
                    length=self.args.max_tokens,
                    temp=self.args.temperature,
                    topk=50,
                    topp=self.args.top_p,
                    repetition_penalty=1.05,
                    llama_vocab_weight=self.llama_vocab_weight,
                )

            samples: List[str] = []
            for i in range(self.args.n):
                completion = gen_ids[i, prompt_len:]
                samples.append(
                    self.tokenizer.decode(completion, skip_special_tokens=True)
                )
            all_outputs.append(samples)

        # Cache if requested (mirrors BaseRunner behaviour)
        if self.args.use_cache:
            for prompt, output in zip(prompts, all_outputs):
                self.cache[prompt] = output
            self.save_cache()

        return all_outputs
