"""
PaCE — Parsimonious Concept Engineering (NeurIPS 2024).

Adapted from ODESteer's pace.py with:
  - tqdm progress bar during initial concept embedding generation
  - No Hydra dependency — plain Python config dict

How this file works
-------------------
1. **ConceptDictionary** — Reads PaCE-1M-style `concept_index.txt` and per-concept
   `*.txt` files of short context strings that *illustrate* each concept.

2. **ConceptPartitioner** — Splits concept indices into "benign" vs "undesirable"
   (e.g. for TruthfulQA, keywords suggestive of falsehood → undesirable). Only
   undesirable directions are removed at inference.

3. **ActivationConceptEncoder** — For each concept, runs the frozen LM on its
   context strings, takes the last-token hidden state at `layer_idx`, averages
   across contexts, and caches a single **CPU** vector per concept (disk cache).

4. **PaCESteerer** — Registers a **forward hook** on that transformer block. On
   each forward, for every token position it expresses the hidden state as a
   linear mix of concept vectors (`decompose_sparse`: SVD-reduced least squares),
    computes a residual and a masked intervention coefficient vector, then
    reconstructs as ``z_new = (z - D c) + D c_masked`` where undesirable
    coefficients are scaled by ``(1 - alpha)``.

Dictionary math stays on **CPU** (NumPy `lstsq`) unless ``pace_gpu`` is set; either
way the cost is dominated by **one full-dictionary** SVD + least-squares per
token position (see ``decompose_sparse``).

**Why runs can take hours**

1. **Concept encoding** (``encode_dictionary``): one forward pass per concept
   (sequential). With tens of thousands of concepts and a large LM, wall time is
   roughly ``#concepts × time_per_forward`` unless vectors are already cached on disk.

2. **Generation with the hook**: on every forward at the hooked layer, for each
   batch and sequence position the code calls ``decompose_sparse`` over **all**
   loaded concept vectors, then sums **undesirable** directions. Complexity is
   roughly ``O(forward_steps × B × T × n_concepts × d)`` in the hook alone — far
   heavier than plain generation. ``pace_gpu`` moves the linear algebra to GPU but
   does not remove the per-token full-dictionary work.

Enable ``pace_token_timing`` in the PaCE cfg to print per-position timings.
"""

from __future__ import annotations

import ast
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sparse coding helpers
# ---------------------------------------------------------------------------

def _svd_embed(X: torch.Tensor, q: int = 500) -> torch.Tensor:
    q = min(q, min(X.shape) - 1)
    U, S, V = torch.svd_lowrank(X, q=q)
    return (V * S).T


def decompose_sparse(
    target: torch.Tensor,
    dictionary: List[torch.Tensor],
    normalize: bool = True,
    use_gpu: bool = False,
    return_timings: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """
    If return_timings is True, returns (coeffs, timings_dict) with keys in seconds:
    stack_embed, svd, norm, lstsq (GPU path) or stack_embed, svd, numpy, lstsq (CPU path).
    """
    t_all = time.perf_counter()

    if use_gpu:
        t0 = time.perf_counter()
        data = torch.stack([target.view(-1)] + [a.view(-1) for a in dictionary], dim=0).float()
        t_stack = time.perf_counter() - t0

        t0 = time.perf_counter()
        embedded = _svd_embed(data.T).T
        if return_timings and embedded.is_cuda:
            torch.cuda.synchronize()
        t_svd = time.perf_counter() - t0

        t0 = time.perf_counter()
        y = embedded[0:1].clone()
        D = embedded[1:].clone()

        if normalize:
            eps = 1e-12
            norm_y = torch.linalg.norm(y, dim=1, keepdim=True).clamp_min(eps)
            norm_D = torch.linalg.norm(D, dim=1, keepdim=True).clamp_min(eps)
            y = y / norm_y
            D = D / norm_D
        else:
            norm_y = torch.ones((1, 1), device=embedded.device, dtype=embedded.dtype)
            norm_D = torch.ones((D.shape[0], 1), device=embedded.device, dtype=embedded.dtype)

        t_norm = time.perf_counter() - t0

        t0 = time.perf_counter()
        c = torch.linalg.lstsq(D.T, y.T).solution.squeeze(-1)
        if return_timings and c.is_cuda:
            torch.cuda.synchronize()
        t_lstsq = time.perf_counter() - t0

        if normalize:
            c = c / norm_D.squeeze(-1) * norm_y.squeeze()

        out = c.to(dtype=torch.float32)
        if return_timings:
            t_total = time.perf_counter() - t_all
            timings = {
                "stack_embed_s": t_stack,
                "svd_s": t_svd,
                "norm_s": t_norm,
                "lstsq_s": t_lstsq,
                "decompose_total_s": t_total,
            }
            return out, timings
        return out

    t0 = time.perf_counter()
    data = torch.stack([target.view(-1)] + [a.view(-1) for a in dictionary], dim=0)
    embedded = _svd_embed(data.T).T
    t_stack_svd = time.perf_counter() - t0

    t0 = time.perf_counter()
    data_np = embedded.numpy().astype(np.float64)
    t_numpy = time.perf_counter() - t0
    y = data_np[0:1].copy()
    D = data_np[1:].copy()

    if normalize:
        norm_y = np.linalg.norm(y) + 1e-12
        norm_D = np.linalg.norm(D, axis=1, keepdims=True) + 1e-12
        y = y / norm_y
        D = D / norm_D
    else:
        norm_y, norm_D = 1.0, np.ones((D.shape[0], 1))

    t0 = time.perf_counter()
    c, *_ = np.linalg.lstsq(D.T, y.T, rcond=None)
    c = c.T[0]
    t_lstsq_np = time.perf_counter() - t0

    if normalize:
        c = c / norm_D.squeeze() * norm_y

    out = torch.tensor(c, dtype=torch.float32)

    if return_timings:
        t_total = time.perf_counter() - t_all
        timings = {
            "stack_svd_s": t_stack_svd,
            "to_numpy_s": t_numpy,
            "lstsq_np_s": t_lstsq_np,
            "decompose_total_s": t_total,
        }
        return out, timings
    return out


# ---------------------------------------------------------------------------
# Concept dictionary loader
# ---------------------------------------------------------------------------

class ConceptDictionary:
    def __init__(
        self,
        index_path: str,
        representation_path: str,
        max_concepts: int = 5000,
    ):
        self.representation_path = Path(representation_path)
        with open(index_path, "r") as f:
            all_concepts: List[str] = ast.literal_eval(f.read())

        if max_concepts > 0:
            all_concepts = all_concepts[:max_concepts]

        self.concepts: List[str] = []
        self.representations: List[List[str]] = []
        skipped_empty = 0

        for concept in all_concepts:
            rep_file = self.representation_path / f"{concept}.txt"
            if rep_file.exists():
                with open(rep_file, "r") as rf:
                    rep = ast.literal_eval(rf.read())
                valid_rep = [r for r in rep if isinstance(r, str) and r.strip()]
                if not valid_rep:
                    skipped_empty += 1
                    continue
                self.concepts.append(concept)
                self.representations.append(valid_rep)

        logger.info(
            "ConceptDictionary: loaded %d / %d concepts (skipped_empty=%d)",
            len(self.concepts),
            len(all_concepts),
            skipped_empty,
        )

    def __len__(self) -> int:
        return len(self.concepts)


# ---------------------------------------------------------------------------
# Concept partitioner
# ---------------------------------------------------------------------------

class ConceptPartitioner:
    _HALLUCINATION_KEYWORDS = {
        "false", "myth", "fake", "fiction", "hoax", "rumor", "rumour",
        "incorrect", "misinformation", "disinformation", "fabricat",
        "conspiracy", "pseudoscience", "debunked", "wrong", "error",
        "lie", "lies", "lying", "untrue", "misleading", "misconception",
        "superstition", "legend", "folklore",
    }

    def __init__(self, mode: str = "heuristic", partition_file: Optional[str] = None):
        self.mode = mode
        self._cache: dict[str, bool] = {}

        if mode == "file":
            if partition_file is None or not Path(partition_file).exists():
                raise FileNotFoundError(f"partition_file={partition_file!r} not found.")
            import json
            with open(partition_file) as f:
                self._cache = json.load(f)

    def is_benign(self, concept: str) -> bool:
        if concept in self._cache:
            return self._cache[concept]
        if self.mode == "file":
            return True
        concept_lower = concept.lower()
        for kw in self._HALLUCINATION_KEYWORDS:
            if kw in concept_lower:
                self._cache[concept] = False
                return False
        self._cache[concept] = True
        return True

    def partition(self, concepts: List[str]) -> Tuple[List[int], List[int]]:
        benign, undesirable = [], []
        for i, c in enumerate(concepts):
            (benign if self.is_benign(c) else undesirable).append(i)
        return benign, undesirable


# ---------------------------------------------------------------------------
# Activation-space concept encoder — with tqdm progress
# ---------------------------------------------------------------------------

class ActivationConceptEncoder:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        layer_idx: int,
        cache_path: str,
        batch_size: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.device = next(model.parameters()).device

    def _encode_contexts(self, contexts: List[str]) -> torch.Tensor:
        contexts = [c for c in contexts if isinstance(c, str) and c.strip()]
        if not contexts:
            raise ValueError("PaCE concept has no valid contexts after filtering.")

        vectors = []
        for i in range(0, len(contexts), self.batch_size):
            batch = contexts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=128,
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**enc, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states[self.layer_idx + 1]
            seq_lens = (enc["attention_mask"].sum(dim=1) - 1).clamp_min(0)
            for b_idx, s_len in enumerate(seq_lens):
                vectors.append(hs[b_idx, s_len].cpu())
        if not vectors:
            raise RuntimeError("PaCE failed to encode any vectors from non-empty contexts.")
        return torch.stack(vectors).mean(dim=0)

    def get_concept_vector(self, concept: str, contexts: List[str]) -> torch.Tensor:
        cache_file = self.cache_path / f"{concept}.pt"
        if cache_file.exists():
            return torch.load(cache_file, map_location="cpu")
        vec = self._encode_contexts(contexts)
        torch.save(vec, cache_file)
        return vec

    def encode_dictionary(
        self,
        concept_dict: ConceptDictionary,
        max_concepts: int = -1,
    ) -> List[torch.Tensor]:
        concepts = concept_dict.concepts
        if max_concepts > 0:
            concepts = concepts[:max_concepts]

        vecs = []
        pairs = list(zip(concept_dict.concepts, concept_dict.representations))
        if max_concepts > 0:
            pairs = pairs[:max_concepts]

        for c, reps in tqdm(pairs, desc="Encoding PaCE concepts", unit="concept"):
            vecs.append(self.get_concept_vector(c, reps))
        return vecs


# ---------------------------------------------------------------------------
# PaCE Steerer
# ---------------------------------------------------------------------------

class PaCESteerer:
    """
    Drop-in PaCE steerer. No Hydra — pass a plain dict as `cfg`.

    Expected cfg keys:
        index_path, representation_path, max_concepts (int),
        partition_mode (str), partition_file (str|None),
        vector_cache_path (str), encode_batch_size (int),
        alpha (float), layer_idx (int)
        pace_gpu (bool): run decompose_sparse on GPU (still expensive: full-dict SVD/lstsq per token).
        pace_token_timing (bool): print per-(batch, seq) position timing for each hook call (verbose).
        reuse_coeff_across_tokens (bool): if True, decompose only once on the first
            seen token and reuse that coefficient vector for all later tokens while
            this hook is registered.
    """

    def __init__(self, cfg: dict, model: nn.Module, tokenizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx: int = cfg["layer_idx"]
        self._hook_handle = None
        self.pace_gpu: bool = bool(cfg.get("pace_gpu", False))
        self.pace_token_timing: bool = bool(cfg.get("pace_token_timing", False))
        self.reuse_coeff_across_tokens: bool = bool(cfg.get("reuse_coeff_across_tokens", False))
        self._timing_token_idx: int = 0
        self._cached_recon_device: Optional[torch.device] = None
        self._cached_recon_base: Optional[torch.Tensor] = None
        self._cached_recon_intervened: Optional[torch.Tensor] = None

        concept_dict = ConceptDictionary(
            index_path=cfg["index_path"],
            representation_path=cfg["representation_path"],
            max_concepts=cfg.get("max_concepts", 5000),
        )

        partitioner = ConceptPartitioner(
            mode=cfg.get("partition_mode", "heuristic"),
            partition_file=cfg.get("partition_file"),
        )
        self.benign_idx, self.undesirable_idx = partitioner.partition(concept_dict.concepts)
        logger.info("PaCESteerer: %d benign, %d undesirable", len(self.benign_idx), len(self.undesirable_idx))

        encoder = ActivationConceptEncoder(
            model=model, tokenizer=tokenizer, layer_idx=self.layer_idx,
            cache_path=cfg.get("vector_cache_path", f"./pace_cache/layer{self.layer_idx}"),
            batch_size=cfg.get("encode_batch_size", 8),
        )
        self.concept_vectors: List[torch.Tensor] = encoder.encode_dictionary(
            concept_dict, max_concepts=cfg.get("max_concepts", 5000),
        )

        self.alpha: float = cfg.get("alpha", 1.0)
        self._concept_matrix_cpu: Optional[torch.Tensor] = None
        self._undesirable_mask_cpu: Optional[torch.Tensor] = None
        if self.concept_vectors:
            self._concept_matrix_cpu = torch.stack(self.concept_vectors, dim=0).to(dtype=torch.float32)
            self._undesirable_mask_cpu = torch.zeros(len(self.concept_vectors), dtype=torch.float32)
            if self.undesirable_idx:
                self._undesirable_mask_cpu[self.undesirable_idx] = 1.0

        self._concept_vectors_gpu: Optional[List[torch.Tensor]] = None
        self._concept_matrix_gpu: Optional[torch.Tensor] = None
        self._undesirable_mask_gpu: Optional[torch.Tensor] = None
        if self.pace_gpu:
            self._concept_vectors_gpu = []

    def _reset_cached_reconstruction(self):
        self._cached_recon_device = None
        self._cached_recon_base = None
        self._cached_recon_intervened = None

    def _ensure_gpu_concepts(self, dev: torch.device):
        if not self.pace_gpu:
            return
        if self._concept_vectors_gpu is None or len(self._concept_vectors_gpu) != len(self.concept_vectors):
            self._concept_vectors_gpu = []
        if len(self._concept_vectors_gpu) == 0 or self._concept_vectors_gpu[0].device != dev:
            self._concept_vectors_gpu = [v.to(device=dev, dtype=torch.float32) for v in self.concept_vectors]
            self._concept_matrix_gpu = None
            self._undesirable_mask_gpu = None
        if self._concept_matrix_gpu is None and self._concept_vectors_gpu:
            self._concept_matrix_gpu = torch.stack(self._concept_vectors_gpu, dim=0)
        if self._undesirable_mask_gpu is None and self._concept_vectors_gpu:
            self._undesirable_mask_gpu = torch.zeros(len(self._concept_vectors_gpu), device=dev, dtype=torch.float32)
            if self.undesirable_idx:
                self._undesirable_mask_gpu[self.undesirable_idx] = 1.0

    def _compute_coeffs(
        self,
        activation: torch.Tensor,
        profile: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        use_gpu = self.pace_gpu
        if use_gpu:
            self._ensure_gpu_concepts(activation.device)
            target = activation.detach().float()
            dictionary = self._concept_vectors_gpu or []
        else:
            target = activation.detach().float().cpu()
            dictionary = self.concept_vectors

        return decompose_sparse(
            target=target,
            dictionary=dictionary,
            normalize=True,
            use_gpu=use_gpu,
            return_timings=profile,
        )

    def _reconstruct_from_coeffs(
        self,
        coeffs: torch.Tensor,
        dev: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pace_gpu:
            self._ensure_gpu_concepts(dev)
            C = self._concept_matrix_gpu
            mask = self._undesirable_mask_gpu
            coeffs_dev = coeffs.to(device=dev, dtype=torch.float32)
        else:
            C = self._concept_matrix_cpu
            mask = self._undesirable_mask_cpu
            coeffs_dev = coeffs.to(device="cpu", dtype=torch.float32)

        if C is None or mask is None:
            zero = torch.zeros_like(coeffs_dev)
            return zero, zero

        base = coeffs_dev @ C
        coeffs_masked = coeffs_dev - (self.alpha * mask * coeffs_dev)
        intervened = coeffs_masked @ C
        return base, intervened

    def _apply_reconstruction(
        self,
        activation: torch.Tensor,
        base: torch.Tensor,
        intervened: torch.Tensor,
    ) -> torch.Tensor:
        dev, dtype = activation.device, activation.dtype
        if base.device != dev:
            base = base.to(device=dev)
        if intervened.device != dev:
            intervened = intervened.to(device=dev)
        residual = activation.detach().float() - base
        out = residual + intervened
        return out.to(dtype=dtype)

    def fit(self, *args, **kwargs):
        return self

    def _steer_activation(
        self,
        activation: torch.Tensor,
        profile: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        if not self.concept_vectors:
            if profile:
                return activation, {"steer_total_s": 0.0, "skipped": True}
            return activation

        t_steer0 = time.perf_counter()
        dev = activation.device

        if (
            self.reuse_coeff_across_tokens
            and self._cached_recon_device == dev
            and self._cached_recon_base is not None
            and self._cached_recon_intervened is not None
        ):
            t0 = time.perf_counter()
            out = self._apply_reconstruction(activation, self._cached_recon_base, self._cached_recon_intervened)
            t_apply = time.perf_counter() - t0
            t_steer = time.perf_counter() - t_steer0
            if profile:
                return out, {
                    "reuse_cached_coeff": True,
                    "apply_reconstruction_s": t_apply,
                    "steer_total_s": t_steer,
                    "n_concepts": len(self.concept_vectors),
                    "n_undesirable": len(self.undesirable_idx),
                }
            return out

        t0 = time.perf_counter()
        dec = self._compute_coeffs(activation, profile=profile)
        t_decompose = time.perf_counter() - t0
        if profile:
            coeffs, dec_tim = dec  # type: ignore[misc]
        else:
            coeffs = dec  # type: ignore[assignment]

        t0 = time.perf_counter()
        base, intervened = self._reconstruct_from_coeffs(coeffs, dev=dev)
        t_recon = time.perf_counter() - t0

        if self.reuse_coeff_across_tokens:
            self._cached_recon_device = dev
            self._cached_recon_base = base.detach()
            self._cached_recon_intervened = intervened.detach()

        t0 = time.perf_counter()
        out = self._apply_reconstruction(activation, base, intervened)
        t_apply = time.perf_counter() - t0
        t_steer = time.perf_counter() - t_steer0

        if profile:
            prof: dict = {
                "reuse_cached_coeff": False,
                "decompose_call_s": t_decompose,
                "reconstruct_s": t_recon,
                "apply_reconstruction_s": t_apply,
                "steer_total_s": t_steer,
                "n_concepts": len(self.concept_vectors),
                "n_undesirable": len(self.undesirable_idx),
            }
            prof.update(dec_tim)
            return out, prof
        return out

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        B, T, D = hidden.shape
        _ms = lambda s: s * 1000.0
        t_hook0 = time.perf_counter()
        steered = hidden.clone()
        t_clone = time.perf_counter() - t_hook0

        for b in range(B):
            if self.pace_token_timing:
                t_batch0 = time.perf_counter()
                prep_s = 0.0
                stack_embed_s = 0.0
                svd_s = 0.0
                norm_s = 0.0
                lstsq_s = 0.0
                reconstruct_s = 0.0
                apply_s = 0.0
                to_dtype_s = 0.0
                decompose_total_s = 0.0
                steer_total_s = 0.0
                token_wall_s = 0.0
                reuse_cached_count = 0
                n_concepts = "?"
                n_undesirable = "?"

            for t in range(T):
                if self.pace_token_timing:
                    t_tok0 = time.perf_counter()
                    out, prof = self._steer_activation(hidden[b, t], profile=True)  # type: ignore[misc]
                    steered[b, t] = out
                    t_tok = time.perf_counter() - t_tok0
                    prep_s += float(prof.get("prep_s", 0.0))
                    stack_embed_s += float(prof.get("stack_embed_s", 0.0))
                    svd_s += float(prof.get("svd_s", prof.get("stack_svd_s", 0.0)))
                    norm_s += float(prof.get("norm_s", 0.0))
                    lstsq_s += float(prof.get("lstsq_s", prof.get("lstsq_np_s", 0.0)))
                    reconstruct_s += float(prof.get("reconstruct_s", 0.0))
                    apply_s += float(prof.get("apply_reconstruction_s", 0.0))
                    to_dtype_s += float(prof.get("to_dtype_s", 0.0))
                    decompose_total_s += float(prof.get("decompose_total_s", prof.get("steer_total_s", 0.0)))
                    steer_total_s += float(prof.get("steer_total_s", 0.0))
                    token_wall_s += t_tok
                    reuse_cached_count += int(bool(prof.get("reuse_cached_coeff", False)))
                    n_concepts = prof.get("n_concepts", "?")
                    n_undesirable = prof.get("n_undesirable", "?")
                else:
                    steered[b, t] = self._steer_activation(hidden[b, t])  # type: ignore[assignment]

            if self.pace_token_timing:
                idx = self._timing_token_idx
                self._timing_token_idx += 1
                t_batch = time.perf_counter() - t_batch0
                print(
                    f"[PaCE timing] i={idx} b={b} T={T} "
                    f"n_concepts={n_concepts} n_undesirable={n_undesirable} "
                    f"reuse_cached={reuse_cached_count}/{T} "
                    f"prep_ms={_ms(prep_s):.2f} "
                    f"stack_embed_ms={_ms(stack_embed_s):.2f} "
                    f"svd_ms={_ms(svd_s):.2f} "
                    f"norm_ms={_ms(norm_s):.2f} "
                    f"lstsq_ms={_ms(lstsq_s):.2f} "
                    f"reconstruct_ms={_ms(reconstruct_s):.2f} "
                    f"apply_ms={_ms(apply_s):.2f} "
                    f"to_dtype_ms={_ms(to_dtype_s):.2f} "
                    f"decompose_total_ms={_ms(decompose_total_s):.2f} "
                    f"steer_total_ms={_ms(steer_total_s):.2f} "
                    f"token_wall_ms={_ms(token_wall_s):.2f} "
                    f"batch_wall_ms={_ms(t_batch):.2f}",
                    flush=True,
                )

        if self.pace_token_timing:
            t_hook = time.perf_counter() - t_hook0
            print(
                f"[PaCE timing] forward_hook layer={self.layer_idx} B={B} T={T} D={D} "
                f"clone_ms={_ms(t_clone):.2f} hook_total_ms={_ms(t_hook):.2f} "
                f"positions={B * T}",
                flush=True,
            )

        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    def register_hook(self):
        self._reset_cached_reconstruction()
        self._timing_token_idx = 0
        layer = self._get_layer(self.layer_idx)
        self._hook_handle = layer.register_forward_hook(self._hook_fn)

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._reset_cached_reconstruction()

    def _get_layer(self, idx: int) -> nn.Module:
        model = self.model
        if hasattr(model, "module"):
            model = model.module
        for attr in ("model", "transformer", "base_model"):
            if hasattr(model, attr):
                model = getattr(model, attr)
                break
        for layers_attr in ("layers", "h", "blocks"):
            if hasattr(model, layers_attr):
                return getattr(model, layers_attr)[idx]
        raise AttributeError(f"Cannot locate transformer layers in {type(model)}")

    def __enter__(self):
        self.register_hook()
        return self

    def __exit__(self, *args):
        self.remove_hook()
