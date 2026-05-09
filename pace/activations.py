"""Layer-L hidden-state probing helpers used by the dictionary builder and the
vector-steerer fitter.

A single ``collect_activations_at_layer`` registers a forward hook on the
transformer block at ``layer_idx`` (or reads the final hidden state when
``layer_idx == -1``), pools each sequence by its EOS position, and returns a
``(N, H)`` CPU tensor. Results and labels are cached on disk per
``(model_id, layer_idx, split, num_samples, max_length)`` so re-runs are cheap.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from tqdm.auto import tqdm


def get_llama_decoder(model):
    """Walk through PEFT / CausalLM wrappers to find the ``LlamaModel`` core.

    Mirrors ``modules._get_llama_model``.
    """
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    if not (hasattr(base, "layers") and hasattr(base, "norm")) and hasattr(base, "model"):
        base = base.model
    if not (hasattr(base, "layers") and hasattr(base, "norm")):
        raise RuntimeError(
            f"Could not locate Llama decoder layers + final norm on {type(model).__name__}."
        )
    return base


def _eos_pool(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Vectorised EOS-position pool. Matches ``utils.eos_pooling`` semantics.

    ``utils.eos_pooling`` does this in a Python loop; we vectorise to keep
    activation collection fast on long batches.
    """
    seq_lens = attention_mask.long().sum(dim=1)
    last_idx = (seq_lens - 1).clamp_min(0)
    batch_idx = torch.arange(token_embeddings.size(0), device=token_embeddings.device)
    return token_embeddings[batch_idx, last_idx]  # (B, H)


def _stable_signature(d: dict) -> str:
    """Order-independent SHA1 signature for a dict of JSON-serialisable values."""
    payload = "|".join(f"{k}={d[k]}" for k in sorted(d))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def make_cache_signature(
    *,
    layer_idx: int,
    split: str,
    extras: Optional[dict] = None,
) -> dict:
    """Return a fully-specified cache signature that callers should pass to
    ``collect_activations_at_layer`` via ``cache_key_extras``.

    ``extras`` MUST include enough information to make the resulting cache
    deterministic w.r.t. tokenization / dataset slicing — at minimum:
      ``{"model_id": ..., "max_length": ..., "n_rows": ..., "max_train_samples": ..., ...}``.
    """
    base = {"layer_idx": int(layer_idx), "split": str(split)}
    if extras:
        base.update({str(k): v for k, v in extras.items()})
    return base


@torch.no_grad()
def collect_activations_at_layer(
    *,
    preLM,
    dataloader,
    layer_idx: int,
    device: torch.device,
    cache_dir: Optional[str] = None,
    cache_tag: Optional[str] = None,
    cache_key_extras: Optional[dict] = None,
    desc: str = "probing activations",
    max_batches: int = 0,
    return_labels: bool = True,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Run ``preLM`` over ``dataloader``, capture EOS-pooled hidden state at
    ``layer_idx`` (use ``-1`` for the final ``last_hidden_state``).

    Returns ``{"acts": (N, H) cpu tensor, "labels": (N, C) cpu tensor or None,
    "cache_path": str | None}``.

    Caching
    -------
    If ``cache_dir`` and ``cache_tag`` are both given, results are stashed on
    disk. ``cache_key_extras`` is a dict of preprocessing knobs (model id,
    ``max_length``, dataset N, slicing flags, ...). It is hashed into the
    filename **and** stored inside the blob — on load, both the filename and
    the embedded signature must match, otherwise the cache is treated as cold.

    Callers MUST pass ``cache_key_extras`` if they care about correctness
    across config changes (filename suffix is the signature hash; the embedded
    dict is the full payload for human inspection).
    """
    cache_path: Optional[Path] = None
    cache_signature: Optional[dict] = None
    cache_signature_hash: Optional[str] = None
    if cache_dir is not None and cache_tag:
        cache_root = Path(cache_dir).expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_signature = make_cache_signature(
            layer_idx=layer_idx, split=cache_tag, extras=cache_key_extras,
        )
        cache_signature_hash = _stable_signature(cache_signature)
        cache_path = cache_root / f"{cache_tag}_{cache_signature_hash}.pt"
        if cache_path.is_file():
            blob = torch.load(cache_path, map_location="cpu")
            stored_sig = blob.get("signature")
            if stored_sig == cache_signature:
                return {
                    "acts": blob["acts"],
                    "labels": blob.get("labels"),
                    "cache_path": str(cache_path),
                }
            print(
                f"[activations] cache signature mismatch for {cache_path.name}; "
                f"stored={stored_sig} vs requested={cache_signature}; recomputing.",
                flush=True,
            )

    decoder = get_llama_decoder(preLM)
    target_layer = decoder.layers[layer_idx] if layer_idx >= 0 else None

    captured: dict[str, Tensor] = {}

    def _hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h

    handle = target_layer.register_forward_hook(_hook) if target_layer is not None else None

    acts_chunks: list[Tensor] = []
    label_chunks: list[Tensor] = []
    preLM.eval()

    iterator = enumerate(dataloader)
    progress = tqdm(iterator, total=len(dataloader), desc=desc)
    for batch_idx, item in progress:
        if isinstance(item, tuple) and len(item) == 2:
            batch, batch_sim = item
        else:
            batch = item
            batch_sim = None

        batch = {k: v.to(device) for k, v in batch.items()}
        out = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        if layer_idx == -1:
            h = out.last_hidden_state  # (B, T, H)
        else:
            h = captured["h"]  # (B, T, H) from hook on layers[L]
        pooled = _eos_pool(h, batch["attention_mask"]).to(dtype=dtype).cpu()  # (B, H)
        acts_chunks.append(pooled)
        if return_labels and batch_sim is not None:
            label_chunks.append(batch_sim.detach().to(dtype=dtype).cpu())  # (B, C_cf)

        if max_batches > 0 and (batch_idx + 1) >= max_batches:
            break

    if handle is not None:
        handle.remove()

    acts = torch.cat(acts_chunks, dim=0) if acts_chunks else torch.empty((0,), dtype=dtype)  # (N, H)
    labels = (
        torch.cat(label_chunks, dim=0)
        if (return_labels and label_chunks)
        else None
    )  # (N, C_cf) or None

    if cache_path is not None:
        torch.save(
            {"acts": acts, "labels": labels, "signature": cache_signature},
            cache_path,
        )

    return {
        "acts": acts,
        "labels": labels,
        "cache_path": str(cache_path) if cache_path is not None else None,
    }


@torch.no_grad()
def collect_token_activations_at_layer(
    *,
    preLM,
    dataloader,
    layer_idx: int,
    device: torch.device,
    max_tokens_per_seq: int = 0,
    max_batches: int = 0,
    desc: str = "probing token activations",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Like ``collect_activations_at_layer`` but returns *all* non-pad token
    activations stacked into ``(M, H)`` (no pooling). Used for percentile-init
    of ``tau`` in PaCE-CBM.
    """
    decoder = get_llama_decoder(preLM)
    target_layer = decoder.layers[layer_idx] if layer_idx >= 0 else None

    captured: dict[str, Tensor] = {}

    def _hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h

    handle = target_layer.register_forward_hook(_hook) if target_layer is not None else None
    chunks: list[Tensor] = []
    preLM.eval()
    progress = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc)
    for batch_idx, item in progress:
        batch, _ = item if isinstance(item, tuple) and len(item) == 2 else (item, None)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        h = out.last_hidden_state if layer_idx == -1 else captured["h"]  # (B, T, H)
        mask = batch["attention_mask"].bool()  # (B, T)
        flat = h[mask]  # (M_b, H) all non-pad positions in batch
        if max_tokens_per_seq > 0 and flat.size(0) > max_tokens_per_seq:
            idx = torch.randperm(flat.size(0), device=flat.device)[:max_tokens_per_seq]
            flat = flat[idx]
        chunks.append(flat.to(dtype=dtype).cpu())
        if max_batches > 0 and (batch_idx + 1) >= max_batches:
            break
    if handle is not None:
        handle.remove()
    return torch.cat(chunks, dim=0) if chunks else torch.empty((0,), dtype=dtype)  # (M_total, H) or degenerate


def caa_per_concept(
    *,
    acts: Tensor,
    labels: Tensor,
) -> Tensor:
    """μ_pos − μ_neg per concept ⇒ (C, H) tensor of CAA-style directions.

    ``acts``: (N, H) float, ``labels``: (N, C) multi-hot float.
    Concepts with zero positive or zero negative samples produce the zero vector.
    """
    if acts.dim() != 2 or labels.dim() != 2:
        raise ValueError(
            f"Expected acts (N,H) and labels (N,C); got {tuple(acts.shape)} and {tuple(labels.shape)}."
        )
    if acts.size(0) != labels.size(0):
        raise ValueError(
            f"Sample-count mismatch: acts={acts.size(0)} labels={labels.size(0)}."
        )

    pos_mask = labels > 0  # (N, C)
    neg_mask = ~pos_mask
    pos_count = pos_mask.sum(dim=0).clamp_min(1).to(acts.dtype)  # (C,)
    neg_count = neg_mask.sum(dim=0).clamp_min(1).to(acts.dtype)  # (C,)
    pos_sum = pos_mask.to(acts.dtype).t() @ acts  # (C, H)
    neg_sum = neg_mask.to(acts.dtype).t() @ acts  # (C, H)
    mu_pos = pos_sum / pos_count.unsqueeze(-1)
    mu_neg = neg_sum / neg_count.unsqueeze(-1)
    diff = mu_pos - mu_neg
    no_pos = (pos_mask.sum(dim=0) == 0)
    no_neg = (neg_mask.sum(dim=0) == 0)
    diff[no_pos | no_neg] = 0.0
    return diff


def per_concept_pos_neg(
    *,
    acts: Tensor,
    labels: Tensor,
    concept_idx: int,
) -> tuple[Tensor, Tensor]:
    """Return ``(pos_X, neg_X)`` for a single concept index ``c``.

    Used by ``train_steerers.py`` to fit ``CAA().fit(pos_X, neg_X)`` per CF tag.
    """
    pos_mask = labels[:, concept_idx] > 0
    return acts[pos_mask], acts[~pos_mask]
