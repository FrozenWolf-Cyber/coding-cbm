"""Frozen PaCE dictionary builder.

Three modes:
  - ``"cf_only"``    : ``D ∈ R^(H × C_cf)`` — one CAA direction per CF tag.
  - ``"pace_only"``  : ``D ∈ R^(H × C_pace)`` — one direction per PaCE concept,
                         built by forwarding the per-concept stimulus strings
                         from ``concept.zip``.
  - ``"hybrid"``     : ``D = [pace_cols | cf_cols]`` (CF block last, ``cf_offset``
                         points to its first column).

Per-concept PaCE vectors are cached as ``{cache_dir}/pace/{concept}.pt``; the
full dictionary plus its sidecar JSON are cached under ``{cache_dir}``.
"""

from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from torch import Tensor
from tqdm.auto import tqdm

from .activations import caa_per_concept, collect_activations_at_layer, get_llama_decoder

REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# PaCE per-concept stimulus loading (concept.zip → pace_data/concept/<name>.txt)
# ─────────────────────────────────────────────────────────────────────────────

def ensure_concept_representations(
    *,
    concept_zip: Path = REPO_ROOT / "concept.zip",
    out_dir: Path = REPO_ROOT / "pace_data" / "concept",
) -> Path:
    """Extract ``concept.zip`` to ``pace_data/concept/`` if not present.

    The zip's internal layout is ``concept/<name>.txt``; we strip the leading
    ``concept/`` so the resulting layout is ``pace_data/concept/<name>.txt``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / ".extracted"
    if sentinel.is_file():
        return out_dir
    if not concept_zip.is_file():
        raise FileNotFoundError(
            f"concept.zip not found at {concept_zip}; cannot build PaCE dictionary."
        )
    with zipfile.ZipFile(concept_zip, "r") as zf:
        for member in zf.infolist():
            name = member.filename
            if name.endswith("/"):
                continue
            inner = name[len("concept/"):] if name.startswith("concept/") else name
            target = out_dir / inner
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
    sentinel.write_text("ok")
    return out_dir


def _load_concept_index(index_path: Path, max_concepts: int) -> List[str]:
    text = index_path.read_text()
    concepts = ast.literal_eval(text)
    if not isinstance(concepts, list):
        raise ValueError(f"{index_path} does not contain a Python list literal.")
    if max_concepts > 0:
        concepts = concepts[:max_concepts]
    return [str(c) for c in concepts]


def _read_representations(rep_dir: Path, concept: str) -> List[str]:
    rep_file = rep_dir / f"{concept}.txt"
    if not rep_file.is_file():
        return []
    raw = ast.literal_eval(rep_file.read_text())
    if not isinstance(raw, list):
        return []
    return [str(s) for s in raw if isinstance(s, str) and s.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# PaCE column construction (forward stimulus strings, mean over EOS hidden state)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _forward_stimuli_pool(
    *,
    preLM,
    tokenizer,
    layer_idx: int,
    device: torch.device,
    stimuli: Sequence[str],
    batch_size: int,
    max_length: int,
) -> Tensor:
    """Tokenize ``stimuli``, forward through ``preLM``, capture hidden state at
    ``layer_idx`` (or last_hidden_state when ``-1``), pool by EOS, return the
    mean across stimuli ⇒ ``(H,)`` vector.
    """
    decoder = get_llama_decoder(preLM)
    target_layer = decoder.layers[layer_idx] if layer_idx >= 0 else None
    captured: dict[str, Tensor] = {}

    def _hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h

    handle = target_layer.register_forward_hook(_hook) if target_layer is not None else None
    pooled_chunks: list[Tensor] = []
    preLM.eval()
    for i in range(0, len(stimuli), batch_size):
        batch_text = list(stimuli[i:i + batch_size])
        enc = tokenizer(
            batch_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = preLM(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        h = out.last_hidden_state if layer_idx == -1 else captured["h"]  # (B_stim, T, H)
        seq_lens = enc["attention_mask"].long().sum(dim=1)
        last_idx = (seq_lens - 1).clamp_min(0)
        b_idx = torch.arange(h.size(0), device=h.device)
        pooled = h[b_idx, last_idx].float().cpu()  # (B_stim, H) EOS per stimulus row
        pooled_chunks.append(pooled)
    if handle is not None:
        handle.remove()
    if not pooled_chunks:
        raise RuntimeError("No stimuli pooled — empty stimulus set?")
    stacked = torch.cat(pooled_chunks, dim=0)  # (n_stimuli, H)
    return stacked.mean(dim=0)  # (H,)


@torch.no_grad()
def build_pace_columns(
    *,
    preLM,
    tokenizer,
    layer_idx: int,
    device: torch.device,
    index_path: Path,
    representation_dir: Path,
    max_concepts: int,
    cache_dir: Path,
    batch_size: int = 8,
    max_length: int = 128,
) -> tuple[List[str], List[Tensor]]:
    """Build per-concept PaCE direction vectors with on-disk caching.

    Concepts whose representation file is missing or empty are skipped.
    Returns ``(concept_names, vectors)`` of equal length.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_concepts = _load_concept_index(index_path, max_concepts)

    names: List[str] = []
    vectors: List[Tensor] = []
    for concept in tqdm(all_concepts, desc=f"PaCE cols (layer {layer_idx})", unit="concept"):
        cache_file = cache_dir / f"{concept}.pt"
        if cache_file.is_file():
            v = torch.load(cache_file, map_location="cpu")
            names.append(concept)
            vectors.append(v)
            continue
        stimuli = _read_representations(representation_dir, concept)
        if not stimuli:
            continue
        vec = _forward_stimuli_pool(
            preLM=preLM,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            device=device,
            stimuli=stimuli,
            batch_size=batch_size,
            max_length=max_length,
        )
        torch.save(vec, cache_file)
        names.append(concept)
        vectors.append(vec)
    return names, vectors


# ─────────────────────────────────────────────────────────────────────────────
# CF column construction (μ_pos − μ_neg from train activations)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_cf_columns(
    *,
    preLM,
    layer_idx: int,
    device: torch.device,
    train_dataloader,
    cf_concept_set: Sequence[str],
    cache_dir: Path,
    cache_tag: str,
    cache_key_extras: dict,
    max_batches: int = 0,
) -> List[Tensor]:
    """CAA-style μ_pos − μ_neg per CF tag from train-split activations.

    Returns a list of ``len(cf_concept_set)`` tensors, each ``(H,)``.
    ``cache_key_extras`` MUST capture all preprocessing knobs that influence
    the activations (model id, max_length, dataset N, slicing limits, ...).
    """
    probe = collect_activations_at_layer(
        preLM=preLM,
        dataloader=train_dataloader,
        layer_idx=layer_idx,
        device=device,
        cache_dir=str(cache_dir),
        cache_tag=cache_tag,
        cache_key_extras=cache_key_extras,
        desc=f"CF probe layer={layer_idx}",
        max_batches=max_batches,
        return_labels=True,
    )
    acts = probe["acts"]
    labels = probe["labels"]
    if labels is None or labels.size(1) != len(cf_concept_set):
        raise RuntimeError(
            f"CF probe labels missing or wrong width "
            f"(got {None if labels is None else tuple(labels.shape)}, "
            f"want C={len(cf_concept_set)})."
        )
    diffs = caa_per_concept(acts=acts, labels=labels)  # (C, H)
    return [diffs[i].clone() for i in range(diffs.size(0))]


# ─────────────────────────────────────────────────────────────────────────────
# Main entry: build full dictionary D and meta
# ─────────────────────────────────────────────────────────────────────────────

def _stable_signature_for_dict(d: dict) -> str:
    import hashlib
    payload = "|".join(f"{k}={d[k]}" for k in sorted(d))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _dict_cache_paths(
    cache_dir: Path, *, layer_idx: int, mode: str, max_pace: int, sig_hash: str,
) -> tuple[Path, Path]:
    tag = f"D_layer{layer_idx}_{mode}_pace{max_pace}_{sig_hash}"
    return cache_dir / f"{tag}.pt", cache_dir / f"{tag}.meta.json"


def build_dictionary(
    *,
    preLM,
    tokenizer,
    layer_idx: int,
    device: torch.device,
    cf_concept_set: Sequence[str],
    cf_train_dataloader,
    cf_cache_key_extras: dict,
    mode: str = "hybrid",
    max_pace_concepts: int = 5000,
    pace_index_path: Path = REPO_ROOT / "pace_data" / "concept_index.txt",
    pace_representation_dir: Optional[Path] = None,
    pace_concept_zip: Path = REPO_ROOT / "concept.zip",
    cache_dir: Path = REPO_ROOT / ".pace_cache" / "dictionary",
    cf_probe_max_batches: int = 0,
    pace_batch_size: int = 8,
    pace_max_length: int = 128,
) -> tuple[Tensor, dict]:
    """Build (or load from cache) ``D ∈ R^(H × C)`` and a sidecar meta dict.

    ``cf_cache_key_extras`` is forwarded to ``collect_activations_at_layer`` so
    the CF-probe cache invalidates correctly when tokenization / dataset
    slicing changes (it should include model id, max_length, dataset N,
    max_train_samples, etc.).
    """
    if mode not in ("cf_only", "pace_only", "hybrid"):
        raise ValueError(f"mode must be cf_only|pace_only|hybrid, got {mode!r}.")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pace_per_concept_cache = cache_dir / f"pace_layer{layer_idx}"

    # The full-dictionary cache key folds in the CF cache extras (so a
    # max_length / sample-cap change rebuilds D), the model id, and the
    # PaCE tokenization knobs.
    full_extras = {
        "mode": mode,
        "max_pace_concepts": int(max_pace_concepts),
        "pace_batch_size": int(pace_batch_size),
        "pace_max_length": int(pace_max_length),
    }
    full_extras.update({f"cf_{k}": v for k, v in cf_cache_key_extras.items()})
    sig_hash = _stable_signature_for_dict(full_extras)
    dict_path, meta_path = _dict_cache_paths(
        cache_dir, layer_idx=layer_idx, mode=mode, max_pace=max_pace_concepts,
        sig_hash=sig_hash,
    )
    if dict_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if meta.get("cache_signature") == full_extras:
            D = torch.load(dict_path, map_location="cpu")
            meta["dictionary_path"] = str(dict_path)
            meta["meta_path"] = str(meta_path)
            return D, meta
        print(
            f"[dictionary] cache signature mismatch at {dict_path.name}; rebuilding.",
            flush=True,
        )

    pace_concepts: List[str] = []
    pace_vectors: List[Tensor] = []
    if mode in ("pace_only", "hybrid"):
        rep_dir = pace_representation_dir
        if rep_dir is None:
            rep_dir = ensure_concept_representations(
                concept_zip=pace_concept_zip,
                out_dir=REPO_ROOT / "pace_data" / "concept",
            )
        pace_concepts, pace_vectors = build_pace_columns(
            preLM=preLM,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            device=device,
            index_path=pace_index_path,
            representation_dir=Path(rep_dir),
            max_concepts=max_pace_concepts,
            cache_dir=pace_per_concept_cache,
            batch_size=pace_batch_size,
            max_length=pace_max_length,
        )

    cf_vectors: List[Tensor] = []
    if mode in ("cf_only", "hybrid"):
        cf_vectors = build_cf_columns(
            preLM=preLM,
            layer_idx=layer_idx,
            device=device,
            train_dataloader=cf_train_dataloader,
            cf_concept_set=cf_concept_set,
            cache_dir=cache_dir,
            cache_tag=f"cf_probe_layer{layer_idx}",
            cache_key_extras=cf_cache_key_extras,
            max_batches=cf_probe_max_batches,
        )

    cols: List[Tensor] = []
    cf_offset = 0
    cf_used: List[str] = []
    pace_used: List[str] = []
    if mode == "hybrid":
        cols.extend(pace_vectors)
        cf_offset = len(pace_vectors)
        cols.extend(cf_vectors)
        pace_used = pace_concepts
        cf_used = list(cf_concept_set)
    elif mode == "pace_only":
        cols.extend(pace_vectors)
        pace_used = pace_concepts
    else:  # cf_only
        cols.extend(cf_vectors)
        cf_used = list(cf_concept_set)

    if not cols:
        raise RuntimeError(
            f"build_dictionary produced 0 columns (mode={mode}); check inputs."
        )

    D = torch.stack(cols, dim=1).contiguous()  # (H, C)
    meta = {
        "layer_idx": int(layer_idx),
        "mode": mode,
        "cf_offset": int(cf_offset),
        "cf_size": len(cf_used),
        "pace_size": len(pace_used),
        "cf_concepts": cf_used,
        "pace_concepts": pace_used,
        "shape": list(D.shape),
        "max_pace_concepts": int(max_pace_concepts),
        "cache_signature": full_extras,
    }
    torch.save(D, dict_path)
    meta_path.write_text(json.dumps(meta, indent=2))
    meta["dictionary_path"] = str(dict_path)
    meta["meta_path"] = str(meta_path)
    return D, meta


def load_dictionary(*, dictionary_path: str, meta_path: str) -> tuple[Tensor, dict]:
    D = torch.load(dictionary_path, map_location="cpu")
    meta = json.loads(Path(meta_path).read_text())
    meta["dictionary_path"] = dictionary_path
    meta["meta_path"] = meta_path
    return D, meta
