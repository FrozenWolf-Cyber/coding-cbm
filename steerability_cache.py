"""
Disk cache for steerability generations: single JSON file per run.

Layout under the checkpoint folder (same directory as epoch weights):

    steerability_outputs/epoch_{E}[_lowscore]/
        samples.json

JSON structure (all keys are strings):

    {
        "<concept_idx>": {
            "<seed>": {
                "<sample_idx>": "<generated_text>"
            }
        }
    }

Used by resume steerability scripts and training eval so partial runs can resume.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence


def sanitize_concept_slug(name: str, max_len: int = 80) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name).strip())
    s = s.strip("_") or "concept"
    return s[:max_len]


def steerability_output_root(ckpt_prefix: str, epoch: int, is_low_score: bool) -> str:
    """Directory root for steerability samples for one evaluated checkpoint."""
    ckpt_prefix = os.path.normpath(ckpt_prefix)
    sfx = "_lowscore" if is_low_score else ""
    return os.path.join(ckpt_prefix, "steerability_outputs", f"epoch_{epoch}{sfx}")


def _json_path(root: str) -> str:
    return os.path.join(root, "samples.json")


def _load_json(root: str) -> Dict:
    p = _json_path(root)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _save_json(root: str, data: Dict) -> None:
    os.makedirs(root, exist_ok=True)
    p = _json_path(root)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_sample(
    root: str,
    concept_idx: int,
    concept_name: str,
    seed: int,
    sample_idx: int,
    text: str,
) -> None:
    """Write a single sample into the run's JSON cache (read-modify-write)."""
    data = _load_json(root)
    ci_key = str(concept_idx)
    s_key = str(seed)
    si_key = str(sample_idx)
    data.setdefault(ci_key, {}).setdefault(s_key, {})[si_key] = text
    _save_json(root, data)


def write_samples_batch(
    root: str,
    samples: List[tuple],
) -> None:
    """
    Write multiple samples in one JSON read-modify-write cycle.

    Each element of *samples* is ``(concept_idx, concept_name, seed, sample_idx, text)``.
    """
    if not samples:
        return
    data = _load_json(root)
    for concept_idx, _concept_name, seed, sample_idx, text in samples:
        ci_key = str(concept_idx)
        s_key = str(seed)
        si_key = str(sample_idx)
        data.setdefault(ci_key, {}).setdefault(s_key, {})[si_key] = text
    _save_json(root, data)


def load_concept_samples(
    cache_root: Optional[str],
    seed: int,
    concept_idx: int,
    concept_name: str,
    n_samples: int,
) -> List[Optional[str]]:
    """Return length-n list; entry k is cached text or None if missing."""
    if not cache_root or n_samples <= 0:
        return [None] * max(0, n_samples)
    data = _load_json(cache_root)
    concept_data = data.get(str(concept_idx), {}).get(str(seed), {})
    return [concept_data.get(str(k)) for k in range(n_samples)]


def save_all_steerability_texts(
    cache_root: str,
    seed: int,
    concept_set: Sequence[str],
    texts_by_concept: Sequence[Sequence[str]],
) -> None:
    """
    Write every sample to the JSON cache (idempotent).
    Skips concepts with empty lists.
    """
    if not cache_root:
        return
    data = _load_json(cache_root)
    s_key = str(seed)
    for ci, texts in enumerate(texts_by_concept):
        if not texts or ci >= len(concept_set):
            continue
        ci_key = str(ci)
        seed_dict = data.setdefault(ci_key, {}).setdefault(s_key, {})
        for si, t in enumerate(texts):
            seed_dict[str(si)] = t
    _save_json(cache_root, data)
