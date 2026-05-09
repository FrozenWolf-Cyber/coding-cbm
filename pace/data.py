"""code_contests data loaders for PaCE-CBM training and evaluation.

Reuses the row-level tokenization from ``train_combined_finegrained`` so the
supervised label scheme (assistant-only ``loss_mask``, dynamic padding, multi-hot
CF-tag supervision) is byte-identical to the existing CBM trainer.

The only thing this module re-implements is the dynamic padding collate, since
the original ``_dynamic_padding_collate`` captures a module-level ``tokenizer``;
``make_collate(pad_id)`` returns the same function parameterised on a pad id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import DownloadConfig, load_dataset

# These are pure functions / dataset class — no side-effects on import.
from train_combined_finegrained import (
    LazyTokenizedClassificationDataset,
    _tokenize_eval_row,
    _tokenize_supervised_row,
)


def make_collate(pad_id: int):
    """Return a dynamic-padding collate fn that pads to the longest seq in the batch.

    Same shape as ``train_combined_finegrained._dynamic_padding_collate`` but
    parameterised on ``pad_id`` so this module has no global state.
    """

    def _collate(batch):
        batch_text, batch_sim = zip(*batch)
        max_len = max(int(x["input_ids"].shape[0]) for x in batch_text)  # longest T in batch → pad others to (B, max_len)

        input_ids: list[torch.Tensor] = []
        attention_mask: list[torch.Tensor] = []
        loss_mask: list[torch.Tensor] = []
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
                lm_target_len = max_len - 1
                cur_lm_len = int(x["loss_mask"].shape[0])
                lm_pad_len = lm_target_len - cur_lm_len
                if lm_pad_len > 0:
                    lm = F.pad(x["loss_mask"], (0, lm_pad_len), value=0)
                else:
                    lm = x["loss_mask"][:lm_target_len]
                loss_mask.append(lm)

        out_text = {
            "input_ids": torch.stack(input_ids, dim=0),  # (B, max_len)
            "attention_mask": torch.stack(attention_mask, dim=0),  # (B, max_len)
        }
        if has_loss_mask:
            out_text["loss_mask"] = torch.stack(loss_mask, dim=0)  # (B, max_len-1)

        out_sim = torch.stack(batch_sim, dim=0)  # (B, C_cf)
        return out_text, out_sim

    return _collate


def build_loaders_param(
    raw_hf_dataset,
    s,
    mode: str,
    tokenizer,
    args,
    *,
    shuffle: Optional[bool] = None,
):
    """Build a DataLoader over ``LazyTokenizedClassificationDataset``.

    ``mode`` is one of ``"train"``, ``"valid"``, ``"test"``. Default shuffle
    is True only for ``"train"``; pass ``shuffle`` explicitly to override.
    """
    if shuffle is None:
        shuffle = (mode == "train")
    dataset = LazyTokenizedClassificationDataset(raw_hf_dataset, s, mode, tokenizer, args)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle,
        collate_fn=make_collate(tokenizer.pad_token_id),
    )


def hf_load_dataset_cache_first(dataset_name: str, cache_dir: str):
    """Cache-first ``load_dataset``: try local-only, then fall back to download.

    Mirrors ``train_combined_finegrained._hf_load_dataset_cache_first`` without
    importing it (that function uses try/except, which our convention forbids in
    new code; we rely on ``DownloadConfig(local_files_only=True)`` raising at
    call-time and resolve via a single fallback in the trainer).
    """
    return load_dataset(
        dataset_name,
        cache_dir=cache_dir,
        download_config=DownloadConfig(local_files_only=False),
    )


def filter_codecontests(
    raw_dataset,
    *,
    cf_concept_lookup,
    keep_python_only_for_train: bool = True,
    max_train_samples: int = 0,
    max_valid_samples: int = 0,
    max_test_samples: int = 0,
):
    """Apply the same filters the existing trainer does.

    - keep rows whose ``cf_tags`` intersect ``cf_concept_lookup``;
    - additionally drop rows without a Python solution from the train split
      (so LM targets exist).
    """
    train = raw_dataset["train"]
    valid = raw_dataset["valid"]
    test = raw_dataset["test"]

    if max_train_samples > 0:
        train = train.select(range(min(max_train_samples, len(train))))
    if max_valid_samples > 0:
        valid = valid.select(range(min(max_valid_samples, len(valid))))
    if max_test_samples > 0:
        test = test.select(range(min(max_test_samples, len(test))))

    def _has_valid_cf_tag(example):
        return any(t in cf_concept_lookup for t in (example.get("cf_tags") or []))

    def _has_python_solution(example):
        sols = example.get("solutions")
        if not isinstance(sols, dict):
            return False
        langs = sols.get("language") or []
        texts = sols.get("solution") or []
        return any(
            lang in (1, 3) and isinstance(sol, str) and sol.strip()
            for lang, sol in zip(langs, texts)
        )

    train = train.filter(_has_valid_cf_tag)
    valid = valid.filter(_has_valid_cf_tag)
    test = test.filter(_has_valid_cf_tag)
    if keep_python_only_for_train:
        train = train.filter(_has_python_solution)
    return train, valid, test


def build_multihot(dataset, concept_set):
    """Multi-hot supervision (N, C) from CF tags, asserting no row is empty.

    Same semantics as ``train_combined_finegrained._build_multihot``.
    """
    concept_idx = {c: i for i, c in enumerate(concept_set)}
    n = len(dataset)
    sim = np.zeros((n, len(concept_set)), dtype=np.float32)
    for i in range(n):
        tags = dataset[i].get("cf_tags") or []
        for t in tags:
            j = concept_idx.get(t)
            if j is not None:
                sim[i, j] = 1.0
        if sim[i].sum() == 0:
            raise ValueError(
                f"Row {i} has no valid CF tags in concept_set; check filtering."
            )
    return sim


def resolve_cache_subdir(root_dir: str, name: str) -> str:
    path = Path(root_dir).expanduser() / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
