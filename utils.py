import torch
import torch.nn.functional as F
import config as CFG
from typing import Sequence

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def eos_pooling(token_embeddings, attention_mask):
    last_index = []
    for i in range(attention_mask.size(0)):
        last_index.append(check_zero(attention_mask[i]))
    last_index = torch.tensor(last_index)
    return token_embeddings[range(len(last_index)), last_index]

def check_zero(mask):
    for i in range(len(mask)):
        if mask[i] == 0:
            return i-1
    return len(mask)-1

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=float('-inf')):
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][:, -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0][indices_to_remove] = filter_value
    return logits

def top_k_top_p_filtering_batched(logits, top_k=0, top_p=0.0, filter_value=float('-inf')):
    """Batched top-k/top-p filtering. logits: (B, vocab_size)"""
    if top_k > 0:
        top_k_vals = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)[0]
        indices_to_remove = logits < top_k_vals[:, -1:]
        logits = logits.masked_fill(indices_to_remove, filter_value)
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits

def elastic_net_penalty(param, alpha=0.99):
    return alpha * torch.abs(param).mean() + (1-alpha) * torch.square(param).mean()

def cos_sim_cubed(cbl_features, target, reduce: bool = True):
    """Cosine similarity after centering and cubing.

    Args:
        cbl_features: (..., D)
        target:       (..., D)
        reduce:       If True (default), return mean over the last non-feature dim.
                       If False, return per-sample similarities (no final mean).
    """
    cbl_features = cbl_features - torch.mean(cbl_features, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)

    cbl_features = F.normalize(cbl_features**3, dim=-1)
    target = F.normalize(target**3, dim=-1)

    sim = torch.sum(cbl_features * target, dim=-1)  # (...,)
    if reduce:
        return sim.mean()
    return sim

def normalize(x, d=-1, mean=None, std=None):
    if mean is not None and std is not None:
        x_mean = mean
        x_std = std
    else:
        x_mean = torch.mean(x, dim=d)
        x_std = torch.std(x, dim=d)
    if d == -1:
        x = x - x_mean.unsqueeze(-1)
        x = x / (x_std.unsqueeze(-1) + 1e-12)
    else:
        x = x - x_mean.unsqueeze(0)
        x = x / (x_std.unsqueeze(0) + 1e-12)
    return x, x_mean, x_std


def load_jsonl_as_dataset(jsonl_path: str, max_samples: int = 0):
    """Load a local JSONL file into a HF Dataset, preserving file order.

    This matches truthful_qa/annotate_llamacpp.py's behavior (read line-by-line json.loads).

    Args:
        jsonl_path: Path to a .jsonl file.
        max_samples: If >0, stop after this many rows.
    """
    import os
    import json

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples and max_samples > 0 and len(rows) >= int(max_samples):
                break

    if len(rows) == 0:
        raise ValueError(f"No rows found in JSONL: {jsonl_path}")

    # Lazy import to avoid making `datasets` a hard dependency for every utils consumer.
    from datasets import Dataset

    return Dataset.from_list(rows)




def build_intervened_concepts_from_similarity(
    concepts: torch.Tensor,
    batch_sim: torch.Tensor,
    intervention_value: float,
    keep_other_concepts: bool,
) -> torch.Tensor:
    """Construct an intervention concept tensor for `cbl.intervene`.

        For each example, set *all* concepts present in `batch_sim` (>0) to
        `intervention_value` for all time steps. All other concepts are set to 0,
        unless `keep_other_concepts=True`, in which case only the tagged concepts
        are overwritten and the rest are left as-is.
    """
    if concepts.dim() != 3:
        raise ValueError(f"Expected concepts to have shape (B, T, C); got {tuple(concepts.shape)}")
    if batch_sim.dim() != 2:
        raise ValueError(f"Expected batch_sim to have shape (B, C); got {tuple(batch_sim.shape)}")
    if concepts.size(0) != batch_sim.size(0) or concepts.size(-1) != batch_sim.size(-1):
        raise ValueError(
            f"Shape mismatch: concepts {tuple(concepts.shape)} vs batch_sim {tuple(batch_sim.shape)}"
        )

    if keep_other_concepts:
        intervened = concepts.detach().clone()
    else:
        intervened = torch.zeros_like(concepts)

    value = float(intervention_value)
    mask = batch_sim > 0  # (B, C)
    if mask.any():
        intervened = intervened.clone()
        intervened[mask.unsqueeze(1).expand_as(intervened)] = value
    return intervened


def compute_multilabel_topk_accuracy(
    prediction_scores: torch.Tensor,
    target_scores: torch.Tensor,
    topk: Sequence[int] = (1, 5, 10),
) -> dict:
    """Compute top-k hit accuracy for multi-label concept targets.

    A sample is counted as correct at k if any ground-truth positive concept index
    appears in the model's top-k predicted indices.
    """
    if prediction_scores.dim() != 2 or target_scores.dim() != 2:
        raise ValueError(
            f"Expected 2D tensors, got prediction_scores={tuple(prediction_scores.shape)} "
            f"target_scores={tuple(target_scores.shape)}"
        )
    if prediction_scores.shape != target_scores.shape:
        raise ValueError(
            f"Shape mismatch: prediction_scores={tuple(prediction_scores.shape)} "
            f"target_scores={tuple(target_scores.shape)}"
        )

    n_samples, n_concepts = prediction_scores.shape
    if n_samples == 0:
        return {f"top{k}_acc": 0.0 for k in topk}

    target_positive = target_scores > 0
    has_positive = target_positive.any(dim=-1)
    if not bool(has_positive.any()):
        return {f"top{k}_acc": 0.0 for k in topk}

    valid_pred = prediction_scores[has_positive]
    valid_target_positive = target_positive[has_positive]
    valid_total = valid_pred.size(0)

    metrics = {}
    for k in topk:
        k_clipped = min(int(k), n_concepts)
        if k_clipped <= 0:
            metrics[f"top{k}_acc"] = 0.0
            continue
        topk_indices = torch.topk(valid_pred, k=k_clipped, dim=-1).indices
        hit = valid_target_positive.gather(1, topk_indices).any(dim=-1)
        metrics[f"top{k}_acc"] = float(hit.float().mean().item()) if valid_total > 0 else 0.0

    return metrics


def compute_multilabel_concept_metrics(
    prediction_scores: torch.Tensor,
    target_scores: torch.Tensor,
    topk: Sequence[int] = (1, 5, 10),
) -> dict:
    """Compute top-k accuracy, top-k IoU, and cosine similarity metrics.

    Returns:
      - top{k}_acc: hit@k over positive-label samples.
      - top{k}_iou: IoU between predicted top-k set and target top-k set.
      - cosine_raw: mean cosine similarity between prediction and target vectors.
      - cosine_cubed: mean centered-cubed cosine similarity.
    """
    if prediction_scores.dim() != 2 or target_scores.dim() != 2:
        raise ValueError(
            f"Expected 2D tensors, got prediction_scores={tuple(prediction_scores.shape)} "
            f"target_scores={tuple(target_scores.shape)}"
        )
    if prediction_scores.shape != target_scores.shape:
        raise ValueError(
            f"Shape mismatch: prediction_scores={tuple(prediction_scores.shape)} "
            f"target_scores={tuple(target_scores.shape)}"
        )

    n_samples, n_concepts = prediction_scores.shape
    if n_samples == 0:
        base = {f"top{k}_acc": 0.0 for k in topk}
        base.update({f"top{k}_iou": 0.0 for k in topk})
        base["cosine_raw"] = 0.0
        base["cosine_cubed"] = 0.0
        return base

    topk_metrics = compute_multilabel_topk_accuracy(
        prediction_scores=prediction_scores,
        target_scores=target_scores,
        topk=topk,
    )

    pred_sorted = torch.argsort(prediction_scores, dim=-1, descending=True)
    topk_iou_sums = {int(k): 0.0 for k in topk}
    for i in range(n_samples):
        row = pred_sorted[i]
        for k in topk_iou_sums.keys():
            k_clipped = min(int(k), n_concepts)
            if k_clipped <= 0:
                continue
            gt_topk = torch.topk(target_scores[i], k=k_clipped, dim=-1).indices.tolist()
            pred_topk = row[:k_clipped].tolist()
            gt_set, pred_set = set(gt_topk), set(pred_topk)
            union = len(gt_set | pred_set)
            if union > 0:
                topk_iou_sums[k] += len(gt_set & pred_set) / union

    iou_metrics = {f"top{k}_iou": (topk_iou_sums[int(k)] / n_samples) for k in topk}

    pred_norm = F.normalize(prediction_scores, p=2, dim=-1)
    target_norm = F.normalize(target_scores, p=2, dim=-1)
    cosine_raw = float((pred_norm * target_norm).sum(dim=-1).mean().item())
    cosine_cubed = float(cos_sim_cubed(prediction_scores, target_scores).item())

    return {
        **topk_metrics,
        **iou_metrics,
        "cosine_raw": cosine_raw,
        "cosine_cubed": cosine_cubed,
    }
