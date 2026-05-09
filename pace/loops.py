"""Pure-function train / validate epochs for PaCE-CBM.

The trainer (``train_pace_cbm.py``) owns the model + optimiser + dataloader and
calls these functions per epoch.  Losses follow the spec:

  - ``word_loss``        — CE on shifted vocab from the **intervened** forward
                           (boost ground-truth CF tags); assistant-only mask.
                           This is the "generation loss similar to current CBM"
                           the user asked for; mirrors the existing
                           ``--intervention_gen_loss`` path semantically.
  - ``concept_loss``     — pooled ``c_sparse[:, :, cf_offset:cf_offset+cf_size]``
                           vs the multi-hot ``batch_sim`` (cosine-cubed or CE).
  - ``sparsity_loss``    — mean |c_sparse| over **non-pad** positions only
                           (``attention_mask``), so padding does not dilute L1.
  - ``identity_loss``    — optional MSE between un-intervened ``z_ctrl`` and
                           ``z_in``; mathematically zero by the residual
                           identity, so enabled only as a numerical sanity
                           guard when float16 accumulation might drift.

A single intervened forward computes ``c_sparse`` (used for concept + sparsity)
because PaCE-CBM's ``c_sparse`` is derived from ``h_L`` (unmodified by the
intervention), so concept supervision is consistent with the no-steer forward.

``eos_pooling`` / concept metrics use the batch dict's ``attention_mask`` —
the same ``T`` as ``c_sparse`` — so EOS indices ignore right-pad tokens.  The
hook does not need the mask; losses run after ``llm(...)`` with ``batch`` still
in scope.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

from utils import (
    compute_multilabel_concept_metrics,
    cos_sim_cubed,
    eos_pooling,
)

from .hook_steerer import PaCECBMSteerer


def _sparsity_loss_masked(c_sparse: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean |c_sparse| over valid (non-pad) token positions only.

    ``c_sparse``: (B, T, C), ``attention_mask``: (B, T) with 1 for real tokens.
    Padding positions would otherwise bias L1 toward arbitrary post-norm values.
    """
    m = attention_mask.float().unsqueeze(-1)  # (B, T, 1)
    num = (c_sparse.abs() * m).sum()
    denom = m.sum() * c_sparse.size(-1)
    return num / denom.clamp_min(1.0)


def _build_word_label(batch: dict) -> Tensor:
    """Shifted-by-1 word_label (B, T-1) with -100 outside the assistant tokens.

    Matches ``train_combined_finegrained.py`` lines ~920-980.
    """
    word_label = torch.where(
        batch["attention_mask"][:, :-1] == 0, -100, batch["input_ids"][:, 1:]
    )  # (B, T-1): CE targets (shifted); -100 = ignore
    if "loss_mask" in batch:
        ignore = torch.full_like(word_label, -100)
        word_label = torch.where(batch["loss_mask"] > 0, word_label, ignore)  # (B, T-1); assistant only
    return word_label


def _flat_valid_mask(batch: dict, *, skip_loss_mask: bool) -> Tensor:
    """Flat (B*(T-1),) bool mask over assistant tokens (or all non-pad if
    ``skip_loss_mask``)."""
    mask = (batch["attention_mask"][:, :-1] != 0).reshape(-1)
    if (not skip_loss_mask) and ("loss_mask" in batch):
        mask = mask & (batch["loss_mask"] > 0).reshape(-1)
    return mask


def _concept_loss_value(
    c_sparse_cf: Tensor,
    batch_sim: Tensor,
    *,
    flat_mask: Tensor,
    loss_type: str,
) -> Tensor:
    """Per-token concept supervision restricted to ``flat_mask`` positions."""
    B, T, C = c_sparse_cf.shape  # c_sparse_cf: (B, T, C_cf)
    # Align with CE on vocab: positions 0..T-2 predict token 1..T-1 → use same span.
    c_slice = c_sparse_cf[:, :-1, :].contiguous().view(-1, C)  # (B*(T-1), C_cf)
    sim_slice = (
        batch_sim.unsqueeze(1).expand(-1, T - 1, -1).contiguous().view(-1, C)  # (B*(T-1), C_cf)
    )
    valid_c = c_slice[flat_mask]
    valid_sim = sim_slice[flat_mask]
    if valid_c.size(0) == 0:
        return torch.zeros((), device=c_sparse_cf.device)
    if loss_type == "cosine_cubed":
        return -cos_sim_cubed(valid_c, valid_sim)
    if loss_type == "ce":
        hard_targets = torch.argmax(valid_sim, dim=-1)
        return F.cross_entropy(valid_c, hard_targets)
    raise ValueError(f"Unknown concept_loss_type: {loss_type}")


def _word_loss_value(vocabs: Tensor, word_label: Tensor) -> Tensor:
    # vocabs: (B, T, V_vocab); word_label: (B, T-1) — CE on each position vs next token
    return F.cross_entropy(
        vocabs[:, :-1, :].reshape(-1, vocabs.size(-1)),  # (B*(T-1), V)
        word_label.reshape(-1),  # (B*(T-1),)
    )


def _identity_loss_value(z_ctrl_unintervened: Tensor, z_in: Tensor, batch: dict, *, skip_loss_mask: bool) -> Tensor:
    """MSE(z_ctrl_no_intervene, z_in), restricted to assistant positions.

    Mirrors the reconstruction-loss masking trick from the existing trainer
    (left-pad shifted ``loss_mask`` by one to align "next-token is assistant"
    with "current position is assistant").
    """
    diff = (z_ctrl_unintervened.float() - z_in.float()) ** 2  # (B, T, H)
    mask = batch["attention_mask"].bool()
    if (not skip_loss_mask) and ("loss_mask" in batch):
        lm_full = F.pad(batch["loss_mask"], (1, 0), value=0).bool()
        mask = mask & lm_full
    mask_f = mask.unsqueeze(-1).to(diff.dtype)
    denom = mask_f.sum().clamp_min(1.0) * diff.size(-1)
    return (diff * mask_f).sum() / denom


def train_one_epoch(
    *,
    pace_cbm,
    llm,
    steerer: PaCECBMSteerer,
    train_loader,
    optimizer,
    args,
    device: torch.device,
    epoch: int,
    max_steps_per_epoch: Optional[int] = None,
    log_fn=None,
) -> dict:
    """Train PaCE-CBM for one epoch with intervened generation loss.

    ``llm`` is a frozen ``LlamaForCausalLM`` (use ``llm.eval()``); ``pace_cbm``
    is set to ``train()``.  ``steerer`` wraps ``pace_cbm`` and is *attached*
    to ``llm`` here.

    ``max_steps_per_epoch``:
      - ``None`` ⇒ no limit (default; full epoch).
      - ``0``    ⇒ break before any optimisation step (true 0-step debug;
                    matches ``--debug_0_step`` semantics in
                    ``train_combined_finegrained.py``).
      - ``N>0``  ⇒ break after ``N`` optimisation steps (debug shortcut).
    """
    pace_cbm.train()
    llm.eval()

    losses: dict[str, list[float]] = {
        "word_loss": [],
        "concept_loss": [],
        "sparsity_loss": [],
        "identity_loss": [],
        "total_loss": [],
    }

    iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"train/epoch_{epoch+1}")
    steerer.attach(llm)
    with steerer:
        for i, (batch, batch_sim) in iterator:
            # Honour true 0-step debug *before* moving anything to device.
            if max_steps_per_epoch is not None and i >= max_steps_per_epoch:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            # batch["input_ids"]: (B, T_max); "loss_mask": (B, T_max-1) when present
            batch_sim = batch_sim.to(device)  # (B, C_cf) multihot CF supervision
            word_label = _build_word_label(batch)
            flat_mask = _flat_valid_mask(batch, skip_loss_mask=args.skip_loss_mask)  # (B*(T-1),)

            steerer.configure_for_batch(
                cf_multihot=batch_sim,
                intervene_value=args.intervention_alpha,
                zero_other_concepts=False,
            )

            outputs = llm(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            vocabs = outputs.logits  # (B, T, V_vocab)
            c_sparse = steerer.last_c_sparse  # (B, T, C_total)
            if c_sparse is None:
                raise RuntimeError("PaCECBMSteerer did not stash c_sparse — hook misfired?")

            c_sparse_cf = c_sparse[:, :, pace_cbm.cf_offset:pace_cbm.cf_offset + pace_cbm.cf_size]  # (B, T, C_cf)

            word_loss = _word_loss_value(vocabs, word_label) * args.word_loss_weight
            concept_loss = _concept_loss_value(
                c_sparse_cf, batch_sim,
                flat_mask=flat_mask,
                loss_type=args.concept_loss_type,
            ) * args.concept_loss_weight
            sparsity_loss = _sparsity_loss_masked(c_sparse, batch["attention_mask"]) * args.sparsity_weight

            total = word_loss + concept_loss + sparsity_loss
            identity_loss = torch.zeros((), device=device)
            if args.identity_weight > 0:
                # No second forward needed: ``steerer.last_h_L`` is the layer
                # output (= z_in to PaCE-CBM) cached by the hook, and
                # ``c_sparse`` is also cached. Reconstruct without intervention
                # and compare. By the residual identity this is mathematically
                # zero; the loss surfaces numerical drift under bf16/fp16.
                z_in_cached = steerer.last_h_L  # (B, T, H)
                z_ctrl_clean = pace_cbm.reconstruct(z_in_cached, c_sparse)  # (B, T, H)
                identity_loss = _identity_loss_value(
                    z_ctrl_clean, z_in_cached, batch, skip_loss_mask=args.skip_loss_mask,
                ) * args.identity_weight
                total = total + identity_loss

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

            losses["word_loss"].append(float(word_loss.detach().cpu().item()))
            losses["concept_loss"].append(float(concept_loss.detach().cpu().item()))
            losses["sparsity_loss"].append(float(sparsity_loss.detach().cpu().item()))
            losses["identity_loss"].append(float(identity_loss.detach().cpu().item()))
            losses["total_loss"].append(float(total.detach().cpu().item()))

            if log_fn is not None:
                log_fn({
                    "epoch": epoch + 1,
                    "batch": i + 1,
                    "word_loss": losses["word_loss"][-1],
                    "concept_loss": losses["concept_loss"][-1],
                    "sparsity_loss": losses["sparsity_loss"][-1],
                    "identity_loss": losses["identity_loss"][-1],
                    "total_loss": losses["total_loss"][-1],
                })

    averages = {
        f"avg_{k}": (sum(v) / len(v)) if v else 0.0
        for k, v in losses.items()
    }
    return averages


@torch.no_grad()
def validate_one_epoch(
    *,
    pace_cbm,
    llm,
    steerer: PaCECBMSteerer,
    valid_loader,
    args,
    device: torch.device,
    epoch: int,
    cf_offset: int,
    cf_size: int,
) -> dict:
    """Validation: identical loss formula on un-intervened vocab is *trivial*
    (z_ctrl ≡ z_in), so we report:

      - intervened word_loss / concept_loss / sparsity (same as train)
      - top-k accuracy / IoU / cosine on EOS-pooled c_sparse[CF block]
        vs ground-truth multi-hot, via ``compute_multilabel_concept_metrics``.
    """
    pace_cbm.eval()
    llm.eval()

    val_losses: dict[str, list[float]] = {
        "word_loss": [], "concept_loss": [], "sparsity_loss": [], "total_loss": [],
    }
    pred_chunks: list[Tensor] = []
    target_chunks: list[Tensor] = []

    steerer.attach(llm)
    with steerer:
        for batch, batch_sim in tqdm(valid_loader, total=len(valid_loader), desc=f"valid/epoch_{epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            # batch["input_ids"]: (B, T_max); "loss_mask": (B, T_max-1) when present
            batch_sim = batch_sim.to(device)  # (B, C_cf) multihot CF supervision
            word_label = _build_word_label(batch)
            flat_mask = _flat_valid_mask(batch, skip_loss_mask=args.skip_loss_mask)  # (B*(T-1),)

            steerer.configure_for_batch(
                cf_multihot=batch_sim,
                intervene_value=args.intervention_alpha,
                zero_other_concepts=False,
            )

            outputs = llm(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            vocabs = outputs.logits  # (B, T, V_vocab)
            c_sparse = steerer.last_c_sparse  # (B, T, C_total)
            c_sparse_cf = c_sparse[:, :, cf_offset:cf_offset + cf_size]  # (B, T, C_cf)

            wl = _word_loss_value(vocabs, word_label)
            cl = _concept_loss_value(
                c_sparse_cf, batch_sim,
                flat_mask=flat_mask, loss_type=args.concept_loss_type,
            )
            sp = _sparsity_loss_masked(c_sparse, batch["attention_mask"])
            tot = (
                args.word_loss_weight * wl
                + args.concept_loss_weight * cl
                + args.sparsity_weight * sp
            )

            val_losses["word_loss"].append(float(wl.detach().cpu().item()))
            val_losses["concept_loss"].append(float(cl.detach().cpu().item()))
            val_losses["sparsity_loss"].append(float(sp.detach().cpu().item()))
            val_losses["total_loss"].append(float(tot.detach().cpu().item()))

            pooled = eos_pooling(c_sparse_cf, batch["attention_mask"]).detach().cpu()  # (B, C_cf)
            pred_chunks.append(pooled)
            target_chunks.append(batch_sim.detach().cpu())

    pred_tensor = torch.cat(pred_chunks, dim=0) if pred_chunks else torch.empty(0, cf_size)  # (N_val, C_cf)
    target_tensor = torch.cat(target_chunks, dim=0) if target_chunks else torch.empty(0, cf_size)  # (N_val, C_cf)
    topk = compute_multilabel_concept_metrics(
        prediction_scores=pred_tensor,
        target_scores=target_tensor,
        topk=(1, 5, 10),
    )

    averages = {
        f"avg_valid_{k}": (sum(v) / len(v)) if v else 0.0
        for k, v in val_losses.items()
    }
    out = {
        **averages,
        "valid_concept_top1_acc": topk["top1_acc"],
        "valid_concept_top5_acc": topk["top5_acc"],
        "valid_concept_top10_acc": topk["top10_acc"],
        "valid_concept_top1_iou": topk["top1_iou"],
        "valid_concept_top5_iou": topk["top5_iou"],
        "valid_concept_top10_iou": topk["top10_iou"],
        "valid_concept_cosine_raw": topk["cosine_raw"],
        "valid_concept_cosine_cubed": topk["cosine_cubed"],
        "valid_loss": averages["avg_valid_total_loss"],
    }
    return out
