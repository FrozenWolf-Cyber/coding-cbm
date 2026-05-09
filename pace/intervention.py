"""Method-agnostic CF-tag → per-row payload resolver.

Eval / training callers pass:
  - ``cf_tags_per_row``: ``List[List[str]]`` (one tag list per problem); or
  - ``cf_multihot``:    ``Tensor (B, C_cf)`` already built.

This module turns either format into the ``configure_for_batch`` call shape
expected by the chosen steerer. It is the single place that knows about the
"ground-truth steering" semantics shared across every method (mirrors
``eval_metrics._build_groundtruth_intervene`` / ``_resolve_intervene`` but
generalised across steerers).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import Tensor

from .hook_steerer import (
    HookSteerer,
    NoSteer,
    PaCECBMSteerer,
    TransformSteerer,
    VecAddSteerer,
)


def cf_tags_to_multihot(
    cf_tags_per_row: Sequence[Sequence[str]],
    cf_concepts: Sequence[str],
    *,
    device: Optional[torch.device] = None,
) -> Tensor:
    """``List[List[str]] → (B, C_cf)`` float multi-hot tensor."""
    cf_index = {c: i for i, c in enumerate(cf_concepts)}
    B = len(cf_tags_per_row)
    C = len(cf_concepts)
    out = torch.zeros(B, C, dtype=torch.float32)  # (B, C_cf)
    for b, tags in enumerate(cf_tags_per_row):
        for t in tags:
            j = cf_index.get(t)
            if j is not None:
                out[b, j] = 1.0
    if device is not None:
        out = out.to(device)
    return out


def configure_steerer(
    steerer: HookSteerer,
    *,
    cf_tags_per_row: Optional[Sequence[Sequence[str]]] = None,
    cf_multihot: Optional[Tensor] = None,
    cf_concepts: Optional[Sequence[str]] = None,
    alpha: float = 1.0,
    zero_other_concepts: bool = False,
    device: Optional[torch.device] = None,
) -> None:
    """Stage the per-batch payload on ``steerer`` for the given CF tags.

    Exactly one of ``cf_tags_per_row`` or ``cf_multihot`` must be provided
    (or both ``None`` / empty for the no-steer baseline).
    """
    if isinstance(steerer, NoSteer):
        return

    if cf_multihot is None and cf_tags_per_row is not None:
        if cf_concepts is None:
            raise ValueError("cf_concepts is required when passing cf_tags_per_row.")
        cf_multihot = cf_tags_to_multihot(cf_tags_per_row, cf_concepts, device=device)

    if isinstance(steerer, PaCECBMSteerer):
        steerer.configure_for_batch(
            cf_multihot=cf_multihot,  # (B, C_cf) or None
            intervene_value=alpha,
            zero_other_concepts=zero_other_concepts,
        )
        return
    if isinstance(steerer, VecAddSteerer):
        steerer.configure_for_batch(cf_multihot=cf_multihot, alpha=alpha)
        return
    if isinstance(steerer, TransformSteerer):
        steerer.configure_for_batch(cf_multihot=cf_multihot)
        return
    raise TypeError(f"Unknown steerer type: {type(steerer).__name__}")


def expand_payload_for_n_samples(steerer: HookSteerer, n_samples: int) -> None:
    """Replicate the staged payload along batch dim for `num_return_sequences`."""
    if n_samples <= 1 or isinstance(steerer, NoSteer):
        return
    expand_fn = getattr(steerer, "_expand_payload_for_n_samples", None)
    if expand_fn is None:
        return
    expand_fn(n_samples)
