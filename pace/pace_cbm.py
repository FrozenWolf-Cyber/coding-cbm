"""PaCE-CBM Architecture 1: learned dense decomposition over a frozen dictionary.

Math
----
Given hidden state ``z_in ∈ R^(B, T, H)`` at the hooked layer L:

    h        = z_in @ D                         # (B, T, C) raw alignments
    compr    = h @ W_A                          # (B, T, k) bottleneck
    c_relu   = ReLU(compr @ W_B)                # (B, T, C) learned scores
    c_sparse = ReLU(c_relu - τ)                 # (B, T, C) sparse scores
    r        = z_in − c_sparse @ Dᵀ             # (B, T, H) non-concept residual
    c_ctrl   = c_sparse with chosen entries set/zeroed
    z_ctrl   = c_ctrl @ Dᵀ + r

When no intervention is applied (``c_ctrl = c_sparse``):  ``z_ctrl ≡ z_in``.
This identity-at-init guarantees generation is never harmed before training.

Trainable
---------
- ``W_A ∈ R^(C × k)`` — concept compressor (params: C·k)
- ``W_B ∈ R^(k × C)`` — concept expander  (params: k·C)
- ``τ   ∈ R^(C,)``    — per-concept threshold (params: C)

Frozen
------
- ``D ∈ R^(H × C)``  — registered as a non-persistent buffer; never optimised.

The dictionary is *not* saved with the module's state-dict (the trainer keeps
it in its own cache + sidecar meta). Use ``state_dict_no_dict`` /
``load_state_dict_with_dict`` for safe round-trips.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class PaCECBM(nn.Module):
    def __init__(
        self,
        *,
        D: Tensor,
        k: int,
        layer_idx: int,
        cf_offset: int,
        cf_size: int,
        compute_dtype: torch.dtype = torch.float32,
    ):
        """``D`` is the dictionary of shape ``(H, C)``; it is registered as a
        non-trainable buffer and is *not* persisted with this module's
        state-dict (it is rebuilt / reloaded from the dictionary cache).
        """
        super().__init__()
        if D.dim() != 2:
            raise ValueError(f"D must be 2-D (H, C); got {tuple(D.shape)}.")
        H, C = D.shape
        self.H = int(H)
        self.C = int(C)
        self.k = int(k)
        self.layer_idx = int(layer_idx)
        self.cf_offset = int(cf_offset)
        self.cf_size = int(cf_size)
        self.compute_dtype = compute_dtype

        # Frozen dictionary as a non-persistent buffer.
        self.register_buffer("D", D.to(compute_dtype).contiguous(), persistent=False)

        W_A, W_B = self._svd_init(self.D, k)
        self.W_A = nn.Parameter(W_A.contiguous())  # (C, k)
        self.W_B = nn.Parameter(W_B.contiguous())  # (k, C)
        self.tau = nn.Parameter(torch.zeros(C, dtype=compute_dtype))

    # ── Initialisation helpers ────────────────────────────────────────────────

    @staticmethod
    @torch.no_grad()
    def _svd_init(D: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """Init from low-rank SVD of D.

        We want ``W_A``: (C, k) such that ``h @ W_A`` projects onto the top-k
        right-singular directions of D in C-space.  ``svd_lowrank(D, q=k)``
        gives ``D ≈ U S Vᵀ`` with ``V ∈ R^(C × k)``; we use ``W_A = V``
        (orthonormal columns) and mirror with ``W_B = W_Aᵀ`` per the spec
        (rows of ``W_B`` mirror columns of ``W_A`` at SVD init).
        """
        H, C = D.shape
        q = max(1, min(int(k), min(H, C) - 1))
        U, S, V = torch.svd_lowrank(D.float(), q=q)  # V : (C, q)
        W_A = V.to(D.dtype)
        if W_A.size(1) < k:
            pad = torch.randn(C, k - W_A.size(1), dtype=D.dtype, device=D.device)
            pad = pad / (k ** 0.5)
            W_A = torch.cat([W_A, pad], dim=1)
        W_B = W_A.t().contiguous()
        return W_A, W_B

    @torch.no_grad()
    def init_tau_from_h_samples(self, h_samples: Tensor, percentile: float = 70.0) -> None:
        """Set ``τ[i]`` to the per-concept percentile of ``|h[:, i]|``.

        ``h_samples``: ``(N, C)`` — typically built by collecting token-level
        ``z_in @ D`` over a small subset of train tokens.
        """
        if h_samples.dim() != 2 or h_samples.size(1) != self.C:
            raise ValueError(
                f"h_samples must be (N, C={self.C}); got {tuple(h_samples.shape)}."
            )
        q = max(0.0, min(1.0, float(percentile) / 100.0))
        # torch.quantile is memory-heavy on huge C; chunk along the concept dim.
        chunk = 512
        out = torch.empty(self.C, dtype=h_samples.dtype)
        for s in range(0, self.C, chunk):
            e = min(s + chunk, self.C)
            out[s:e] = torch.quantile(h_samples[:, s:e].abs(), q, dim=0)
        self.tau.data.copy_(out.to(self.tau.dtype, device=self.tau.device))

    # ── Forward primitives ────────────────────────────────────────────────────

    def _to_dtype(self, x: Tensor) -> Tensor:
        return x.to(self.compute_dtype) if x.dtype != self.compute_dtype else x

    def forward_concepts(self, z_in: Tensor) -> tuple[Tensor, Tensor]:
        """``z_in: (B, T, H)`` ⇒ ``(h, c_sparse)`` both in ``(B, T, C)``."""
        z = self._to_dtype(z_in)
        h = z @ self.D  # (B, T, C)
        compr = h @ self.W_A  # (B, T, k)
        c_relu = torch.relu(compr @ self.W_B)  # (B, T, C)
        c_sparse = torch.relu(c_relu - self.tau)  # (B, T, C)
        return h, c_sparse

    def reconstruct(
        self,
        z_in: Tensor,
        c_sparse: Tensor,
        *,
        intervene_row_mask: Optional[Tensor] = None,
        intervene_value: float = 0.0,
        zero_other_concepts: bool = False,
    ) -> Tensor:
        """Reconstruct ``z_ctrl`` from ``c_sparse`` with optional intervention.

        ``intervene_row_mask``: bool tensor of shape ``(B, C)`` selecting
        per-row concept indices to override; expanded to ``(B, T, C)`` here.
        ``intervene_value``: scalar value to set those entries to (use 0.0 to
        zero-out concepts; positive values to amplify).

        If ``zero_other_concepts`` is True, all entries *outside* the row mask
        are zeroed (equivalent to "keep only the active CF tags"), matching the
        ``--intervention_keep_other_concepts`` semantics from the original CBM.

        The residual identity ``z_ctrl = c_ctrl @ Dᵀ + r`` with
        ``r = z - c_sparse @ Dᵀ`` holds **positionwise** for every index,
        including padding positions (pure tensor algebra).  Losses should
        still ignore pad positions where inputs do not affect the objective
        (e.g. sparsity uses ``attention_mask`` in ``loops.py``).
        """
        z = self._to_dtype(z_in)
        D_T = self.D.t()  # (C, H)

        c_DT_baseline = c_sparse @ D_T  # (B, T, H)
        r = z - c_DT_baseline  # (B, T, H)

        if intervene_row_mask is None:
            c_ctrl = c_sparse
        else:
            B, C = intervene_row_mask.shape
            if C != self.C:
                raise ValueError(
                    f"intervene_row_mask width {C} != C={self.C}."
                )
            mask_bt = intervene_row_mask.unsqueeze(1).expand(B, c_sparse.size(1), C)  # (B, T, C)
            value_t = torch.full_like(c_sparse, float(intervene_value))  # (B, T, C)
            if zero_other_concepts:
                # Only the masked entries survive; everything else is zero.
                c_ctrl = torch.where(mask_bt, value_t, torch.zeros_like(c_sparse))
            else:
                c_ctrl = torch.where(mask_bt, value_t, c_sparse)

        z_ctrl = c_ctrl @ D_T + r  # (B, T, H) = (B,T,C) @ (C,H) + (B,T,H)
        return z_ctrl

    def apply(
        self,
        z_in: Tensor,
        *,
        intervene_row_mask: Optional[Tensor] = None,
        intervene_value: float = 0.0,
        zero_other_concepts: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """One-shot: ``forward_concepts`` then ``reconstruct``.

        Returns ``(z_ctrl, c_sparse)`` so callers can compute concept losses.
        """
        _, c_sparse = self.forward_concepts(z_in)
        z_ctrl = self.reconstruct(
            z_in,
            c_sparse,
            intervene_row_mask=intervene_row_mask,
            intervene_value=intervene_value,
            zero_other_concepts=zero_other_concepts,
        )
        return z_ctrl, c_sparse

    # ── Persistence ───────────────────────────────────────────────────────────

    def state_dict_no_dict(self) -> dict:
        """Standard ``state_dict`` — ``D`` is non-persistent so already excluded."""
        return self.state_dict()

    def load_state_dict_with_dict(self, state: dict, *, D: Tensor) -> "PaCECBM":
        """Load a saved state but accept the dictionary tensor explicitly.

        Use this after constructing the module with the same dictionary
        retrieved from the dictionary cache.
        """
        if D.shape != self.D.shape:
            raise ValueError(
                f"Loaded D shape {tuple(D.shape)} != module D shape {tuple(self.D.shape)}."
            )
        self.D.data.copy_(D.to(self.D.dtype, device=self.D.device))
        self.load_state_dict(state, strict=True)
        return self
