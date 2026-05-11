"""PaCE-style **dictionary decomposition** steering for code eval (inference only).

This mirrors the NeurIPS-style hook in ``nrp_scripts/pace.py`` / ``steer/pace.py``:

1. Build coefficients ``c`` by sparse-coding the hidden state at an **anchor**
   position onto a fixed dictionary (one row per Codeforces concept), using
   ``steer.pace.decompose_sparse`` (SVD-reduced least squares).
2. Apply a **CF-tag intervention** on ``c`` (boost selected coordinates; optional
   ``zero_other_concepts``).
3. Reconstruct in hidden space: ``h' = h - Dᵀc + Dᵀc_ctrl`` where rows of ``D``
   are the dictionary vectors (same layout as ``VecAddSteerer``'s ``steer_vecs``).
4. **Reuse** (default **on**): the pair ``(Dᵀc, Dᵀc_ctrl)`` from the anchor is cached
   for **all later** hook calls in the same ``generate`` (same semantics as
   ``reuse_coeff_across_tokens`` in ``steer/pace.PaCESteerer``). Disable with
   ``reuse_anchor_coeffs=False`` to re-decompose every step (slow).

**Defaults**

- ``use_gpu_lstsq=True``: run ``decompose_sparse`` on GPU when ``h_L`` is on CUDA
  (falls back to CPU if no CUDA). Set ``False`` to force CPU.
- ``reuse_anchor_coeffs=True``: cache anchor ``base``/``intervened`` across steps.

**Anchor rule**

- ``T > 1`` (prefill, ``intervene_phase="all"``): anchor = **last** prompt token
  (the site that predicts the first generated token).
- ``T == 1`` (decode step, or single-token forward): anchor = that token.

Under ``decode_only``, prefill is skipped; the first steered forward is the
first **generated** token — decomposition runs there, then coefficients are reused
for all following decode steps.

Trainable PaCE-CBM lives in ``pace_cbm.py``; this module is **steer-only** and
expects the same ``vec_pack.pt`` tensor as **CAA** (``(C_cf, H)``) — eval loads
that file via ``eval_steerable_cli`` (see ``PACE_DECOMP_VEC_SOURCE`` there).
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from steer.pace import decompose_sparse

from .hook_steerer import HookSteerer


class PaCEDecompSteerer(HookSteerer):
    """Sparse dictionary coding on an anchor hidden state; reuse reconstruction by default."""

    method_name = "pace_decomp"

    def __init__(
        self,
        concept_matrix: Tensor,
        *,
        layer_idx: int,
        method_name: str = "PaCE",
        intervene_phase: str = "decode_only",
        normalize_decomposition: bool = True,
        use_gpu_lstsq: bool = True,
        reuse_anchor_coeffs: bool = True,
    ):
        super().__init__(layer_idx=layer_idx, intervene_phase=intervene_phase)
        if concept_matrix.dim() != 2:
            raise ValueError(f"concept_matrix must be (C_cf, H); got {tuple(concept_matrix.shape)}.")
        self.method_name = method_name
        self.C_cf = int(concept_matrix.size(0))
        self.H = int(concept_matrix.size(1))
        self._concept_matrix = concept_matrix.float().contiguous()  # (C_cf, H) CPU
        self._dict_list: List[Tensor] = [
            self._concept_matrix[i].contiguous() for i in range(self.C_cf)
        ]
        self.normalize_decomposition = bool(normalize_decomposition)
        self.use_gpu_lstsq = bool(use_gpu_lstsq)
        self.reuse_anchor_coeffs = bool(reuse_anchor_coeffs)
        self._cached_base: Optional[Tensor] = None  # (B, H) on last seen device
        self._cached_intervened: Optional[Tensor] = None
        # Safety: only permit batch-repeat alignment when the caller explicitly
        # expanded the batch (e.g. repeat_interleave for num_return_sequences).
        # Otherwise, treat B mismatches as an error instead of silently repeating.
        self._allow_batch_repeat: bool = False
        self._batch_repeat_factor: int = 1

    def configure_for_batch(
        self,
        *,
        cf_multihot: Optional[Tensor] = None,
        alpha: float = 1.0,
        zero_other_concepts: bool = False,
    ) -> None:
        self._cached_base = None
        self._cached_intervened = None
        self._allow_batch_repeat = False
        self._batch_repeat_factor = 1
        if cf_multihot is None or cf_multihot.numel() == 0:
            self._payload = None
            return
        if cf_multihot.dim() != 2 or cf_multihot.size(1) != self.C_cf:
            raise ValueError(
                f"cf_multihot must be (B, C_cf={self.C_cf}); got {tuple(cf_multihot.shape)}."
            )
        self._payload = {
            "cf_multihot": cf_multihot.float().contiguous(),
            "alpha": float(alpha),
            "zero_other": bool(zero_other_concepts),
        }

    def _expand_payload_for_n_samples(self, n_samples: int) -> None:
        if self._payload is None or n_samples <= 1:
            return
        # The generation driver expanded the batch dimension via repeat_interleave.
        # Mark that we may need to align cached (B_prompt, H) vectors to the
        # expanded (B_prompt*n_samples, ...) batch.
        self._allow_batch_repeat = True
        self._batch_repeat_factor = int(n_samples)
        self._payload["cf_multihot"] = HookSteerer.expand_payload_for_n_samples(
            self._payload["cf_multihot"], n_samples,
        )

    @staticmethod
    def _apply_cf_to_coeffs(
        c: Tensor,
        cf_row: Tensor,
        *,
        alpha: float,
        zero_other: bool,
    ) -> Tensor:
        """``c``: (C_cf,) on CPU float; ``cf_row``: (C_cf,) multi-hot / weights."""
        m = (cf_row > 0).float()
        if not bool(m.any()):
            return c
        if zero_other:
            return torch.where(m > 0, c + float(alpha), torch.zeros_like(c))
        return c + float(alpha) * m

    def _ensure_cache(self, h_L: Tensor, payload: dict) -> None:
        """Fill ``_cached_base`` / ``_cached_intervened`` from anchor positions."""
        B, T, Hdim = h_L.shape
        if Hdim != self.H:
            raise ValueError(f"h_L hidden dim {Hdim} != dictionary H={self.H}.")
        t_anchor = T - 1 if T > 1 else 0
        h_anchor = h_L[:, t_anchor, :].detach().float()  # (B, H)

        cf = payload["cf_multihot"]
        if cf.size(0) != B:
            if B % cf.size(0) == 0:
                factor = B // cf.size(0)
                cf = cf.repeat_interleave(factor, dim=0)
            else:
                raise RuntimeError(
                    f"PaCE decomp batch mismatch: cf rows={cf.size(0)}, h_L B={B}."
                )

        alpha = payload["alpha"]
        zero_other = payload["zero_other"]

        base_rows: List[Tensor] = []
        int_rows: List[Tensor] = []
        dev = h_L.device
        D = self._concept_matrix.to(device=dev, dtype=torch.float32)  # (C, H)

        for b in range(B):
            if self.use_gpu_lstsq and torch.cuda.is_available() and h_L.is_cuda:
                target = h_anchor[b].detach().float().to(device=h_L.device)
                dict_use = [t.to(device=h_L.device, dtype=torch.float32) for t in self._dict_list]
                use_gpu = True
            else:
                target = h_anchor[b].detach().float().cpu()
                dict_use = self._dict_list
                use_gpu = False
            c = decompose_sparse(
                target=target,
                dictionary=dict_use,
                normalize=self.normalize_decomposition,
                use_gpu=use_gpu,
                return_timings=False,
            )
            if not isinstance(c, Tensor):
                c = torch.as_tensor(c, dtype=torch.float32)
            c = c.view(-1).float().detach().cpu()
            if c.numel() != self.C_cf:
                raise RuntimeError(
                    f"decompose_sparse returned length {c.numel()}; expected C_cf={self.C_cf}."
                )
            c_ctrl = self._apply_cf_to_coeffs(c, cf[b].cpu(), alpha=alpha, zero_other=zero_other)
            c = c.to(device=dev, dtype=torch.float32)
            c_ctrl = c_ctrl.to(device=dev, dtype=torch.float32)
            base_rows.append(c @ D)
            int_rows.append(c_ctrl @ D)

        self._cached_base = torch.stack(base_rows, dim=0)
        self._cached_intervened = torch.stack(int_rows, dim=0)

    def _align_cache_batch(self, h_L: Tensor) -> tuple[Tensor, Tensor]:
        B = h_L.size(0)
        base = self._cached_base
        inter = self._cached_intervened
        assert base is not None and inter is not None
        if base.size(0) != B:
            if not self._allow_batch_repeat:
                raise RuntimeError(
                    "PaCEDecompSteerer batch mismatch: cached base/intervened were computed for "
                    f"B_cache={base.size(0)} but hook received B={B}. "
                    "This usually means the generation driver expanded the batch (e.g. "
                    "repeat_interleave for num_return_sequences) without calling "
                    "expand_payload_for_n_samples on the steerer. "
                    "Refusing to silently repeat cached rows."
                )
            expected = base.size(0) * int(self._batch_repeat_factor)
            if expected == B:
                factor = int(self._batch_repeat_factor)
                base = base.repeat_interleave(factor, dim=0)
                inter = inter.repeat_interleave(factor, dim=0)
            else:
                raise RuntimeError(
                    "PaCEDecompSteerer batch mismatch under allowed repeat: "
                    f"B_cache={base.size(0)} repeat_factor={self._batch_repeat_factor} "
                    f"⇒ expected B={expected}, but got B={B}."
                )
        return base, inter

    def _hook_fn(self, module, args, output):
        h_L = output[0] if isinstance(output, tuple) else output
        if not self._should_intervene_now(h_L):
            return output
        payload = self._payload
        if payload is None:
            return output

        if not self.reuse_anchor_coeffs:
            self._cached_base = None
            self._cached_intervened = None
        if self._cached_base is None or self._cached_intervened is None:
            self._ensure_cache(h_L, payload)

        base, inter = self._align_cache_batch(h_L)
        base = base.to(device=h_L.device, dtype=torch.float32)
        inter = inter.to(device=h_L.device, dtype=torch.float32)
        # (B,T,H) - (B,1,H) + (B,1,H)
        z_ctrl = h_L.float() - base.unsqueeze(1) + inter.unsqueeze(1)
        z_ctrl = z_ctrl.to(dtype=h_L.dtype)
        if isinstance(output, tuple):
            return (z_ctrl,) + tuple(output[1:])
        return z_ctrl
