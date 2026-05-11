"""Single hook contract used by every steerer.

Every steerer registers exactly one forward hook on the chosen Llama decoder
layer and rewrites that layer's hidden state.  Two phases:

  1. ``configure_for_batch(...)`` — caller stages a per-row payload (e.g.
     CF-tag multi-hot or per-row additive vectors).
  2. ``__enter__`` (via ``with steerer:``) — the hook is registered, the LM
     forward / generate fires, the hook reads the staged payload, mutates
     ``h_L``, returns it. ``__exit__`` removes the hook.

There is **only one** hook active per run; PaCE-CBM and the vector / transform
steerers are mutually exclusive (composition is intentionally out of scope).

KV cache / post-layer steering (HuggingFace ``generate``)
-----------------------------------------------------------
The hook replaces the **output** hidden state of layer ``L`` after the decoder
block runs.  Self-attention inside block ``L`` has already computed
``past_key_value`` for that forward; the substituted tensor is what layer ``L+1``
receives.  That matches common CAA-style hooks: an approximation of a different
residual stream, not a full K/V recomputation.  Cached K/V for **previous**
positions reflect whatever ran on those steps (with ``intervene_phase="decode_only"``,
prefill is unsteered).

Decode steps only pass ``h_L`` of shape ``(B, 1, H)`` into the steerer — only
the current slice is transformed inside the hook.

To disable incremental KV caching entirely (slower, sometimes used for
ablations), pass ``use_cache=False`` through
``_generate_with_steerer_batched`` into ``llm.generate``.

HuggingFace decoder output is a tuple
-------------------------------------
``LlamaDecoderLayer`` returns ``(hidden_states, self_attn_weights, past_key_value)``.
When that happens, the hook must return ``(z_ctrl,) + tuple(output[1:])``, not
a bare tensor, so weights and cache metadata are preserved.  All concrete
``_hook_fn`` implementations here follow that rule.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Llama layer locator (handles raw LlamaModel, LlamaForCausalLM, and PEFT).
# ─────────────────────────────────────────────────────────────────────────────

def get_llama_layers(model: nn.Module) -> tuple[nn.ModuleList, nn.Module]:
    """Return ``(layers, decoder_module)`` for an arbitrarily wrapped Llama.

    Mirrors ``modules._get_llama_model`` but returns the layers list directly.
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
    return base.layers, base


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

VALID_INTERVENE_PHASES = ("all", "decode_only")


class HookSteerer(ABC):
    """Abstract single-layer forward-hook steerer.

    Subclasses implement ``_hook_fn(module, args, output) -> output`` and
    ``configure_for_batch(...)`` (which stages per-row payload state).

    ``intervene_phase`` controls *when* the steering vector is applied during
    a ``model.generate(...)`` call:

      - ``"all"``         — modify ``h_L`` at every forward step. This is what
                            **training** wants: the single supervised forward
                            has no prefill/decode split, and we need the
                            steered prompt context to influence the loss on
                            assistant tokens (the gradient signal that trains
                            ``W_A`` / ``W_B`` / ``τ``).
      - ``"decode_only"`` — skip when ``h_L.size(1) > 1`` (i.e. the prefill
                            forward of ``generate``); apply only when
                            ``h_L.size(1) == 1`` (each newly generated token).
                            This is what **eval generation** wants: the prompt
                            is interpreted by the unmodified model, and only
                            the autoregressively generated tokens are steered.

    Note on first generated token under ``"decode_only"``: the very first
    sampled token is drawn from logits computed during prefill, so it is
    *not* steered. Every subsequent token is. This matches the user
    requirement "only during newly generated tokens".
    """

    method_name: str = "abstract"

    def __init__(self, *, layer_idx: int, intervene_phase: str = "all"):
        if intervene_phase not in VALID_INTERVENE_PHASES:
            raise ValueError(
                f"intervene_phase must be one of {VALID_INTERVENE_PHASES}; got {intervene_phase!r}."
            )
        self.layer_idx = int(layer_idx)
        self.intervene_phase = intervene_phase
        self._handle = None
        self._payload: Optional[dict] = None
        # Safety for generation-time batch expansion: by default we treat any
        # mismatch between payload batch size and hook batch size as an error.
        # Only allow repeating payload rows when the generation driver explicitly
        # expanded the batch (e.g. repeat_interleave for num_return_sequences)
        # and called the steerer payload expander.
        self._allow_batch_repeat: bool = False
        self._batch_repeat_factor: int = 1

    def _mark_batch_repeat_allowed(self, n_samples: int) -> None:
        """Allow batch repeating for expanded generation batches.

        Called by each concrete steerer in its ``_expand_payload_for_n_samples``.
        """
        if n_samples is None or int(n_samples) <= 1:
            return
        self._allow_batch_repeat = True
        self._batch_repeat_factor = int(n_samples)

    def _should_intervene_now(self, h_L: Tensor) -> bool:
        """Return False when the current forward should be passed through
        unmodified (prefill under ``decode_only``).
        """
        # h_L is (B, T, H); generate prefill has T>1, each decode step T==1.
        if self.intervene_phase == "decode_only" and h_L.size(1) > 1:
            return False
        return True

    # Context-manager protocol — the hook lives only inside the ``with`` block.
    def attach(self, model: nn.Module) -> "HookSteerer":
        self._target_model = model
        return self

    def __enter__(self) -> "HookSteerer":
        if not hasattr(self, "_target_model"):
            raise RuntimeError("HookSteerer: call .attach(model) before entering the context.")
        layers, decoder = get_llama_layers(self._target_model)
        target = decoder.norm if self.layer_idx == -1 else layers[self.layer_idx]
        self._handle = target.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *args) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    # Optional: clear any stale per-batch payload between runs.
    def clear_payload(self) -> None:
        self._payload = None
        self._allow_batch_repeat = False
        self._batch_repeat_factor = 1

    @abstractmethod
    def _hook_fn(self, module, args, output):
        ...

    @abstractmethod
    def configure_for_batch(self, *args, **kwargs) -> None:
        ...

    @staticmethod
    def expand_payload_for_n_samples(payload_tensor: Tensor, n_samples: int) -> Tensor:
        """Expand ``(B, …)`` payload to ``(B*n_samples, …)`` via repeat_interleave.

        Required when generation does ``num_return_sequences=n_samples`` via a
        manual ``repeat_interleave`` of input_ids (the eval driver does this).
        """
        if n_samples <= 1:
            return payload_tensor
        return payload_tensor.repeat_interleave(n_samples, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# NoSteer — identity hook (registered to keep the with-block uniform)
# ─────────────────────────────────────────────────────────────────────────────

class NoSteer(HookSteerer):
    method_name = "none"

    def _hook_fn(self, module, args, output):
        return output

    def configure_for_batch(self, *args, **kwargs) -> None:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PaCECBMSteerer — wraps PaCECBM, hook applies dictionary intervention
# ─────────────────────────────────────────────────────────────────────────────

class PaCECBMSteerer(HookSteerer):
    method_name = "pace_cbm"

    def __init__(
        self,
        pace_cbm,
        *,
        layer_idx: Optional[int] = None,
        intervene_phase: str = "all",
    ):
        super().__init__(
            layer_idx=layer_idx if layer_idx is not None else pace_cbm.layer_idx,
            intervene_phase=intervene_phase,
        )
        self.pace_cbm = pace_cbm
        # Last computed concept tensor, exposed for losses (kept on autograd graph).
        self.last_c_sparse: Optional[Tensor] = None
        self.last_h_L: Optional[Tensor] = None

    def configure_for_batch(
        self,
        *,
        cf_multihot: Optional[Tensor] = None,
        intervene_value: float = 0.0,
        zero_other_concepts: bool = False,
    ) -> None:
        """Stage the per-batch intervention.

        - ``cf_multihot``: ``(B, C_cf)`` float / bool.  When provided, builds
          a row mask over the CF-tag block of the dictionary
          ``[cf_offset : cf_offset + cf_size]``.  Pass ``None`` (or the all-zero
          tensor) for a no-intervention forward; by the residual identity
          ``z_ctrl ≡ z_in`` in that case, while ``last_c_sparse`` is still
          stashed for concept-loss / concept-accuracy computation.
        - ``intervene_value``: scalar to write into selected concept positions.
        - ``zero_other_concepts``: if True, zero non-selected concepts too.
        """
        if cf_multihot is None or cf_multihot.numel() == 0:
            self._payload = None
            self._allow_batch_repeat = False
            self._batch_repeat_factor = 1
            return
        if cf_multihot.dim() != 2 or cf_multihot.size(1) != self.pace_cbm.cf_size:
            raise ValueError(
                f"cf_multihot must be (B, C_cf={self.pace_cbm.cf_size}); "
                f"got {tuple(cf_multihot.shape)}."
            )
        B = cf_multihot.size(0)  # B = batch rows
        # row_mask: (B, C_total); True only on CF columns selected by multihot.
        row_mask = torch.zeros(
            B, self.pace_cbm.C, dtype=torch.bool, device=cf_multihot.device,
        )
        row_mask[:, self.pace_cbm.cf_offset:self.pace_cbm.cf_offset + self.pace_cbm.cf_size] = (
            cf_multihot > 0
        )
        self._payload = {
            "row_mask": row_mask,
            "value": float(intervene_value),
            "zero_other": bool(zero_other_concepts),
        }

    def _expand_payload_for_n_samples(self, n_samples: int) -> None:
        """Repeat the staged row mask along the batch dim."""
        if self._payload is None or n_samples <= 1:
            return
        self._mark_batch_repeat_allowed(n_samples)
        rm = self._payload["row_mask"]
        self._payload["row_mask"] = rm.repeat_interleave(n_samples, dim=0)

    def _hook_fn(self, module, args, output):
        # h_L: (B, T, H) — T = full seq in one LM forward (generate prefill or train).
        h_L = output[0] if isinstance(output, tuple) else output

        # ``decode_only``: pass the prefill forward through unmodified so the
        # prompt's K/V cache at layers L+1..N is built from the *clean* h_L.
        # The hook still re-fires on every decode step (T==1) where it does
        # apply the intervention.
        if not self._should_intervene_now(h_L):
            return output

        payload = self._payload
        if payload is None:
            row_mask = None
            value = 0.0
            zero_other = False
        else:
            row_mask = payload["row_mask"]
            B_payload = row_mask.size(0)
            B_hL = h_L.size(0)
            if B_payload != B_hL:
                if not self._allow_batch_repeat:
                    raise RuntimeError(
                        "PaCECBMSteerer batch mismatch: payload row_mask was staged for "
                        f"B_payload={B_payload} but hook received B={B_hL}. "
                        "This usually means generation expanded the batch (e.g. "
                        "repeat_interleave for num_return_sequences) without calling "
                        "expand_payload_for_n_samples on the steerer. Refusing to "
                        "silently repeat the payload."
                    )
                expected = B_payload * int(self._batch_repeat_factor)
                if expected != B_hL:
                    raise RuntimeError(
                        "PaCECBMSteerer batch mismatch under allowed repeat: "
                        f"B_payload={B_payload} repeat_factor={self._batch_repeat_factor} "
                        f"⇒ expected B={expected}, but got B={B_hL}."
                    )
                row_mask = row_mask.repeat_interleave(int(self._batch_repeat_factor), dim=0)
            row_mask = row_mask.to(h_L.device)
            value = payload["value"]
            zero_other = payload["zero_other"]

        # z_ctrl, c_sparse: same (B, T, H) and (B, T, C_total) as h_L inputs.
        z_ctrl, c_sparse = self.pace_cbm.apply(
            h_L,
            intervene_row_mask=row_mask,  # (B, C_total) or None
            intervene_value=value,
            zero_other_concepts=zero_other,
        )
        self.last_c_sparse = c_sparse  # (B, T, C_total)
        self.last_h_L = h_L  # (B, T, H)
        z_ctrl = z_ctrl.to(h_L.dtype)
        if isinstance(output, tuple):
            return (z_ctrl,) + tuple(output[1:])
        return z_ctrl


# ─────────────────────────────────────────────────────────────────────────────
# VecAddSteerer — additive steering (CAA / ITI / RepE).
# ─────────────────────────────────────────────────────────────────────────────

class VecAddSteerer(HookSteerer):
    """Adds ``alpha * Σ_t steer_vecs[t]`` to ``h_L`` for each row.

    ``steer_vecs`` is a ``(C_cf, H)`` tensor (one row per CF tag, in concept-set
    order).  ``configure_for_batch(cf_multihot, alpha)`` precomputes the per-row
    sum so the hook only does an addition.
    """

    def __init__(
        self,
        steer_vecs: Tensor,
        *,
        layer_idx: int,
        method_name: str = "vec_add",
        intervene_phase: str = "all",
    ):
        super().__init__(layer_idx=layer_idx, intervene_phase=intervene_phase)
        if steer_vecs.dim() != 2:
            raise ValueError(f"steer_vecs must be (C_cf, H); got {tuple(steer_vecs.shape)}.")
        self.method_name = method_name
        # Stored as a buffer-style attribute on CPU; moved to h_L's device at hook time.
        self.steer_vecs = steer_vecs.float().contiguous()
        self.C_cf = int(steer_vecs.size(0))
        self.H = int(steer_vecs.size(1))

    def configure_for_batch(
        self,
        *,
        cf_multihot: Optional[Tensor] = None,
        alpha: float = 1.0,
    ) -> None:
        if cf_multihot is None or cf_multihot.numel() == 0:
            self._payload = None
            self._allow_batch_repeat = False
            self._batch_repeat_factor = 1
            return
        if cf_multihot.dim() != 2 or cf_multihot.size(1) != self.C_cf:
            raise ValueError(
                f"cf_multihot must be (B, C_cf={self.C_cf}); got {tuple(cf_multihot.shape)}."
            )
        # cf_multihot: (B, C_cf) @ steer_vecs (C_cf, H) -> add_vec: (B, H)
        add_vec = (cf_multihot.float().to(self.steer_vecs.device) @ self.steer_vecs) * float(alpha)
        self._payload = {"add_vec": add_vec}

    def _expand_payload_for_n_samples(self, n_samples: int) -> None:
        if self._payload is None or n_samples <= 1:
            return
        self._mark_batch_repeat_allowed(n_samples)
        av = self._payload["add_vec"]
        self._payload["add_vec"] = av.repeat_interleave(n_samples, dim=0)

    def _hook_fn(self, module, args, output):
        h_L = output[0] if isinstance(output, tuple) else output  # (B, T, H)
        if not self._should_intervene_now(h_L):
            return output
        payload = self._payload
        if payload is None:
            return output
        add_vec = payload["add_vec"]  # (B, H)
        if add_vec.size(0) != h_L.size(0):
            if not self._allow_batch_repeat:
                raise RuntimeError(
                    "VecAddSteerer batch mismatch: add_vec was staged for "
                    f"B_payload={add_vec.size(0)} but hook received B={h_L.size(0)}. "
                    "This usually means generation expanded the batch (e.g. "
                    "repeat_interleave for num_return_sequences) without calling "
                    "expand_payload_for_n_samples on the steerer. Refusing to "
                    "silently repeat the payload."
                )
            expected = add_vec.size(0) * int(self._batch_repeat_factor)
            if expected != h_L.size(0):
                raise RuntimeError(
                    "VecAddSteerer batch mismatch under allowed repeat: "
                    f"B_payload={add_vec.size(0)} repeat_factor={self._batch_repeat_factor} "
                    f"⇒ expected B={expected}, but got B={h_L.size(0)}."
                )
            add_vec = add_vec.repeat_interleave(int(self._batch_repeat_factor), dim=0)
        add_vec_b = add_vec.to(device=h_L.device, dtype=h_L.dtype).unsqueeze(1)  # (B, 1, H)
        z_ctrl = h_L + add_vec_b  # (B, T, H) + broadcast (B, 1, H) -> (B, T, H)
        if isinstance(output, tuple):
            return (z_ctrl,) + tuple(output[1:])
        return z_ctrl


# ─────────────────────────────────────────────────────────────────────────────
# TransformSteerer — wraps a per-tag ``Steer`` instance (LinAcT / MiMiC).
# ─────────────────────────────────────────────────────────────────────────────

class TransformSteerer(HookSteerer):
    """Applies a per-tag affine map ``Steer.steer(X)`` at the hook.

    Holds a list of ``Steer`` objects (one per CF tag).  Per-batch, the caller
    selects a single tag index per row (e.g. the first ground-truth tag);
    rows whose ``selected_idx`` is ``-1`` are passed through unchanged.

    LinAcT/MiMiC don't compose linearly across multiple per-tag affine maps,
    so we deliberately apply a single tag per row instead of an arbitrary mix.
    """

    def __init__(
        self,
        steerers: Sequence,
        *,
        layer_idx: int,
        method_name: str = "transform",
        alpha: float = 1.0,
        intervene_phase: str = "all",
    ):
        super().__init__(layer_idx=layer_idx, intervene_phase=intervene_phase)
        self.steerers = list(steerers)
        self.method_name = method_name
        self.alpha = float(alpha)
        self.C_cf = len(self.steerers)

    def configure_for_batch(
        self,
        *,
        cf_multihot: Optional[Tensor] = None,
        per_row_idx: Optional[List[int]] = None,
    ) -> None:
        """Either pass ``per_row_idx`` directly (List[int] of length B,
        ``-1`` = passthrough) or pass ``cf_multihot`` and we will pick the
        first nonzero index per row.
        """
        if per_row_idx is not None:
            self._payload = {"per_row_idx": list(per_row_idx)}
            self._allow_batch_repeat = False
            self._batch_repeat_factor = 1
            return
        if cf_multihot is None or cf_multihot.numel() == 0:
            self._payload = None
            self._allow_batch_repeat = False
            self._batch_repeat_factor = 1
            return
        idx_list: List[int] = []
        for row in cf_multihot:
            nz = (row > 0).nonzero(as_tuple=False).flatten()
            idx_list.append(int(nz[0].item()) if nz.numel() > 0 else -1)
        self._payload = {"per_row_idx": idx_list}

    def _expand_payload_for_n_samples(self, n_samples: int) -> None:
        if self._payload is None or n_samples <= 1:
            return
        self._mark_batch_repeat_allowed(n_samples)
        idx = self._payload["per_row_idx"]
        expanded: List[int] = []
        for i in idx:
            expanded.extend([i] * n_samples)
        self._payload["per_row_idx"] = expanded

    def _hook_fn(self, module, args, output):
        h_L = output[0] if isinstance(output, tuple) else output  # (B, T, H)
        if not self._should_intervene_now(h_L):
            return output
        payload = self._payload
        if payload is None:
            return output
        per_row_idx = payload["per_row_idx"]  # length B (after expansion matches B_hL)
        if len(per_row_idx) != h_L.size(0):
            if not self._allow_batch_repeat:
                raise RuntimeError(
                    "TransformSteerer batch mismatch: per_row_idx was staged for "
                    f"B_payload={len(per_row_idx)} but hook received B={h_L.size(0)}. "
                    "This usually means generation expanded the batch (e.g. "
                    "repeat_interleave for num_return_sequences) without calling "
                    "expand_payload_for_n_samples on the steerer. Refusing to "
                    "silently repeat the payload."
                )
            expected = len(per_row_idx) * int(self._batch_repeat_factor)
            if expected != h_L.size(0):
                raise RuntimeError(
                    "TransformSteerer batch mismatch under allowed repeat: "
                    f"B_payload={len(per_row_idx)} repeat_factor={self._batch_repeat_factor} "
                    f"⇒ expected B={expected}, but got B={h_L.size(0)}."
                )
            per_row_idx = [i for i in per_row_idx for _ in range(int(self._batch_repeat_factor))]

        original_dtype = h_L.dtype
        z_ctrl = h_L.clone()  # (B, T, H)
        for b, ti in enumerate(per_row_idx):
            if ti < 0 or ti >= self.C_cf:
                continue
            steer_obj = self.steerers[ti]
            row = h_L[b].float()  # (T, H) single sequence in batch
            transformed = steer_obj.steer(row, T=self.alpha).to(dtype=original_dtype)  # (T, H)
            z_ctrl[b] = transformed  # (T, H) back into row b
        if isinstance(output, tuple):
            return (z_ctrl,) + tuple(output[1:])
        return z_ctrl
