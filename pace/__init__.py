"""PaCE-CBM trainer + steerable evaluation framework.

Layout
------
- ``data``           : code_contests data loaders parameterised on tokenizer.
- ``activations``    : layer-L hidden state probing (forward-hook based).
- ``dictionary``     : builds frozen ``D ∈ R^(H × C)`` (CF + PaCE columns).
- ``pace_cbm``       : ``PaCECBM`` module — Arch 1 (W_A, W_B, τ over frozen D).
- ``hook_steerer``   : single hook contract for PaCE-CBM, vector and transform steerers.
- ``intervention``   : CF-tag → per-row payload routing.
- ``loops``          : pure-function train / validate epochs.
- ``eval_steerable`` : steerer-agnostic evaluation cascade (reuses ``eval_metrics``).

Existing CBM code (``train_combined_finegrained.py``, ``modules.py``,
``eval_metrics.py``, ``utils.py``, ``config.py``, ``steer/*``) is read-only
input — every entry point in this package is additive.
"""

from .pace_cbm import PaCECBM
from .hook_steerer import (
    HookSteerer,
    NoSteer,
    PaCECBMSteerer,
    VecAddSteerer,
    TransformSteerer,
    get_llama_layers,
)

__all__ = [
    "PaCECBM",
    "HookSteerer",
    "NoSteer",
    "PaCECBMSteerer",
    "VecAddSteerer",
    "TransformSteerer",
    "get_llama_layers",
]
