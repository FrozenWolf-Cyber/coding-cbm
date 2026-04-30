"""
CAA — Contrastive Activation Addition (ACL 2024).
steer_vec = mean(pos) − mean(neg)
"""

import torch
from torch import Tensor
from .base import VecSteer


class CAA(VecSteer):
    @torch.no_grad()
    def fit(self, pos_X: Tensor, neg_X: Tensor) -> "CAA":
        self.steer_vec = pos_X.mean(dim=0) - neg_X.mean(dim=0)
        return self
