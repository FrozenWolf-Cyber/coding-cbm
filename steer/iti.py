"""
ITI — Inference-Time Intervention (NeurIPS 2023).
Logistic probe on pos/neg activations; normalised coefficient as steer vector.
"""

import torch
from torch import Tensor
from sklearn.linear_model import LogisticRegression
from .base import VecSteer


class ITI(VecSteer):
    def __init__(self):
        super().__init__()
        self.clf = LogisticRegression(max_iter=1000)

    @torch.no_grad()
    def fit(self, pos_X: Tensor, neg_X: Tensor) -> "ITI":
        Xs = torch.cat([pos_X, neg_X], dim=0)
        labels = torch.cat([torch.ones(len(pos_X)), torch.zeros(len(neg_X))], dim=0)
        self.clf.fit(Xs.cpu().numpy(), labels.cpu().numpy())
        self.steer_vec = torch.as_tensor(
            self.clf.coef_.ravel(), device=pos_X.device, dtype=pos_X.dtype,
        )
        self.steer_vec = self.steer_vec / self.steer_vec.norm()
        return self
