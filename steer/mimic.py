"""
MiMiC — Representation Surgery: affine steering (ICML 2024).
Logistic classifier + linear optimal transport.
"""

import torch
from torch import Tensor
from sklearn.linear_model import LogisticRegression
import ot

from .base import Steer


class MiMiC(Steer):
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000)
        self.ot_linear = ot.da.LinearTransport(reg=1e-2)

    def fit(self, pos_X: Tensor, neg_X: Tensor):
        train_X = torch.cat([pos_X, neg_X], dim=0).cpu().numpy()
        train_Y = torch.cat([
            torch.ones(len(pos_X)), torch.zeros(len(neg_X)),
        ], dim=0).numpy()
        self.clf.fit(train_X, train_Y)
        self.ot_linear.fit(Xs=neg_X.cpu().numpy(), Xt=pos_X.cpu().numpy())
        return self

    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        y_pred = self.clf.predict(X.detach().cpu().numpy())
        steered = X.detach().cpu().numpy().copy()
        steered[y_pred == 0] = self.ot_linear.transform(steered[y_pred == 0])
        return torch.as_tensor(steered, device=X.device, dtype=X.dtype)

    def vector_field(self, X: Tensor) -> Tensor:
        return torch.zeros_like(X)
