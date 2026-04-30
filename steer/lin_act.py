"""
LinAcT — Linear Activation Transport (ICLR 2024).
Per-dimension affine map from neg → pos cloud via sorted quantile matching.
"""

import torch
from torch import Tensor, nn
from .base import Steer


class LinOT(nn.Module):
    def __init__(self):
        super().__init__()
        self.fitted = False

    def fit(self, X_target: Tensor, X_source: Tensor):
        n = min(len(X_target), len(X_source))
        X_target, X_source = X_target[:n], X_source[:n]
        m_target = X_target.mean(dim=0)
        m_source = X_source.mean(dim=0)
        X_target_c = (X_target - m_target).sort(dim=0).values
        X_source_c = (X_source - m_source).sort(dim=0).values
        w_num = (X_target_c * X_source_c).sum(dim=0)
        w_den = (X_source_c ** 2).sum(dim=0)
        self.register_buffer("w", w_num / (w_den + 1e-10))
        self.register_buffer("b", m_target - self.w * m_source)
        self.fitted = True
        return self

    def forward(self, X: Tensor) -> Tensor:
        return X * self.w + self.b


class LinAcT(Steer):
    def __init__(self):
        self.lin_ot = LinOT()

    def fit(self, pos_X: Tensor, neg_X: Tensor):
        self.lin_ot.fit(pos_X, neg_X)
        return self

    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        self.lin_ot.to(X.device)
        return self.lin_ot(X)

    def vector_field(self, X: Tensor) -> Tensor:
        self.lin_ot.to(X.device)
        return self.lin_ot(X) - X
