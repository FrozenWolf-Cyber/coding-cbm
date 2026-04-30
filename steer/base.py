"""
Base steering classes: abstract Steer and VecSteer (additive steering vector).
Adapted from ODESteer's _base_steer.py.
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Steer(ABC):
    @abstractmethod
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def vector_field(self, X: Tensor) -> Tensor:
        raise NotImplementedError


class VecSteer(Steer):
    def __init__(self):
        self.steer_vec = None

    @abstractmethod
    def fit(self, pos_X: Tensor, neg_X: Tensor) -> "VecSteer":
        raise NotImplementedError

    @torch.no_grad()
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        return X + T * self.steer_vec.to(X.device)

    def vector_field(self, X: Tensor) -> Tensor:
        return self.steer_vec.broadcast_to(X.shape)
