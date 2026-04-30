"""Single-step ODE steering (Euler-style). Vendored from ODESteer."""

from abc import abstractmethod

from torch import Tensor

from .base import Steer
from .kernels import KernelClassifier, NormedPolyClassifier, RFFClassifier


class BaseStepODESteer(Steer):
    def __init__(self, **kwargs):
        super().__init__()
        self.clf = self._init_clf(**kwargs)

    def fit(self, pos_X: Tensor, neg_X_or_labels: Tensor) -> "BaseStepODESteer":
        self.clf.fit(pos_X, neg_X_or_labels)
        return self

    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        vf = self.vector_field(X)
        return X + T * vf

    def vector_field(self, X: Tensor) -> Tensor:
        self.clf.to(X.device)
        raw_grad = self.clf.grad(X)
        return raw_grad / (raw_grad.norm(dim=-1, keepdim=True) + 1e-10)

    @abstractmethod
    def _init_clf(self, **kwargs) -> KernelClassifier:
        raise NotImplementedError


class StepODESteer(BaseStepODESteer):
    """One-step ODESteer with NormedPolyCntSketch classifier."""

    def _init_clf(self, **kwargs) -> NormedPolyClassifier:
        return NormedPolyClassifier(**kwargs)


class RFFStepODESteer(BaseStepODESteer):
    """One-step ODESteer with RFF classifier (ablation)."""

    def _init_clf(self, **kwargs) -> RFFClassifier:
        return RFFClassifier(**kwargs)
