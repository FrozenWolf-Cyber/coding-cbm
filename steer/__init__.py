from .base import Steer, VecSteer
from .caa import CAA
from .iti import ITI
from .repe import RepE
from .lin_act import LinAcT
from .mimic import MiMiC
from .ode_steer import BaseODESteer, ODESteer, RFFODESteer
from .pace import PaCESteerer
from .step_ode_steer import BaseStepODESteer, RFFStepODESteer, StepODESteer

_REGISTRY = {
    "CAA": CAA,
    "ITI": ITI,
    "RepE": RepE,
    "LinAcT": LinAcT,
    "MiMiC": MiMiC,
    "ODESteer": ODESteer,
    "RFFODESteer": RFFODESteer,
    "StepODESteer": StepODESteer,
    "RFFStepODESteer": RFFStepODESteer,
}


def get_steer_model(name: str, **kwargs):
    if name == "NoSteer" or name is None:
        return None
    if name == "PaCE":
        raise ValueError("PaCE requires special initialization — use PaCESteerer directly.")
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown steer method: {name}. Available: {sorted(_REGISTRY)}")
    return cls(**kwargs)


__all__ = [
    "Steer",
    "VecSteer",
    "CAA",
    "ITI",
    "RepE",
    "LinAcT",
    "MiMiC",
    "PaCESteerer",
    "get_steer_model",
    "BaseODESteer",
    "ODESteer",
    "RFFODESteer",
    "BaseStepODESteer",
    "StepODESteer",
    "RFFStepODESteer",
]
