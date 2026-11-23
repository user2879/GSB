"""JoinLab: deterministic join-order, costing and scheduler laboratory."""

from .errors import LabError
from .loader import load_workload

__all__ = ["ExperimentSession", "LabError", "load_workload"]
__version__ = "1.0.0"


def __getattr__(name: str):
    if name == "ExperimentSession":
        from .session import ExperimentSession

        return ExperimentSession
    raise AttributeError(name)
