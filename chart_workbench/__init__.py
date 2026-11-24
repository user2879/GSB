"""Deterministic synthetic-chart calibration workbench."""

__all__ = ["Workbench", "load_config", "run_workbench", "generate_samples"]
__version__ = "1.0.0"


def __getattr__(name):
    if name in {"Workbench", "load_config", "run_workbench"}:
        from .engine import Workbench, load_config, run_workbench

        return {
            "Workbench": Workbench,
            "load_config": load_config,
            "run_workbench": run_workbench,
        }[name]
    if name == "generate_samples":
        from .generate import generate_samples

        return generate_samples
    raise AttributeError(name)
