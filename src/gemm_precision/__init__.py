"""Deterministic GEMM/BMM precision and error-propagation observability toolkit."""

from .analysis import analyse_combination
from .config_io import load_config
from .errors import BusinessError
from .kernel import KernelResult, bmm, gemm
from .model import AccuracyTolerance, BlockShape, ExperimentConfig, SampleSpec
from .reference import reference_bmm, reference_gemm
from .runner import run_experiment
from .samples import SampleData, generate_sample, generate_samples

__all__ = [
    "AccuracyTolerance",
    "BlockShape",
    "BusinessError",
    "ExperimentConfig",
    "KernelResult",
    "SampleData",
    "SampleSpec",
    "analyse_combination",
    "bmm",
    "gemm",
    "generate_sample",
    "generate_samples",
    "load_config",
    "reference_bmm",
    "reference_gemm",
    "run_experiment",
]
