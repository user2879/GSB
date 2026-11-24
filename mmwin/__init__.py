"""Deterministic, local multimodal event-window aggregation."""

from .config import WindowConfig, load_config, parse_config
from .engine import AggregationEngine, canonical_business_projection, run_engine
from .events import load_events
from .plan import InjectionPlan, load_plan, parse_plan

__all__ = [
    "AggregationEngine",
    "InjectionPlan",
    "WindowConfig",
    "canonical_business_projection",
    "load_config",
    "load_events",
    "load_plan",
    "parse_config",
    "parse_plan",
    "run_engine",
]

__version__ = "1.0.0"
