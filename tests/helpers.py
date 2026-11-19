from __future__ import annotations

from pathlib import Path

from tactical_path.config import AppConfig, load_config
from tactical_path.mapgen import generate_bundle
from tactical_path.models import MapBundle


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "demo.yaml"


def config() -> AppConfig:
    return load_config(CONFIG_PATH)


def bundle(tier: str) -> MapBundle:
    cfg = config()
    return generate_bundle(
        cfg.tiers[tier],
        seed=cfg.seed,
        mobility_rules=cfg.mobility,
    )

