from __future__ import annotations

from pathlib import Path

from rope_extrap.loader import load_position_config, load_samples


ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "samples"
CONFIG_LARGE = SAMPLES_DIR / "config_large_budget.json"
CONFIG_SMALL = SAMPLES_DIR / "config_small_budget.json"
SAMPLES_JSONL = SAMPLES_DIR / "samples.jsonl"
COMMANDS_JSON = SAMPLES_DIR / "commands.json"


def load_config_and_samples(path: str | Path = CONFIG_LARGE, count: int | None = None):
    config, config_failures = load_position_config(path)
    assert config is not None, config_failures
    samples, sample_failures = load_samples(
        SAMPLES_JSONL,
        dimension=config.dimension,
        target_length=config.extrapolation_target_length,
    )
    if count is not None:
        samples = dict(list(samples.items())[:count])
    return config, samples, sample_failures

