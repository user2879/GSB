from __future__ import annotations

APP_NAME = "deterministic-instruction-factory"
SCHEMA_VERSION = 1

DIMENSION_ORDER = ("clarity", "correctness", "format", "length")
STAGES = ("sampling", "scoring", "dedupe", "aggregate")
HARD_MAX_CONCURRENCY = 16
MIN_DECIMAL_PLACES = 0
MAX_DECIMAL_PLACES = 6
ALLOWED_QUANTIZATION_STEPS = (
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
)

E_SEED_INVALID = "E_SEED_INVALID"
E_WEIGHT_INVALID = "E_WEIGHT_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_TASK_BUDGET_EXHAUSTED = "E_TASK_BUDGET_EXHAUSTED"
E_SCORE_BUDGET_EXHAUSTED = "E_SCORE_BUDGET_EXHAUSTED"
E_GENERATION_BUDGET_EXHAUSTED = "E_GENERATION_BUDGET_EXHAUSTED"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"

DEFAULT_INPUT_FILES = {
    "seeds": "seeds.jsonl",
    "lexicon": "lexicon.json",
    "scoring": "scoring.json",
    "config": "config.json",
    "plan": "plan.json",
}

SAFE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"

