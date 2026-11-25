from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VERDICT_CODES = {
    "reject": -1,
    "borderline": 0,
    "accept": 1,
}
VERDICT_NAMES = {-1: "reject", 0: "borderline", 1: "accept"}
DECLARED_VERDICT_NAMES = frozenset(VERDICT_CODES)
DECLARED_QUANT_SCHEMES = frozenset({"Q_COARSE", "Q_BALANCED", "Q_FINE"})
QUANT_STEP_UNITS = {"Q_COARSE": 5, "Q_BALANCED": 2, "Q_FINE": 1}
ALLOWED_DECIMAL_PLACES = frozenset({2, 4, 6})
ALLOWED_SUBWORD_RULES = frozenset({"sw-v1", "sw-v2"})
ALLOWED_EXCHANGE_VERSIONS = frozenset({"g1", "g2"})
ALLOWED_LEDGER_GENERATIONS = frozenset({"L1", "L2"})


@dataclass(frozen=True)
class Issue:
    code: str
    stage: str
    target: str
    message: str
    injection_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreferenceRecord:
    session_id: str
    turn: int
    prompt: str
    chosen: str
    rejected: str
    annotator_id: str
    event_ms: int

    @property
    def key(self) -> tuple[str, int]:
        return self.session_id, self.turn

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Config:
    feature_dim: int
    hash_bucket_count: int
    subword_rule_version: str
    normalize: bool
    controlled_vocabulary: tuple[str, ...]
    window_lengths: tuple[int, ...]
    partition_boundaries: tuple[int, ...]
    fixed_decimal_places: int
    quant_scheme: str
    exchange_version: str = "g2"
    ledger_generation: str = "L2"
    decision_threshold_permille: int = 0
    score_weights: tuple[int, ...] = field(default_factory=tuple)
    window_budget: int = 10_000
    session_quota: int = 10_000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoadedInput:
    config: Config | None
    records: list[PreferenceRecord]
    weights: dict[str, int]
    issues: list[Issue]

    @property
    def ok(self) -> bool:
        return self.config is not None and not self.issues
