from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .errors import AppError, E_DECODE_PARAM_INVALID, E_INPUT_INVALID


@dataclass(frozen=True)
class VocabEntry:
    token: str
    pos: str
    slot_start: bool
    slot: Optional[str]
    subwords: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "pos": self.pos,
            "slot_start": self.slot_start,
            "slot": self.slot,
            "subwords": list(self.subwords),
        }


@dataclass
class Vocabulary:
    version: str
    entries: Tuple[VocabEntry, ...]
    token_to_id: Dict[str, int]
    bigrams: Dict[Tuple[str, str], float]
    start_token: str
    smoothing_token: str
    smoothing_token_id: int
    subword_rules: Dict[str, Any]
    source_fingerprint: str = ""

    def __len__(self) -> int:
        return len(self.entries)

    def token(self, token_id: int) -> str:
        return self.entries[token_id].token

    def entry(self, token_id: int) -> VocabEntry:
        return self.entries[token_id]

    def bigram_score(self, previous: str, token: str, backoff: float) -> float:
        return self.bigrams.get((previous, token), backoff)

    def render(self, token_ids: Sequence[int]) -> str:
        tokens = [self.entries[i].token for i in token_ids]
        separator = str(self.subword_rules.get("token_separator", ""))
        return separator.join(tokens)

    def subwords(self, token_ids: Sequence[int]) -> List[str]:
        result: List[str] = []
        for token_id in token_ids:
            entry = self.entries[token_id]
            result.extend(entry.subwords or (entry.token,))
        return result

    def subword_text(self, token_ids: Sequence[int]) -> str:
        joiner = str(self.subword_rules.get("subword_separator", ""))
        return joiner.join(self.subwords(token_ids))


@dataclass(frozen=True)
class ShardRecord:
    session_id: str
    shard_seq: int
    start_frame: int
    end_frame: int
    arrived: bool = True
    revision: int = 0

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def global_frames(self) -> Tuple[int, ...]:
        return tuple(range(self.start_frame, self.end_frame + 1))

    def sort_key(self) -> Tuple[str, int, int, int]:
        return (self.session_id, self.shard_seq, self.start_frame, self.revision)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "shard_seq": self.shard_seq,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "arrived": self.arrived,
            "revision": self.revision,
        }


@dataclass
class SessionDataset:
    session_id: str
    shards: Dict[int, ShardRecord] = field(default_factory=dict)
    scores: Dict[int, Dict[int, Tuple[float, ...]]] = field(default_factory=dict)
    # The ingest order is retained only for validation diagnostics.  Canonical
    # output ordering always uses (session_id, shard_seq, global frame).
    ingest_order: List[int] = field(default_factory=list)
    input_valid: bool = True
    input_failures: List[Dict[str, Any]] = field(default_factory=list)

    def ordered_shards(self, include_not_arrived: bool = False) -> List[ShardRecord]:
        values = [
            self.shards[seq]
            for seq in sorted(self.shards)
            if include_not_arrived or self.shards[seq].arrived
        ]
        return values

    def clone(self) -> "SessionDataset":
        return SessionDataset(
            session_id=self.session_id,
            shards=dict(self.shards),
            scores={seq: dict(rows) for seq, rows in self.scores.items()},
            ingest_order=list(self.ingest_order),
            input_valid=self.input_valid,
            input_failures=[dict(item) for item in self.input_failures],
        )


@dataclass
class Dataset:
    vocab: Vocabulary
    sessions: Dict[str, SessionDataset]
    score_source: str = ""

    def sorted_session_ids(self) -> List[str]:
        return sorted(self.sessions)

    def clone(self) -> "Dataset":
        return Dataset(
            vocab=self.vocab,
            sessions={sid: session.clone() for sid, session in self.sessions.items()},
            score_source=self.score_source,
        )


@dataclass(frozen=True)
class DecodeConfig:
    overlap_frames: int
    min_stable_frames: int
    lm_weight: float
    slot_min_frames: int
    slot_max_frames: int
    response_budget_ms: int
    thread_pool_concurrency: int
    snapshot_interval_shards: int
    strategy: str = "lookahead"
    lookahead_frames: int = 3
    lm_backoff_score: float = -2.5
    smoothing_score: float = -12.0
    candidate_missing_below: float = -100.0
    stable_margin: float = 0.05
    snapshot_root: str = "artifacts/snapshots"
    report_path: str = "artifacts/diagnostic_report.json"
    test_compute_delay_ms: float = 0.0

    REQUIRED_KEYS = (
        "overlap_frames",
        "min_stable_frames",
        "lm_weight",
        "slot_min_frames",
        "slot_max_frames",
        "response_budget_ms",
        "thread_pool_concurrency",
        "snapshot_interval_shards",
    )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DecodeConfig":
        missing = [key for key in cls.REQUIRED_KEYS if key not in raw]
        if missing:
            raise AppError(
                E_INPUT_INVALID,
                "missing required decode config keys: " + ", ".join(missing),
                stage="config_load",
            )
        try:
            config = cls(
                overlap_frames=int(raw["overlap_frames"]),
                min_stable_frames=int(raw["min_stable_frames"]),
                lm_weight=float(raw["lm_weight"]),
                slot_min_frames=int(raw["slot_min_frames"]),
                slot_max_frames=int(raw["slot_max_frames"]),
                response_budget_ms=int(raw["response_budget_ms"]),
                thread_pool_concurrency=int(raw["thread_pool_concurrency"]),
                snapshot_interval_shards=int(raw["snapshot_interval_shards"]),
                strategy=str(raw.get("strategy", "lookahead")).lower(),
                lookahead_frames=int(raw.get("lookahead_frames", 3)),
                lm_backoff_score=float(raw.get("lm_backoff_score", -2.5)),
                smoothing_score=float(raw.get("smoothing_score", -12.0)),
                candidate_missing_below=float(raw.get("candidate_missing_below", -100.0)),
                stable_margin=float(raw.get("stable_margin", 0.05)),
                snapshot_root=str(raw.get("snapshot_root", "artifacts/snapshots")),
                report_path=str(raw.get("report_path", "artifacts/diagnostic_report.json")),
                test_compute_delay_ms=float(raw.get("test_compute_delay_ms", 0.0)),
            )
        except (TypeError, ValueError) as exc:
            raise AppError(
                E_INPUT_INVALID,
                f"decode config contains a non-numeric value: {exc}",
                stage="config_load",
            ) from exc
        config.validate_global()
        return config

    def validate_global(self) -> None:
        if self.overlap_frames < 0:
            raise AppError(E_INPUT_INVALID, "overlap_frames must be >= 0", stage="config_validation")
        if self.min_stable_frames <= 0:
            raise AppError(E_INPUT_INVALID, "min_stable_frames must be > 0", stage="config_validation")
        if self.slot_min_frames <= 0 or self.slot_max_frames <= 0:
            raise AppError(
                E_INPUT_INVALID,
                "slot frame bounds must both be positive",
                stage="config_validation",
            )
        if self.slot_min_frames > self.slot_max_frames:
            raise AppError(
                E_INPUT_INVALID,
                "slot_min_frames must not exceed slot_max_frames",
                stage="config_validation",
            )
        if self.thread_pool_concurrency <= 0:
            raise AppError(
                E_INPUT_INVALID,
                "thread_pool_concurrency must be > 0",
                stage="config_validation",
            )
        if self.snapshot_interval_shards <= 0:
            raise AppError(
                E_INPUT_INVALID,
                "snapshot_interval_shards must be > 0",
                stage="config_validation",
            )
        if self.response_budget_ms < 0:
            raise AppError(E_INPUT_INVALID, "response_budget_ms must be >= 0", stage="config_validation")
        if self.strategy not in ("greedy", "lookahead"):
            raise AppError(
                E_DECODE_PARAM_INVALID,
                "strategy must be 'greedy' or 'lookahead'",
                stage="decode_validation",
            )
        if self.strategy == "lookahead" and self.lookahead_frames <= 0:
            raise AppError(
                E_DECODE_PARAM_INVALID,
                "lookahead strategy requires lookahead_frames > 0",
                stage="decode_validation",
            )
        if self.lookahead_frames < 0:
            raise AppError(E_DECODE_PARAM_INVALID, "lookahead_frames must be >= 0", stage="decode_validation")
        if not np.isfinite(self.lm_weight):
            raise AppError(E_INPUT_INVALID, "lm_weight must be finite", stage="config_validation")
        if not np.isfinite(self.lm_backoff_score) or not np.isfinite(self.smoothing_score):
            raise AppError(E_INPUT_INVALID, "LM/smoothing scores must be finite", stage="config_validation")

    def validate_for_session(self, shortest_shard_frames: int) -> None:
        if shortest_shard_frames > 0 and self.overlap_frames >= shortest_shard_frames:
            raise AppError(
                E_INPUT_INVALID,
                "overlap_frames must be smaller than the shortest shard length",
                stage="config_validation",
                details={
                    "overlap_frames": self.overlap_frames,
                    "shortest_shard_frames": shortest_shard_frames,
                },
            )
        # Lookahead validation is session-specific because an injection can set
        # zero only for one session while greedy zero remains valid.
        if self.strategy == "lookahead" and self.lookahead_frames <= 0:
            raise AppError(
                E_DECODE_PARAM_INVALID,
                "lookahead strategy requires lookahead_frames > 0",
                stage="decode_validation",
            )

    def with_overrides(self, overrides: Optional[Mapping[str, Any]]) -> "DecodeConfig":
        if not overrides:
            return self
        allowed = {
            "strategy",
            "lookahead_frames",
            "overlap_frames",
            "min_stable_frames",
            "lm_weight",
            "lm_backoff_score",
            "slot_min_frames",
            "slot_max_frames",
            "smoothing_score",
            "candidate_missing_below",
            "stable_margin",
        }
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise AppError(
                E_COMMAND_INVALID,
                "unknown config override(s): " + ", ".join(unknown),
                stage="command_validation",
            )
        values = dict(overrides)
        if "strategy" in values:
            values["strategy"] = str(values["strategy"]).lower()
        try:
            updated = replace(self, **values)
        except TypeError as exc:
            raise AppError(E_COMMAND_INVALID, str(exc), stage="command_validation") from exc
        try:
            updated.validate_global()
        except AppError:
            raise
        except (TypeError, ValueError) as exc:
            raise AppError(
                E_COMMAND_INVALID,
                f"invalid config override value: {exc}",
                stage="command_validation",
            ) from exc
        return updated

    def strategy_fingerprint_payload(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "lookahead_frames": self.lookahead_frames,
            "overlap_frames": self.overlap_frames,
            "lm_weight": self.lm_weight,
            "lm_backoff_score": self.lm_backoff_score,
            "smoothing_score": self.smoothing_score,
            "candidate_missing_below": self.candidate_missing_below,
        }

    def grid_fingerprint_payload(self) -> Dict[str, Any]:
        # Strategy and overlap policy deliberately excluded: changing either
        # must not rebuild the lattice for the same authoritative frame set.
        return {
            "lm_weight": self.lm_weight,
            "lm_backoff_score": self.lm_backoff_score,
            "smoothing_score": self.smoothing_score,
            "candidate_missing_below": self.candidate_missing_below,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "overlap_frames": self.overlap_frames,
            "min_stable_frames": self.min_stable_frames,
            "lm_weight": self.lm_weight,
            "slot_min_frames": self.slot_min_frames,
            "slot_max_frames": self.slot_max_frames,
            "response_budget_ms": self.response_budget_ms,
            "thread_pool_concurrency": self.thread_pool_concurrency,
            "snapshot_interval_shards": self.snapshot_interval_shards,
            "strategy": self.strategy,
            "lookahead_frames": self.lookahead_frames,
            "lm_backoff_score": self.lm_backoff_score,
            "smoothing_score": self.smoothing_score,
            "candidate_missing_below": self.candidate_missing_below,
            "stable_margin": self.stable_margin,
            "snapshot_root": self.snapshot_root,
            "report_path": self.report_path,
            "test_compute_delay_ms": self.test_compute_delay_ms,
        }


@dataclass(frozen=True)
class Candidate:
    token_id: int
    token: str
    emission: float

    def as_dict(self) -> Dict[str, Any]:
        return {"token_id": self.token_id, "token": self.token, "emission": round(self.emission, 8)}


@dataclass
class ShardBuild:
    session_id: str
    shard_seq: int
    revision: int
    start_frame: int
    end_frame: int
    frames: List[Tuple[int, Tuple[float, ...], bool]]
    failure: Optional[Dict[str, Any]] = None

    @property
    def sort_key(self) -> Tuple[str, int, int, int]:
        return (self.session_id, self.shard_seq, self.start_frame, self.revision)


@dataclass
class OwnedFrame:
    global_frame: int
    shard_seq: int
    revision: int
    frame_in_shard: int
    scores: Tuple[float, ...]
    covered: Tuple[Dict[str, Any], ...] = ()

    def as_dict(self, *, include_scores: bool = False) -> Dict[str, Any]:
        result = {
            "global_frame": self.global_frame,
            "owner": {
                "shard_seq": self.shard_seq,
                "revision": self.revision,
                "frame_in_shard": self.frame_in_shard,
            },
            "overwritten": list(self.covered),
        }
        if include_scores:
            result["scores"] = [round(value, 8) for value in self.scores]
        return result


@dataclass
class GridFrame:
    global_frame: int
    owner_shard: int
    owner_revision: int
    frame_in_shard: int
    candidates: Tuple[Candidate, ...]
    smoothed: bool
    smoothed_token_id: Optional[int]


@dataclass
class TokenDecision:
    global_frame: int
    token_id: int
    token: str
    score: float
    second_score: float
    margin: float
    smoothed: bool
    owner_shard: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "global_frame": self.global_frame,
            "token_id": self.token_id,
            "token": self.token,
            "score": round(self.score, 8),
            "second_score": round(self.second_score, 8),
            "margin": round(self.margin, 8),
            "smoothed": self.smoothed,
            "owner_shard": self.owner_shard,
        }


@dataclass
class DecodeResult:
    decisions: Tuple[TokenDecision, ...]
    greedy_decisions: Tuple[TokenDecision, ...]
    stable_length: int
    strategy: str
    lookahead_frames: int
    strategy_agreement_prefix: int
    reused_prefix_frames: int = 0
    decode_mode: str = "full"

    @property
    def token_ids(self) -> Tuple[int, ...]:
        return tuple(item.token_id for item in self.decisions)

    @property
    def stable_token_ids(self) -> Tuple[int, ...]:
        return tuple(item.token_id for item in self.decisions[: self.stable_length])

    @property
    def tail_token_ids(self) -> Tuple[int, ...]:
        return tuple(item.token_id for item in self.decisions[self.stable_length :])


@dataclass
class SessionArtifacts:
    session_id: str
    config: DecodeConfig
    owned_frames: List[OwnedFrame]
    grid: List[GridFrame]
    decode: Optional[DecodeResult]
    failures: List[Dict[str, Any]]
    effective_overlap: int
    fingerprints: Dict[str, str]
    cache_actions: Dict[str, Any] = field(default_factory=dict)
    edit_affected: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return len(self.grid)
