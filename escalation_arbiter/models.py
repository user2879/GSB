"""Domain models for tickets, rule packages, config and stage artifacts."""

from dataclasses import asdict, dataclass, field

MAX_CONCURRENCY = 16
STAGES = ("decay", "relation", "arbitration", "conversion")


@dataclass(frozen=True)
class Evidence:
    ticket_id: str
    seq: int
    etype: str
    role: str
    ts: int
    terms: tuple
    confidence: int

    def to_dict(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "seq": self.seq,
            "etype": self.etype,
            "role": self.role,
            "ts": self.ts,
            "terms": list(self.terms),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Evidence":
        return cls(
            ticket_id=data["ticket_id"],
            seq=int(data["seq"]),
            etype=data["etype"],
            role=data["role"],
            ts=int(data["ts"]),
            terms=tuple(data["terms"]),
            confidence=int(data["confidence"]),
        )


@dataclass
class Ticket:
    tid: str
    evidences: list = field(default_factory=list)

    def sorted_evidence(self) -> list:
        return sorted(self.evidences, key=lambda e: e.seq)

    def base_ts(self) -> int:
        return min(e.ts for e in self.evidences)

    def to_dict(self) -> dict:
        return {"tid": self.tid,
                "evidences": [e.to_dict() for e in self.sorted_evidence()]}


@dataclass(frozen=True)
class RelationEntry:
    rid: str
    kind: str  # "mutex" | "support"
    left: str
    right: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RelationEntry":
        return cls(rid=data["rid"], kind=data["kind"],
                   left=data["left"], right=data["right"])


@dataclass(frozen=True)
class PriorityTier:
    limit: int
    level: int
    label: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PriorityTier":
        return cls(limit=int(data["limit"]), level=int(data["level"]),
                   label=data["label"])


@dataclass
class RulePackage:
    type_weights: dict
    role_cred: dict
    mutex: list
    support: list
    tiers: list
    reason_dict: dict

    def all_relations(self) -> list:
        return list(self.mutex) + list(self.support)


@dataclass(frozen=True)
class ArbConfig:
    decimals: int
    quantum: int
    half_life_ms: int
    deduct_permille: int
    bonus_permille: int
    fusion: str
    zero_floor: int
    budget_ms: int
    concurrency: int
    snapshot_interval: int
    conf_min_permille: int
    conf_max_permille: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ArbConfig":
        return cls(
            decimals=int(data["decimals"]),
            quantum=int(data["quantum"]),
            half_life_ms=int(data["half_life_ms"]),
            deduct_permille=int(data["deduct_permille"]),
            bonus_permille=int(data["bonus_permille"]),
            fusion=data["fusion"],
            zero_floor=int(data["zero_floor"]),
            budget_ms=int(data["budget_ms"]),
            concurrency=int(data["concurrency"]),
            snapshot_interval=int(data["snapshot_interval"]),
            conf_min_permille=int(data.get("conf_min_permille", 0)),
            conf_max_permille=int(data.get("conf_max_permille", 1000)),
        )


@dataclass
class EvidenceScore:
    seq: int
    etype: str
    role: str
    ts: int
    delta_ms: int
    decay: int
    weight: int
    role_cred: int
    confidence: int
    factor_1: int
    factor_2: int
    factor_3: int
    round_dir_1: str
    round_dir_2: str
    round_dir_3: str
    raw_score: int
    quantized_score: int
    residual: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvidenceScore":
        return cls(**data)


@dataclass
class RelationContribution:
    rid: str
    kind: str
    left_seq: int
    right_seq: int
    reason_code: str
    left_before: int
    right_before: int
    left_delta: int
    right_delta: int
    left_after: int
    right_after: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RelationContribution":
        return cls(**data)


@dataclass
class TicketDecision:
    tid: str
    base_ts: int
    evidence_scores: list
    final_evidence_scores: dict
    contributions: list
    decision_score: int
    boundary: bool
    priority_level: int
    priority_label: str
    escalate: bool
    reason_codes: list
    stage_keys: dict
    failure: object = None

    def to_dict(self) -> dict:
        return {
            "tid": self.tid,
            "base_ts": self.base_ts,
            "evidence_scores": [e.to_dict() for e in self.evidence_scores],
            "final_evidence_scores": {str(k): v for k, v in
                                      self.final_evidence_scores.items()},
            "contributions": [c.to_dict() for c in self.contributions],
            "decision_score": self.decision_score,
            "boundary": self.boundary,
            "priority_level": self.priority_level,
            "priority_label": self.priority_label,
            "escalate": self.escalate,
            "reason_codes": list(self.reason_codes),
            "stage_keys": dict(self.stage_keys),
            "failure": None if self.failure is None else self.failure.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TicketDecision":
        from .errors import Failure
        return cls(
            tid=data["tid"],
            base_ts=int(data["base_ts"]),
            evidence_scores=[EvidenceScore.from_dict(r)
                             for r in data["evidence_scores"]],
            final_evidence_scores={int(k): int(v)
                                   for k, v in data["final_evidence_scores"].items()},
            contributions=[RelationContribution.from_dict(r)
                           for r in data["contributions"]],
            decision_score=int(data["decision_score"]),
            boundary=bool(data["boundary"]),
            priority_level=int(data["priority_level"]),
            priority_label=data["priority_label"],
            escalate=bool(data["escalate"]),
            reason_codes=list(data["reason_codes"]),
            stage_keys=dict(data["stage_keys"]),
            failure=None if data.get("failure") is None
            else Failure.from_dict(data["failure"]),
        )
