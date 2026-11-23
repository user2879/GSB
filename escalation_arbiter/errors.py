"""Business error codes and structured failure records."""

from dataclasses import dataclass, field
from typing import Optional

E_EVIDENCE_INVALID = "E_EVIDENCE_INVALID"
E_RULE_UNKNOWN = "E_RULE_UNKNOWN"
E_RULE_CYCLE = "E_RULE_CYCLE"
E_DICT_DUPLICATE = "E_DICT_DUPLICATE"
E_THRESHOLD_ORDER = "E_THRESHOLD_ORDER"
E_FIXED_WIDTH = "E_FIXED_WIDTH"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_INTERACT_BUDGET_EXCEEDED = "E_INTERACT_BUDGET_EXCEEDED"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"

ALL_CODES = (
    E_EVIDENCE_INVALID,
    E_RULE_UNKNOWN,
    E_RULE_CYCLE,
    E_DICT_DUPLICATE,
    E_THRESHOLD_ORDER,
    E_FIXED_WIDTH,
    E_SNAPSHOT_STALE,
    E_INTERACT_BUDGET_EXCEEDED,
    E_COMMAND_INVALID,
    E_CONFIG_INVALID,
)


@dataclass(frozen=True)
class Failure:
    """A single structured failure entry.

    ``inj_id`` is set when the failure was triggered by an injected fault,
    which makes injection hits auditable in the diagnostic report.
    """

    code: str
    scope: str
    ref: str
    detail: str
    inj_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "scope": self.scope,
            "ref": self.ref,
            "detail": self.detail,
            "inj_id": self.inj_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Failure":
        return cls(
            code=data["code"],
            scope=data["scope"],
            ref=data["ref"],
            detail=data["detail"],
            inj_id=data.get("inj_id"),
        )


@dataclass
class FailureList:
    """Failures kept in a deterministic, insertion-ordered sequence."""

    items: list = field(default_factory=list)

    def add(self, code: str, scope: str, ref: str, detail: str,
            inj_id: Optional[str] = None) -> Failure:
        failure = Failure(code, scope, ref, detail, inj_id)
        self.items.append(failure)
        return failure

    def codes(self) -> list:
        return [f.code for f in self.items]

    def to_list(self) -> list:
        return [f.to_dict() for f in self.items]

    @classmethod
    def from_list(cls, data: list) -> "FailureList":
        obj = cls()
        obj.items = [Failure.from_dict(row) for row in data]
        return obj
