"""Deterministic business error codes.

Every validation / runtime failure is expressed with a stable integer code so
that reports and ledgers remain reproducible across runs and environments.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


class BizCode:
    OK = 0

    # --- loading / validation (1xxx) ---
    DUPLICATE_REQUEST = 1001
    UNKNOWN_REQUEST = 1002
    DUPLICATE_SEQUENCE = 1003
    INVALID_SEQUENCE_ID = 1004
    NON_POSITIVE_STEP = 1005
    STEP_HOLE = 1006
    STEP_REGRESSION = 1007
    NON_FINITE_LOGPROB = 1008
    POSITIVE_LOGPROB = 1009
    INVALID_TEMPERATURE = 1010
    TOPK_OUT_OF_RANGE = 1011
    TOPP_OUT_OF_RANGE = 1012
    INVALID_WIDTH = 1013
    INVALID_TIER = 1014
    INVALID_CHANNEL = 1015
    EMPTY_REQUEST = 1016
    EMPTY_STEP = 1017
    INVALID_TOKEN_CODE = 1018
    MISSING_FIELD = 1019
    REQUEST_FAILED = 1099

    # --- exchange formats (2xxx) ---
    V1_LINE_TOO_SHORT = 2001
    V1_ILLEGAL_CHAR = 2002
    V1_BAD_CHECKSUM = 2003
    V1_BAD_HEADER = 2004
    V2_BAD_JSON = 2010
    V2_UNKNOWN_VERSION = 2011
    V2_BAD_PROBABILITY = 2012
    V2_BAD_TOKEN = 2013
    SIDECAR_MISSING = 2020

    # --- scheduling / budgets (3xxx) ---
    REQUEST_QUOTA_EXCEEDED = 3001
    BATCH_BUDGET_EXHAUSTED = 3002
    STEP_RECOMPUTE_CAP = 3003
    STALE_SNAPSHOT_VERSION = 3004
    CONFIG_HASH_MISMATCH = 3005
    TODO_PENDING = 3006

    # --- tolerance / attribution (4xxx) ---
    TOLERANCE_EXCEEDED = 4001

    # --- injected fault marker (5xxx) ---
    INJECTED_FAULT = 5001

    @classmethod
    def name(cls, code: int) -> str:
        for key, value in vars(cls).items():
            if not key.startswith("_") and isinstance(value, int) and value == code:
                return key
        return f"UNKNOWN_{code}"


@dataclass(frozen=True)
class BizError(Exception):
    """A single located business failure.

    ``location`` is a deterministic structured path, e.g.
    ``request=R002/sequence=3/step=4/token=17``.
    Inherits Exception so callers can either collect it or raise it.
    """

    code: int
    location: str
    message: str
    injection_id: str | None = None

    def __str__(self) -> str:
        tag = f" [{self.injection_id}]" if self.injection_id else ""
        return f"{self.code_name}({self.code}){tag} {self.location}: {self.message}"

    @property
    def code_name(self) -> str:
        return BizCode.name(self.code)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["code_name"] = self.code_name
        return data


class DecodeError(Exception):
    """Raised for unrecoverable internal misuse (validation uses BizError)."""

    def __init__(self, code: int, message: str):
        super().__init__(f"[{BizCode.name(code)}] {message}")
        self.code = code
        self.message = message
