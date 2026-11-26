"""Core domain models for the decoding sampling chain."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any

from .config import (
    CHANNELS,
    TEMPERATURE_MAX_INCLUSIVE,
    TEMPERATURE_MIN_EXCLUSIVE,
    canonical_json,
)
from .errors import BizCode, BizError
import math


@dataclass
class Step:
    """One decoded step.

    logprobs maps an integer token code to a finite, non-positive log prob.
    index is the step ordinal (1-based within the sequence).
    """

    index: int
    logprobs: dict[int, float]

    def sorted_tokens(self) -> list[int]:
        return sorted(self.logprobs)

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "logprobs": {str(tok): self.logprobs[tok]
                         for tok in sorted(self.logprobs)},
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_json(self.fingerprint_payload()).encode("utf-8")
        ).hexdigest()


@dataclass
class CandidateSequence:
    seq_id: int
    seed_token: int
    channel: str
    temperature: float
    steps: list[Step] = field(default_factory=list)

    def step_map(self) -> dict[int, Step]:
        return {step.index: step for step in self.steps}

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "seq_id": self.seq_id,
            "seed_token": self.seed_token,
            "channel": self.channel,
            "temperature": self.temperature,
            "steps": [st.fingerprint_payload() for st in sorted(
                self.steps, key=lambda st: st.index)],
        }

    def step_fingerprint(self, step_index: int) -> str:
        step = self.step_map().get(step_index)
        if step is None:
            raise KeyError(step_index)
        payload = {
            "request_anchor": self.seq_id,
            "seed_token": self.seed_token,
            "channel": self.channel,
            "temperature": self.temperature,
            "step": step.fingerprint_payload(),
        }
        return hashlib.sha256(
            canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass
class Request:
    request_id: str
    sequences: list[CandidateSequence] = field(default_factory=list)

    def sequence_map(self) -> dict[int, CandidateSequence]:
        return {seq.seq_id: seq for seq in self.sequences}

    def all_steps(self) -> list[tuple[int, Step]]:
        out: list[tuple[int, Step]] = []
        for seq in sorted(self.sequences, key=lambda s: s.seq_id):
            for step in sorted(seq.steps, key=lambda st: st.index):
                out.append((seq.seq_id, step))
        return out

    def content_fingerprint(self) -> str:
        """Hash of the request content only (config-independent)."""
        payload = {
            "request_id": self.request_id,
            "sequences": [s.fingerprint_payload()
                          for s in sorted(self.sequences,
                                          key=lambda x: x.seq_id)],
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "sequences": [
                {
                    "seq_id": seq.seq_id,
                    "seed_token": seq.seed_token,
                    "channel": seq.channel,
                    "temperature": seq.temperature,
                    "steps": [
                        {"index": st.index,
                         "logprobs": {str(k): v for k, v in
                                      sorted(st.logprobs.items())}}
                        for st in sorted(seq.steps, key=lambda x: x.index)
                    ],
                }
                for seq in sorted(self.sequences, key=lambda x: x.seq_id)
            ],
        }


def stable_task_key(request_id: str, seq_id: int, step_index: int) -> str:
    """Deterministic stable sort key carried by every scheduled task."""
    return f"{request_id}|{int(seq_id):08d}|{int(step_index):08d}"


def validate_request(request: Request) -> list[BizError]:
    """Structural + range validation for one request.

    Every problem is reported with a precise location; validation never stops
    at the first error.
    """
    errs: list[BizError] = []
    loc_req = f"request={request.request_id}"

    if not request.sequences:
        errs.append(BizError(BizCode.EMPTY_REQUEST, loc_req,
                             "request has no candidate sequences"))
        return errs

    seen_seq: set[int] = set()
    for seq in request.sequences:
        loc_seq = f"{loc_req}/sequence={seq.seq_id}"
        seq_id_valid = (isinstance(seq.seq_id, int)
                        and not isinstance(seq.seq_id, bool)
                        and seq.seq_id > 0)
        if not seq_id_valid:
            errs.append(BizError(BizCode.INVALID_SEQUENCE_ID, loc_seq,
                                 "sequence id must be a positive integer"))
        elif seq.seq_id in seen_seq:
            errs.append(BizError(BizCode.DUPLICATE_SEQUENCE, loc_seq,
                                 "duplicate candidate sequence id"))
        if seq_id_valid:
            seen_seq.add(seq.seq_id)

        if not isinstance(seq.seed_token, int) or isinstance(seq.seed_token, bool) \
                or seq.seed_token < 0:
            errs.append(BizError(BizCode.INVALID_TOKEN_CODE,
                                 f"{loc_seq}/seed_token",
                                 "seed token must be a non-negative integer"))

        if seq.channel not in CHANNELS:
            errs.append(BizError(BizCode.INVALID_CHANNEL,
                                 f"{loc_seq}/channel",
                                 f"channel={seq.channel!r} not in {CHANNELS}"))

        temp = seq.temperature
        if not isinstance(temp, (int, float)) or isinstance(temp, bool) \
                or not math.isfinite(float(temp)) \
                or not (TEMPERATURE_MIN_EXCLUSIVE < float(temp)
                        <= TEMPERATURE_MAX_INCLUSIVE):
            errs.append(BizError(BizCode.INVALID_TEMPERATURE,
                                 f"{loc_seq}/temperature",
                                 f"temperature={temp!r} must be finite and "
                                 f"in (0,{TEMPERATURE_MAX_INCLUSIVE}]"))

        if not seq.steps:
            errs.append(BizError(BizCode.EMPTY_STEP, loc_seq,
                                 "sequence has no decoded steps"))
            continue

        expected = 1
        seen_indexes: set[int] = set()
        for step in seq.steps:
            loc_step = f"{loc_seq}/step={step.index}"
            index_valid = (isinstance(step.index, int)
                           and not isinstance(step.index, bool)
                           and step.index > 0)
            if not index_valid:
                errs.append(BizError(BizCode.NON_POSITIVE_STEP, loc_step,
                                     "step index must be a positive integer"))
                continue
            if step.index in seen_indexes:
                errs.append(BizError(BizCode.DUPLICATE_SEQUENCE, loc_step,
                                     f"duplicate step index {step.index}"))
            elif step.index < expected:
                errs.append(BizError(BizCode.STEP_REGRESSION, loc_step,
                                     f"step index regressed; expected >="
                                     f" {expected}"))
            elif step.index > expected:
                errs.append(BizError(BizCode.STEP_HOLE, loc_step,
                                     f"step hole: expected {expected}, got "
                                     f"{step.index}"))
            if index_valid:
                seen_indexes.add(step.index)
                expected = max(expected, step.index + 1)

            if not step.logprobs:
                errs.append(BizError(BizCode.EMPTY_STEP, loc_step,
                                     "step has no token logprobs"))
            for tok, lp in step.logprobs.items():
                loc_tok = f"{loc_step}/token={tok}"
                if not isinstance(tok, int) or isinstance(tok, bool) or tok < 0:
                    errs.append(BizError(BizCode.INVALID_TOKEN_CODE, loc_tok,
                                         "token code must be non-negative int"))
                if not isinstance(lp, (int, float)) or isinstance(lp, bool):
                    errs.append(BizError(BizCode.NON_FINITE_LOGPROB, loc_tok,
                                         f"logprob={lp!r} not numeric"))
                    continue
                if not math.isfinite(float(lp)):
                    errs.append(BizError(BizCode.NON_FINITE_LOGPROB, loc_tok,
                                         f"non-finite logprob={lp!r}"))
                elif float(lp) > 0.0:
                    errs.append(BizError(BizCode.POSITIVE_LOGPROB, loc_tok,
                                         f"positive logprob={lp!r}"))
    return errs


def validate_requests(requests: list[Request]) -> list[BizError]:
    errs: list[BizError] = []
    seen: set[str] = set()
    for req in requests:
        loc = f"request={req.request_id}"
        if req.request_id in seen:
            errs.append(BizError(BizCode.DUPLICATE_REQUEST, loc,
                                 "duplicate request id"))
        seen.add(req.request_id)
        errs.extend(validate_request(req))
    return errs


def request_asdict(request: Request) -> dict:
    return asdict(request)
