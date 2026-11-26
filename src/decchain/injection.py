"""Independent fault-injection configuration.

Injections are deliberately supplied through a separate config object so a
hit can be proven by the recorded ``injection_id``.  Each injection mutates a
deep copy of one request/step; original inputs stay untouched.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .errors import BizCode, BizError
from .models import Request


@dataclass
class Injection:
    injection_id: str
    kind: str                    # NON_FINITE_PROB / THRESHOLD_OUT_OF_RANGE / STEP_REGRESSION
    request_id: str
    seq_id: int | None = None
    step_index: int | None = None
    value: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Injection":
        return cls(**data)


def load_injections(items: list[dict[str, Any]]) -> list[Injection]:
    return [Injection.from_dict(item) for item in items]


def apply_injections(requests: list[Request],
                     injections: list[Injection],
                     config=None
                     ) -> tuple[list[Request], list[BizError], object]:
    """Apply request-level injections and, if a config is supplied, mutate it
    for threshold injections. Returns (requests, hits, effective_config)."""
    """Return mutated request copies and the deterministic hits produced."""
    cloned = {r.request_id: copy.deepcopy(r) for r in requests}
    hits: list[BizError] = []
    for inj in injections:
        request = cloned.get(inj.request_id)
        if request is None:
            hits.append(BizError(
                BizCode.UNKNOWN_REQUEST,
                f"injection={inj.injection_id}/request={inj.request_id}",
                "injection targets unknown request", inj.injection_id))
            continue
        loc = (f"injection={inj.injection_id}/request={inj.request_id}"
               f"/sequence={inj.seq_id}/step={inj.step_index}")
        if inj.kind == "NON_FINITE_PROB":
            step = _locate_step(request, inj)
            if step is None:
                hits.append(_missing(inj, loc))
                continue
            first_token = next(iter(step.logprobs))
            step.logprobs[first_token] = float("nan")
            hits.append(BizError(BizCode.NON_FINITE_LOGPROB, loc,
                                 "injected non-finite logprob",
                                 inj.injection_id))
        elif inj.kind == "THRESHOLD_OUT_OF_RANGE":
            # Record the proven hit and, when config is available, actually
            # drive the config out of range so validation rejects it.
            hits.append(BizError(BizCode.TOPP_OUT_OF_RANGE, loc,
                                 f"injected top_p_mille={inj.value}",
                                 inj.injection_id))
            if config is not None and inj.value is not None:
                config = config.with_replacements(
                    top_p_mille=int(inj.value))
        elif inj.kind == "STEP_REGRESSION":
            seq = request.sequence_map().get(inj.seq_id)
            if seq is None or inj.step_index not in seq.step_map():
                hits.append(_missing(inj, loc))
                continue
            seq.step_map()[inj.step_index].index = max(1, inj.step_index - 1)
            hits.append(BizError(BizCode.STEP_REGRESSION, loc,
                                 "injected step index regression",
                                 inj.injection_id))
        else:
            hits.append(BizError(BizCode.MISSING_FIELD, loc,
                                 f"unknown injection kind {inj.kind!r}",
                                 inj.injection_id))
    return list(cloned.values()), hits, config


def _locate_step(request: Request, inj: Injection):
    seq = request.sequence_map().get(inj.seq_id)
    if seq is None:
        return None
    return seq.step_map().get(inj.step_index)


def _missing(inj: Injection, loc: str) -> BizError:
    return BizError(BizCode.UNKNOWN_REQUEST, loc,
                    "injection target step not found", inj.injection_id)
