"""Deterministic local sample generation (no network, no external data)."""

from __future__ import annotations

import json
import math

from .models import Request, CandidateSequence, Step


def _logprobs(token_count: int, seed: int) -> dict[int, float]:
    # Deterministic, finite, non-positive log probabilities.
    out = {}
    for i in range(token_count):
        value = -((i + 1) * (seed % 3 + 1) * 0.37)
        out[100 + i] = round(min(0.0, -abs(value)), 6)
    return out


def build_sample_requests() -> list[Request]:
    requests: list[Request] = []
    for r in range(1, 4):
        request_id = f"R{r:03d}"
        sequences = []
        for seq_id in range(1, 3):
            steps = [
                Step(index=step,
                     logprobs=_logprobs(5 + r, r * 7 + seq_id * 3 + step))
                for step in range(1, 4)
            ]
            sequences.append(CandidateSequence(
                seq_id=seq_id,
                seed_token=10 + seq_id,
                channel="SAMPLE" if seq_id == 1 else "GREEDY",
                temperature=round(0.7 + 0.2 * r + 0.1 * seq_id, 4),
                steps=steps,
            ))
        requests.append(Request(request_id=request_id, sequences=sequences))
    return requests


def requests_to_payload(requests: list[Request]) -> dict:
    return {"requests": [r.to_dict() for r in requests]}


def write_sample_file(path: str) -> list[Request]:
    requests = build_sample_requests()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(requests_to_payload(requests), handle,
                  sort_keys=True, indent=2, ensure_ascii=False)
    return requests
