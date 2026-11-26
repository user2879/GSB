"""File loading and validation entry points."""

from __future__ import annotations

import json

from .config import DecodeConfig
from .errors import BizCode, BizError
from .models import (
    CandidateSequence,
    Request,
    Step,
    validate_requests,
)


def load_config(data: dict) -> tuple[DecodeConfig | None, list[BizError]]:
    try:
        config = DecodeConfig(
            width=int(data.get("width", 4)),
            tier=str(data.get("tier", "ROUND_HALF_UP")),
            top_k=int(data.get("top_k", 8)),
            top_p_mille=int(data.get("top_p_mille", 900)),
            tolerance_units=int(data.get("tolerance_units", 2)),
        )
    except (TypeError, ValueError) as exc:
        return None, [BizError(BizCode.MISSING_FIELD, "config", str(exc))]
    return config, config.validate()


def load_requests(payload: dict) -> tuple[list[Request], list[BizError]]:
    requests: list[Request] = []
    errors: list[BizError] = []
    for idx, item in enumerate(payload.get("requests", [])):
        loc = f"payload.requests[{idx}]"
        try:
            sequences = []
            for seq_item in item.get("sequences", []):
                steps = [
                    Step(index=int(st["index"]),
                         logprobs={int(tok): float(lp)
                                   for tok, lp in st["logprobs"].items()})
                    for st in seq_item.get("steps", [])
                ]
                sequences.append(CandidateSequence(
                    seq_id=int(seq_item["seq_id"]),
                    seed_token=int(seq_item["seed_token"]),
                    channel=str(seq_item["channel"]),
                    temperature=float(seq_item["temperature"]),
                    steps=steps,
                ))
            requests.append(Request(
                request_id=str(item["request_id"]), sequences=sequences))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(BizError(BizCode.MISSING_FIELD, loc, str(exc)))
    errors.extend(validate_requests(requests))
    return requests, errors


def load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
