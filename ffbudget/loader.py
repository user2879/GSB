"""Stage 1: loading and validation of config, slot declarations, backfill
batches (NDJSON) and ranking requests. Invalid items become business-coded
failure records instead of raising.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import errors
from .fingerprint import content_fingerprint
from .models import (LAYER_REQUEST, VALID_LAYERS, LoadResult,
                     Receipt, SlotSpec)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _failure(code: str, message: str, **context) -> dict:
    record = {"code": code, "message": message}
    record.update(context)
    return record


def load_config(path: str | Path) -> tuple[dict, list[dict]]:
    failures: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    required = ("config_version", "format_version", "fixed_point", "staleness",
                "budget", "scheduler", "slots")
    for name in required:
        if name not in config:
            failures.append(_failure(errors.E_CONFIG_MISMATCH,
                                     f"missing config section: {name}",
                                     section=name))
    if failures:
        return config, failures

    fp_cfg = config["fixed_point"]
    if fp_cfg.get("width_tier") != "i64q16" or fp_cfg.get("frac_bits") != 16:
        failures.append(_failure(errors.E_CONFIG_MISMATCH,
                                 "unsupported fixed-point width tier",
                                 section="fixed_point"))

    staleness = config["staleness"]
    for name in ("max_staleness_ms", "penalty_fresh_q16",
                 "stale_penalty_per_ms_q16", "stale_floor_q16",
                 "penalty_unavailable_q16"):
        if not _is_int(staleness.get(name)):
            failures.append(_failure(errors.E_CONFIG_MISMATCH,
                                     f"staleness.{name} must be integer",
                                     section="staleness", field=name))

    budget = config["budget"]
    for name in ("score_point_budget", "slot_point_quota"):
        if not _is_int(budget.get(name)) or budget[name] <= 0:
            failures.append(_failure(errors.E_CONFIG_MISMATCH,
                                     f"budget.{name} must be positive int",
                                     section="budget", field=name))

    scheduler = config["scheduler"]
    if not _is_int(scheduler.get("max_concurrency")) or scheduler["max_concurrency"] < 1:
        failures.append(_failure(errors.E_CONFIG_MISMATCH,
                                 "scheduler.max_concurrency must be >= 1",
                                 section="scheduler"))

    slots: dict[str, SlotSpec] = {}
    seen_keys: set[str] = set()
    for index, raw_slot in enumerate(config["slots"]):
        key = raw_slot.get("key", f"<#{index}>")
        layer = raw_slot.get("layer")
        if key in seen_keys:
            failures.append(_failure(errors.E_DUPLICATE_SLOT_KEY,
                                     "duplicate slot key", slot_key=key))
            continue
        seen_keys.add(key)
        if layer not in VALID_LAYERS:
            failures.append(_failure(errors.E_UNKNOWN_LAYER,
                                     "slot layer must be request/item/cross",
                                     slot_key=key, layer=str(layer)))
            continue
        retention = raw_slot.get("retention")
        if not _is_int(retention) or retention < 1:
            failures.append(_failure(errors.E_BAD_RETENTION,
                                     "retention must be positive integer",
                                     slot_key=key))
            continue
        dim = raw_slot.get("dim")
        if not _is_int(dim) or dim < 1:
            failures.append(_failure(errors.E_BAD_DIMENSION,
                                     "dim must be positive integer",
                                     slot_key=key))
            continue
        weight = raw_slot.get("weight")
        if not _is_int(weight) or weight < 0:
            failures.append(_failure(errors.E_CONFIG_MISMATCH,
                                     "weight must be non-negative integer",
                                     slot_key=key))
            continue
        vector = raw_slot.get("default_vector")
        source = raw_slot.get("default_vector_source")
        if not isinstance(vector, list) or len(vector) != dim or \
                not all(_is_int(element) for element in vector):
            failures.append(_failure(errors.E_VECTOR_DIM_MISMATCH,
                                     "default_vector must be int list of dim",
                                     slot_key=key, dim=dim))
            continue
        if not isinstance(source, str) or not source.strip():
            failures.append(_failure(errors.E_CONFIG_MISMATCH,
                                     "default_vector_source must be non-empty",
                                     slot_key=key))
            continue
        slots[key] = SlotSpec(key=key, layer=layer, dim=dim,
                              retention=retention, weight=weight,
                              default_vector=tuple(vector),
                              default_vector_source=source)
    if not slots:
        failures.append(_failure(errors.E_CONFIG_MISMATCH,
                                 "no valid slot declarations"))
    config["_slot_order"] = [slot.key for slot in slots.values()]
    return config, failures


def _group_key(slot: SlotSpec, item_id: str | None) -> str:
    if slot.layer == LAYER_REQUEST:
        return slot.key
    return f"{slot.key}#{item_id}"


def _make_receipt(raw: dict, batch_seq: int, slots: dict,
                  failures: list, source: str, line_no: int) -> Receipt | None:
    slot_key = raw.get("slot_key")
    slot = slots.get(slot_key)
    if slot is None:
        failures.append(_failure(errors.E_UNKNOWN_SLOT,
                                 "receipt references undeclared slot",
                                 source=source, line=line_no,
                                 slot_key=str(slot_key)))
        return None
    effective_ms = raw.get("effective_ms")
    if not _is_int(effective_ms) or effective_ms < 0:
        failures.append(_failure(errors.E_BAD_EFFECTIVE_MS,
                                 "effective_ms must be non-negative integer ms",
                                 source=source, line=line_no, slot_key=slot_key))
        return None
    points = raw.get("points")
    if not _is_int(points) or points < 1:
        failures.append(_failure(errors.E_BAD_POINTS,
                                 "points must be positive integer",
                                 source=source, line=line_no, slot_key=slot_key))
        return None
    rev_seq = raw.get("rev_seq")
    if not _is_int(rev_seq) or rev_seq < 0:
        failures.append(_failure(errors.E_REV_SEQ_NOT_MONOTONIC,
                                 "rev_seq must be non-negative integer",
                                 source=source, line=line_no, slot_key=slot_key))
        return None
    vector = raw.get("vector")
    if not isinstance(vector, list) or len(vector) != slot.dim or \
            not all(_is_int(element) for element in vector):
        failures.append(_failure(errors.E_VECTOR_DIM_MISMATCH,
                                 "vector must be int list matching slot dim",
                                 source=source, line=line_no, slot_key=slot_key,
                                 dim=slot.dim))
        return None
    item_id = raw.get("item_id")
    if slot.layer == LAYER_REQUEST:
        if item_id is not None:
            failures.append(_failure(errors.E_CONFIG_MISMATCH,
                                     "request-level slot must not carry item_id",
                                     source=source, line=line_no,
                                     slot_key=slot_key))
            return None
    else:
        if not isinstance(item_id, str) or not item_id:
            failures.append(_failure(errors.E_CONFIG_MISMATCH,
                                     "item/cross slot requires non-empty item_id",
                                     source=source, line=line_no,
                                     slot_key=slot_key))
            return None
    digest_payload = [batch_seq, rev_seq, slot_key, effective_ms, points,
                      list(vector), item_id]
    digest = content_fingerprint(digest_payload)
    group_key = _group_key(slot, item_id)
    return Receipt(batch_seq=batch_seq, rev_seq=rev_seq, slot_key=slot_key,
                   effective_ms=effective_ms, points=points,
                   vector=tuple(vector), item_id=item_id, digest=digest,
                   group_key=group_key)


def load_batch_file(path: str | Path, slots: dict) -> tuple[dict, list, list[dict]]:
    failures: list[dict] = []
    receipts: list[Receipt] = []
    source = Path(path).name
    file_batch_seq: int | None = None
    last_rev_by_slot: dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(_failure(errors.E_CONFIG_MISMATCH,
                                         f"invalid NDJSON: {exc.msg}",
                                         source=source, line=line_no))
                continue
            batch_seq = raw.get("batch_seq")
            if not _is_int(batch_seq) or batch_seq < 0:
                failures.append(_failure(errors.E_BATCH_SEQ_NOT_STRICT,
                                         "batch_seq must be non-negative integer",
                                         source=source, line=line_no))
                continue
            if file_batch_seq is not None and batch_seq != file_batch_seq:
                failures.append(_failure(errors.E_BATCH_SEQ_NOT_STRICT,
                                         "one NDJSON file must carry one batch_seq",
                                         source=source, line=line_no,
                                         batch_seq=batch_seq,
                                         previous=file_batch_seq))
                continue
            file_batch_seq = batch_seq
            slot_key = raw.get("slot_key", "?")
            rev_seq = raw.get("rev_seq")
            if _is_int(rev_seq) and slot_key in last_rev_by_slot and \
                    rev_seq < last_rev_by_slot[slot_key]:
                failures.append(_failure(errors.E_REV_SEQ_NOT_MONOTONIC,
                                         "rev_seq must be non-decreasing within batch/slot",
                                         source=source, line=line_no,
                                         slot_key=slot_key, rev_seq=rev_seq,
                                         previous=last_rev_by_slot[slot_key]))
                continue
            receipt = _make_receipt(raw, batch_seq, slots, failures,
                                    source, line_no)
            if receipt is None:
                continue
            last_rev_by_slot[slot_key] = receipt.rev_seq
            receipts.append(receipt)
    batches = {receipt.batch_seq: source for receipt in receipts}
    return batches, receipts, failures


def load_requests(path: str | Path) -> tuple[list[dict], list[dict]]:
    failures: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    requests = payload.get("requests", [])
    seen: set[str] = set()
    valid: list[dict] = []
    for index, raw in enumerate(requests):
        request_id = raw.get("request_id", f"<#{index}>")
        if request_id in seen:
            failures.append(_failure(errors.E_CONFIG_MISMATCH,
                                     "duplicate request_id",
                                     request_id=request_id))
            continue
        seen.add(request_id)
        rank_ms = raw.get("rank_ms")
        if not _is_int(rank_ms) or rank_ms < 0:
            failures.append(_failure(errors.E_BAD_EFFECTIVE_MS,
                                     "rank_ms must be non-negative integer",
                                     request_id=request_id))
            continue
        candidates = raw.get("candidates")
        if not isinstance(candidates, list) or not candidates or \
                len(set(candidates)) != len(candidates) or \
                not all(isinstance(item, str) and item for item in candidates):
            failures.append(_failure(errors.E_CONFIG_MISMATCH,
                                     "candidates must be non-empty unique strings",
                                     request_id=request_id))
            continue
        valid.append({"request_id": request_id, "rank_ms": rank_ms,
                      "candidates": list(candidates)})
    return valid, failures


def load_workspace(config_path: str | Path, batch_dir: str | Path,
                   requests_path: str | Path) -> LoadResult:
    config, config_failures = load_config(config_path)
    failures = list(config_failures)
    slots = _slots_from_config(config)
    all_batches: dict = {}
    all_receipts: list[Receipt] = []
    if slots:
        for batch_path in sorted(Path(batch_dir).glob("*.ndjson")):
            batches, receipts, batch_failures = load_batch_file(batch_path, slots)
            failures.extend(batch_failures)
            all_batches.update(batches)
            all_receipts.extend(receipts)
    requests, request_failures = load_requests(requests_path)
    failures.extend(request_failures)
    return LoadResult(config=config, slots=slots, batches=all_batches,
                      receipts=all_receipts, failures=failures), requests


def _slots_from_config(config: dict) -> dict:
    from .models import SlotSpec
    slots: dict[str, SlotSpec] = {}
    for raw in config.get("slots", []):
        key = raw.get("key")
        if not isinstance(key, str):
            continue
        try:
            slots[key] = SlotSpec(
                key=key, layer=raw["layer"], dim=raw["dim"],
                retention=raw["retention"], weight=raw["weight"],
                default_vector=tuple(raw["default_vector"]),
                default_vector_source=raw["default_vector_source"])
        except (KeyError, TypeError):
            continue
    return slots
