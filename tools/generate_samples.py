"""Generate deterministic local samples from a fixed integer seed.

No real accounts, online endpoints or external datasets are used. Outputs:
  samples/config.json
  samples/requests.json
  samples/batches/0001_first.ndjson
  samples/batches/0002_second.ndjson
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20261125
ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
BATCH_DIR = SAMPLES / "batches"

ITEMS = ["item_a", "item_b", "item_c", "item_d"]
REQUESTS = ["req_01", "req_02", "req_03", "req_04", "req_05"]

SLOT_ORDER = [
    ("req_context", "request", 3, 2, 4),
    ("req_geo", "request", 2, 2, 2),
    ("item_profile", "item", 4, 3, 5),
    ("item_quality", "item", 2, 2, 3),
    ("cross_cooccur", "cross", 3, 2, 4),
    ("cross_affinity", "cross", 2, 2, 2),
]


def _vector(rng: random.Random, dim: int, scale: int) -> list:
    return [rng.randint(-scale, scale) for _ in range(dim)]


def build_config() -> dict:
    rng = random.Random(SEED)
    slots = []
    for key, layer, dim, retention, weight in SLOT_ORDER:
        slots.append({
            "key": key,
            "layer": layer,
            "dim": dim,
            "retention": retention,
            "weight": weight,
            "default_vector": [rng.randint(-2, 2) for _ in range(dim)],
            "default_vector_source":
                f"config://slots/{key}/default_vector#seed={SEED}",
        })
    return {
        "config_version": "ffbudget-sample-v1",
        "format_version": 1,
        "fixed_point": {"width_tier": "i64q16", "frac_bits": 16},
        "staleness": {
            "max_staleness_ms": 4000,
            "penalty_fresh_q16": 65536,
            "stale_penalty_per_ms_q16": 8,
            "stale_floor_q16": 16384,
            "penalty_unavailable_q16": 6554,
        },
        "budget": {"score_point_budget": 2000, "slot_point_quota": 600},
        "scheduler": {"max_concurrency": 4},
        "slots": slots,
    }


def _receipt(rng, batch_seq, rev_seq, slot, item_id, effective_ms):
    key, layer, dim, _, _ = slot
    vector = _vector(rng, dim, 20)
    payload = {
        "batch_seq": batch_seq,
        "rev_seq": rev_seq,
        "slot_key": key,
        "effective_ms": effective_ms,
        "points": dim,
        "vector": vector,
    }
    if layer != "request":
        payload["item_id"] = item_id
    return payload


def build_batches(config: dict):
    rng = random.Random(SEED + 1)
    first_lines, second_lines, redelivery_lines = [], [], []
    rev_counter: dict[str, int] = {}

    def next_rev(slot_key: str) -> int:
        rev_counter[slot_key] = rev_counter.get(slot_key, 0) + 1
        return rev_counter[slot_key]

    # Batch 1001: request slots + item_profile for items a/b/c + cross for a/b
    for key, layer, dim, _, _ in SLOT_ORDER:
        if layer == "request":
            first_lines.append(_receipt(
                rng, 1001, next_rev(key), (key, layer, dim, 0, 0),
                None, effective_ms=500))
        elif key == "item_profile":
            for item_id in ITEMS[:3]:
                first_lines.append(_receipt(
                    rng, 1001, next_rev(key), (key, layer, dim, 0, 0),
                    item_id, effective_ms=800))
        elif key == "cross_cooccur":
            for item_id in ITEMS[:2]:
                first_lines.append(_receipt(
                    rng, 1001, next_rev(key), (key, layer, dim, 0, 0),
                    item_id, effective_ms=900))

    # Exact redelivery of one batch-1001 receipt (same batch_seq + content),
    # exercised through a separate file to simulate duplicate delivery.
    redelivery_lines.append(dict(first_lines[2]))

    # Batch 1002: newer revisions, including a future-effective receipt and a
    # full-tie pair (identical rev_seq/effective/batch ordering fields are
    # produced by giving item_d two same-rev receipts across redelivery).
    for key, layer, dim, _, _ in SLOT_ORDER:
        if layer == "request":
            effective_ms = 6000 if key == "req_geo" else 1500
            second_lines.append(_receipt(
                rng, 1002, next_rev(key), (key, layer, dim, 0, 0),
                None, effective_ms=effective_ms))
        elif key == "item_profile":
            for item_id in ITEMS[:3]:
                second_lines.append(_receipt(
                    rng, 1002, next_rev(key), (key, layer, dim, 0, 0),
                    item_id, effective_ms=1600))
            # brand-new slot group, never backfilled before
            second_lines.append(_receipt(
                rng, 1002, next_rev(key), (key, layer, dim, 0, 0),
                "item_d", effective_ms=1700))
        elif key == "item_quality":
            for item_id in ITEMS:
                second_lines.append(_receipt(
                    rng, 1002, next_rev(key), (key, layer, dim, 0, 0),
                    item_id, effective_ms=1800))
        elif key == "cross_cooccur":
            for item_id in ITEMS[:3]:
                second_lines.append(_receipt(
                    rng, 1002, next_rev(key), (key, layer, dim, 0, 0),
                    item_id, effective_ms=2000))
        elif key == "cross_affinity":
            for item_id in ITEMS[:2]:
                second_lines.append(_receipt(
                    rng, 1002, next_rev(key), (key, layer, dim, 0, 0),
                    item_id, effective_ms=2100))

    # Batch 1003 arrives "late" (out of order relative to ranking time): it
    # carries a lower rev_seq than batch 1002 for one group, so adjudication
    # must refuse it without touching the effective revision.
    late_lines = []
    target = "item_quality"
    late_lines.append(_receipt(
        rng, 1003, 1, (target, "item", 2, 0, 0), "item_a",
        effective_ms=1000))

    # Batch 1004 holds a full-tie pair for one group: identical rev_seq,
    # effective_ms and batch_seq, different content. The digest tie-break is
    # the only thing that can decide the winner.
    tie_slot = ("cross_affinity", "cross", 2, 0, 0)
    tie_a = _receipt(rng, 1004, 9, tie_slot, "item_c",
                     effective_ms=2500)
    tie_b = _receipt(rng, 1004, 9, tie_slot, "item_c",
                     effective_ms=2500)
    tie_lines = [tie_a, tie_b]

    return first_lines, redelivery_lines, second_lines, late_lines, tie_lines


def build_requests() -> dict:
    # Ranking timestamps exercise fresh, stale and future-fallback views.
    timeline = {
        "req_01": 1000,
        "req_02": 1700,
        "req_03": 3000,
        "req_04": 5500,
        "req_05": 7000,
    }
    return {
        "request_set_version": "ffbudget-requests-v1",
        "requests": [
            {"request_id": request_id, "rank_ms": rank_ms,
             "candidates": list(ITEMS)}
            for request_id, rank_ms in sorted(timeline.items())
        ],
    }


def write_ndjson(path: Path, lines: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(json.dumps(line, sort_keys=True,
                                    ensure_ascii=False) + "\n")


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    config = build_config()
    with open(SAMPLES / "config.json", "w", encoding="utf-8",
              newline="\n") as handle:
        json.dump(config, handle, sort_keys=True, ensure_ascii=False, indent=2)
    with open(SAMPLES / "requests.json", "w", encoding="utf-8",
              newline="\n") as handle:
        json.dump(build_requests(), handle, sort_keys=True,
                  ensure_ascii=False, indent=2)

    first, redelivery, second, late, tie = build_batches(config)
    write_ndjson(BATCH_DIR / "0001_first.ndjson", first)
    write_ndjson(BATCH_DIR / "0002_redelivery.ndjson", redelivery)
    write_ndjson(BATCH_DIR / "0003_second.ndjson", second)
    write_ndjson(BATCH_DIR / "0004_late.ndjson", late)
    write_ndjson(BATCH_DIR / "0005_tie.ndjson", tie)
    print(f"generated samples under {SAMPLES} (seed={SEED})")


if __name__ == "__main__":
    main()
