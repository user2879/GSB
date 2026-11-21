"""Deterministic NDJSON sample generator (stdlib only, fixed seed).

Rows, weight distribution and stack depth are fully controlled by CLI
arguments, and the random generator is seeded explicitly, so re-running with
the same arguments reproduces byte-identical files and input summaries.

Per-service timestamps are emitted monotone non-decreasing: each service keeps
its own millisecond clock while rows are round-robin interleaved across
services in file order.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Sequence, Tuple

SERVICES = ("gateway", "billing", "inventory", "notify")
METHODS = (
    "http", "auth", "cache", "db", "query", "serialize", "retry",
    "enqueue", "render", "rpc", "poll", "flush",
)


def _frame(service: str, rng: random.Random) -> str:
    return f"{service}.{rng.choice(METHODS)}"


def _build_stack(service: str, depth: int,
                 rng: random.Random) -> List[str]:
    frames = [f"{service}.entry"]
    current_service = service
    for level in range(depth - 1):
        # Mostly stay inside the service; occasionally cross a service boundary.
        if level > 0 and rng.random() < 0.18:
            current_service = rng.choice(
                [s for s in SERVICES if s != current_service])
        frames.append(_frame(current_service, rng))
    return frames


def generate(*, rows: int, services: int, min_depth: int, max_depth: int,
             weight_mode: str, seed: int,
             output: str) -> Tuple[Path, dict]:
    if services <= 0 or services > len(SERVICES):
        raise ValueError(
            f"services must be in 1..{len(SERVICES)}, got {services}")
    if min_depth <= 0 or max_depth < min_depth:
        raise ValueError("depth bounds must satisfy 0 < min_depth <= max_depth")
    chosen_services = SERVICES[:services]
    rng = random.Random(seed)

    clocks = {service: 1_700_000_000_000 for service in chosen_services}
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_weight = 0
    depth_sum = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(rows):
            service = chosen_services[index % services]
            depth = rng.randint(min_depth, max_depth)
            frames = _build_stack(service, depth, rng)
            if weight_mode == "uniform":
                weight = rng.randint(1, 5)
            elif weight_mode == "skewed":
                # Long-tail: mostly 1..4, occasional heavy samples.
                weight = (
                    rng.randint(50, 200) if rng.random() < 0.05
                    else rng.randint(1, 4))
            elif weight_mode == "unit":
                weight = 1
            else:
                raise ValueError(f"unknown weight_mode={weight_mode!r}")
            duration = rng.randint(5, 500)
            clocks[service] += rng.randint(0, 25)
            row = {
                "request_id": f"req-{index:08d}",
                "timestamp_ms": clocks[service],
                "duration_ms": duration,
                "service_id": service,
                "weight": weight,
                "frames": frames,
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            total_weight += weight
            depth_sum += depth

    summary = {
        "rows": rows,
        "services": list(chosen_services),
        "depth_range": [min_depth, max_depth],
        "weight_mode": weight_mode,
        "seed": seed,
        "total_weight": total_weight,
        "avg_depth": depth_sum / rows if rows else 0.0,
        "output": str(out_path),
    }
    return out_path, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic NDJSON stack samples.")
    parser.add_argument("--rows", type=int, default=5_000)
    parser.add_argument("--services", type=int, default=3)
    parser.add_argument("--min-depth", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument(
        "--weight-mode",
        choices=("uniform", "skewed", "unit"), default="skewed")
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument(
        "--output", type=str, default="data/samples.ndjson")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _, summary = generate(
        rows=args.rows, services=args.services,
        min_depth=args.min_depth, max_depth=args.max_depth,
        weight_mode=args.weight_mode, seed=args.seed,
        output=args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
