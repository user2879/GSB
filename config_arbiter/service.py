from __future__ import annotations

import argparse
import json
import socketserver
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .canonical import canonical_json, normalize_decimal, sha256_hex, stable_seed64
from .fixedpoint import sat_add, saturate, scale_factor, trunc_div


@dataclass(frozen=True)
class AggregateComputation:
    aggregates: dict[str, int]
    sample_digest: str


def _request_error(field: str, message: str) -> dict[str, Any]:
    return {"error": {"code": "E_FIELD", "field": field, "message": message}}


def _require_int(payload: dict[str, Any], field: str, minimum: int) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def compute_aggregates(
    *,
    seed: int,
    scale_exponent: int,
    sample_count: int,
    group_size: int,
    noise_bound: int,
    slice_payload: dict[str, Any],
) -> AggregateComputation:
    version = slice_payload.get("version")
    values = slice_payload.get("values")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("slice.version must be an integer >= 1")
    if not isinstance(values, dict) or not values:
        raise ValueError("slice.values must be a non-empty object")

    groups = sample_count // group_size
    aggregate_summary: dict[str, int] = {}
    sample_traces: dict[str, list[int]] = {}
    factor = scale_factor(scale_exponent)
    for key in sorted(values):
        _, decimal_value = normalize_decimal(values[key], f"slice.values.{key}")
        config_units = saturate(int(decimal_value * factor))

        derived_seed = stable_seed64(seed, key, version)
        rng = np.random.Generator(np.random.PCG64(derived_seed))
        if noise_bound:
            raw = rng.integers(
                -noise_bound,
                noise_bound + 1,
                size=(groups, group_size),
                dtype=np.int64,
            )
            group_sums = raw.sum(axis=1, dtype=np.int64)
            group_offsets = np.floor_divide(group_sums, group_size)
            raw = raw - group_offsets[:, None]
            remainder = raw.sum(axis=1, dtype=np.int64)
            raw[:, -1] -= remainder
        else:
            raw = np.zeros((groups, group_size), dtype=np.int64)

        # Every group is centered to a zero integer sum. The grouped fixed-point
        # mean therefore isolates the slice value while the seed still determines
        # the exact sample trace and its digest.
        sample_values = [
            [saturate(config_units + int(noise)) for noise in group.tolist()]
            for group in raw
        ]
        group_means = [
            trunc_div(sum(group), group_size) for group in sample_values
        ]
        group_total = 0
        for group_mean in group_means:
            group_total = sat_add(group_total, group_mean)
        aggregate_summary[key] = trunc_div(group_total, groups)
        sample_traces[key] = [item for group in sample_values for item in group]

    sample_digest = sha256_hex(
        {
            "seed": seed,
            "scale_exponent": scale_exponent,
            "version": version,
            "samples": sample_traces,
        }
    )
    return AggregateComputation(aggregates=aggregate_summary, sample_digest=sample_digest)


def build_receipt(
    *,
    request_id: str,
    service_id: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    scale_exponent = _require_int(request, "scale_exponent", 0)
    seed = _require_int(request, "seed", 0)
    sample_count = _require_int(request, "sample_count", 2)
    group_size = _require_int(request, "group_size", 2)
    noise_bound = _require_int(request, "noise_bound", 0)
    if sample_count % group_size != 0:
        raise ValueError("sample_count must be divisible by group_size")
    slice_payload = request.get("slice")
    if not isinstance(slice_payload, dict):
        raise ValueError("slice must be an object")

    computation = compute_aggregates(
        seed=seed,
        scale_exponent=scale_exponent,
        sample_count=sample_count,
        group_size=group_size,
        noise_bound=noise_bound,
        slice_payload=slice_payload,
    )
    slice_version = int(slice_payload["version"])
    digest_input = {
        "service_id": service_id,
        "slice_version": slice_version,
        "aggregate_summary": computation.aggregates,
        "sample_digest": computation.sample_digest,
    }
    elapsed_ms = (time.monotonic_ns() - started_ns) // 1_000_000
    return {
        "request_id": request_id,
        "service_id": service_id,
        "slice_version": slice_version,
        "aggregate_summary": computation.aggregates,
        "sample_digest": computation.sample_digest,
        "digest": sha256_hex(digest_input),
        "elapsed_ms_monotonic": int(elapsed_ms),
    }


class _ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = False
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], service_id: str, seed: int):
        self.service_id = service_id
        self.seed = seed
        super().__init__(server_address, _RequestHandler)


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            line = self.rfile.readline(1_048_577)
            if not line:
                return
            if len(line) > 1_048_576:
                self._write(_request_error("$", "request line exceeds 1 MiB"))
                return
            try:
                request = json.loads(line.decode("utf-8"))
                if not isinstance(request, dict):
                    raise ValueError("request must be a JSON object")
                request_id = request.get("request_id")
                if not isinstance(request_id, str) or not request_id:
                    raise ValueError("request_id must be a non-empty string")
                if request.get("service_id") != self.server.service_id:
                    raise ValueError("service_id does not match this service")
                receipt = build_receipt(
                    request_id=request_id,
                    service_id=self.server.service_id,
                    request=request,
                )
            except Exception as exc:  # Return protocol failures, never crash the server.
                self._write(_request_error("request", str(exc)))
                continue
            self._write(receipt)

    def _write(self, payload: dict[str, Any]) -> None:
        try:
            self.wfile.write(canonical_json(payload).encode("utf-8") + b"\n")
            self.wfile.flush()
        except OSError:
            return


def run_service(service_id: str, port: int, seed: int) -> int:
    try:
        server = _ThreadingServer(("127.0.0.1", port), service_id=service_id, seed=seed)
    except OSError as exc:
        print(
            canonical_json(
                {
                    "event": "ERROR",
                    "code": "E_PORT",
                    "field": "bind",
                    "message": f"cannot bind 127.0.0.1:{port}: {exc}",
                }
            ),
            flush=True,
        )
        return 2
    print(
        canonical_json(
            {
                "event": "READY",
                "service_id": service_id,
                "host": "127.0.0.1",
                "port": port,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic loopback environment service")
    parser.add_argument("--service-id", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    return run_service(args.service_id, args.port, args.seed)


if __name__ == "__main__":
    sys.exit(main())
