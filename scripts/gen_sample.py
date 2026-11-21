"""Deterministic sample generator (NumPy RNG, fixed seed).

Writes samples/spec.json and samples/requests.ndjson using loopback
high ports only. The sample exercises every path: a healthy instance,
a delayed instance (probe timeouts -> unhealthy -> ejected), an
error-response instance, and a refused instance with no tenants.
"""

import argparse
import json
import os

import numpy as np

BASE_PORT = 18301


def build_spec() -> dict:
    return {
        "tenant_set_version": 1,
        "instance_set_version": 1,
        "rule_version": 1,
        "global_concurrency": 2,
        "total_share_cap": 100,
        "duration_factor_ms": 20,
        "probe_policy": {
            "policy_version": 1,
            "interval_ms": 100,
            "timeout_ms": 150,
            "failure_threshold": 2,
            "eject_threshold": 3,
        },
        "tenants": [
            {"tenant_id": "alpha", "share": 40, "max_concurrency": 1, "burst": 1},
            {"tenant_id": "beta", "share": 25, "max_concurrency": 1, "burst": 1},
            {"tenant_id": "gamma", "share": 20, "max_concurrency": 2, "burst": 0},
        ],
        "instances": [
            {"instance_id": "svc-a", "host": "127.0.0.1", "port": BASE_PORT,
             "weight": 2, "behavior": "normal", "hosts": ["alpha"]},
            {"instance_id": "svc-b", "host": "127.0.0.1", "port": BASE_PORT + 1,
             "weight": 1, "behavior": "delay", "delay_ms": 5000, "hosts": ["beta"]},
            {"instance_id": "svc-c", "host": "127.0.0.1", "port": BASE_PORT + 2,
             "weight": 1, "behavior": "error", "hosts": ["gamma"]},
            {"instance_id": "svc-d", "host": "127.0.0.1", "port": BASE_PORT + 3,
             "weight": 1, "behavior": "refused", "hosts": []},
        ],
    }


def build_requests(seed: int, count: int) -> list:
    rng = np.random.default_rng(seed)
    tenants = np.array(["alpha", "alpha", "beta", "gamma"])
    picks = tenants[rng.integers(0, len(tenants), size=count)]
    demands = rng.integers(1, 7, size=count)
    gaps = rng.integers(0, 40, size=count)
    arrivals = np.cumsum(gaps)
    requests = []
    for idx in range(count):
        requests.append({
            "request_id": f"req-{idx:03d}",
            "tenant_id": str(picks[idx]),
            "demand": int(demands[idx]),
            "arrival_ms": int(arrivals[idx]),
        })
    return requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="samples")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    spec_path = os.path.join(args.out, "spec.json")
    req_path = os.path.join(args.out, "requests.ndjson")
    with open(spec_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(build_spec(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(req_path, "w", encoding="utf-8", newline="\n") as fh:
        for req in build_requests(args.seed, args.count):
            fh.write(json.dumps(req, sort_keys=True) + "\n")
    print(f"wrote {spec_path} and {req_path} (seed={args.seed}, count={args.count})")


if __name__ == "__main__":
    main()
