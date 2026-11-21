"""Shared test helpers: free-port allocation and spec construction."""

import json
import os
import socket
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def free_ports(count, start=19100):
    ports = []
    candidate = start
    while len(ports) < count:
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", candidate))
            except OSError:
                candidate += 1
                continue
        ports.append(candidate)
        candidate += 1
    return ports


def make_spec_dict(ports, **overrides):
    spec = {
        "tenant_set_version": 1,
        "instance_set_version": 1,
        "rule_version": 1,
        "global_concurrency": 2,
        "total_share_cap": 100,
        "duration_factor_ms": 20,
        "probe_policy": {
            "policy_version": 1,
            "interval_ms": 60,
            "timeout_ms": 120,
            "failure_threshold": 2,
            "eject_threshold": 3,
        },
        "tenants": [
            {"tenant_id": "alpha", "share": 30, "max_concurrency": 1, "burst": 1},
            {"tenant_id": "beta", "share": 20, "max_concurrency": 1, "burst": 0},
        ],
        "instances": [
            {"instance_id": "svc-a", "host": "127.0.0.1", "port": ports[0],
             "weight": 1, "behavior": "normal", "hosts": ["alpha"]},
            {"instance_id": "svc-b", "host": "127.0.0.1", "port": ports[1],
             "weight": 1, "behavior": "error", "hosts": ["beta"]},
        ],
    }
    spec.update(overrides)
    return spec


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def write_ndjson(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def make_requests(count=12, tenant_cycle=("alpha", "beta"), gap_ms=25, demand=2):
    rows = []
    for idx in range(count):
        rows.append({
            "request_id": f"r{idx:03d}",
            "tenant_id": tenant_cycle[idx % len(tenant_cycle)],
            "demand": demand,
            "arrival_ms": idx * gap_ms,
        })
    return rows
