"""Shared helpers for the workbench test suite."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from workbench.authn import expected_signature  # noqa: E402

TEST_PORTS = {"svc-a": 24101, "svc-b": 24102, "svc-c": 24103}


def make_spec_files(tmpdir, script=None, services=None, keys=None):
    """Write services.json / keys.json / requests.json into tmpdir."""
    if services is None:
        services = [
            {"service_id": "svc-a", "host": "127.0.0.1",
             "port": TEST_PORTS["svc-a"], "capacity_tokens": 3,
             "refill_interval_ms": 50, "cost_tokens": 1,
             "signature_window_ms": 1000, "bucket_keys": ["alpha"]},
            {"service_id": "svc-b", "host": "127.0.0.1",
             "port": TEST_PORTS["svc-b"], "capacity_tokens": 2,
             "refill_interval_ms": 100, "cost_tokens": 1,
             "signature_window_ms": 1000, "bucket_keys": ["beta"]},
        ]
    if keys is None:
        keys = {
            "svc-a": [
                {"version": 1, "secret": "a-v1", "retire_ms": 1000},
                {"version": 2, "secret": "a-v2", "retire_ms": None},
            ],
            "svc-b": [
                {"version": 1, "secret": "b-v1", "retire_ms": None},
            ],
        }
    if script is None:
        script = [
            ("c1", "svc-a", 0, "alpha", 2, None),
            ("c1", "svc-a", 10, "alpha", 2, None),
            ("c1", "svc-a", 20, "alpha", 2, None),
            ("c1", "svc-a", 30, "alpha", 2, None),
            ("c1", "svc-a", 200, "alpha", 2, None),
            ("c2", "svc-b", 300, "beta", 1, None),
            ("c2", "svc-b", 310, "beta", 1, None),
            ("c2", "svc-b", 320, "beta", 1, None),
        ]
    requests = []
    for seq, (client, sid, arrival, bk, ver, sig) in enumerate(script, 1):
        if sig is None:
            secret = None
            for kv in keys[sid]:
                if kv["version"] == ver:
                    secret = kv["secret"]
            sig = ("0" * 64 if secret is None else
                   expected_signature(secret, client, sid, seq, ver))
        requests.append({"seq": seq, "client_key": client,
                         "service_id": sid, "arrival_ms": arrival,
                         "bucket_key": bk, "cred_version": ver,
                         "signature": sig})
    _write(tmpdir, "services.json", {"services": services})
    _write(tmpdir, "keys.json", {"keys": keys})
    _write(tmpdir, "requests.json", {"requests": requests})
    return tmpdir


def _write(tmpdir, name, obj):
    with open(os.path.join(tmpdir, name), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")
