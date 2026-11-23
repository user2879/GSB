"""Generate the deterministic local sample into data/.

Computes signatures with the declared rule so the sample exercises:
allowed, E_RATE (bucket exhausted / cost > capacity), E_AUTH mismatch,
E_AUTH expired, old-version-in-window, E_TIMEOUT and E_RECEIPT paths.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from workbench.authn import expected_signature  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

SERVICES = {
    "services": [
        {"service_id": "svc-a", "host": "127.0.0.1", "port": 23101,
         "capacity_tokens": 4, "refill_interval_ms": 50,
         "cost_tokens": 1, "signature_window_ms": 2000,
         "bucket_keys": ["alpha", "beta"]},
        {"service_id": "svc-b", "host": "127.0.0.1", "port": 23102,
         "capacity_tokens": 2, "refill_interval_ms": 100,
         "cost_tokens": 1, "signature_window_ms": 2000,
         "bucket_keys": ["gamma"]},
        {"service_id": "svc-c", "host": "127.0.0.1", "port": 23103,
         "capacity_tokens": 2, "refill_interval_ms": 500,
         "cost_tokens": 2, "signature_window_ms": 1000,
         "bucket_keys": ["delta"]},
    ]
}

KEYS = {
    "keys": {
        "svc-a": [
            {"version": 1, "secret": "svc-a-secret-v1", "retire_ms": 1000},
            {"version": 2, "secret": "svc-a-secret-v2", "retire_ms": 5000},
            {"version": 3, "secret": "svc-a-secret-v3", "retire_ms": None},
        ],
        "svc-b": [
            {"version": 1, "secret": "svc-b-secret-v1", "retire_ms": 2000},
            {"version": 2, "secret": "svc-b-secret-v2", "retire_ms": None},
        ],
        "svc-c": [
            {"version": 1, "secret": "svc-c-secret-v1", "retire_ms": None},
        ],
    }
}

# (client_key, service, arrival_ms, bucket_key, cred_version, sig_override)
SCRIPT = [
    ("client-1", "svc-a", 0, "alpha", 3, None),
    ("client-1", "svc-a", 10, "alpha", 3, None),
    ("client-2", "svc-a", 20, "alpha", 3, None),
    ("client-1", "svc-a", 30, "alpha", 3, None),
    ("client-1", "svc-a", 40, "alpha", 3, None),
    ("client-1", "svc-a", 200, "alpha", 3, None),
    ("client-3", "svc-a", 2500, "beta", 1, None),
    ("client-3", "svc-a", 4000, "beta", 1, None),
    ("client-1", "svc-a", 4100, "beta", 3, "bad_sig"),
    ("client-9", "svc-a", 4200, "beta", 9, None),
    ("client-1", "svc-b", 4300, "gamma", 2, None),
    ("client-2", "svc-b", 4310, "gamma", 2, None),
    ("client-1", "svc-b", 4320, "gamma", 2, None),
    ("client-1", "svc-b", 4800, "gamma", 2, None),
    ("__timeout__", "svc-b", 4900, "gamma", 2, None),
    ("__badreceipt__", "svc-b", 5000, "gamma", 2, None),
    ("client-1", "svc-c", 5100, "delta", 1, None),
    ("client-1", "svc-c", 5700, "delta", 1, None),
    ("client-2", "svc-c", 5800, "delta", 1, None),
    ("client-2", "svc-c", 6900, "delta", 1, None),
]


def main():
    requests = []
    for seq, (client, sid, arrival, bk, ver, sig) in enumerate(SCRIPT, 1):
        if sig is None:
            secret = None
            for kv in KEYS["keys"][sid]:
                if kv["version"] == ver:
                    secret = kv["secret"]
            if secret is None:
                sig = "0" * 64
            else:
                sig = expected_signature(secret, client, sid, seq, ver)
        requests.append({
            "seq": seq, "client_key": client, "service_id": sid,
            "arrival_ms": arrival, "bucket_key": bk,
            "cred_version": ver, "signature": sig})
    os.makedirs(DATA, exist_ok=True)
    for name, obj in (("services.json", SERVICES), ("keys.json", KEYS),
                      ("requests.json", {"requests": requests})):
        path = os.path.join(DATA, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1)
            fh.write("\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
