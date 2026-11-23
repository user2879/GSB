"""Deterministic local sample generation (integer seeds, numpy RandomState).

Running this module twice with the same seed file produces byte-identical
JSON/NDJSON outputs. No network or external account is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .util import write_json

SERVICES = ["svc-auth", "svc-gateway", "svc-order", "svc-payment",
            "svc-inventory", "svc-notify"]
# Edges: caller -> dependency. acyclic in topological order.
DEPENDS_ON = {
    "svc-gateway": ["svc-auth", "svc-order"],
    "svc-order": ["svc-payment", "svc-inventory"],
    "svc-payment": ["svc-inventory"],
    "svc-notify": ["svc-order"],
    "svc-auth": [],
    "svc-inventory": [],
}
SEVERITIES = ["critical", "major", "minor", "warning", "info"]


def build_graph_doc() -> dict:
    return {"graph_version": 1,
            "services": [{"service_id": sid,
                          "depends_on": sorted(DEPENDS_ON[sid])}
                         for sid in SERVICES]}


def build_rules_doc(seed: int, declared_version: int = 1) -> dict:
    rng = np.random.RandomState(seed + 101)
    windows = [0, 500, 1000, 2000, 5000]
    fps = [f"fp_{i:012x}" for i in range(4)]
    return {
        "declared_version": declared_version,
        "schema_version": 3,
        "default_silence_window_ms": 1000,
        "window_version": 1,
        "rule_version": 1,
        "silence_overrides": {
            "by_fingerprint": {
                fps[1]: int(rng.choice(windows)),
                fps[2]: int(rng.choice(windows)),
            },
            "by_service": {"svc-payment": 2000},
        },
        "root_cause_rules": [
            {"rule_id": "rc-upstream-all", "match_service": "*",
             "include_upstream": True, "include_downstream": False,
             "max_layer": 3},
            {"rule_id": "rc-gateway-both", "match_service": "svc-gateway",
             "include_upstream": True, "include_downstream": True,
             "max_layer": 2},
        ],
    }


def build_seed_doc(seed: int) -> dict:
    return {
        "master_seed": int(seed),
        "alerts_per_source": 18,
        "source_count": 4,
        "start_timestamp_ms": 1_700_000_000_000,
        "timestamp_jitter_choices_ms": [0, 100, 250, 500, 900, 1200, 3000],
        "numpy": np.__version__,
    }


def _stream_rows(seed_doc: dict, rules_doc: dict) -> list[dict]:
    seed = int(seed_doc["master_seed"])
    per_source = int(seed_doc["alerts_per_source"])
    source_count = int(seed_doc["source_count"])
    ts = int(seed_doc["start_timestamp_ms"])
    jitter = seed_doc["timestamp_jitter_choices_ms"]
    fps = [f"fp_{i:012x}" for i in range(4)]
    rows: list[dict] = []
    clock = ts
    for source in range(source_count):
        rng = np.random.RandomState(seed + source * 97)
        service = SERVICES[source % len(SERVICES)]
        source_clock = clock
        for idx in range(per_source):
            source_clock += int(rng.choice(jitter))
            severity = SEVERITIES[int(rng.randint(0, len(SEVERITIES)))]
            version = ("v1", "v2", "v3")[int(rng.randint(0, 3))]
            record = {"version": version,
                      "timestamp_ms": int(source_clock),
                      "service_id": service,
                      "severity": severity}
            if version in ("v2", "v3"):
                record["fingerprint"] = fps[int(rng.randint(0, len(fps)))]
                record["silence_window_ms"] = int(
                    rng.choice([0, 500, 1000, 2000]))
            if version == "v3":
                refs = [SERVICES[int(rng.randint(0, len(SERVICES)))]]
                record["root_cause_refs"] = refs
                record["strategy_id"] = f"strategy_{int(rng.randint(0, 3))}"
                if rng.randint(0, 2):
                    record["labels"] = {"env": "prod",
                                        "zone": f"z{int(rng.randint(0, 3))}"}
            rows.append(record)
    # Global monotonic merge of the per-source streams (stable by ts then row).
    rows.sort(key=lambda r: (r["timestamp_ms"], r["service_id"],
                             r["severity"]))
    return rows


def write_ndjson(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True,
                                ensure_ascii=False) + "\n")


def generate_all(out_dir: str | Path, seed: int = 20261124) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graph_doc = build_graph_doc()
    rules_doc = build_rules_doc(seed)
    seed_doc = build_seed_doc(seed)
    rows = _stream_rows(seed_doc, rules_doc)
    paths = {
        "graph": out / "service_graph.json",
        "rules": out / "root_cause_rules.json",
        "seeds": out / "seeds.json",
        "stream": out / "alerts.ndjson",
    }
    write_json(paths["graph"], graph_doc)
    write_json(paths["rules"], rules_doc)
    write_json(paths["seeds"], seed_doc)
    write_ndjson(paths["stream"], rows)
    return {k: str(v) for k, v in paths.items()}


# --------------------------------------------------------------- fault samples
def generate_injected(base_dir: str | Path) -> dict:
    """Generate three injected-invalid samples for the required error codes."""
    base = Path(base_dir)
    inj = base / "injected"
    inj.mkdir(parents=True, exist_ok=True)
    paths = generate_all(base / "base", seed=4242)

    # 1) graph cycle -> E_SPEC
    cyclic = build_graph_doc()
    cyclic["services"][0]["depends_on"] = ["svc-inventory"]
    # svc-inventory -> svc-gateway creates inventory -> gateway -> inventory
    for svc in cyclic["services"]:
        if svc["service_id"] == "svc-inventory":
            svc["depends_on"] = ["svc-gateway"]
    write_json(inj / "graph_cycle.json", cyclic)

    # 2) negative silence window -> E_FIELD
    neg_rows = []
    with open(paths["stream"], encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            rec = json.loads(line)
            if rec.get("version") in ("v2", "v3"):
                rec["silence_window_ms"] = -1
                neg_rows.append(json.dumps(rec, sort_keys=True))
                neg_rows.extend(fh.read().splitlines())
                break
            neg_rows.append(json.dumps(rec, sort_keys=True))
    with open(inj / "alerts_negative_window.ndjson", "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write("\n".join(neg_rows) + "\n")

    # 3) snapshot tampering is performed at runtime on the SQLite file; the
    # marker file documents the injection parameter.
    write_json(inj / "snapshot_injection.json",
               {"table": "aggregates", "tamper": "record_json",
                "expected_code": "E_SNAPSHOT"})
    return {"injected_dir": str(inj), **paths}
