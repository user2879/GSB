"""Startup-parameter driven fault injection.

Injection changes only input files or startup flags - never source code:

* ``schema_missing_table`` drops a sample table from the database file, which
  the schema gate then reports as ``E_SCHEMA``;
* ``config_quota_exceed`` rewrites the JSON pool quotas so their sum exceeds
  ``max_connections``, which config validation reports as ``E_CONFIG``;
* ``snapshot_missing_version`` strips ``param_version`` out of a snapshot
  document, which snapshot loading reports as ``E_SNAPSHOT``;
* ``replay_data_version`` bumps the sample sidecar version so replay reports
  ``E_REPLAY`` (a deterministic non-reproducibility demonstration).
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from .sample_db import version_sidecar_path


def inject_schema_missing_table(db_path: str | Path, table: str) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
    finally:
        conn.close()


def inject_config_quota_exceed(config_path: str | Path) -> Path:
    source = Path(config_path)
    target = source.with_name(source.stem + ".injected.json")
    data = json.loads(source.read_text(encoding="utf-8"))
    quotas = data["pool"]["workload_quotas"]
    first = next(iter(quotas))
    quotas[first] = int(data["pool"]["max_connections"]) + 1
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def inject_snapshot_missing_version(snapshot_path: str | Path) -> Path:
    source = Path(snapshot_path)
    target = source.with_name(source.stem + ".injected.json")
    document = json.loads(source.read_text(encoding="utf-8"))
    document["pool_config"].pop("param_version", None)
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return target


def inject_replay_data_version(db_path: str | Path) -> str:
    sidecar = version_sidecar_path(db_path)
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    previous = str(meta.get("sample_data_version", "1"))
    meta["sample_data_version"] = previous + "-tampered"
    sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return meta["sample_data_version"]
