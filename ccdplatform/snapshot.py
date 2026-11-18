"""会话快照：SQLite 持久化、内容摘要、版本识别与整体替换恢复。"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import GridError
from .fixedpoint import GridSpec
from .graph import DependencyGraph
from .model import (
    CompiledItem,
    DispatchPlan,
    DispatchRecord,
    Receipt,
    StateTransition,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS versions (
    version INTEGER PRIMARY KEY,
    content_digest TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    grid_json TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    fingerprints_json TEXT NOT NULL,
    graph_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS compiled_items (
    version INTEGER NOT NULL,
    key TEXT NOT NULL,
    item_json TEXT NOT NULL,
    canonical_b64 TEXT NOT NULL,
    PRIMARY KEY (version, key)
);
CREATE TABLE IF NOT EXISTS receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL,
    receipt_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dispatch_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL,
    record_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL,
    transition_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS error_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER,
    code TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def content_digest(
    version: int,
    grid: GridSpec,
    compiled: Dict[str, CompiledItem],
    plan: DispatchPlan,
    graph: DependencyGraph,
) -> str:
    """对版本状态的规范化字节做 sha256，作为内容摘要。"""
    h = hashlib.sha256()
    h.update(str(version).encode("ascii"))
    h.update(json.dumps(grid.to_json(), sort_keys=True).encode("ascii"))
    for key in sorted(compiled):
        h.update(compiled[key].canonical_bytes())
    h.update(
        json.dumps(plan.to_json(), sort_keys=True, ensure_ascii=True).encode("ascii")
    )
    h.update(json.dumps(graph.edges, sort_keys=True).encode("ascii"))
    return h.hexdigest()


@dataclass
class VersionSnapshot:
    version: int
    content_digest: str
    grid: GridSpec
    plan: DispatchPlan
    compiled: Dict[str, CompiledItem]
    fingerprints: Dict[str, bytes]
    graph: DependencyGraph
    receipts: List[Receipt]
    records: List[DispatchRecord]
    transitions: List[StateTransition]
    manifest_json: str


class SnapshotStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------- meta / grid
    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def load_grid(self) -> GridSpec:
        raw = self.get_meta("grid")
        if raw is None:
            raise GridError(
                "快照缺失精度格度参数 grid",
                {"missing_field": "meta.grid"},
            )
        return GridSpec.from_json(json.loads(raw))

    # --------------------------------------------------------------- versions
    def list_versions(self) -> List[int]:
        rows = self.conn.execute(
            "SELECT version FROM versions ORDER BY version"
        ).fetchall()
        return [int(r["version"]) for r in rows]

    def latest_version(self) -> Optional[int]:
        versions = self.list_versions()
        return versions[-1] if versions else None

    def save_version(
        self,
        version: int,
        grid: GridSpec,
        compiled: Dict[str, CompiledItem],
        plan: DispatchPlan,
        graph: DependencyGraph,
        fingerprints: Dict[str, bytes],
        manifest_json: str,
        digest: str,
        created_at: str,
    ) -> None:
        import base64

        self.conn.execute(
            "INSERT OR REPLACE INTO versions"
            "(version,content_digest,manifest_json,grid_json,plan_json,"
            "fingerprints_json,graph_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (
                version,
                digest,
                manifest_json,
                json.dumps(grid.to_json()),
                json.dumps(plan.to_json(), ensure_ascii=True),
                json.dumps(
                    {k: base64.b64encode(v).decode("ascii") for k, v in fingerprints.items()}
                ),
                json.dumps(graph.edges),
                created_at,
            ),
        )
        self.conn.execute("DELETE FROM compiled_items WHERE version=?", (version,))
        self.conn.executemany(
            "INSERT INTO compiled_items(version,key,item_json,canonical_b64)"
            " VALUES(?,?,?,?)",
            [
                (
                    version,
                    key,
                    json.dumps(compiled[key].to_json(), ensure_ascii=True),
                    base64.b64encode(compiled[key].canonical_bytes()).decode("ascii"),
                )
                for key in sorted(compiled)
            ],
        )
        self.conn.commit()

    def append_receipts(self, version: int, receipts: List[Receipt]) -> None:
        self.conn.executemany(
            "INSERT INTO receipts(version,receipt_json) VALUES(?,?)",
            [(version, json.dumps(r.to_json(), ensure_ascii=True)) for r in receipts],
        )
        self.conn.commit()

    def append_records(self, version: int, records: List[DispatchRecord]) -> None:
        self.conn.executemany(
            "INSERT INTO dispatch_records(version,record_json) VALUES(?,?)",
            [(version, json.dumps(r.to_json(), ensure_ascii=True)) for r in records],
        )
        self.conn.commit()

    def append_transitions(
        self, version: int, transitions: List[StateTransition]
    ) -> None:
        self.conn.executemany(
            "INSERT INTO transitions(version,transition_json) VALUES(?,?)",
            [
                (version, json.dumps(t.to_json(), ensure_ascii=True))
                for t in transitions
            ],
        )
        self.conn.commit()

    def append_error(
        self, code: str, payload: Dict[str, Any], version: Optional[int]
    ) -> None:
        from datetime import datetime, timezone

        self.conn.execute(
            "INSERT INTO error_events(version,code,payload_json,created_at)"
            " VALUES(?,?,?,?)",
            (
                version,
                code,
                json.dumps(payload, ensure_ascii=True, sort_keys=True),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def load_version(self, version: int) -> VersionSnapshot:
        import base64

        row = self.conn.execute(
            "SELECT * FROM versions WHERE version=?", (version,)
        ).fetchone()
        if row is None:
            raise KeyError(f"快照中不存在版本 {version}")
        grid = GridSpec.from_json(json.loads(row["grid_json"]))
        plan = DispatchPlan.from_json(json.loads(row["plan_json"]))
        edges = {
            k: tuple(v) for k, v in json.loads(row["graph_json"]).items()
        }
        graph = DependencyGraph(edges)
        fp_raw = json.loads(row["fingerprints_json"])
        fingerprints = {
            k: base64.b64decode(v) for k, v in fp_raw.items()
        }
        item_rows = self.conn.execute(
            "SELECT item_json FROM compiled_items WHERE version=? ORDER BY key",
            (version,),
        ).fetchall()
        compiled: Dict[str, CompiledItem] = {}
        for r in item_rows:
            ci = CompiledItem.from_json(json.loads(r["item_json"]))
            compiled[ci.key] = ci
        receipts = [
            Receipt.from_json(json.loads(r["receipt_json"]))
            for r in self.conn.execute(
                "SELECT receipt_json FROM receipts WHERE version=? ORDER BY id",
                (version,),
            ).fetchall()
        ]
        records = [
            DispatchRecord(**json.loads(r["record_json"]))
            for r in self.conn.execute(
                "SELECT record_json FROM dispatch_records WHERE version=? ORDER BY id",
                (version,),
            ).fetchall()
        ]
        transitions = [
            StateTransition(**json.loads(r["transition_json"]))
            for r in self.conn.execute(
                "SELECT transition_json FROM transitions WHERE version=? ORDER BY id",
                (version,),
            ).fetchall()
        ]
        return VersionSnapshot(
            version=version,
            content_digest=row["content_digest"],
            grid=grid,
            plan=plan,
            compiled=compiled,
            fingerprints=fingerprints,
            graph=graph,
            receipts=receipts,
            records=records,
            transitions=transitions,
            manifest_json=row["manifest_json"],
        )

    def digest_of(self, version: int) -> Optional[str]:
        row = self.conn.execute(
            "SELECT content_digest FROM versions WHERE version=?", (version,)
        ).fetchone()
        return row["content_digest"] if row else None
