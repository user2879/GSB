"""SQLite 账本：缓存条目、租约、传播轨迹与请求事件的持久化。

账本携带整数格式版本与内容摘要；快照为账本文件的整体拷贝，
恢复时做版本识别与摘要校验，不一致则整体替换而非增量合并。
"""

import json
import os
import shutil
import sqlite3
import threading

from .errors import SnapshotError, VersionError
from .spec import content_digest

FORMAT_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS contents (
    path TEXT PRIMARY KEY,
    etag TEXT NOT NULL,
    body_hex TEXT NOT NULL,
    ttl_ms INTEGER NOT NULL,
    derived_from TEXT NOT NULL,
    combine TEXT
);
CREATE TABLE IF NOT EXISTS entries (
    node_id TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    path TEXT NOT NULL,
    etag TEXT NOT NULL,
    body_hex TEXT NOT NULL,
    ttl_ms INTEGER NOT NULL,
    write_tick_ms INTEGER NOT NULL,
    entry_version INTEGER NOT NULL,
    source TEXT NOT NULL,
    last_access_tick_ms INTEGER NOT NULL,
    PRIMARY KEY (node_id, cache_key)
);
CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    paths TEXT NOT NULL,
    content_versions TEXT NOT NULL,
    lease_version INTEGER NOT NULL,
    issued_mono_ms INTEGER NOT NULL,
    expires_mono_ms INTEGER NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL
);
"""


def _locked(fn):
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)

    wrapper.__name__ = fn.__name__
    return wrapper


class Ledger:
    def __init__(self, db_path):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    @_locked
    def close(self):
        self.conn.close()

    # ---- 初始化 ----
    @_locked
    def initialize(self, spec, policy_version):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM meta")
        cur.execute("DELETE FROM contents")
        cur.execute("DELETE FROM entries")
        cur.execute("DELETE FROM leases")
        cur.execute("DELETE FROM events")
        cur.execute(
            "INSERT INTO meta(key, value) VALUES ('format_version', ?)",
            (str(FORMAT_VERSION),),
        )
        cur.execute(
            "INSERT INTO meta(key, value) VALUES ('content_digest', ?)",
            (spec.digest,),
        )
        cur.execute(
            "INSERT INTO meta(key, value) VALUES ('policy_version', ?)",
            (str(policy_version),),
        )
        for c in spec.contents.values():
            cur.execute(
                "INSERT INTO contents(path, etag, body_hex, ttl_ms, derived_from, combine)"
                " VALUES (?,?,?,?,?,?)",
                (
                    c.path,
                    c.etag,
                    c.body.hex(),
                    c.ttl_ms,
                    json.dumps(list(c.derived_from)),
                    c.combine,
                ),
            )
        self.conn.commit()

    @_locked
    def get_meta(self, key):
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else None

    @_locked
    def set_meta(self, key, value):
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )
        self.conn.commit()

    # ---- 内容 ----
    @_locked
    def load_contents(self):
        from .spec import Content, ContentSpec

        rows = self.conn.execute("SELECT * FROM contents ORDER BY path").fetchall()
        contents = {}
        for r in rows:
            contents[r["path"]] = Content(
                path=r["path"],
                etag=r["etag"],
                body=bytes.fromhex(r["body_hex"]),
                ttl_ms=r["ttl_ms"],
                derived_from=tuple(json.loads(r["derived_from"])),
                combine=r["combine"],
            )
        return ContentSpec(
            spec_version=int(self.get_meta("spec_version") or 1),
            contents=contents,
            digest=content_digest(contents),
        )

    @_locked
    def replace_contents(self, spec):
        """整体替换内容集合与派生关系（不清空事件轨迹）。"""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM contents")
        for c in spec.contents.values():
            cur.execute(
                "INSERT INTO contents(path, etag, body_hex, ttl_ms, derived_from, combine)"
                " VALUES (?,?,?,?,?,?)",
                (
                    c.path,
                    c.etag,
                    c.body.hex(),
                    c.ttl_ms,
                    json.dumps(list(c.derived_from)),
                    c.combine,
                ),
            )
        cur.execute(
            "INSERT INTO meta(key, value) VALUES ('content_digest', ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (spec.digest,),
        )
        self.conn.commit()

    # ---- 缓存条目 ----
    @_locked
    def upsert_entry(self, node_id, entry):
        self.conn.execute(
            "INSERT INTO entries(node_id, cache_key, path, etag, body_hex, ttl_ms,"
            " write_tick_ms, entry_version, source, last_access_tick_ms)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(node_id, cache_key) DO UPDATE SET"
            " path=excluded.path, etag=excluded.etag, body_hex=excluded.body_hex,"
            " ttl_ms=excluded.ttl_ms, write_tick_ms=excluded.write_tick_ms,"
            " entry_version=excluded.entry_version, source=excluded.source,"
            " last_access_tick_ms=excluded.last_access_tick_ms",
            (
                node_id,
                entry.cache_key,
                entry.path,
                entry.etag,
                entry.body.hex(),
                entry.ttl_ms,
                entry.write_tick_ms,
                entry.entry_version,
                entry.source,
                entry.last_access_tick_ms,
            ),
        )
        self.conn.commit()

    @_locked
    def remove_entry(self, node_id, cache_key):
        self.conn.execute(
            "DELETE FROM entries WHERE node_id=? AND cache_key=?",
            (node_id, cache_key),
        )
        self.conn.commit()

    @_locked
    def entries_for(self, node_id):
        from .cache import Entry

        rows = self.conn.execute(
            "SELECT * FROM entries WHERE node_id=? ORDER BY cache_key", (node_id,)
        ).fetchall()
        return [
            Entry(
                cache_key=r["cache_key"],
                path=r["path"],
                etag=r["etag"],
                body=bytes.fromhex(r["body_hex"]),
                ttl_ms=r["ttl_ms"],
                write_tick_ms=r["write_tick_ms"],
                entry_version=r["entry_version"],
                source=r["source"],
                last_access_tick_ms=r["last_access_tick_ms"],
            )
            for r in rows
        ]

    # ---- 租约 ----
    @_locked
    def record_lease(self, lease):
        self.conn.execute(
            "INSERT INTO leases(lease_id, node_id, paths, content_versions,"
            " lease_version, issued_mono_ms, expires_mono_ms, status)"
            " VALUES (?,?,?,?,?,?,?,?)"
            " ON CONFLICT(lease_id) DO UPDATE SET"
            " lease_version=excluded.lease_version,"
            " expires_mono_ms=excluded.expires_mono_ms,"
            " status=excluded.status",
            (
                lease["lease_id"],
                lease["node_id"],
                json.dumps(lease["paths"]),
                json.dumps(lease["content_versions"], sort_keys=True),
                lease["lease_version"],
                lease["issued_mono_ms"],
                lease["expires_mono_ms"],
                lease["status"],
            ),
        )
        self.conn.commit()

    @_locked
    def leases(self):
        rows = self.conn.execute("SELECT * FROM leases ORDER BY lease_id").fetchall()
        return [dict(r) for r in rows]

    # ---- 事件（请求结果 / 逐出 / 传播 / 调度尝试）----
    @_locked
    def record_event(self, kind, payload):
        cur = self.conn.execute(
            "INSERT INTO events(kind, payload) VALUES (?, ?)",
            (kind, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        )
        self.conn.commit()
        return cur.lastrowid

    @_locked
    def events(self, kind=None):
        if kind:
            rows = self.conn.execute(
                "SELECT seq, kind, payload FROM events WHERE kind=? ORDER BY seq",
                (kind,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT seq, kind, payload FROM events ORDER BY seq"
            ).fetchall()
        return [
            {"seq": r["seq"], "kind": r["kind"], "payload": json.loads(r["payload"])}
            for r in rows
        ]

    # ---- 快照 ----
    @_locked
    def save_snapshot(self, snapshot_path):
        self.conn.commit()
        shutil.copyfile(self.db_path, snapshot_path)
        return snapshot_path


def _read_meta_map(conn):
    return {r[0]: r[1] for r in conn.execute("SELECT key, value FROM meta")}


def open_snapshot(snapshot_path):
    """打开快照并做版本识别与完整性校验。损坏/截断 -> E_SNAPSHOT。"""
    if not os.path.exists(snapshot_path):
        raise SnapshotError("快照文件不存在", snapshot=snapshot_path)
    try:
        conn = sqlite3.connect(f"file:{snapshot_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise SnapshotError(
                "快照完整性校验失败", snapshot=snapshot_path, integrity=integrity
            )
        meta = _read_meta_map(conn)
    except SnapshotError:
        raise
    except sqlite3.DatabaseError as exc:
        raise SnapshotError(
            "快照文件损坏或截断", snapshot=snapshot_path, error=str(exc)
        )
    fmt = meta.get("format_version")
    if fmt != str(FORMAT_VERSION):
        conn.close()
        raise SnapshotError(
            "快照格式版本不兼容",
            snapshot=snapshot_path,
            snapshot_format_version=fmt,
            expected=FORMAT_VERSION,
        )
    return conn, meta


def restore_snapshot(snapshot_path, ledger_path, current_spec, logger=None):
    """整体替换恢复。

    1. 校验快照格式版本与完整性（E_SNAPSHOT）；
    2. 校验快照内内容摘要与其内容集合一致（E_VERSION）；
    3. 用快照整体替换当前账本；
    4. 比较快照内容集合与磁盘当前声明：不一致则整体替换全部
       条目、租约与派生关系，并稳定列出新增/删除/版本变更路径。
    旧快照保留为只读。
    """
    snap_conn, meta = open_snapshot(snapshot_path)
    from .spec import Content

    rows = snap_conn.execute("SELECT * FROM contents ORDER BY path").fetchall()
    snap_contents = {}
    for r in rows:
        snap_contents[r["path"]] = Content(
            path=r["path"],
            etag=r["etag"],
            body=bytes.fromhex(r["body_hex"]),
            ttl_ms=r["ttl_ms"],
            derived_from=tuple(json.loads(r["derived_from"])),
            combine=r["combine"],
        )
    snap_digest = content_digest(snap_contents)
    if snap_digest != meta.get("content_digest"):
        snap_conn.close()
        raise VersionError(
            "快照内容摘要与内容集合不一致",
            snapshot=snapshot_path,
            recorded_digest=meta.get("content_digest"),
            computed_digest=snap_digest,
        )
    snap_conn.close()

    # 整体替换账本文件
    if os.path.exists(ledger_path):
        os.remove(ledger_path)
    shutil.copyfile(snapshot_path, ledger_path)
    os.chmod(snapshot_path, 0o444)  # 旧快照保留为只读

    ledger = Ledger(ledger_path)
    snap_etags = {p: c.etag for p, c in snap_contents.items()}
    disk_etags = {p: c.etag for p, c in current_spec.contents.items()}
    added = sorted(set(disk_etags) - set(snap_etags))
    removed = sorted(set(snap_etags) - set(disk_etags))
    changed = sorted(
        p for p in set(snap_etags) & set(disk_etags) if snap_etags[p] != disk_etags[p]
    )
    replaced = False
    if added or removed or changed:
        # 版本不一致隔离：整体替换条目、租约与派生关系，不做增量合并
        cur = ledger.conn.cursor()
        cur.execute("DELETE FROM entries")
        cur.execute("DELETE FROM leases")
        ledger.replace_contents(current_spec)
        replaced = True
        if logger:
            logger.log(
                "ledger_version_isolation",
                code="E_VERSION",
                added=added,
                removed=removed,
                changed=changed,
            )
    diff = {"added": added, "removed": removed, "changed": changed}
    return ledger, diff, replaced
