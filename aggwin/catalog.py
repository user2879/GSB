"""SQLite catalog: monotonic query-parameter versions and canonical ids."""

import sqlite3

from .canon import digest_obj


def canonical_query_id(query):
    """Stable canonical identifier for a single query definition."""
    body = {"id": query.qid, "kind": query.kind, "table": query.table,
            "group_by": query.group_by, "partition_by": query.partition_by,
            "value_column": query.value_column,
            "aggregates": query.aggregates, "interval": query.interval,
            "max_rows": query.max_rows, "out_of_order": query.out_of_order}
    return "q-" + digest_obj(body)[:16]


def param_set_fingerprint(query_set):
    """Fingerprint of everything that influences results."""
    session = query_set.session
    body = {
        "queries": [
            {"id": q.qid, "kind": q.kind, "table": q.table,
             "group_by": q.group_by, "partition_by": q.partition_by,
             "value_column": q.value_column, "aggregates": q.aggregates,
             "interval": q.interval, "max_rows": q.max_rows,
             "out_of_order": q.out_of_order}
            for q in query_set.queries
        ],
        "scale": session["scale"],
        "truncation": session["truncation"],
    }
    return digest_obj(body)


class Catalog:
    """Maps parameter-set fingerprints to monotonically increasing versions."""

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS query_param_sets ("
            " fingerprint TEXT PRIMARY KEY,"
            " version INTEGER NOT NULL UNIQUE,"
            " created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS snapshots ("
            " snapshot_id TEXT PRIMARY KEY,"
            " path TEXT NOT NULL,"
            " query_param_version INTEGER NOT NULL,"
            " format_version INTEGER NOT NULL,"
            " digest TEXT NOT NULL,"
            " stale INTEGER NOT NULL DEFAULT 0,"
            " created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
        self.conn.commit()

    def param_version(self, fingerprint):
        row = self.conn.execute(
            "SELECT version FROM query_param_sets WHERE fingerprint=?",
            (fingerprint,)).fetchone()
        if row:
            return row[0]
        current = self.conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM query_param_sets"
        ).fetchone()[0]
        version = current + 1
        self.conn.execute(
            "INSERT INTO query_param_sets (fingerprint, version) VALUES (?,?)",
            (fingerprint, version))
        self.conn.commit()
        return version

    def register_snapshot(self, snapshot_id, path, query_param_version,
                          format_version, digest):
        self.conn.execute(
            "INSERT OR REPLACE INTO snapshots"
            " (snapshot_id, path, query_param_version, format_version, digest,"
            "  stale) VALUES (?,?,?,?,?,0)",
            (snapshot_id, path, query_param_version, format_version, digest))
        self.conn.commit()

    def mark_stale_before(self, query_param_version):
        """Mark every snapshot of older param versions as stale."""
        self.conn.execute(
            "UPDATE snapshots SET stale=1 WHERE query_param_version<?",
            (query_param_version,))
        self.conn.commit()

    def list_snapshots(self):
        return self.conn.execute(
            "SELECT snapshot_id, path, query_param_version, format_version,"
            " digest, stale, created_at FROM snapshots ORDER BY created_at"
        ).fetchall()

    def close(self):
        self.conn.close()
