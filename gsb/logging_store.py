from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import GSBError


class StructuredLogger:
    def __init__(self, database_path: str | Path | None = None, jsonl_path: str | Path | None = None) -> None:
        self.database_path = Path(database_path) if database_path else None
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self.database_path is not None:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()

    def _initialize(self) -> None:
        assert self.database_path is not None
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    level TEXT NOT NULL,
                    code TEXT NOT NULL,
                    event TEXT NOT NULL,
                    details_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_code ON events(code)"
            )
            connection.commit()

    def log(self, level: str, code: str, event: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": level.upper(),
            "code": code,
            "event": event,
            "details": details or {},
        }
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        if self.database_path is not None:
            with sqlite3.connect(self.database_path) as connection:
                connection.execute(
                    "INSERT INTO events(timestamp_utc, level, code, event, details_json) VALUES (?, ?, ?, ?, ?)",
                    (
                        record["timestamp_utc"],
                        record["level"],
                        record["code"],
                        record["event"],
                        json.dumps(record["details"], ensure_ascii=False, sort_keys=True, default=str),
                    ),
                )
                connection.commit()
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")
        return record

    def error(self, error: GSBError, event: str = "error") -> None:
        self.log(
            "ERROR",
            error.code,
            event,
            {"code": error.code, "message": error.message, "details": error.details},
        )

    def read_events(self, limit: int = 200) -> list[dict[str, Any]]:
        if self.database_path is None:
            return []
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                "SELECT timestamp_utc, level, code, event, details_json FROM events ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            {
                "timestamp_utc": row[0],
                "level": row[1],
                "code": row[2],
                "event": row[3],
                "details": json.loads(row[4]),
            }
            for row in rows
        ]


class NullLogger(StructuredLogger):
    def __init__(self) -> None:
        super().__init__(None, None)
