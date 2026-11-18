"""High-level facade used by scripts, CLI and the Tkinter workbench."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

from .diagnostics import DiagnosisEngine, generate_candidates
from .errors import WorkbenchError, fail
from .logging_utils import JsonlLogger
from .query import Query, parse_select
from .samples import DEFAULT_SAMPLES_DIR, load_queries, load_samples
from .session import Session


F = TypeVar("F", bound=Callable[..., Any])


def _logged(method: F) -> F:
    @functools.wraps(method)
    def wrapper(self: "Workbench", *args: Any, **kwargs: Any) -> Any:
        try:
            return method(self, *args, **kwargs)
        except WorkbenchError as exc:
            self.logger.error(
                "operation_failed",
                exc.message,
                code=exc.code,
                details=exc.details,
                operation=method.__name__,
                session_path=str(self.path),
            )
            raise

    return wrapper  # type: ignore[return-value]


class Workbench:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.logger = session.logger
        self.engine = DiagnosisEngine(session)

    @property
    def path(self) -> Path:
        return self.session.path

    @property
    def conn(self):
        return self.session.conn

    @classmethod
    def create(
        cls,
        path: str | Path,
        name: str | None = None,
        *,
        overwrite: bool = False,
        logger_path: str | Path | None = None,
    ) -> "Workbench":
        logger = JsonlLogger(logger_path) if logger_path else None
        return cls(Session.create(path, name=name, overwrite=overwrite, logger=logger))

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        expected_versions: Mapping[str, int] | None = None,
        logger_path: str | Path | None = None,
    ) -> "Workbench":
        logger = JsonlLogger(logger_path) if logger_path else None
        return cls(Session.open(path, expected_versions=expected_versions, logger=logger))

    @_logged
    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "Workbench":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @_logged
    def info(self) -> dict[str, Any]:
        return self.session.session_info()

    @_logged
    def versions(self) -> dict[str, Any]:
        return self.session.version_info()

    @_logged
    def save_as(self, path: str | Path, name: str | None = None) -> "Workbench":
        new_session = self.session.save_as(path, name=name)
        return Workbench(new_session)

    @_logged
    def initialize_samples(
        self,
        samples_dir: str | Path = DEFAULT_SAMPLES_DIR,
        *,
        corrupt_input: bool = False,
    ) -> dict[str, Any]:
        return load_samples(self.session, samples_dir, corrupt_input=corrupt_input)

    @_logged
    def sample_queries(self, samples_dir: str | Path = DEFAULT_SAMPLES_DIR) -> list[dict[str, Any]]:
        return load_queries(samples_dir)

    @_logged
    def create_table(self, sql: str) -> dict[str, Any]:
        return self.session.create_domain_table(sql)

    @_logged
    def append_rows(
        self, table: str, columns: Sequence[str], rows: Iterable[Sequence[Any]]
    ) -> dict[str, Any]:
        return self.session.append_rows(table, columns, rows)

    @_logged
    def add_index(
        self,
        table: str,
        columns: Sequence[str],
        *,
        name: str | None = None,
        unique: bool = False,
    ) -> dict[str, Any]:
        return self.session.add_index(table, columns, name=name, unique=unique)

    @_logged
    def drop_index(self, name: str) -> dict[str, Any]:
        return self.session.drop_index(name)

    @_logged
    def refresh_statistics(self, table: str | None = None) -> dict[str, Any]:
        return self.session.refresh_statistics(table)

    @_logged
    def indexes(self) -> list[dict[str, Any]]:
        return self.session.list_indexes()

    @_logged
    def schema(self) -> list[dict[str, Any]]:
        return self.session.schema_snapshot()

    @_logged
    def set_thresholds(self, threshold_ms: int, threshold_rows: int) -> dict[str, int]:
        return self.session.set_thresholds(threshold_ms, threshold_rows)

    @_logged
    def thresholds(self) -> dict[str, int]:
        return self.session.thresholds()

    @_logged
    def execute_select(self, sql: str, *, timeout_ms: int | float | None = None) -> dict[str, Any]:
        return self.session.execute_select(sql, timeout_ms=timeout_ms)

    @_logged
    def execute_query(self, query: Query, *, timeout_ms: int | float | None = None) -> dict[str, Any]:
        execution = self.session.execute_observed(query, kind="original", timeout_ms=timeout_ms)
        if execution.outcome == "timeout":
            raise fail(
                "E_TIMEOUT",
                execution.error_message or "query timed out",
                {"query_id": execution.query_id, "elapsed_ms": execution.elapsed_ms},
            )
        if execution.outcome != "success":
            raise fail("E_DATA", execution.error_message or "query failed")
        return execution.to_dict()

    @_logged
    def slow_queries(
        self,
        threshold_ms: int | None = None,
        threshold_rows: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.session.slow_queries(threshold_ms, threshold_rows)

    @_logged
    def execution_history(
        self, query_id: str | None = None, *, kind: str | None = None
    ) -> list[dict[str, Any]]:
        return self.session.execution_history(query_id=query_id, kind=kind)

    @_logged
    def classify_history(self, threshold_ms: int, threshold_rows: int) -> list[dict[str, Any]]:
        return self.session.classify_history(threshold_ms, threshold_rows)

    @_logged
    def generate_candidates(self, query_or_sql: Query | str) -> list[dict[str, Any]]:
        query = self.session.query_by_id(query_or_sql) if isinstance(query_or_sql, str) and query_or_sql.startswith("q_") else (
            query_or_sql if isinstance(query_or_sql, Query) else parse_select(query_or_sql)
        )
        return [candidate.to_dict() for candidate in generate_candidates(query, self.session)]

    @_logged
    def diagnose(self, query_or_sql: Query | str, **options: Any) -> dict[str, Any]:
        return self.engine.diagnose(query_or_sql, **options)

    @_logged
    def conclusions(self, *, include_stale: bool = True) -> list[dict[str, Any]]:
        return self.session.conclusions(include_stale=include_stale)

    @_logged
    def current_best(self, query_id: str | None = None):
        return self.session.current_best(query_id)

    @_logged
    def change_events(self) -> list[dict[str, Any]]:
        return self.session.change_events()

    @_logged
    def save_snapshot(self, path: str | Path) -> Path:
        return self.session.save_snapshot(path)

    @_logged
    def restore_snapshot(self, path: str | Path) -> dict[str, Any]:
        return self.session.restore_snapshot(path)

    @_logged
    def replay_conclusions(
        self, query_ids: Sequence[str] | None = None, **options: Any
    ) -> dict[str, Any]:
        return self.engine.replay_conclusions(query_ids, **options)

    @_logged
    def log_event(self, event: str, message: str = "", **fields: Any) -> dict[str, Any]:
        return self.logger.info(event, message, **fields)
