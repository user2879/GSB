from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .config import AppConfig, config_from_mapping
from .editing import append_document, delete_document, update_document
from .engine import Engine, RunResult
from .loader import discover_inputs
from .snapshot import SnapshotStore
from .util import atomic_write_json


class InteractiveSession:
    def __init__(
        self,
        *,
        data_dir: Path,
        state_dir: Path,
        config: AppConfig,
        config_failures: list[dict[str, Any]] | None = None,
        engine_factory: Callable[..., Engine] = Engine,
    ):
        self.data_dir = Path(data_dir).resolve()
        self.state_dir = Path(state_dir).resolve()
        self.config = config
        self.config_failures = list(config_failures or [])
        self.engine_factory = engine_factory
        self.stop_event = threading.Event()
        self._condition = threading.Condition(threading.RLock())
        self._active_thread: threading.Thread | None = None
        self._active_generation: int | None = None
        self._latest_generation = -1
        self._generation = 0
        self._latest_report: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._pending_rebuild = False
        self.quit_requested = False
        self._runtime_path = self.state_dir / "runtime_config.json"
        store = SnapshotStore(self.state_dir)
        previous = store.previous_report()
        if isinstance(previous, dict):
            self._latest_report = previous
            self._latest_generation = 0
            prior_config = previous.get("config")
            if isinstance(prior_config, dict):
                candidate, failures = config_from_mapping({**prior_config, **config.to_dict()})
                if not failures:
                    self.config = candidate
            if previous.get("run", {}).get("config_hash") == self.config.config_hash:
                self._latest_generation = 0
            else:
                self._latest_generation = -1

    def summary(self) -> dict[str, Any]:
        with self._condition:
            if self._latest_report is not None:
                report = self._latest_report
                summary = dict(report.get("summary", {}))
                summary["latest_global_version"] = report.get("run", {}).get("global_version")
                summary["latest_status"] = report.get("run", {}).get("status")
                summary["latest_config_hash"] = report.get("run", {}).get("config_hash")
                summary["failure_count"] = len(report.get("failures", []))
                summary["completed_shard_count"] = len(summary.get("completed_shards", []))
                summary["current_block_count"] = summary.get("block_count", 0)
                summary["stale"] = self._latest_generation < self._generation
                summary["pending"] = self._active_thread is not None or summary["stale"]
                return summary
            return {
                "completed_shards": [],
                "partial_shards": [],
                "pending_shards": [],
                "block_count": 0,
                "compression_tier": self.config.compression_tier,
                "affected_shards": [],
                "accepted_documents": 0,
                "accepted_queries": 0,
                "failure_count": 0,
                "completed_shard_count": 0,
                "current_block_count": 0,
                "latest_global_version": None,
                "latest_status": "not_run",
                "latest_config_hash": None,
                "stale": True,
                "pending": True,
            }

    def latest_report(self) -> dict[str, Any] | None:
        with self._condition:
            return self._latest_report

    def _start_engine_locked(self) -> None:
        if self._active_thread is not None:
            self._pending_rebuild = True
            return
        generation = self._generation
        config = self.config
        self._pending_rebuild = False

        def runner() -> None:
            result: RunResult | None = None
            error: str | None = None
            try:
                engine = self.engine_factory(
                    data_dir=self.data_dir,
                    state_dir=self.state_dir,
                    config=config,
                    config_failures=self.config_failures,
                    stop_event=self.stop_event,
                )
                result = engine.run()
            except Exception as exc:  # pragma: no cover - defensive boundary for the REPL
                error = f"{type(exc).__name__}: {exc}"
            with self._condition:
                if error is not None:
                    self._last_error = error
                if result is not None and generation == self._generation:
                    self._latest_report = result.report
                    self._latest_generation = generation
                    self._last_error = None
                self._active_thread = None
                self._active_generation = None
                should_restart = self._pending_rebuild or self._generation != generation
                if should_restart and not self.quit_requested:
                    self._start_engine_locked()
                self._condition.notify_all()

        thread = threading.Thread(target=runner, name=f"iidx-interactive-{generation}", daemon=True)
        self._active_thread = thread
        self._active_generation = generation
        thread.start()

    def _persist_runtime_config(self) -> None:
        atomic_write_json(self._runtime_path, self.config.to_dict())

    def _apply_config_overrides(self, overrides: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
        candidate, failures = config_from_mapping({**self.config.to_dict(), **overrides})
        if failures:
            return False, failures
        with self._condition:
            self.config = candidate
            self._generation += 1
            self._persist_runtime_config()
            self._start_engine_locked()
        return True, []

    def _wait_for_generation(self, generation: int, deadline: float) -> bool:
        with self._condition:
            while self._latest_generation < generation:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def _response(
        self,
        *,
        command: str,
        started: float,
        code: str = "OK",
        message: str | None = None,
        data: Any = None,
    ) -> dict[str, Any]:
        summary = self.summary()
        if code == "OK" and summary.get("stale"):
            code = "E_INTERACT_BUDGET"
            message = message or "latest requested parameters are not complete; returning the latest available summary"
        response: dict[str, Any] = {
            "code": code,
            "command": command,
            "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
            "summary": summary,
            "pending": bool(summary.get("pending")),
        }
        if message:
            response["message"] = message
        if data is not None:
            response["data"] = data
        return response

    def execute_command(self, line: str, *, force_wait: bool = False) -> dict[str, Any]:
        started = time.monotonic()
        line = line.strip()
        if not line:
            return self._response(command=line, started=started, code="E_COMMAND", message="empty command")
        name, _, rest = line.partition(" ")
        name = name.strip().lower()
        rest = rest.strip()
        budget_ms = self.config.interactive_latency_budget_ms
        deadline = time.monotonic() + budget_ms / 1000.0

        try:
            if name in {"quit", "exit"}:
                self.quit_requested = True
                return self._response(command=line, started=started, data={"quit": True})
            if name in {"help", "?"}:
                return self._response(command=line, started=started, data={"commands": HELP_COMMANDS})
            if name in {"status", "summary", "s"}:
                return self._response(command=line, started=started)
            if name in {"run", "rebuild"}:
                with self._condition:
                    self._generation += 1
                    generation = self._generation
                    self._start_engine_locked()
                self._wait_for_generation(generation, deadline)
                return self._response(command=line, started=started)
            if name in {"tier", "set-tier", "compression-tier"}:
                if not rest:
                    return self._response(command=line, started=started, code="E_COMMAND", message="tier needs auto, 1 or 2")
                ok, failures = self._apply_config_overrides({"compression_tier": rest})
                if not ok:
                    return self._response(command=line, started=started, code="E_CONFIG_INVALID", data={"failures": failures})
                with self._condition:
                    generation = self._generation
                self._wait_for_generation(generation, deadline)
                return self._response(command=line, started=started)
            if name in {"chunk", "chunk-size", "block-size"}:
                return self._integer_override(line, started, "chunk_size", rest, deadline)
            if name in {"budget", "global-budget"}:
                return self._integer_override(line, started, "global_token_budget", rest, deadline)
            if name in {"quota", "shard-quota"}:
                return self._integer_override(line, started, "per_shard_token_quota", rest, deadline)
            if name in {"threads", "thread-pool"}:
                return self._integer_override(line, started, "thread_pool_size", rest, deadline)
            if name in {"batch", "batch-size", "batch-max-docs"}:
                return self._integer_override(line, started, "batch_max_docs", rest, deadline)
            if name in {"latency", "latency-budget"}:
                return self._integer_override(line, started, "interactive_latency_budget_ms", rest, deadline)
            if name == "switch-points":
                try:
                    points = [int(item.strip()) for item in rest.split(",") if item.strip()]
                except ValueError:
                    return self._response(command=line, started=started, code="E_COMMAND", message="switch-points expects comma-separated integers")
                ok, failures = self._apply_config_overrides({"compression_switch_points": points})
                if not ok:
                    return self._response(command=line, started=started, code="E_CONFIG_INVALID", data={"failures": failures})
                with self._condition:
                    generation = self._generation
                self._wait_for_generation(generation, deadline)
                return self._response(command=line, started=started)
            if name in {"append-doc"}:
                payload = json.loads(rest)
                result = append_document(discover_inputs(self.data_dir).documents, payload)
                return self._trigger_after_edit(line, started, deadline, result)
            if name in {"update-doc"}:
                doc_id, _, payload_text = rest.partition(" ")
                payload = json.loads(payload_text)
                result = update_document(discover_inputs(self.data_dir).documents, doc_id, payload)
                return self._trigger_after_edit(line, started, deadline, result)
            if name in {"delete-doc"}:
                result = delete_document(discover_inputs(self.data_dir).documents, rest)
                return self._trigger_after_edit(line, started, deadline, result)
            if name in {"query", "q"}:
                report = self.latest_report()
                if report is None:
                    return self._response(command=line, started=started, code="E_NOT_READY", message="no completed report")
                for query in report.get("queries", []):
                    if query.get("query_id") == rest:
                        return self._response(command=line, started=started, data=query)
                return self._response(command=line, started=started, code="E_NOT_FOUND", message=f"query not found: {rest}")
            if name == "blocks":
                report = self.latest_report()
                if report is None:
                    return self._response(command=line, started=started, code="E_NOT_READY", message="no completed report")
                if rest:
                    try:
                        block_no = int(rest)
                    except ValueError:
                        return self._response(command=line, started=started, code="E_COMMAND", message="blocks expects an integer block number")
                    block = next((item for item in report.get("blocks", []) if item.get("block_no") == block_no), None)
                    if block is None:
                        return self._response(command=line, started=started, code="E_NOT_FOUND", message=f"block not found: {block_no}")
                    return self._response(command=line, started=started, data=block)
                return self._response(command=line, started=started, data=report.get("blocks", []))
            if name == "ledger":
                report = self.latest_report()
                if report is None:
                    return self._response(command=line, started=started, code="E_NOT_READY", message="no completed report")
                return self._response(command=line, started=started, data=report.get("budget_ledger", []))
            if name == "compression":
                report = self.latest_report()
                if report is None:
                    return self._response(command=line, started=started, code="E_NOT_READY", message="no completed report")
                return self._response(command=line, started=started, data=report.get("compression_decisions", []))
            if name == "failures":
                report = self.latest_report()
                if report is None:
                    return self._response(command=line, started=started, code="E_NOT_READY", message="no completed report")
                return self._response(command=line, started=started, data=report.get("failures", []))
            if name == "report":
                report = self.latest_report()
                if report is None:
                    return self._response(command=line, started=started, code="E_NOT_READY", message="no completed report")
                if rest:
                    destination = Path(rest).expanduser().resolve()
                    atomic_write_json(destination, report)
                else:
                    destination = SnapshotStore(self.state_dir).write_latest_report(report)
                return self._response(command=line, started=started, data={"report_path": str(destination)})
            return self._response(command=line, started=started, code="E_COMMAND", message=f"unknown command: {name}")
        except (ValueError, json.JSONDecodeError) as exc:
            return self._response(command=line, started=started, code="E_COMMAND", message=str(exc))
        except Exception as exc:  # defensive command boundary
            return self._response(
                command=line,
                started=started,
                code="E_RUN_FAILED",
                message=f"{type(exc).__name__}: {exc}",
            )

    def _integer_override(
        self,
        line: str,
        started: float,
        field: str,
        rest: str,
        deadline: float,
    ) -> dict[str, Any]:
        try:
            value = int(rest)
        except ValueError:
            return self._response(command=line, started=started, code="E_COMMAND", message=f"{field} expects an integer")
        ok, failures = self._apply_config_overrides({field: value})
        if not ok:
            return self._response(command=line, started=started, code="E_CONFIG_INVALID", data={"failures": failures})
        if field == "interactive_latency_budget_ms":
            deadline = time.monotonic() + max(0, value) / 1000.0
        with self._condition:
            generation = self._generation
        self._wait_for_generation(generation, deadline)
        return self._response(command=line, started=started)

    def _trigger_after_edit(
        self,
        line: str,
        started: float,
        deadline: float,
        edit_result: dict[str, Any],
    ) -> dict[str, Any]:
        with self._condition:
            self._generation += 1
            generation = self._generation
            self._start_engine_locked()
        self._wait_for_generation(generation, deadline)
        response = self._response(command=line, started=started)
        response["edit"] = edit_result
        return response

    def close(self, timeout: float = 10.0) -> None:
        self.stop_event.set()
        with self._condition:
            thread = self._active_thread
        if thread is not None:
            thread.join(timeout=timeout)


HELP_COMMANDS = {
    "status": "show the latest completed summary without waiting for a rebuild",
    "run": "request a rebuild and wait up to the interactive latency budget",
    "tier auto|1|2": "change compression tier policy and rebuild",
    "chunk-size N": "change inverted block term count and rebuild",
    "budget N": "change the global token budget and rebuild",
    "quota N": "change the per-shard token quota and rebuild",
    "threads N": "change thread-pool concurrency (subject to the configured quota)",
    "batch-size N": "change maximum documents per scheduling batch",
    "switch-points A,B": "set strictly increasing auto-tier switch points",
    "latency N": "change the interactive latency budget in milliseconds",
    "append-doc JSON": "append a document JSON object and rebuild",
    "update-doc ID JSON": "update a document by ID and rebuild",
    "delete-doc ID": "delete a document by ID and rebuild",
    "query ID": "show one query result from the latest report",
    "blocks [N]": "show all block fingerprints or one block",
    "ledger": "show token-budget deductions",
    "compression": "show compression tier decisions",
    "failures": "show the stable failure list",
    "report [PATH]": "write the latest report JSON",
    "quit": "exit the interactive session",
}


def run_repl(session: InteractiveSession, input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print) -> int:
    output_fn("iidx interactive shell. Type 'help' for commands.")
    while not session.quit_requested:
        try:
            line = input_fn("iidx> ")
        except EOFError:
            break
        if not line.strip():
            continue
        response = session.execute_command(line)
        output_fn(json.dumps(response, ensure_ascii=False, sort_keys=False))
    session.close()
    return 0
