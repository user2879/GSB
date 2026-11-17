"""Interactive command-driven session with latency-budget semantics.

Each modifying command triggers an incremental recompute. The response is
returned within the configured budget (ms); if the fresh result is not ready
in time, the session answers E_INTERACT_BUDGET_PENDING with the most recent
completed summary while computation continues in the background.
"""
from __future__ import annotations

import json
import os
import signal
import threading

from .common import (
    E_CONFIG_INVALID,
    E_INTERACT_BUDGET_PENDING,
    E_UNKNOWN_COMMAND,
    OK,
    BusinessError,
    atomic_write_json,
)
from .config import validate_config, validate_layer_params
from .engine import Engine
from .report import build_record, build_report

MODIFYING_PREFIXES = ("set-bits", "set-scheme", "set-group", "set-per-channel", "set-distill")


class Session:
    def __init__(self, cfg: dict, weights, calib, workdir: str):
        self.cfg = cfg
        self.weights = weights
        self.calib = calib
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.record_path = os.path.join(workdir, "experiment_record.json")
        self.report_path = os.path.join(workdir, "report.json")
        self.log_path = os.path.join(workdir, "interactive_log.jsonl")
        self.engine = Engine(cfg, weights, calib)
        self.busy = threading.Lock()  # held while a recompute runs
        self.stop_requested = False
        self.command_count = 0
        self._log_fh = open(self.log_path, "a", encoding="utf-8")

    # -- helpers ------------------------------------------------------------
    def _emit(self, response: dict) -> dict:
        self._log_fh.write(json.dumps(response, ensure_ascii=False, sort_keys=True) + "\n")
        self._log_fh.flush()
        return response

    def _commit_record(self) -> None:
        results = self.engine.last_results
        if results is None:
            return
        atomic_write_json(self.record_path, build_record(self.cfg, self.weights, self.calib, results))

    def _summary_response(self, code: str, command: str, stale: bool) -> dict:
        results = self.engine.last_results
        summary = results["summary"] if results else None
        invalidation = self.engine.last_invalidation
        return {
            "command": command,
            "business_code": code,
            "stale": stale,
            "summary": summary,
            "invalidated": invalidation["invalidated"],
            "reused": invalidation["reused"],
            "failures": results["failures"] if results else [],
            "progress": (
                f"{summary['completed_layers']}/{summary['total_layers']} layers evaluated"
                if summary
                else "0/0 layers evaluated"
            ),
        }

    def _ensure_results(self) -> None:
        if self.engine.last_results is None:
            self.engine.compute()
            self._commit_record()

    # -- command handling -----------------------------------------------------
    def _apply_modify(self, tokens: list) -> None:
        cmd = tokens[0]
        overrides = self.cfg["layers"].setdefault("overrides", {})
        if cmd == "set-distill":
            if len(tokens) != 3:
                raise BusinessError(E_CONFIG_INVALID, "usage: set-distill <temperature> <alpha>")
            self.cfg["distill"]["temperature"] = float(tokens[1])
            self.cfg["distill"]["soft_label_weight"] = float(tokens[2])
        else:
            if len(tokens) != 3:
                raise BusinessError(E_CONFIG_INVALID, f"usage: {cmd} <layer> <value>")
            layer, value = tokens[1], tokens[2]
            if layer not in self.weights.layer_names:
                raise BusinessError(E_CONFIG_INVALID, f"unknown layer {layer!r}")
            entry = overrides.setdefault(layer, {})
            if cmd == "set-bits":
                entry["bits"] = int(value)
            elif cmd == "set-scheme":
                entry["scheme"] = value
            elif cmd == "set-group":
                entry["group_size"] = int(value)
            elif cmd == "set-per-channel":
                if value not in ("on", "off"):
                    raise BusinessError(E_CONFIG_INVALID, "value must be on|off")
                entry["per_channel"] = value == "on"
            merged = dict(self.cfg["layers"]["default"])
            merged.update(entry)
            validate_layer_params(merged, layer)
        validate_config(self.cfg, self.weights)

    def execute(self, line: str) -> dict:
        line = line.strip()
        if not line or line.startswith("#"):
            return self._emit({"command": line, "business_code": OK, "skipped": True})
        tokens = line.split()
        cmd = tokens[0]
        try:
            if cmd == "quit":
                self.stop_requested = True
                with self.busy:  # wait for any background recompute to commit
                    pass
                self._ensure_results()
                self._commit_record()
                atomic_write_json(
                    self.report_path,
                    build_report(self.cfg, self.weights, self.calib, self.engine.last_results),
                )
                return self._emit({"command": line, "business_code": OK, "exit_code": 0})
            if cmd == "status":
                return self._emit(self._summary_response(OK, line, stale=False))
            if cmd == "report":
                with self.busy:
                    pass
                self._ensure_results()
                self._commit_record()
                atomic_write_json(
                    self.report_path,
                    build_report(self.cfg, self.weights, self.calib, self.engine.last_results),
                )
                return self._emit(self._summary_response(OK, line, stale=False))
            if cmd not in MODIFYING_PREFIXES:
                raise BusinessError(E_UNKNOWN_COMMAND, f"unknown command {cmd!r}")

            self._apply_modify(tokens)
            self.command_count += 1
            done = threading.Event()

            def work():
                with self.busy:
                    self.engine.compute()
                    if self.command_count % self.cfg["interaction"]["snapshot_interval"] == 0:
                        self._commit_record()
                    done.set()

            worker = threading.Thread(target=work, daemon=True)
            worker.start()
            budget_s = self.cfg["interaction"]["budget_ms"] / 1000.0
            if done.wait(budget_s):
                return self._emit(self._summary_response(OK, line, stale=False))
            # Budget exceeded: return the most recent completed summary; the
            # background computation keeps running (never described as stopped).
            return self._emit(self._summary_response(E_INTERACT_BUDGET_PENDING, line, stale=True))
        except BusinessError as exc:
            return self._emit({"command": line, "business_code": exc.code, "error": exc.message})

    def request_stop(self, *_args) -> None:
        self.stop_requested = True

    def run_script(self, commands) -> int:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)
        self._ensure_results()  # baseline: a completed summary always exists
        for line in commands:
            self.execute(line)
            if self.stop_requested:
                break
        with self.busy:  # ensure background work committed before exit
            pass
        self._ensure_results()
        self._commit_record()
        if self.engine.last_results:
            atomic_write_json(
                self.report_path,
                build_report(self.cfg, self.weights, self.calib, self.engine.last_results),
            )
        self._log_fh.close()
        return 0

    def run_stdin(self) -> int:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)
        self._ensure_results()  # baseline: a completed summary always exists
        print("qdemo interactive session; commands: set-bits|set-scheme|set-group|"
              "set-per-channel|set-distill|status|report|quit")
        while not self.stop_requested:
            try:
                line = input("qdemo> ")
            except EOFError:
                break
            response = self.execute(line)
            print(json.dumps(response, ensure_ascii=False))
        with self.busy:
            pass
        self._ensure_results()
        self._commit_record()
        if self.engine.last_results:
            atomic_write_json(
                self.report_path,
                build_report(self.cfg, self.weights, self.calib, self.engine.last_results),
            )
        self._log_fh.close()
        return 0
