"""Interactive workbench (requirements 5, 7 and 8).

Supports five command classes:
    set-weight <channel> <value>
    set-temperature <value>
    set-precision <frac_bits> <quantum>
    replace-turn <turn_id> <json record>
    add-tool <json record> / remove-tool <name>
plus:
    run / summary / quit

Every command triggers a rerun whose stage products are cached under
content-fingerprint keys, so only affected turns and their downstream
stages are recomputed.  Each command responds within the configured
interactive latency budget: when the fresh result is not ready in time,
the most recent completed summary is returned with E_PREVIEW_BUDGET and
its explicit version.  Snapshots are committed every
`snapshot_commit_interval` runs via staged wholesale replacement.
"""

from __future__ import annotations

import json
import threading

from . import codes, loader
from .cache import StageCache
from .fingerprint import fingerprint
from .pipeline import Pipeline
from .snapshot import Snapshotter, snapshot_artifacts


class Workbench:
    def __init__(self, turns, acoustics, tools, lexicon, config, load_failures,
                 run_dir, cache=None):
        self.turns = list(turns)
        self.acoustics = dict(acoustics)
        self.tools = dict(tools)
        self.lexicon = lexicon
        self.config = dict(config)
        self.load_failures = list(load_failures)
        self.cache = cache if cache is not None else StageCache()
        self.snapshotter = Snapshotter(run_dir)
        self.version = 0
        self.last_report = None
        self.last_summary = None
        self.affected_turns = []
        self._lock = threading.Lock()
        self._worker = None
        self.startup_findings = self.snapshotter.check_stale(fingerprint(self.config))

    # ------------------------------------------------------------------
    # Command handling
    # ------------------------------------------------------------------

    def apply(self, command_line):
        """Apply one command line; returns the response dict."""
        parts = command_line.strip().split()
        if not parts:
            return {"ok": False, "error": "empty command"}
        cmd, args = parts[0], parts[1:]
        handler = getattr(self, f"_cmd_{cmd.replace('-', '_')}", None)
        if handler is None:
            return {"ok": False, "error": f"unknown command {cmd!r}"}
        return handler(args)

    def _cmd_set_weight(self, args):
        channel, value = args[0], float(args[1])
        if channel not in self.config["weights"]:
            return {"ok": False, "error": f"unknown channel {channel!r}"}
        self.config["weights"][channel] = value
        affected = sorted(t["turn_id"] for t in self.turns)
        return self._rerun(f"set-weight {channel} {value}", affected)

    def _cmd_set_temperature(self, args):
        self.config["temperature"] = float(args[0])
        affected = sorted(t["turn_id"] for t in self.turns)
        return self._rerun(f"set-temperature {args[0]}", affected)

    def _cmd_set_precision(self, args):
        frac_bits, quantum = int(args[0]), int(args[1])
        allowed_bits = self.config["precision"].get("allowed_frac_bits", [])
        allowed_quantum = self.config["precision"].get("allowed_quantum", [])
        if frac_bits not in allowed_bits or quantum not in allowed_quantum:
            return {"ok": False, "code": codes.E_LOAD_PRECISION,
                    "error": f"precision ({frac_bits}, {quantum}) outside allowed set"}
        self.config["precision"]["frac_bits"] = frac_bits
        self.config["precision"]["quantum"] = quantum
        affected = sorted(t["turn_id"] for t in self.turns)
        return self._rerun(f"set-precision {frac_bits} {quantum}", affected)

    def _cmd_replace_turn(self, args):
        turn_id = args[0]
        record = json.loads(" ".join(args[1:]))
        record["turn_id"] = turn_id
        replaced = False
        for idx, turn in enumerate(self.turns):
            if turn["turn_id"] == turn_id:
                self.turns[idx] = record
                replaced = True
                break
        if not replaced:
            return {"ok": False, "error": f"turn {turn_id!r} not found"}
        return self._rerun(f"replace-turn {turn_id}", [turn_id])

    def _cmd_add_tool(self, args):
        record = json.loads(" ".join(args))
        name = record.get("name")
        if not name:
            return {"ok": False, "error": "tool record missing name"}
        self.tools[name] = record
        affected = sorted(t["turn_id"] for t in self.turns
                          if any(c.get("tool") == name for c in t.get("tool_calls", [])))
        return self._rerun(f"add-tool {name}", affected)

    def _cmd_remove_tool(self, args):
        name = args[0]
        if name not in self.tools:
            return {"ok": False, "error": f"tool {name!r} not found"}
        del self.tools[name]
        affected = sorted(t["turn_id"] for t in self.turns
                          if any(c.get("tool") == name for c in t.get("tool_calls", [])))
        return self._rerun(f"remove-tool {name}", affected)

    def _cmd_run(self, args):
        affected = sorted(t["turn_id"] for t in self.turns)
        return self._rerun("run", affected)

    def _cmd_summary(self, args):
        return {"ok": True, "summary": self.last_summary, "version": self.version}

    # ------------------------------------------------------------------
    # Rerun with latency budget
    # ------------------------------------------------------------------

    def _rerun(self, label, affected):
        self.affected_turns = sorted(affected)
        latency_s = self.config["interactive_latency_ms"] / 1000.0
        result_box = {}
        self.version += 1
        requested_version = self.version

        def work():
            report = self._execute()
            with self._lock:
                result_box["report"] = report

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()
        self._worker.join(timeout=latency_s)
        with self._lock:
            report = result_box.get("report")
        if report is None:
            # Latency budget exhausted: return the latest completed summary
            # with its explicit version, flagged E_PREVIEW_BUDGET.
            stale = dict(self.last_summary) if self.last_summary else {"version": None}
            stale["code"] = codes.E_PREVIEW_BUDGET
            stale["requested_version"] = requested_version
            return {"ok": True, "command": label, "preview": stale,
                    "affected_turns": self.affected_turns}
        return {"ok": True, "command": label, "report": report,
                "summary": self.last_summary, "affected_turns": self.affected_turns}

    def _execute(self):
        pipeline = Pipeline(self.turns, self.acoustics, self.tools, self.lexicon,
                            self.config, cache=self.cache)
        report = pipeline.run()
        report["version"] = self.version
        report["failures"] = self.load_failures + report["failures"]
        if self.startup_findings:
            report["startup_findings"] = self.startup_findings
        self.last_report = report
        self.last_summary = {
            "version": self.version,
            "completed_turns": report["summary"]["completed_turns"],
            "total_turns": report["summary"]["total_turns"],
            "confidence_distribution": report["summary"]["confidence_distribution"],
            "affected_turns": self.affected_turns,
            "flips": report["summary"]["flips"],
            "progress": 1.0 if not report["summary"]["pending_turns"] else
                report["summary"]["completed_turns"] / max(1, report["summary"]["total_turns"]),
        }
        interval = self.config.get("snapshot_commit_interval", 1)
        if self.version % max(1, interval) == 0:
            artifacts = snapshot_artifacts(report)
            self.snapshotter.commit(self.version, report["config_fingerprint"], artifacts)
        return report
