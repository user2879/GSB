"""Interactive command processor and one-shot scripted edit sequences."""

from __future__ import annotations

import json
import shlex
from dataclasses import replace
from pathlib import Path

from .config import MonitorConfig, load_config
from .errors import DriftError
from .runner import run_pipeline


class Session:
    def __init__(self, config: MonitorConfig, tier_name: str, output_dir: str):
        self.config = config
        self.tier_name = tier_name
        self.output_dir = output_dir
        self.history: list[dict] = []

    def apply(self, command: str) -> dict:
        parts = shlex.split(command)
        if not parts:
            return {"status": "noop"}
        verb, args = parts[0], parts[1:]
        if verb == "run":
            report = run_pipeline(self.config, self.tier_name,
                                  self.output_dir, resume=False)
            entry = {"command": command, "status": "ok",
                     "recomputed": report["incremental"]["recomputed_count"],
                     "reused": report["incremental"]["reused_count"],
                     "failures": len(report["failure_list"]),
                     "snapshot_version": report["snapshot"]["global_version"]}
            self.history.append(entry)
            return entry
        if verb == "resume":
            report = run_pipeline(self.config, self.tier_name,
                                  self.output_dir, resume=True)
            entry = {"command": command, "status": "ok",
                     "resume_status": report["snapshot"]["resume_status"]}
            self.history.append(entry)
            return entry
        if verb == "set-windows":
            self.config = replace(self.config, window_count=int(args[0]))
        elif verb == "set-threshold":
            self.config = replace(self.config,
                                  significance_threshold=float(args[0]))
        elif verb == "set-tier":
            self._set_quantization(args[0])
        elif verb == "set-concurrency":
            self.config = replace(self.config, concurrency=int(args[0]))
        elif verb == "set-size-tier":
            self.tier_name = args[0]
        elif verb == "set-metric":
            column, metric = args[0], args[1]
            overrides = dict(self.config.metric_overrides)
            overrides[column] = metric
            self.config = replace(self.config, metric_overrides=overrides)
        elif verb == "unset-metric":
            overrides = dict(self.config.metric_overrides)
            overrides.pop(args[0], None)
            self.config = replace(self.config, metric_overrides=overrides)
        elif verb == "set-approx":
            flag = args[0].lower() in ("1", "true", "yes", "on")
            self.config = replace(self.config, enable_approx_quantile=flag)
        elif verb == "set-window-cap":
            self.config = replace(self.config, window_cap=int(args[0]))
        elif verb == "set-fail-keys":
            self.config = replace(self.config, fail_keys=tuple(args))
        elif verb == "show":
            return self._snapshot()
        elif verb == "history":
            return {"history": self.history}
        elif verb == "exit":
            return {"status": "exit"}
        else:
            return {"status": "unknown_command", "command": verb}
        self.history.append({"command": command, "status": "edited"})
        return self._snapshot()

    def _set_quantization(self, tier_name: str) -> None:
        if tier_name not in self.config.tier_names:
            raise KeyError(f"unknown quantization tier {tier_name}")
        self.config = replace(self.config, active_tier=tier_name)

    def _snapshot(self) -> dict:
        return {
            "status": "edited",
            "window_count": self.config.window_count,
            "threshold": self.config.significance_threshold,
            "tiers": list(self.config.tier_names),
            "active_tier": self.config.active_tier,
            "metric_overrides": dict(self.config.metric_overrides),
            "size_tier": self.tier_name,
            "concurrency": self.config.concurrency,
        }


def open_session(config_path: str, tier_name: str, output_dir: str) -> Session:
    config = load_config(config_path)
    return Session(config, tier_name, output_dir)


def replay(session: Session, commands: list[str]) -> list[dict]:
    results = []
    for command in commands:
        try:
            results.append(session.apply(command))
        except DriftError as exc:
            results.append({"command": command, "status": exc.code,
                            "field": exc.field, "message": exc.message})
        except (ValueError, KeyError) as exc:
            results.append({"command": command, "status": "E_CONFIG_INVALID",
                            "message": str(exc)})
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Interactive drift monitor CLI")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tier", default="small")
    parser.add_argument("--output", default="output")
    parser.add_argument("--command", action="append",
                        help="scripted command; repeatable. Omit for REPL.")
    args = parser.parse_args()
    session = open_session(args.config, args.tier, args.output)
    if args.command:
        print(json.dumps(replay(session, args.command), indent=2,
                         ensure_ascii=False, default=str))
        return
    print("drift-monitor REPL. Commands: run | resume | set-windows N | "
          "set-threshold X | set-tier NAME | set-metric COL METRIC | "
          "set-size-tier NAME | set-concurrency N | show | history | exit")
    while True:
        try:
            line = input("drift> ").strip()
        except EOFError:
            break
        if not line:
            continue
        try:
            result = session.apply(line)
        except DriftError as exc:
            result = exc.to_dict()
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        if result.get("status") == "exit":
            break


if __name__ == "__main__":
    main()
