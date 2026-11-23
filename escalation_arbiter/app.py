"""Application orchestration: load, run, snapshot, diagnose."""

import json
import os

from .engine import (EngineState, diagnostic_report, execute_command,
                     full_rebuild, restore_or_rebuild, write_snapshot)
from .loader import (load_commands, load_config, load_injections,
                     load_rule_package, load_tickets)
from .serialization import round_trip_stats, serialize


class ArbiterApp:
    def __init__(self, samples_dir, work_dir):
        self.samples_dir = samples_dir
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self.config, config_failures = load_config(
            os.path.join(samples_dir, "config.json"))
        self.rules, rule_failures = load_rule_package(
            os.path.join(samples_dir, "rules.json"))
        self.tickets, ticket_failures = load_tickets(
            os.path.join(samples_dir, "evidence.jsonl"),
            self.rules, self.config)
        self.commands, command_failures = load_commands(
            os.path.join(samples_dir, "commands.json"))
        self.injections = load_injections(
            os.path.join(samples_dir, "injections.json"))
        self.load_failures = []
        for bucket in (config_failures, rule_failures, ticket_failures,
                       command_failures):
            self.load_failures.extend(bucket.items)
        self.state = EngineState(self.tickets, self.rules, self.config,
                                 self.injections, work_dir)
        for failure in self.load_failures:
            self.state.failures.items.append(failure)

    def run_full(self):
        return full_rebuild(self.state)

    def restore(self):
        return restore_or_rebuild(self.state)

    def run_commands(self, compute_delay_ms=0):
        histories = []
        for command in self.commands:
            if command["op"] == "rebuild":
                summary = full_rebuild(self.state)
            else:
                summary = execute_command(
                    self.state, command,
                    compute_delay_ms=compute_delay_ms)
            histories.append({"command": command, "summary": summary})
        return histories

    def serialize_current(self, version=None):
        version = version or self.state.exchange_version
        return serialize(version,
                         [self.state.decisions[tid]
                          for tid in self.state.ordered_tids()
                          if self.state.decisions.get(tid) is not None],
                         self.rules, self.config,
                         os.path.join(self.work_dir, "exchange"))

    def round_trip_report(self):
        exchange_dir = os.path.join(self.work_dir, "exchange")
        os.makedirs(exchange_dir, exist_ok=True)
        decisions = [self.state.decisions[tid]
                     for tid in self.state.ordered_tids()
                     if self.state.decisions.get(tid) is not None]
        return round_trip_stats(decisions, self.rules, self.config,
                                exchange_dir)

    def report(self):
        write_snapshot(self.state)
        round_trip = self.round_trip_report()
        report = diagnostic_report(self.state, round_trip=round_trip)
        path = os.path.join(self.work_dir, "diagnostic_report.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2,
                      sort_keys=True)
        return report, path


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Escalation arbiter")
    parser.add_argument("--samples", default="samples")
    parser.add_argument("--workdir", default="workdir")
    parser.add_argument("--delay-ms", type=float, default=0)
    args = parser.parse_args(argv)
    app = ArbiterApp(args.samples, args.workdir)
    app.run_full()
    histories = app.run_commands(compute_delay_ms=args.delay_ms)
    report, path = app.report()
    print(json.dumps({
        "commands": len(histories),
        "report": path,
        "completed": report["completed_tickets"],
        "failures": report["failures"],
        "injection_hits": report["injection_hits"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
