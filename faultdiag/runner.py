"""运行编排: 规格加载 -> 账本恢复 -> 快照校验 -> 事件批处理 ->
指令下发 -> 收敛判定 -> 快照落盘, 并支持三类异常注入。"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from .converge import check_domain_convergence
from .dispatch import Dispatcher
from .errors import DiagError
from .ledger import Ledger
from .snapshot import build_payload, open_or_replace, write_snapshot
from .spec import (Event, domains_version, events_digest, instances_version,
                   load_domains, load_events, load_instances)
from .state import Engine
from .structured_log import StructuredLog
from .util import digest_obj


@dataclass
class RunConfig:
    instances_path: str
    domains_path: str
    events_path: str
    workdir: str
    batch_size: int = 3
    global_limit: int = 4
    domain_quota: int = 2
    base_port: int = 24100
    handshake_timeout: float = 2.0
    command_timeout: float = 2.0
    inject: str | None = None          # "event" | "level" | "snapshot"
    modes: dict = field(default_factory=dict)


def _apply_injection(config: RunConfig, domains, instances, events) -> None:
    if config.inject == "event":
        first = sorted(instances)[0]
        events.append(Event(len(events) + 1, events[-1].ts_ms, first, "__bogus__"))
    elif config.inject == "level":
        first = sorted(domains)[0]
        domains[first] = dataclasses.replace(domains[first], isolation_level=99)
    elif config.inject == "snapshot":
        snapdir = Path(config.workdir) / "snapshots"
        snapdir.mkdir(parents=True, exist_ok=True)
        path = snapdir / "snapshot.json"
        if not path.exists():
            path.write_text(json.dumps(build_payload(
                "0", "0", "0", 0, {}, {}, "0")), encoding="utf-8")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["digest"] = "0" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")


def _receipts_digest(results) -> str:
    return digest_obj([[r.cmd_seq, r.instance, r.action, r.attempt, r.status,
                        json.dumps(r.receipt, sort_keys=True) if r.receipt else None]
                       for r in results])


def run(config: RunConfig) -> dict:
    log = StructuredLog(config.workdir)
    log.info("run started", details={"workdir": config.workdir,
                                     "inject": config.inject})
    domains = load_domains(config.domains_path)
    instances = load_instances(config.instances_path, domains)
    events = load_events(config.events_path, instances)
    _apply_injection(config, domains, instances, events)

    versions = {"instances_version": instances_version(instances),
                "domains_version": domains_version(domains),
                "events_digest": events_digest(events)}
    ledger = Ledger(Path(config.workdir) / "ledger.db", versions)
    for did, dspec in domains.items():
        ledger.set_meta(f"cooldown:{did}", str(dspec.recovery_cooldown_ms))
        ledger.set_meta(f"level:{did}", str(dspec.isolation_level))
    ledger.conn.commit()
    unknown_commands = ledger.recover()
    if unknown_commands:
        log.info("uncommitted commands marked unknown, rerun required",
                 details={"commands": unknown_commands})

    engine = Engine(instances, domains)
    restored = ledger.restore_state()
    cursor, cursor_ts = ledger.cursor()
    engine.restore(restored["health"], restored["domain_state"],
                   restored["isolated_since_ts"], {}, cursor, cursor_ts)

    payload = build_payload(versions["instances_version"],
                            versions["domains_version"],
                            versions["events_digest"], cursor,
                            engine.domain_state, engine.health, ledger.digest())
    _, replace_report = open_or_replace(config.workdir, payload)
    if replace_report:
        log.info("snapshot whole-replaced", details=replace_report)

    dispatcher = Dispatcher(instances, base_port=config.base_port,
                            global_limit=config.global_limit,
                            domain_quota=config.domain_quota,
                            handshake_timeout=config.handshake_timeout,
                            command_timeout=config.command_timeout,
                            modes=config.modes)
    processed = 0
    command_count = 0
    try:
        dispatcher.start_services()
        next_cmd_seq = ledger.max_cmd_seq() + 1
        pending = [e for e in events if e.seq > cursor]
        for start in range(0, len(pending), config.batch_size):
            batch_events = pending[start:start + config.batch_size]
            with ledger.batch():
                recorded = len(engine.transitions)
                for event in batch_events:
                    intents = engine.apply_event(event)
                    results = dispatcher.submit(intents, next_cmd_seq)
                    next_cmd_seq += len(intents)
                    command_count += len(results)
                    for res in results:
                        ledger.add_command(res)
                    for tr in engine.transitions[recorded:]:
                        ledger.add_transition(tr)
                    recorded = len(engine.transitions)
                    for domain in sorted({i.domain for i in intents}):
                        check_domain_convergence(engine, domain, results)
                        if any(i.action == "isolate" for i in intents
                               if i.domain == domain):
                            engine.confirm_isolated(domain)
                            ledger.set_domain_mark(
                                domain, engine.isolated_since_ts[domain])
                        if any(i.action == "recover" for i in intents
                               if i.domain == domain):
                            engine.confirm_normal(domain)
                    for tr in engine.transitions[recorded:]:
                        ledger.add_transition(tr)
                    recorded = len(engine.transitions)
                    expected_views = {did: engine.expected_view(did)
                                      for did in sorted(engine.domain_state)}
                    ledger.add_event_state(event.seq, dict(engine.health),
                                           dict(engine.domain_state),
                                           expected_views,
                                           _receipts_digest(results))
                    ledger.set_cursor(event.seq, event.ts_ms)
                    processed += 1
        final_payload = build_payload(
            versions["instances_version"], versions["domains_version"],
            versions["events_digest"], ledger.cursor()[0],
            dict(engine.domain_state), dict(engine.health), ledger.digest())
        write_snapshot(config.workdir, final_payload)
        report = {"status": "ok",
                  "events_processed": processed,
                  "cursor": ledger.cursor()[0],
                  "transitions": len(engine.transitions),
                  "commands": command_count,
                  "unknown_commands": unknown_commands,
                  "max_concurrent": dispatcher.max_concurrent,
                  "domain_max_concurrent": dispatcher.domain_max,
                  "ledger_digest": ledger.digest(),
                  "snapshot_replace": replace_report,
                  "domain_states": dict(engine.domain_state),
                  "instance_health": dict(engine.health)}
        log.info("run finished", details={"cursor": report["cursor"],
                                          "ledger_digest": report["ledger_digest"]})
        return report
    except DiagError as exc:
        log.error("run failed", code=exc.code, details=exc.details)
        raise
    finally:
        dispatcher.stop_services()
        ledger.close()
