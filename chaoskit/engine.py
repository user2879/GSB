"""Experiment orchestrator: drives the deterministic timeline end to end.

Injections alter only runtime inputs and process state:

* ``network_delay``  -> delay knob on the local forwarding proxy;
* ``error_rate``     -> service runtime rule sent over the control channel;
* ``suspend``        -> real OS-level process suspend;
* ``terminate``      -> real forced process termination (service is restarted
                        afterwards so later injections remain executable).

Observations are sampled before, during and after each injection at a fixed
interval, and each completed injection is persisted in one SQLite transaction.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import observe as obs
from .errors import ChaosError, TargetError
from .ledger import LEDGER_FILENAME, Ledger, open_or_replace_ledger
from .rng import mix_u64
from .spec import ExperimentSpec
from .target import DelayProxy, ManagedService
from .timeline import TimelineEntry, build_timeline, timeline_digest, verify_timeline

EventCallback = Callable[[Dict[str, Any]], None]
PHASES = ("baseline", "active", "post")


@dataclass
class InjectionResult:
    entry: TimelineEntry
    status: str
    observed_state: str
    records_by_phase: Dict[str, List[obs.RequestRecord]] = field(default_factory=dict)
    summaries: List[Dict[str, Any]] = field(default_factory=list)


class ChaosEngine:
    def __init__(self, spec: ExperimentSpec, workdir: str | Path,
                 on_event: Optional[EventCallback] = None) -> None:
        self.spec = spec
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir = self.workdir / "runtime"
        self.on_event = on_event or (lambda event: None)
        self.entries = build_timeline(spec)
        self.timeline_digest_value = timeline_digest(self.entries)
        self.ledger: Optional[Ledger] = None
        self.services: Dict[str, ManagedService] = {}
        self.proxies: Dict[str, DelayProxy] = {}
        self._parallel_slots = threading.BoundedSemaphore(spec.budget.max_parallel)

    # --------------------------------------------------------------- lifecycle
    def setup(self) -> Dict[str, Any]:
        for service_spec in self.spec.services.values():
            service = ManagedService(service_spec, self.runtime_dir)
            service.start()
            self.services[service_spec.service_id] = service
            proxy = DelayProxy(
                service_spec.host,
                service_spec.proxy_port,
                service_spec.host,
                service_spec.data_port,
            )
            proxy.start()
            self.proxies[service_spec.service_id] = proxy
            self._event("service_ready", service_id=service_spec.service_id,
                        data_port=service_spec.data_port, proxy_port=service_spec.proxy_port)

        self.ledger, open_event = open_or_replace_ledger(
            self.workdir / LEDGER_FILENAME, self.spec, self.entries
        )
        stored = self.ledger.experiment_row()
        if stored is not None and stored["timeline_digest"] != self.timeline_digest_value:
            regenerated = build_timeline(self.spec)
            stored_entries = self._stored_entries()
            verify_timeline(stored_entries, regenerated)
        self._event("ledger", **open_event)
        return open_event

    def _stored_entries(self) -> List[TimelineEntry]:
        import json
        rows = self.ledger.conn.execute("SELECT * FROM injections ORDER BY idx").fetchall()
        entries = []
        for row in rows:
            entries.append(
                TimelineEntry(
                    index=row["idx"],
                    start_offset_ms=row["start_offset_ms"],
                    target_id=row["target_id"],
                    injection_id=row["injection_id"],
                    injection_type=row["injection_type"],
                    duration_ms=row["duration_ms"],
                    params=json.loads(row["params_json"]),
                )
            )
        return entries

    def teardown(self) -> None:
        for proxy in self.proxies.values():
            proxy.stop()
        for service in self.services.values():
            service.stop()
        if self.ledger is not None:
            self.ledger.close()

    # ------------------------------------------------------------------- run
    def run(self) -> Dict[str, Any]:
        if self.ledger is None:
            raise ChaosError("engine.setup() must be called before run()")
        results: List[InjectionResult] = []
        for entry in self.entries:
            row = self.ledger.injection_row(entry.index)
            if row is not None and row["status"] == "committed":
                self._event("skip_committed", idx=entry.index, injection_id=entry.injection_id)
                continue
            with self._parallel_slots:
                result = self._run_one(entry)
            results.append(result)
        report = self.ledger.recovery_report(self.spec)
        report["timeline_digest"] = self.timeline_digest_value
        report["content_digest"] = self.ledger.content_digest()
        return report

    def _run_one(self, entry: TimelineEntry) -> InjectionResult:
        self._event("injection_start", idx=entry.index, type=entry.injection_type,
                    target=entry.target_id)
        self.ledger.mark_running(entry.index)
        service = self.services[entry.target_id]
        proxy = self.proxies[entry.target_id]
        interval = self.spec.observation.sample_interval_ms
        min_samples = self.spec.observation.min_samples

        records_by_phase: Dict[str, List[obs.RequestRecord]] = {}
        records_by_phase["baseline"] = obs.collect_phase(
            service.spec.host, proxy.listen_port, "baseline", interval, min_samples
        )

        observed_state, rollback = self._apply(entry, service, proxy)
        try:
            # The duration gates how long the fault stays active; the active
            # phase always takes a fixed sample count so request sequences and
            # their seed-derived accept/reject decisions reproduce across runs.
            self._sleep(entry.duration_ms / 1000.0)
            records_by_phase["active"] = obs.collect_phase(
                service.spec.host, proxy.listen_port, "active", interval, min_samples
            )
        finally:
            rollback()

        records_by_phase["post"] = obs.collect_phase(
            service.spec.host, proxy.listen_port, "post", interval, min_samples
        )

        summaries = [
            obs.aggregate(records_by_phase[phase], phase, min_samples).to_dict()
            for phase in PHASES
        ]
        # Persist the state observed while the injection was in effect so the
        # ledger can distinguish a real suspend from a real termination even
        # though the rollback restarts/resumes the process afterwards.
        final_state = observed_state
        self.ledger.commit_injection(
            entry.index,
            final_state,
            records_by_phase,
            summaries,
            interval,
        )
        self._event("injection_committed", idx=entry.index, observed_state=final_state,
                    active_success=self._phase_success(summaries, "active"))
        return InjectionResult(entry, "committed", final_state, records_by_phase, summaries)

    @staticmethod
    def _phase_success(summaries: List[Dict[str, Any]], phase: str) -> Optional[int]:
        for summary in summaries:
            if summary["phase"] == phase:
                return summary["success_rate_bps"]
        return None

    def _apply(self, entry: TimelineEntry, service: ManagedService,
               proxy: DelayProxy) -> tuple[str, Callable[[], None]]:
        kind = entry.injection_type
        if kind == "network_delay":
            proxy.set_delay(entry.params["delay_ms"])

            def rollback() -> None:
                proxy.set_delay(0)

            return "running", rollback

        if kind == "error_rate":
            rule_seed = mix_u64([self.spec.seed, entry.index, entry.injection_id])
            service.apply_error_rule(entry.index, rule_seed, entry.params["error_bps"])

            def rollback() -> None:
                service.clear_error_rule()

            return "running", rollback

        if kind == "suspend":
            service.suspend()
            if not self._wait_unreachable(service, timeout=3.0):
                raise TargetError(
                    f"service {service.spec.service_id} still reachable after suspend",
                    details={"service_id": service.spec.service_id},
                )

            def rollback() -> None:
                service.resume()

            return "suspended", rollback

        if kind == "terminate":
            service.terminate()
            if not self._wait_unreachable(service, timeout=3.0):
                raise TargetError(
                    f"service {service.spec.service_id} still reachable after termination",
                    details={"service_id": service.spec.service_id},
                )

            def rollback() -> None:
                service.start()

            return "terminated", rollback

        raise TargetError(f"unsupported injection type {kind}")

    @staticmethod
    def _wait_unreachable(service: ManagedService, timeout: float) -> bool:
        # Signal delivery (SIGSTOP) and socket teardown are asynchronous; poll
        # until the data port genuinely refuses connections or time out.
        import time as _time
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            if not service.is_reachable(timeout=0.25):
                return True
            _time.sleep(0.1)
        return not service.is_reachable(timeout=0.25)

    @staticmethod
    def _sleep(seconds: float) -> None:
        import time as _time
        _time.sleep(max(0.0, seconds))

    def recovery_report(self) -> Dict[str, Any]:
        return self.ledger.recovery_report(self.spec)

    def _event(self, name: str, **payload: Any) -> None:
        event = {"event": name, **payload}
        self.on_event(event)
        return event
