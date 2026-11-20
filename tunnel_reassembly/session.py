"""会话编排：加载声明/脚本、确定性裁决、账本写入、快照与恢复。"""

from __future__ import annotations

import time
from pathlib import Path

from .clock import ReplayClock
from .config import canonical_digest, load_channels, load_trace
from .engine import adjudicate
from .errors import BusinessError, E_SNAPSHOT
from .ledger import Ledger
from .model import ChannelSpec, FragmentEvent, ReassemblyResult
from .replay import compare_results


class Session:
    def __init__(self, db_path: str | Path) -> None:
        self.ledger = Ledger(db_path)
        self.specs: list[ChannelSpec] = []
        self.config_version: str | None = None
        self.trace_version: str | None = None
        self.trace: dict[str, list[dict]] = {}
        self.results: dict[str, ReassemblyResult] = {}

    def load_inputs(self, config_obj: dict, trace_obj: dict) -> None:
        self.specs, self.config_version = load_channels(config_obj)
        self.trace, self.trace_version = load_trace(trace_obj, self.specs)
        for spec in self.specs:
            self.ledger.register_channel(spec.__dict__, self.config_version, self.trace_version)
        self.ledger.set_meta("config_version", self.config_version)
        self.ledger.set_meta("trace_version", self.trace_version)

    def _events_from_trace(self, channel_id: str) -> list[FragmentEvent]:
        events: list[FragmentEvent] = []
        for group in self.trace[channel_id]:
            for frag in group["fragments"]:
                events.append(
                    FragmentEvent(
                        channel_id=channel_id,
                        group_seq=group["group_seq"],
                        frag_offset=frag["frag_offset"],
                        frag_length=frag["frag_length"],
                        total_frags=group["total_frags"],
                        send_ms=group["send_ms"],
                        arrive_ms=group["arrive_ms"],
                        payload=frag["payload"],
                    )
                )
        return events

    def run_deterministic(self) -> dict[str, ReassemblyResult]:
        """依据脚本声明的 send/arrive 时刻做逐位可复现裁决。"""
        for spec in self.specs:
            events = self._events_from_trace(spec.channel_id)
            expected = {
                g["group_seq"]: g["expected_digest"] for g in self.trace[spec.channel_id]
            }
            result = adjudicate(spec, events, expected)
            self.ledger.save_result(result)
            self.results[spec.channel_id] = result
        return self.results

    def snapshot(self, snapshot_id: str) -> str:
        return self.ledger.write_snapshot(
            snapshot_id,
            time.monotonic_ns() // 1_000_000,
            self.config_version,
            self.trace_version,
            list(self.results.values()),
        )

    def restore_and_replay(self) -> tuple[dict, dict[str, ReassemblyResult]]:
        """恢复快照并重新执行同一脚本，全量比对；UNKNOWN 组要求重跑。"""
        payload = self.ledger.load_latest_snapshot(self.config_version, self.trace_version)
        unknown = self.ledger.mark_unknown_inflight()
        rerun = self.run_deterministic()
        for cid, result in rerun.items():
            expected_item = next(x for x in payload["results"] if x["channel_id"] == cid)
            snapshot_result = ReassemblyResult(
                channel_id=cid,
                estimates=self.ledger.load_estimates(cid),
                window_events=self.ledger.load_window_events(cid),
                flips=self.ledger.load_flips(cid),
                outcomes=self.ledger.load_outcomes(cid),
                fragments=self.ledger.load_fragments(cid),
            )
            compare_results(snapshot_result, result)
        payload["unknown_inflight"] = unknown
        return payload, rerun

    def close(self) -> None:
        self.ledger.close()
