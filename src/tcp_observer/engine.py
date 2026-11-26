from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any

import numpy as np

from .canonical import sha256_hex
from .congestion import CongestionState, step_congestion
from .spec import SessionSpec


def _raw_bound_max(bounds: dict[str, Any]) -> int:
    return max(
        int(bounds["cwnd_scaled"]),
        int(bounds["ssthresh_scaled"]),
        int(bounds["srtt_scaled"]),
        int(bounds["rttvar_scaled"]),
        int(bounds["rto_scaled"]),
    )


def _item_id(event_seq: int) -> str:
    return f"item-{event_seq:06d}"


def _batch_id(batch_index: int) -> str:
    return f"batch-{batch_index:04d}"


def _window_id(window_index: int) -> str:
    return f"window-{window_index:04d}"


def _interval_ref(spec: SessionSpec, batch_index: int, start_seq: int, end_seq: int) -> dict[str, Any]:
    return {
        "kind": "batch_interval",
        "batch_index": batch_index,
        "batch_id": _batch_id(batch_index),
        "start_event_seq": start_seq,
        "end_event_seq": end_seq,
        "script_version": spec.script_version,
        "script_digest": spec.script_digest,
    }


def build_batch(spec: SessionSpec, batch_index: int, batch_items: list[dict[str, Any]]) -> dict[str, Any]:
    start_seq = batch_items[0]["event_seq"]
    end_seq = batch_items[-1]["event_seq"]
    last = batch_items[-1]
    state = last["state_after"]
    scale = spec.scale
    ack_count = sum(int(item["acks_processed"]) for item in batch_items)
    duration_ms = sum(int(item["duration_ms"]) for item in batch_items)
    max_bound = max(_raw_bound_max(item["error_bounds_raw"]) for item in batch_items)
    affected = [bool(item.get("edit_affected", True)) for item in batch_items]
    interval = _interval_ref(spec, batch_index, start_seq, end_seq)
    core = {
        "integer_format_version": 1,
        "observation_format_version": 3,
        "batch_index": batch_index,
        "batch_id": _batch_id(batch_index),
        "start_seq": start_seq,
        "end_seq": end_seq,
        "event_count": len(batch_items),
        "item_ids": [item["item_id"] for item in batch_items],
        "item_digests": [item["content_digest"] for item in batch_items],
        "ack_count": ack_count,
        "cwnd": state["cwnd"],
        "cwnd_scaled": state["cwnd_scaled"],
        "ssthresh": state["ssthresh"],
        "ssthresh_scaled": state["ssthresh_scaled"],
        "srtt_ms": state["srtt_ms"],
        "srtt_scaled": state["srtt_scaled"],
        "rttvar_scaled": state["rttvar_scaled"],
        "rto_ms": state["rto_ms"],
        "rto_scaled": state["rto_scaled"],
        "outstanding_packets": state["outstanding_packets"],
        "outstanding_bytes": state["outstanding_bytes"],
        "duration_ms": duration_ms,
        "peer_id": last.get("peer_id", f"peer-{spec.session_id}"),
        "scale": scale,
        "error_bound_scaled": max_bound,
        "affected": affected,
        "script_version": spec.script_version,
        "scale_version": spec.scale_version,
        "script_digest": spec.script_digest,
        "scale_caliber_digest": scale_caliber_digest(spec),
        "interval_ref": interval,
        "metric_refs": {
            "cwnd": {"value": state["cwnd_scaled"], "unit": "scaled_MSS_packets", "interval": interval},
            "ssthresh": {"value": state["ssthresh_scaled"], "unit": "scaled_MSS_packets", "interval": interval},
            "outstanding": {"value_bytes": state["outstanding_bytes"], "value_packets": state["outstanding_packets"], "interval": interval},
            "rto": {"value_scaled_ms": state["rto_scaled"], "value_ms_ceil": state["rto_ms"], "interval": interval},
        },
    }
    core["content_digest"] = sha256_hex({k: v for k, v in core.items() if k != "content_digest"})
    return core


def scale_caliber(spec: SessionSpec) -> dict[str, Any]:
    return {
        "scale": spec.scale,
        "truncation": "floor toward zero for non-negative numerators",
        "overflow": "saturate cwnd/ssthresh at window_max*scale and other int values at int64 max",
        "rtt_alpha_denominator": 8,
        "rtt_beta_denominator": 4,
    }


def scale_caliber_digest(spec: SessionSpec) -> str:
    return sha256_hex(scale_caliber(spec))


def build_window(spec: SessionSpec, window_index: int, window_items: list[dict[str, Any]], batches: list[dict[str, Any]]) -> dict[str, Any]:
    start_seq = window_items[0]["event_seq"]
    end_seq = window_items[-1]["event_seq"]
    full = len(window_items) == spec.seal_length
    last_state = window_items[-1]["state_after"]
    window_batches = [
        batch
        for batch in batches
        if batch["start_seq"] >= start_seq and batch["end_seq"] <= end_seq
    ]
    core = {
        "integer_format_version": 1,
        "window_index": window_index,
        "window_id": _window_id(window_index),
        "start_seq": start_seq,
        "end_seq": end_seq,
        "expected_length": spec.seal_length,
        "actual_length": len(window_items),
        "sealed": full,
        "sealed_at_event_seq": end_seq if full else None,
        "item_ids": [item["item_id"] for item in window_items],
        "item_digests": [item["content_digest"] for item in window_items],
        "batch_ids": [batch["batch_id"] for batch in window_batches],
        "batch_digests": [batch["content_digest"] for batch in window_batches],
        "final_state": {
            "cwnd": last_state["cwnd"],
            "cwnd_scaled": last_state["cwnd_scaled"],
            "ssthresh": last_state["ssthresh"],
            "ssthresh_scaled": last_state["ssthresh_scaled"],
            "srtt_ms": last_state["srtt_ms"],
            "srtt_scaled": last_state["srtt_scaled"],
            "rto_ms": last_state["rto_ms"],
            "rto_scaled": last_state["rto_scaled"],
            "outstanding_bytes": last_state["outstanding_bytes"],
        },
        "error_bound_scaled": max(_raw_bound_max(item["error_bounds_raw"]) for item in window_items),
        "script_version": spec.script_version,
        "scale_version": spec.scale_version,
        "script_digest": spec.script_digest,
        "scale_caliber_digest": scale_caliber_digest(spec),
    }
    core["content_digest"] = sha256_hex({k: v for k, v in core.items() if k != "content_digest"})
    return core


def build_summary(spec: SessionSpec, items: list[dict[str, Any]], batches: list[dict[str, Any]], windows: list[dict[str, Any]]) -> dict[str, Any]:
    last = items[-1]["state_after"]
    bound_matrix = np.array(
        [[int(item["error_bounds_raw"][key]) for key in ("cwnd_scaled", "ssthresh_scaled", "srtt_scaled", "rttvar_scaled", "rto_scaled")] for item in items],
        dtype=np.int64,
    )
    body = {
        "integer_format_version": 1,
        "observation_format_version": 3,
        "session_id": spec.session_id,
        "seed": spec.seed,
        "initial_seq": spec.initial_seq,
        "event_count": spec.event_count,
        "batch_size": spec.batch_size,
        "seal_length": spec.seal_length,
        "window_max": spec.window_max,
        "mss": spec.mss,
        "scale": spec.scale,
        "script_version": spec.script_version,
        "scale_version": spec.scale_version,
        "config_digest": spec.config_digest,
        "script_digest": spec.script_digest,
        "scale_caliber_digest": scale_caliber_digest(spec),
        "batch_count": len(batches),
        "window_count": len(windows),
        "sealed_window_count": sum(1 for window in windows if window["sealed"]),
        "final_state": last,
        "totals": {
            "fast_retransmit_count": last["fast_retransmit_count"],
            "timeout_count": last["timeout_count"],
            "saturation_count": last["saturation_count"],
            "ack_count": sum(int(item["acks_processed"]) for item in items),
            "max_error_bound_scaled": int(bound_matrix.max()),
            "numpy_bound_matrix_shape": list(bound_matrix.shape),
        },
        "batches": [
            {
                "batch_index": batch["batch_index"],
                "batch_id": batch["batch_id"],
                "start_seq": batch["start_seq"],
                "end_seq": batch["end_seq"],
                "content_digest": batch["content_digest"],
            }
            for batch in batches
        ],
        "windows": [
            {
                "window_index": window["window_index"],
                "window_id": window["window_id"],
                "start_seq": window["start_seq"],
                "end_seq": window["end_seq"],
                "sealed": window["sealed"],
                "content_digest": window["content_digest"],
            }
            for window in windows
        ],
    }
    body["content_digest"] = sha256_hex(body)
    return body


class ObservationWorkspace:
    def __init__(
        self,
        spec: SessionSpec,
        exchange: dict[str, Any] | None = None,
        initial_events: int | None = None,
    ) -> None:
        self.spec = spec
        self.exchange = exchange
        self.initial_events = initial_events
        self.response_by_seq: dict[int, dict[str, Any]] = {}
        if exchange:
            for response in exchange.get("responses", []):
                if isinstance(response.get("event_seq"), int):
                    self.response_by_seq[int(response["event_seq"])] = response
        self.state = CongestionState.initial(spec)
        self.pre_states: list[CongestionState] = []
        self.items: list[dict[str, Any]] = []
        self._build(initial_events)

    def _build(self, event_limit: int | None = None) -> None:
        limit = self.spec.event_count if event_limit is None else event_limit
        for event_index in range(limit):
            seq = self.spec.initial_seq + event_index
            entry = self.spec.entry_for_seq(seq)
            if entry is None:
                from .errors import ScriptError

                raise ScriptError(f"missing script entry for event {seq}", event_seq=seq, details={"gap_start_seq": seq})
            self.append_event(entry.event_seq)
        self.rebuild_views()

    def append_event(self, seq: int) -> dict[str, Any]:
        event_index = seq - self.spec.initial_seq
        entry = self.spec.entry_for_seq(seq)
        if entry is None:
            raise KeyError(seq)
        self.pre_states.append(copy.deepcopy(self.state))
        response = self.response_by_seq.get(seq)
        self.state, detail = step_congestion(self.state, entry, self.spec, response)
        item = {
            "item_id": _item_id(seq),
            "event_seq": seq,
            "event_index": event_index,
            "batch_index": event_index // self.spec.batch_size,
            "batch_id": _batch_id(event_index // self.spec.batch_size),
            "window_index": event_index // self.spec.seal_length,
            "window_id": _window_id(event_index // self.spec.seal_length),
            "script_version": self.spec.script_version,
            "scale_version": self.spec.scale_version,
            "script_digest": self.spec.script_digest,
            "peer_id": (response or {}).get("peer_id", f"peer-{self.spec.session_id}"),
            "duration_ms": int((response or {}).get("duration_ms", entry.delay_ms)),
            "response_kind": (response or {}).get("kind", entry.type),
            **detail,
        }
        item["error_bounds_raw"] = asdict(self.state.bounds)
        item["content_digest"] = sha256_hex({k: v for k, v in item.items() if k != "content_digest"})
        self.items.append(item)
        return item

    def rebuild_views(self) -> dict[str, Any]:
        for item in self.items:
            index = int(item["event_index"])
            item["batch_index"] = index // self.spec.batch_size
            item["batch_id"] = _batch_id(item["batch_index"])
            item["window_index"] = index // self.spec.seal_length
            item["window_id"] = _window_id(item["window_index"])
        batches: list[dict[str, Any]] = []
        for batch_index in range((self.spec.event_count + self.spec.batch_size - 1) // self.spec.batch_size):
            batch_items = [item for item in self.items if item["batch_index"] == batch_index]
            if batch_items:
                batches.append(build_batch(self.spec, batch_index, batch_items))
        windows: list[dict[str, Any]] = []
        max_window = (self.spec.event_count - 1) // self.spec.seal_length
        for window_index in range(max_window + 1):
            window_items = [item for item in self.items if item["window_index"] == window_index]
            if window_items:
                windows.append(build_window(self.spec, window_index, window_items, batches))
        self.batches = batches
        self.windows = windows
        self.summary = build_summary(self.spec, self.items, batches, windows)
        return self.result()

    def result(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "items": self.items,
            "batches": self.batches,
            "windows": self.windows,
        }

    def sealed_digests(self) -> dict[int, str]:
        return {window["window_index"]: window["content_digest"] for window in self.windows if window["sealed"]}

    def append_next_event_incrementally(self) -> dict[str, Any]:
        if len(self.items) >= self.spec.event_count:
            raise IndexError("no remaining event")
        old_sealed = self.sealed_digests()
        event_index = len(self.items)
        event_seq = self.spec.initial_seq + event_index
        old_batches = list(self.batches)
        old_windows = list(self.windows)
        item = self.append_event(event_seq)

        batch_index = event_index // self.spec.batch_size
        batches = list(old_batches[:batch_index])
        for candidate_index in range(batch_index, len(old_batches) + 1):
            batch_items = [candidate for candidate in self.items if candidate["batch_index"] == candidate_index]
            if batch_items:
                batches.append(build_batch(self.spec, candidate_index, batch_items))
        self.batches = batches

        window_index = event_index // self.spec.seal_length
        windows = list(old_windows[:window_index])
        max_window = (self.spec.event_count - 1) // self.spec.seal_length
        for candidate_index in range(window_index, max_window + 1):
            window_items = [candidate for candidate in self.items if candidate["window_index"] == candidate_index]
            if window_items:
                windows.append(build_window(self.spec, candidate_index, window_items, self.batches))
        self.windows = windows
        self.summary = build_summary(self.spec, self.items, self.batches, self.windows)
        new_sealed = self.sealed_digests()
        return {
            "new_event_seq": event_seq,
            "new_item_id": item["item_id"],
            "affected_items": [item["item_id"]],
            "recomputed_items": [item["item_id"]],
            "reused_item_count": event_index,
            "affected_batch_id": _batch_id(batch_index),
            "affected_window_id": _window_id(window_index),
            "sealed_windows_byte_unchanged": old_sealed == {k: v for k, v in new_sealed.items() if k in old_sealed},
            "propagation_basis": item["propagation_basis"],
            "sealed_digests_before": old_sealed,
            "sealed_digests_after": new_sealed,
            "result": self.result(),
        }

    def apply_open_window_script_edit(self, new_spec: SessionSpec, changed_seq: int) -> dict[str, Any]:
        old_sealed = self.sealed_digests()
        changed_index = changed_seq - self.spec.initial_seq
        sealed_count = (self.spec.event_count // self.spec.seal_length) * self.spec.seal_length
        if changed_index < sealed_count:
            from .errors import ScriptError

            raise ScriptError(
                "edit intersects a sealed window; versioned whole-snapshot replacement is required",
                event_seq=changed_seq,
                details={"changed_index": changed_index, "sealed_count": sealed_count},
            )
        self.spec = new_spec
        self.state = copy.deepcopy(self.pre_states[changed_index])
        self.items = self.items[:changed_index]
        self.pre_states = self.pre_states[:changed_index]
        old_batches = list(self.batches)
        old_windows = list(self.windows)
        for seq in range(changed_seq, new_spec.last_event_seq + 1):
            self.append_event(seq)

        first_batch = changed_index // new_spec.batch_size
        batches = list(old_batches[:first_batch])
        batch_count = (new_spec.event_count + new_spec.batch_size - 1) // new_spec.batch_size
        for batch_index in range(first_batch, batch_count):
            batch_items = [item for item in self.items if item["batch_index"] == batch_index]
            if batch_items:
                batches.append(build_batch(new_spec, batch_index, batch_items))
        self.batches = batches

        first_window = changed_index // new_spec.seal_length
        windows = list(old_windows[:first_window])
        max_window = (new_spec.event_count - 1) // new_spec.seal_length
        for window_index in range(first_window, max_window + 1):
            window_items = [item for item in self.items if item["window_index"] == window_index]
            if window_items:
                windows.append(build_window(new_spec, window_index, window_items, self.batches))
        self.windows = windows
        self.summary = build_summary(new_spec, self.items, self.batches, self.windows)
        result = self.result()
        new_sealed = self.sealed_digests()
        unchanged = old_sealed == {idx: digest for idx, digest in new_sealed.items() if idx in old_sealed}
        affected = [_item_id(seq) for seq in range(changed_seq, new_spec.last_event_seq + 1)]
        propagation_steps = []
        for offset, item in enumerate(self.items[changed_index:]):
            seq = item["event_seq"]
            propagation_steps.append(
                {
                    "item_id": item["item_id"],
                    "event_seq": seq,
                    "depends_on": None if offset == 0 else _item_id(seq - 1),
                    "basis": item["propagation_basis"],
                }
            )
        return {
            "changed_event_seq": changed_seq,
            "affected_items": affected,
            "recomputed_items": affected,
            "reused_item_count": changed_index,
            "sealed_window_digests_before": old_sealed,
            "sealed_window_digests_after": new_sealed,
            "sealed_windows_byte_unchanged": unchanged,
            "propagation_steps": propagation_steps,
            "result": result,
        }

    def apply_seal_length_view_edit(self, new_spec: SessionSpec) -> dict[str, Any]:
        old_digests = self.sealed_digests()
        old_length = self.spec.seal_length
        sealed_windows = [window for window in self.windows if window["sealed"]]
        open_windows = [window for window in self.windows if not window["sealed"]]
        open_start = (self.spec.event_count // old_length) * old_length
        affected = [_item_id(self.spec.initial_seq + i) for i in range(open_start, self.spec.event_count)]
        updated_open_windows: list[dict[str, Any]] = []
        for window in open_windows:
            updated = dict(window)
            updated["seal_length_before_edit"] = old_length
            updated["expected_length"] = new_spec.seal_length
            updated["seal_length_after_edit"] = new_spec.seal_length
            updated["sealed"] = len(window["item_ids"]) == new_spec.seal_length
            updated["sealed_at_event_seq"] = window["end_seq"] if updated["sealed"] else None
            updated["content_digest_before_seal_length_reference"] = window["content_digest"]
            updated["content_digest"] = sha256_hex(
                {k: v for k, v in updated.items() if k != "content_digest"}
            )
            updated_open_windows.append(updated)
        self.spec = new_spec
        self.windows = sealed_windows + updated_open_windows
        new_digests = {window["window_index"]: window["content_digest"] for window in self.windows if window["sealed"]}
        return {
            "old_seal_length": old_length,
            "new_seal_length": new_spec.seal_length,
            "affected_view_items": affected,
            "metric_items_recomputed": [],
            "basis": "historical sealed windows are retained; seal_length changes only the open-window threshold/reference",
            "old_sealed_digests": old_digests,
            "new_sealed_digests": new_digests,
            "sealed_window_digests_unchanged": old_digests == {k: v for k, v in new_digests.items() if k in old_digests},
            "open_windows": updated_open_windows,
        }


def build_observation(spec: SessionSpec, exchange: dict[str, Any] | None = None) -> dict[str, Any]:
    return ObservationWorkspace(spec, exchange).result()


def diff_script_entries(old: SessionSpec, new: SessionSpec) -> list[int]:
    old_by_seq = {entry.event_seq: entry.to_json() for entry in old.entries}
    new_by_seq = {entry.event_seq: entry.to_json() for entry in new.entries}
    changed: list[int] = []
    for seq in sorted(set(old_by_seq) | set(new_by_seq)):
        if old_by_seq.get(seq) != new_by_seq.get(seq):
            changed.append(seq)
    return changed
