#!/usr/bin/env python3
"""Generate the deterministic local sample set used by the workbench."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lossdiag.model import FLAG_DATA, FLAG_REORDER, FLAG_RETRANSMIT, FiveTuple
from lossdiag.util import atomic_write_text, pretty_json
from lossdiag.wire import encode_frame


FLOW_A = FiveTuple("10.10.0.1", "10.20.0.1", 40000, 5000, "tcp")
FLOW_B = FiveTuple("10.10.0.2", "10.20.0.1", 41000, 5001, "tcp")
FLOW_C = FiveTuple("10.10.0.3", "10.20.0.1", 42000, 5002, "udp")


def payload(seq: int, size: int = 12) -> bytes:
    return bytes((seq + index) % 251 for index in range(size))


def frame(flow: FiveTuple, seq: int, send_ms: int, arrival: int, flags: int = FLAG_DATA) -> bytes:
    return encode_frame(
        five_tuple=flow,
        seq=seq,
        send_ms=send_ms,
        flags=flags,
        arrival_ordinal=arrival,
        payload=payload(seq),
    )


def flow_dict(flow: FiveTuple) -> dict:
    return flow.to_dict()


def event(flow: FiveTuple, seq: int, count: int, **extra: object) -> dict:
    record = {"five_tuple": flow_dict(flow), "event_seq": seq, "loss_count": count}
    record.update(extra)
    return record


def main() -> int:
    samples = ROOT / "samples"
    samples.mkdir(parents=True, exist_ok=True)

    frames = [
        frame(FLOW_A, 1, 1000, 1),
        frame(FLOW_A, 2, 1010, 2),
        frame(FLOW_A, 4, 1030, 3),
        frame(FLOW_A, 5, 1040, 4),
        frame(FLOW_B, 20, 1200, 5),
        frame(FLOW_B, 21, 1210, 6),
        frame(FLOW_B, 22, 1220, 7),
        frame(FLOW_B, 24, 1240, 8),
        frame(FLOW_B, 23, 1230, 9, FLAG_DATA | FLAG_REORDER),
        frame(FLOW_A, 5, 1050, 10, FLAG_DATA | FLAG_RETRANSMIT),
        frame(FLOW_A, 7, 1070, 11, FLAG_DATA | FLAG_REORDER),
        frame(FLOW_A, 6, 1060, 12),
        frame(FLOW_A, 8, 1080, 13),
        frame(FLOW_A, 10, 1100, 14),
        frame(FLOW_C, 40, 1400, 15),
        frame(FLOW_C, 41, 1410, 16),
        frame(FLOW_C, 42, 1420, 17),
        frame(FLOW_C, 43, 1430, 18),
        frame(FLOW_C, 45, 1450, 19),
        frame(FLOW_B, 26, 1260, 20),
        frame(FLOW_B, 28, 1280, 21),
        frame(FLOW_B, 29, 1290, 22),
        frame(FLOW_B, 30, 1300, 23),
        frame(FLOW_C, 47, 1470, 24),
        frame(FLOW_C, 48, 1480, 25),
        frame(FLOW_C, 49, 1490, 26),
        frame(FLOW_C, 50, 1500, 27),
    ]
    (samples / "observations.bin").write_bytes(b"".join(frames))
    atomic_write_text(
        samples / "wire_manifest.json",
        pretty_json(
            {
                "model_version": "obs-frame-v1",
                "source_id": "capture-local-01",
                "time_basis": {
                    "clock": "sender_monotonic_ms",
                    "unit": "ms",
                    "epoch": "sender-boot-7",
                },
                "alignment_version": "monotonic-offset-v1",
                "loopback_port": 47731,
            }
        )
        + "\n",
    )

    common_basis = {
        "clock": "collector_monotonic_ms",
        "unit": "ms",
        "epoch": "collector-boot-3",
    }
    alignment = {
        "offset_to_frame_ms": 1000,
        "basis_version": "monotonic-offset-v1",
    }

    v1 = {
        "schema": "loss-events",
        "version": "v1",
        "source_id": "events-v1-local",
        "time_basis": common_basis,
        "alignment": alignment,
        "normalization_defaults": {
            "link_id": None,
            "time_window_ms": None,
            "retransmit": False,
            "reorder_depth": 0,
        },
        "records": [
            event(FLOW_A, 3, 1),
            event(FLOW_A, 5, 1),
            event(FLOW_A, 6, 2),
            event(FLOW_A, 9, 1),
            event(FLOW_B, 23, 2),
            event(FLOW_B, 25, 1),
            event(FLOW_B, 27, 1),
            event(FLOW_C, 44, 1),
            event(FLOW_C, 46, 1),
        ],
    }
    atomic_write_text(samples / "events_v1.json", pretty_json(v1) + "\n")

    def window(start: int, end: int) -> dict:
        return {"start_ms": start, "end_ms": end}

    v2 = {
        "schema": "loss-events",
        "version": "v2",
        "source_id": "events-v2-local",
        "time_basis": common_basis,
        "alignment": alignment,
        "normalization_defaults": {"retransmit": False, "reorder_depth": 0},
        "records": [
            event(FLOW_A, 3, 1, link_id="link-A", time_window_ms=window(15, 35)),
            event(FLOW_A, 5, 1, link_id="link-A", time_window_ms=window(40, 50)),
            event(FLOW_A, 6, 2, link_id="link-A", time_window_ms=window(60, 70)),
            event(FLOW_A, 9, 1, link_id="link-A", time_window_ms=window(100, 110)),
            event(FLOW_B, 23, 2, link_id="link-B", time_window_ms=window(225, 245)),
            event(FLOW_B, 25, 1, link_id="link-B", time_window_ms=window(255, 265)),
            event(FLOW_B, 27, 1, link_id="link-B", time_window_ms=window(275, 285)),
            event(FLOW_C, 44, 1, link_id="link-C", time_window_ms=window(445, 455)),
            event(FLOW_C, 46, 1, link_id="link-C", time_window_ms=window(465, 475)),
        ],
    }
    atomic_write_text(samples / "events_v2.json", pretty_json(v2) + "\n")

    v3 = {
        "schema": "loss-events",
        "version": "v3",
        "source_id": "events-v3-local",
        "time_basis": common_basis,
        "alignment": alignment,
        "normalization_defaults": {},
        "records": [
            event(FLOW_A, 3, 1, link_id="link-A", time_window_ms=window(15, 35), retransmit=False, reorder_depth=0),
            event(FLOW_A, 5, 1, link_id="link-A", time_window_ms=window(40, 50), retransmit=True, reorder_depth=0),
            event(FLOW_A, 6, 2, link_id="link-A", time_window_ms=window(60, 70), retransmit=False, reorder_depth=1),
            event(FLOW_A, 9, 1, link_id="link-A", time_window_ms=window(100, 110), retransmit=False, reorder_depth=0),
            event(FLOW_B, 23, 2, link_id="link-B", time_window_ms=window(225, 245), retransmit=False, reorder_depth=2),
            event(FLOW_B, 25, 1, link_id="link-B", time_window_ms=window(255, 265), retransmit=False, reorder_depth=0),
            event(FLOW_B, 27, 1, link_id="link-B", time_window_ms=window(275, 285), retransmit=False, reorder_depth=0),
            event(FLOW_C, 44, 1, link_id="link-C", time_window_ms=window(445, 455), retransmit=False, reorder_depth=0),
            event(FLOW_C, 46, 1, link_id="link-C", time_window_ms=window(465, 475), retransmit=False, reorder_depth=0),
        ],
    }
    atomic_write_text(samples / "events_v3.json", pretty_json(v3) + "\n")
    print(f"generated deterministic samples in {samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
