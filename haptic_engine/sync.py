"""Inter-device synchronization anchor alignment diagnostics."""

from __future__ import annotations

from .envelope import ms_to_samples
from .errors import E_SYNC, HapticError
from .models import Device, Manifest, SyncAnchor
from .packing import Block


def _command_start_at_sample(blocks: list[Block], sample: int) -> int | None:
    """Return the start sample of the command observable at ``sample``."""
    for block in blocks:
        for command in block.commands:
            if command.start_sample <= sample < command.end_sample:
                return command.start_sample
    return None


def evaluate_anchors(
    manifest: Manifest,
    device_map: dict[str, Device],
    block_streams: dict[str, list[Block]],
) -> tuple[list[dict], list[dict]]:
    """Return per-anchor diagnostics and any violations.

    Deviation is measured against the reference device both in samples and in
    milliseconds (converted with each device's own sample rate). A violation
    raises :class:`HapticError` with ``E_SYNC``; the report rows are still
    returned so callers can persist them alongside valid command streams.
    """
    rows = build_anchor_rows(manifest, device_map, block_streams)
    violations: list[dict] = []
    for anchor, row in zip(manifest.sync_anchors, rows):
        for device_row in row["devices"]:
            if device_row["abs_deviation_samples"] > anchor.allowed_samples:
                violations.append(
                    {
                        "anchor_id": anchor.anchor_id,
                        "device_id": device_row["device_id"],
                        "deviation_samples": device_row["deviation_samples"],
                        "allowed_samples": anchor.allowed_samples,
                    }
            )
    if violations:
        raise HapticError(
            E_SYNC,
            "synchronization anchor deviation exceeds tolerance",
            violations=violations,
        )
    return rows, violations


def build_anchor_rows(
    manifest: Manifest,
    device_map: dict[str, Device],
    block_streams: dict[str, list[Block]],
) -> list[dict]:
    return [
        _evaluate_one(anchor, device_map, block_streams)
        for anchor in manifest.sync_anchors
    ]


def _evaluate_one(
    anchor: SyncAnchor,
    device_map: dict[str, Device],
    block_streams: dict[str, list[Block]],
) -> dict:
    device_rows = []
    for device_id in anchor.device_ids:
        device = device_map[device_id]
        expected_sample = ms_to_samples(anchor.time_ms, device.sample_rate_hz)
        command_start = _command_start_at_sample(block_streams[device_id], expected_sample)
        if command_start is None:
            deviation_samples: int | None = None
            deviation_ms: float | None = None
        else:
            # Local-grid sample offset between the actual command start and the
            # ideal anchor sample; milliseconds use the device sample period.
            deviation_samples = command_start - expected_sample
            deviation_ms = deviation_samples * 1000.0 / device.sample_rate_hz
        device_rows.append(
            {
                "device_id": device_id,
                "expected_sample": expected_sample,
                "command_start_sample": command_start,
                "deviation_samples": deviation_samples,
                "abs_deviation_samples": abs(deviation_samples) if deviation_samples is not None else 10**9,
                "deviation_ms": deviation_ms,
            }
        )
    reference_row = next(row for row in device_rows if row["device_id"] == anchor.reference_device)
    reference_deviation = reference_row["deviation_samples"] or 0
    for row in device_rows:
        if row["device_id"] == anchor.reference_device or row["deviation_samples"] is None:
            continue
        # Relative deviation versus the reference device, kept in this device's
        # sample grid plus a millisecond representation.
        row["relative_to_reference_samples"] = row["deviation_samples"] - reference_deviation
        row["relative_to_reference_ms"] = row["deviation_ms"]
    return {
        "anchor_id": anchor.anchor_id,
        "time_ms": anchor.time_ms,
        "reference_device": anchor.reference_device,
        "reference_sample": reference_row["expected_sample"],
        "devices": device_rows,
    }
