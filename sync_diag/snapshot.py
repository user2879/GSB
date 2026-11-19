from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .errors import SyncError, require
from .hashing import sha256_file, sha256_json, write_json


def _snapshot_path(store_dir: Path, version: int, content_hash: str) -> Path:
    return store_dir / f"sync_snapshot.v{version:06d}.{content_hash[:16]}.json"


def _read_current(store_dir: Path) -> dict[str, Any] | None:
    pointer_path = store_dir / "current_snapshot.json"
    if not pointer_path.exists():
        return None
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SyncError("E_SNAPSHOT_STALE", "current snapshot pointer is unreadable", {"path": str(pointer_path), "reason": str(exc)}) from exc
    snapshot_path = store_dir / str(pointer.get("file", ""))
    if not snapshot_path.exists():
        raise SyncError("E_SNAPSHOT_STALE", "current snapshot file is missing", {"pointer": pointer})
    actual_file_hash = sha256_file(snapshot_path)
    require(
        actual_file_hash == pointer.get("file_sha256"),
        "E_SNAPSHOT_STALE",
        "current snapshot file hash does not match pointer",
        expected=pointer.get("file_sha256"),
        actual=actual_file_hash,
    )
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SyncError("E_SNAPSHOT_STALE", "snapshot JSON is unreadable", {"path": str(snapshot_path), "reason": str(exc)}) from exc
    require(
        sha256_json(snapshot.get("payload")) == snapshot.get("content_hash"),
        "E_SNAPSHOT_STALE",
        "snapshot content hash is invalid",
        expected=snapshot.get("content_hash"),
        actual=sha256_json(snapshot.get("payload")),
    )
    return snapshot


def publish_snapshot(store_dir: Path, version: int, payload: dict[str, Any]) -> dict[str, Any]:
    store_dir.mkdir(parents=True, exist_ok=True)
    content_hash = sha256_json(payload)
    current = _read_current(store_dir)
    if current is not None:
        current_version = int(current["version"])
        current_hash = str(current["content_hash"])
        if version < current_version:
            raise SyncError(
                "E_SNAPSHOT_STALE",
                "snapshot version is older than the current snapshot",
                {"incoming_version": version, "current_version": current_version},
            )
        if version == current_version and content_hash != current_hash:
            raise SyncError(
                "E_SNAPSHOT_STALE",
                "snapshot version is immutable and already has different content",
                {
                    "version": version,
                    "incoming_content_hash": content_hash,
                    "current_content_hash": current_hash,
                },
            )
        if version == current_version and content_hash == current_hash:
            pointer = json.loads((store_dir / "current_snapshot.json").read_text(encoding="utf-8"))
            return {**pointer, "idempotent": True}
    snapshot = {
        "schema_version": 1,
        "version": int(version),
        "content_hash": content_hash,
        "payload": payload,
    }
    path = _snapshot_path(store_dir, version, content_hash)
    write_json(path, snapshot)
    pointer = {
        "schema_version": 1,
        "version": int(version),
        "content_hash": content_hash,
        "file": path.name,
        "file_sha256": sha256_file(path),
    }
    write_json(store_dir / "current_snapshot.json", pointer)
    return {**pointer, "idempotent": False}


def load_current_snapshot(store_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot = _read_current(store_dir)
    if snapshot is None:
        raise SyncError("E_SNAPSHOT_STALE", "no current snapshot exists", {"store": str(store_dir)})
    pointer = json.loads((store_dir / "current_snapshot.json").read_text(encoding="utf-8"))
    return snapshot, pointer


def _polyline(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def render_diagnostic_view(store_dir: Path, output_path: Path) -> str:
    snapshot, pointer = load_current_snapshot(store_dir)
    payload = snapshot["payload"]
    estimate = payload["estimate"]
    verification = payload["verification"]
    sample_rate = int(estimate["sample_rate_hz"])
    frames = int(payload["capture"]["analysis_frames"])
    duration = frames / sample_rate
    width = 1400
    left = 145
    right = 35
    top = 105
    row_height = 82
    row_gap = 10
    plot_width = width - left - right
    height = top + 8 * (row_height + row_gap) + 55
    colors = ["#7a869a", "#1f77b4", "#d17b0f", "#2a9d8f", "#c44e52", "#7b5aa6", "#4f8f3b", "#b24c7a"]
    verification_by_channel = {int(channel["channel"]): channel for channel in verification["channels"]}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#252a34}.small{font-size:11px}.label{font-size:13px;font-weight:600}</style>',
        f'<text x="{left}" y="30" class="label">Clock Sync Diagnostic View</text>',
        f'<text x="{left}" y="50" class="small">snapshot v{snapshot["version"]} · hash {html.escape(snapshot["content_hash"][:16])} · {duration:.3f} s · 48 kHz · PCM24 · 8 ch</text>',
        f'<text x="{left}" y="68" class="small">drift curve (line) + post-alignment pulse residuals (dots); residual scale is per channel with ±max(|residual|, 0.01) sample</text>',
        f'<line x1="{left}" y1="{top - 12}" x2="{left + plot_width}" y2="{top - 12}" stroke="#8a8f98" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        seconds = duration * tick / 5.0
        x = left + plot_width * tick / 5.0
        elements.append(f'<line x1="{x:.2f}" y1="{top - 16}" x2="{x:.2f}" y2="{height - 35}" stroke="#e7e3dc" stroke-width="1"/>')
        elements.append(f'<text x="{x:.2f}" y="{height - 17}" text-anchor="middle" class="small">{seconds:.1f}s</text>')
    for channel_estimate in estimate["channels"]:
        channel = int(channel_estimate["channel"])
        row = channel
        y0 = top + row * (row_height + row_gap)
        center_y = y0 + row_height / 2.0
        color = colors[channel]
        elements.append(f'<text x="12" y="{y0 + 18}" class="label">CH {channel}</text>')
        role = "reference" if channel == 0 else "aligned slave"
        elements.append(f'<text x="12" y="{y0 + 36}" class="small">{role}</text>')
        if channel > 0:
            elements.append(f'<text x="12" y="{y0 + 52}" class="small">rate {channel_estimate.get("global_linear_rate_ppm", 0.0):+.4f} ppm</text>')
        elements.append(f'<rect x="{left}" y="{y0}" width="{plot_width}" height="{row_height}" rx="3" fill="#ffffff" stroke="#ded8ce"/>')
        curve = channel_estimate.get("curve", {})
        curve_x = np.asarray(curve.get("x_samples", []), dtype=np.float64)
        curve_y = np.asarray(curve.get("offset_samples", []), dtype=np.float64)
        if curve_x.size > 1:
            drift_scale = max(float(np.max(np.abs(curve_y - curve_y[0]))), 0.01)
            points = []
            for x_value, offset in zip(curve_x, curve_y):
                px = left + plot_width * min(max(x_value / frames, 0.0), 1.0)
                py = center_y - 24.0 * (offset - curve_y[0]) / drift_scale
                points.append((px, py))
            elements.append(f'<polyline points="{_polyline(points)}" fill="none" stroke="{color}" stroke-width="1.6"/>')
        channel_verification = verification_by_channel.get(channel)
        if channel_verification:
            pulses = channel_verification["pulses"]
            residuals = [float(pulse["residual_after_samples"]) for pulse in pulses]
            residual_scale = max(max((abs(value) for value in residuals), default=0.0), 0.01)
            for pulse in pulses:
                px = left + plot_width * min(max(float(pulse["reference_sample"]) / frames, 0.0), 1.0)
                py = center_y + 24.0 * float(pulse["residual_after_samples"]) / residual_scale
                elements.append(f'<line x1="{px:.2f}" y1="{y0 + 2}" x2="{px:.2f}" y2="{y0 + 7}" stroke="#c9c2b8" stroke-width="0.8"/>')
                elements.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="1.9" fill="{color}"/>')
            elements.append(f'<text x="{left + plot_width - 4}" y="{y0 + row_height - 5}" text-anchor="end" class="small">residual rms {channel_verification["after_rmse_samples"]:.5f} smp</text>')
    elements.append(f'<text x="{left}" y="{height - 5}" class="small">source snapshot sha256 {html.escape(pointer["file_sha256"])}</text>')
    elements.append("</svg>")
    data = ("\n".join(elements) + "\n").encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return sha256_file(output_path)
