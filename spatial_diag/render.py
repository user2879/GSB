"""Deterministic equirectangular diagnostic rendering from snapshot payloads."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from .canonical import atomic_write_json, canonical_hash, file_records
from .snapshot import SnapshotError, SnapshotStore


def _validate_snapshot(snapshot: dict[str, Any]) -> None:
    payload = {
        key: value
        for key, value in snapshot.items()
        if key not in {"version", "integrity"}
    }
    if canonical_hash(payload) != snapshot.get("integrity", {}).get("content_sha256"):
        raise SnapshotError("refusing to render a snapshot whose content hash does not verify")


def _difference_field(fields: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    reconstructed = np.asarray(fields["reconstructed_normalized"], dtype=np.float64)
    reference = np.asarray(fields["reference_normalized"], dtype=np.float64)
    difference = (reconstructed - reference) / np.maximum(reference, 1.0e-12)
    return np.clip(difference, -1.0, 1.0).reshape(shape)


def render_snapshot_to_directory(snapshot: dict[str, Any], output_dir: Path) -> list[Path]:
    _validate_snapshot(snapshot)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    version = snapshot["version"]
    for tier_name, tier_data in snapshot["tiers"].items():
        fields = tier_data["fields"]
        shape = tuple(int(v) for v in fields["shape"])
        reconstructed = np.asarray(fields["reconstructed_normalized"], dtype=np.float64).reshape(shape)
        reference = np.asarray(fields["reference_normalized"], dtype=np.float64).reshape(shape)
        relative = np.asarray(fields["relative_error"], dtype=np.float64).reshape(shape)
        low_grid = np.asarray(fields["low_confidence_grid"], dtype=np.int8).reshape(shape)
        difference = _difference_field(fields, shape)
        azimuth = np.asarray(fields["azimuth_deg"], dtype=np.float64)
        elevation = np.asarray(fields["elevation_deg"], dtype=np.float64)
        projection = tier_data["projection"]
        decision = tier_data["decision"]
        validation = tier_data["validation"]
        sample_directions = np.asarray(projection["sample_directions"], dtype=np.float64)
        low_samples = np.asarray(projection["low_confidence"], dtype=np.int8).astype(bool)
        sample_lon = np.degrees(np.arctan2(sample_directions[:, 1], sample_directions[:, 0]))
        sample_lat = np.degrees(np.arcsin(np.clip(sample_directions[:, 2], -1.0, 1.0)))

        fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6), constrained_layout=True)
        energy_image = axes[0, 0].imshow(
            reconstructed,
            origin="lower",
            extent=(-180.0, 180.0, -90.0, 90.0),
            aspect="auto",
            cmap="viridis",
        )
        axes[0, 0].set_title("Reconstructed directional energy")
        if np.any(low_samples):
            axes[0, 0].scatter(
                sample_lon[low_samples],
                sample_lat[low_samples],
                marker="x",
                s=22,
                linewidths=0.9,
                color="#ff2d95",
                label="low confidence",
            )
            axes[0, 0].legend(loc="upper right", fontsize=7)
        fig.colorbar(energy_image, ax=axes[0, 0], shrink=0.78, label="normalized energy")

        diff_image = axes[0, 1].imshow(
            difference,
            origin="lower",
            extent=(-180.0, 180.0, -90.0, 90.0),
            aspect="auto",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )
        axes[0, 1].set_title("Reconstructed - reference (relative)")
        fig.colorbar(diff_image, ax=axes[0, 1], shrink=0.78, label="relative difference")

        axes[1, 0].imshow(
            np.where(low_grid > 0, 1.0, 0.0),
            origin="lower",
            extent=(-180.0, 180.0, -90.0, 90.0),
            aspect="auto",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        axes[1, 0].set_title("Low-confidence projection mask")
        axes[1, 0].set_xlabel("azimuth (deg)")
        axes[1, 0].set_ylabel("elevation (deg)")

        ledger = axes[1, 1]
        ledger.axis("off")
        ledger.set_title("Decision ledger and error interval")
        lines = [
            f"snapshot: {version}",
            f"tier: {tier_name}  sources={decision['inputs']['source_positions']}  sphere={decision['inputs']['sphere_points']}",
            f"strategy: {decision['selected_strategy']}  order={decision['selected_order']}",
            f"peak memory: {decision['peak_memory_bytes']} / {decision['inputs']['memory_budget_bytes']} bytes",
            f"error bound: {decision['resolution_error_bound_deg']:.3f} deg (budget {decision['inputs']['resolution_budget_deg']:.3f})",
            f"relative error p50/p95/max: {validation['selected']['error_interval']['p50']:.4f} / {validation['selected']['error_interval']['p95']:.4f} / {validation['selected']['error_interval']['max']:.4f}",
            f"over-tolerance directions: {validation['selected']['over_tolerance_direction_ratio']:.3%}",
            f"worst direction: az={validation['selected']['worst_direction']['azimuth_deg']:.1f} lat={validation['selected']['worst_direction']['elevation_deg']:.1f}",
            "degradations:",
        ]
        if decision["degradations"]:
            for item in decision["degradations"]:
                lines.append(f"  - {item['kind']}: {item.get('reason', '')}")
        else:
            lines.append("  - none")
        lines.append("low-confidence directions never enter equal-weight fitting")
        ledger.text(0.01, 0.96, "\n".join(lines), va="top", ha="left", fontsize=8.4, family="DejaVu Sans")
        for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
            axis.set_xlabel("azimuth (deg)")
            axis.set_ylabel("elevation (deg)")
            axis.set_xticks([-180, -90, 0, 90, 180])
            axis.set_yticks([-90, -45, 0, 45, 90])
        fig.suptitle(
            f"Spatial acoustic diagnostic | {tier_name} | immutable snapshot {version}",
            fontsize=13,
        )
        image_path = output_dir / f"{tier_name}.png"
        fig.savefig(
            image_path,
            dpi=120,
            format="png",
            metadata={"Software": "spatial-diagnostic/1.0.0"},
        )
        plt.close(fig)
        outputs.append(image_path)

    manifest = {
        "schema": "spatial-diagnostic-view-manifest/1",
        "snapshot_version": version,
        "snapshot_content_sha256": snapshot["integrity"]["content_sha256"],
        "files": file_records(outputs, output_dir),
        "policy": "rendered only from the verified immutable snapshot payload",
    }
    manifest["content_hash"] = canonical_hash(manifest)
    manifest_path = output_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    outputs.append(manifest_path)
    return outputs


def render_committed_snapshot(store: SnapshotStore) -> dict[str, Any]:
    snapshot = store.load_current()
    existing = store.views / snapshot["version"]
    if existing.is_dir() and (existing / "manifest.json").is_file():
        import json

        with (existing / "manifest.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("snapshot_content_sha256") == snapshot["integrity"]["content_sha256"]:
            return {
                "view_dir": existing,
                "version": snapshot["version"],
                "reused": True,
                "files": manifest.get("files", []),
            }
    staging = Path(tempfile.mkdtemp(prefix="views-", dir=str(store.staging)))
    try:
        outputs = render_snapshot_to_directory(snapshot, staging)
        final = store.publish_views(snapshot["version"], staging)
        with (final / "manifest.json").open("r", encoding="utf-8") as handle:
            import json

            manifest = json.load(handle)
        return {
            "view_dir": final,
            "version": snapshot["version"],
            "reused": False,
            "files": manifest.get("files", []),
            "rendered_files": [path.name for path in outputs],
        }
    except Exception:
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
        raise

