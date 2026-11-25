"""Content-addressed, directory-atomic session snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import BizError, issue
from .fixed import CHANNEL_ORDER
from .io_utils import atomic_write_bytes, atomic_write_json, atomic_write_jsonl, read_json
from .material import channel_png, composite_png
from .models import (
    BudgetConfig,
    ChannelArtifact,
    ChannelSpec,
    CompositeArtifact,
    FixedConfig,
    LayerArtifact,
    LayerPartial,
    LayerSpec,
    LayerWeight,
    MaterialConfig,
    RenderConfig,
    RenderState,
    TextureData,
)

SNAPSHOT_FORMAT = "fixedtex-snapshot-v1"
CURRENT_FILE = "CURRENT"
SNAPSHOT_DIR = "snapshots"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def config_from_dict(value: dict[str, Any]) -> RenderConfig:
    layers = tuple(
        LayerSpec(
            index=int(item["index"]),
            frequency_milli=int(item["frequency_milli"]),
            amplitude_milli=int(item["amplitude_milli"]),
            seed=int(item["seed"]),
            valid=bool(item.get("valid", True)),
        )
        for item in value.get("layers", [])
    )
    fixed = value.get("fixed", {})
    budgets = value.get("budgets", {})
    return RenderConfig(
        name=str(value.get("name", "snapshot")),
        texture_path=str(value.get("texture_path", "")),
        resolution_tiers=tuple(int(item) for item in value.get("resolution_tiers", [])),
        resolution=int(value.get("resolution", 64)),
        seed=int(value.get("seed", 0)),
        algorithm=str(value.get("algorithm", "value2d")),
        layers=layers,
        fixed=FixedConfig(
            fraction_bits=int(fixed.get("fraction_bits", 12)),
            quant_step=int(fixed.get("quant_step", 4)),
            tie_rule=str(fixed.get("tie_rule", "ties_to_even")),
            guard_bits=int(fixed.get("guard_bits", 2)),
        ),
        budgets=BudgetConfig(
            layer_points=int(budgets.get("layer_points", 65536)),
            batch_points=int(budgets.get("batch_points", 262144)),
        ),
        blend_mode=str(value.get("blend_mode", "weighted")),
        preview_budget_ms=int(value.get("preview_budget_ms", 250)),
        thread_pool_size=int(value.get("thread_pool_size", 2)),
        commit_interval_commands=int(value.get("commit_interval_commands", 1)),
        error_tolerance_quant_steps=int(value.get("error_tolerance_quant_steps", 1)),
        source_path=str(value.get("source_path", "")),
        source_hash=str(value.get("source_hash", "")),
        raw=value.get("raw", {}),
    )


def material_from_dict(value: dict[str, Any]) -> MaterialConfig:
    channels: dict[str, ChannelSpec] = {}
    for name in CHANNEL_ORDER:
        item = value.get("channels", {}).get(name, {})
        weights = tuple(
            LayerWeight(
                index=int(layer["index"]),
                weight_q=int(layer["weight_q"]),
                raw_weight=float(layer.get("raw_weight", 0.0)),
            )
            for layer in item.get("layer_weights", [])
        )
        channels[name] = ChannelSpec(
            name=name,
            base_weight_q=int(item.get("base_weight_q", 1000)),
            layer_weights=weights,
            valid=bool(item.get("valid", True)),
        )
    return MaterialConfig(
        name=str(value.get("name", "snapshot")),
        channels=channels,
        tie_rule=str(value.get("tie_rule", "ties_to_even")),
        equal_weight_rule=str(value.get("equal_weight_rule", "lower_layer_index")),
        source_path=str(value.get("source_path", "")),
        source_hash=str(value.get("source_hash", "")),
        raw=value.get("raw", {}),
    )


@dataclass
class LoadedSnapshot:
    version: int
    state: RenderState
    config: RenderConfig
    material: MaterialConfig
    texture: TextureData
    manifest: dict[str, Any]
    snapshot_path: Path


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.snapshots = self.root / SNAPSHOT_DIR
        self.current_path = self.root / CURRENT_FILE

    def ensure(self) -> None:
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def current_info(self) -> dict[str, Any] | None:
        if not self.current_path.exists():
            return None
        try:
            value = read_json(self.current_path)
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def next_version(self) -> int:
        current = self.current_info()
        if current is None:
            versions: list[int] = []
            if self.snapshots.exists():
                for path in self.snapshots.iterdir():
                    if path.is_dir() and path.name.startswith("v"):
                        try:
                            versions.append(int(path.name[1:]))
                        except ValueError:
                            continue
            return (max(versions) if versions else 0) + 1
        try:
            return int(current.get("version", 0)) + 1
        except (TypeError, ValueError):
            return 1

    def save(
        self,
        state: RenderState,
        config: RenderConfig,
        material: MaterialConfig,
        texture: TextureData,
        *,
        source_config_hash: str,
        source_material_hash: str,
        version: int | None = None,
    ) -> dict[str, Any]:
        self.ensure()
        version = self.next_version() if version is None else int(version)
        temp_dir = self.snapshots / f".tmp-{uuid.uuid4().hex}"
        artifact_dir = temp_dir / "artifacts"
        output_dir = temp_dir / "outputs"
        artifact_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        artifact_files: dict[str, dict[str, str]] = {}

        def npz(path: Path, **arrays: Any) -> None:
            np.savez_compressed(path, **arrays)
            artifact_files[str(path.relative_to(temp_dir))] = {
                "sha256": _sha256_file(path),
                "logical": str(path.relative_to(temp_dir)),
            }

        for index, artifact in sorted(state.layers.items()):
            npz(
                artifact_dir / f"layer-{index:04d}.npz",
                raw=artifact.raw,
                quantized=artifact.quantized,
                residual=artifact.residual,
                decisions=artifact.decisions,
            )
        for index, partial in sorted(state.partials.items()):
            npz(
                artifact_dir / f"partial-{index:04d}.npz",
                raw=partial.raw,
                quantized=partial.quantized,
                residual=partial.residual,
                decisions=partial.decisions,
            )
        for name, artifact in state.channels.items():
            npz(artifact_dir / f"channel-{name}.npz", array=artifact.array)
        if state.composite is not None:
            npz(artifact_dir / "composite.npz", array=state.composite.array)
        npz(
            artifact_dir / "texture.npz",
            rgb=texture.rgb,
            metadata=np.asarray(
                [
                    texture.width,
                    texture.height,
                    texture.png_color_type,
                    texture.png_bit_depth,
                ],
                dtype=np.int64,
            ),
        )

        manifest = {
            "format": SNAPSHOT_FORMAT,
            "version": version,
            "source_config_hash": source_config_hash,
            "source_material_hash": source_material_hash,
            "effective_config_hash": state.effective_config_hash,
            "texture_content_key": texture.content_key,
            "texture_file_sha256": texture.file_sha256,
            "texture_path": texture.path,
            "texture_mode": texture.mode,
            "last_sequence": state.last_sequence,
            "last_texture_key": state.last_texture_key,
            "pending": state.pending,
            "failures": state.failures,
            "injection_hits": state.injection_hits,
            "last_preview": state.last_preview,
            "layers": {
                str(index): {
                    "key": item.key,
                    "texture_key": item.texture_key,
                    "parameter_key": item.parameter_key,
                    "frac_bits": item.frac_bits,
                    "quant_step": item.quant_step,
                    "file": f"artifacts/layer-{index:04d}.npz",
                }
                for index, item in sorted(state.layers.items())
            },
            "partials": {
                str(index): {
                    "key": item.key,
                    "texture_key": item.texture_key,
                    "parameter_key": item.parameter_key,
                    "frac_bits": item.frac_bits,
                    "quant_step": item.quant_step,
                    "points_done": item.points_done,
                    "file": f"artifacts/partial-{index:04d}.npz",
                }
                for index, item in sorted(state.partials.items())
            },
            "channels": {
                name: {
                    "key": item.key,
                    "layer_keys": list(item.layer_keys),
                    "layer_indices": list(item.layer_indices),
                    "frac_bits": item.frac_bits,
                    "quant_step": item.quant_step,
                    "tie_count": item.tie_count,
                    "rounding_count": item.rounding_count,
                    "parameter_key": item.parameter_key,
                    "blend_mode": item.blend_mode,
                    "file": f"artifacts/channel-{name}.npz",
                }
                for name, item in state.channels.items()
            },
            "composite": None
            if state.composite is None
            else {
                "key": state.composite.key,
                "channel_keys": list(state.composite.channel_keys),
                "frac_bits": state.composite.frac_bits,
                "quant_step": state.composite.quant_step,
                "parameter_key": state.composite.parameter_key,
                "file": "artifacts/composite.npz",
            },
            "ledger_count": len(state.ledger),
            "artifacts": artifact_files,
            "config": config.to_dict(),
            "material": material.to_dict(),
        }
        atomic_write_json(temp_dir / "config.json", manifest["config"])
        atomic_write_json(temp_dir / "material.json", manifest["material"])
        atomic_write_jsonl(temp_dir / "ledger.jsonl", state.ledger)
        atomic_write_json(temp_dir / "manifest.json", manifest)

        for channel in material.ordered_channels():
            if channel.name in state.channels:
                image = channel_png(state.channels[channel.name].array, state.channels[channel.name].frac_bits)
                image.save(output_dir / f"{channel.name}.png", format="PNG")
        if state.composite is not None:
            composite_png(state.composite.array, state.composite.frac_bits).save(
                output_dir / "output.png", format="PNG"
            )

        for path in temp_dir.rglob("*"):
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        target = self.snapshots / f"v{version:08d}"
        if target.exists():
            raise BizError("E_SNAPSHOT_IO", f"snapshot target already exists: {target}")
        os.replace(temp_dir, target)
        _fsync_directory(self.snapshots)

        pointer = {
            "version": version,
            "snapshot": target.name,
            "source_config_hash": source_config_hash,
            "effective_config_hash": state.effective_config_hash,
        }
        atomic_write_json(self.current_path, pointer)
        self.root.mkdir(parents=True, exist_ok=True)
        self._publish_outputs(target, state)
        atomic_write_jsonl(self.root / "ledger.jsonl", state.ledger)
        atomic_write_bytes(self.root / "VERSION", f"{version}\n".encode("ascii"))
        try:
            link_temp = self.root / f".current-{uuid.uuid4().hex}"
            os.symlink(os.path.relpath(target, self.root), link_temp)
            os.replace(link_temp, self.root / "current")
        except OSError:
            pass
        return manifest

    def _publish_outputs(self, snapshot_path: Path, state: RenderState) -> None:
        output_dir = snapshot_path / "outputs"
        if (output_dir / "output.png").exists():
            atomic_write_bytes(self.root / "output.png", (output_dir / "output.png").read_bytes())
        else:
            output_path = self.root / "output.png"
            if output_path.exists():
                output_path.unlink()
        channel_dir = self.root / "channels"
        channel_dir.mkdir(parents=True, exist_ok=True)
        for name in CHANNEL_ORDER:
            source = output_dir / f"{name}.png"
            target = channel_dir / f"{name}.png"
            if source.exists():
                atomic_write_bytes(target, source.read_bytes())
            elif target.exists():
                target.unlink()

    def load(self) -> tuple[LoadedSnapshot | None, list[dict[str, Any]]]:
        issues: list[dict[str, Any]] = []
        current = self.current_info()
        if current is None:
            return None, issues
        try:
            version = int(current["version"])
            snapshot_name = str(current["snapshot"])
        except (KeyError, TypeError, ValueError):
            return None, [
                issue("E_SNAPSHOT_CORRUPT", "CURRENT pointer is malformed", path=str(self.current_path), stage="snapshot")
            ]
        snapshot_path = self.snapshots / snapshot_name
        manifest_path = snapshot_path / "manifest.json"
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            return None, [
                issue("E_SNAPSHOT_CORRUPT", f"cannot read snapshot manifest: {exc}", path=str(manifest_path), stage="snapshot")
            ]
        if not isinstance(manifest, dict) or manifest.get("format") != SNAPSHOT_FORMAT:
            return None, [
                issue("E_SNAPSHOT_CORRUPT", "snapshot format is invalid", path=str(manifest_path), stage="snapshot")
            ]
        if int(manifest.get("version", -1)) != version:
            return None, [
                issue(
                    "E_SNAPSHOT_VERSION",
                    "CURRENT version disagrees with snapshot manifest",
                    path=str(manifest_path),
                    stage="snapshot",
                    pointer_version=version,
                    manifest_version=manifest.get("version"),
                )
            ]
        for relative, info in manifest.get("artifacts", {}).items():
            path = snapshot_path / relative
            if not path.exists() or _sha256_file(path) != info.get("sha256"):
                return None, [
                    issue(
                        "E_SNAPSHOT_CORRUPT",
                        "snapshot artifact hash mismatch",
                        path=str(path),
                        stage="snapshot",
                    )
                ]
        try:
            config = config_from_dict(manifest["config"])
            material = material_from_dict(manifest["material"])
            texture_npz = np.load(snapshot_path / "artifacts/texture.npz", allow_pickle=False)
            rgb = np.asarray(texture_npz["rgb"], dtype=np.uint8)
            metadata = np.asarray(texture_npz["metadata"], dtype=np.int64)
            texture = TextureData(
                path=str(manifest.get("texture_path", "")),
                width=int(metadata[0]),
                height=int(metadata[1]),
                mode=str(manifest.get("texture_mode", "RGB")),
                png_color_type=int(metadata[2]),
                png_bit_depth=int(metadata[3]),
                rgb=rgb,
                content_key=str(manifest["texture_content_key"]),
                file_sha256=str(manifest.get("texture_file_sha256", "")),
            )
        except (KeyError, TypeError, ValueError, OSError) as exc:
            return None, [
                issue("E_SNAPSHOT_CORRUPT", f"cannot deserialize snapshot payload: {exc}", path=str(snapshot_path), stage="snapshot")
            ]

        state = RenderState(
            version=version,
            last_sequence=int(manifest.get("last_sequence", 0)),
            effective_config_hash=str(manifest.get("effective_config_hash", "")),
            last_texture_key=str(manifest.get("last_texture_key", "")),
            pending=manifest.get("pending", []),
            failures=manifest.get("failures", []),
            injection_hits=manifest.get("injection_hits", []),
            last_preview=manifest.get("last_preview", {}),
        )
        try:
            for index_text, info in manifest.get("layers", {}).items():
                arrays = np.load(snapshot_path / info["file"], allow_pickle=False)
                index = int(index_text)
                state.layers[index] = LayerArtifact(
                    index=index,
                    key=str(info["key"]),
                    texture_key=str(info["texture_key"]),
                    parameter_key=str(info["parameter_key"]),
                    frac_bits=int(info["frac_bits"]),
                    quant_step=int(info["quant_step"]),
                    raw=np.asarray(arrays["raw"], dtype=np.int64),
                    quantized=np.asarray(arrays["quantized"], dtype=np.int64),
                    residual=np.asarray(arrays["residual"], dtype=np.int64),
                    decisions=np.asarray(arrays["decisions"], dtype=np.int8),
                )
            for index_text, info in manifest.get("partials", {}).items():
                arrays = np.load(snapshot_path / info["file"], allow_pickle=False)
                index = int(index_text)
                state.partials[index] = LayerPartial(
                    index=index,
                    key=str(info["key"]),
                    texture_key=str(info["texture_key"]),
                    parameter_key=str(info["parameter_key"]),
                    frac_bits=int(info["frac_bits"]),
                    quant_step=int(info["quant_step"]),
                    raw=np.asarray(arrays["raw"], dtype=np.int64),
                    quantized=np.asarray(arrays["quantized"], dtype=np.int64),
                    residual=np.asarray(arrays["residual"], dtype=np.int64),
                    decisions=np.asarray(arrays["decisions"], dtype=np.int8),
                    points_done=int(info["points_done"]),
                )
            for name, info in manifest.get("channels", {}).items():
                arrays = np.load(snapshot_path / info["file"], allow_pickle=False)
                state.channels[name] = ChannelArtifact(
                    name=name,
                    key=str(info["key"]),
                    layer_keys=tuple(str(item) for item in info["layer_keys"]),
                    layer_indices=tuple(int(item) for item in info["layer_indices"]),
                    array=np.asarray(arrays["array"], dtype=np.int64),
                    frac_bits=int(info["frac_bits"]),
                    quant_step=int(info["quant_step"]),
                    tie_count=int(info["tie_count"]),
                    rounding_count=int(info["rounding_count"]),
                    parameter_key=str(info.get("parameter_key", "")),
                    blend_mode=str(info.get("blend_mode", "")),
                )
            composite_info = manifest.get("composite")
            if composite_info:
                arrays = np.load(snapshot_path / composite_info["file"], allow_pickle=False)
                state.composite = CompositeArtifact(
                    key=str(composite_info["key"]),
                    channel_keys=tuple(str(item) for item in composite_info["channel_keys"]),
                    array=np.asarray(arrays["array"], dtype=np.int64),
                    frac_bits=int(composite_info["frac_bits"]),
                    quant_step=int(composite_info["quant_step"]),
                    parameter_key=str(composite_info.get("parameter_key", "")),
                )
            ledger_path = snapshot_path / "ledger.jsonl"
            state.ledger = []
            if ledger_path.exists():
                for line in ledger_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        if isinstance(row, dict):
                            state.ledger.append(row)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            return None, [
                issue("E_SNAPSHOT_CORRUPT", f"cannot restore render state: {exc}", path=str(snapshot_path), stage="snapshot")
            ]
        return (
            LoadedSnapshot(
                version=version,
                state=state,
                config=config,
                material=material,
                texture=texture,
                manifest=manifest,
                snapshot_path=snapshot_path,
            ),
            issues,
        )

    def clear_active(self) -> None:
        if self.current_path.exists():
            self.current_path.unlink()
