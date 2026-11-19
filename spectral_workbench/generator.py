"""Deterministic local spectral-cube generator and truth manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .config import CubeConfig, InjectionConfig, resolve_centers
from .errors import ErrorCode, WorkbenchError
from .io_utils import hash_tree, key_for, read_json, sha256_array, sha256_bytes, write_bytes, write_json


@dataclass(frozen=True)
class GeneratedCube:
    root: Path
    manifest_path: Path
    content_hash: str
    frame_count: int


def _endmember_spectra(config: CubeConfig) -> Tuple[np.ndarray, List[str]]:
    bands = np.arange(config.frames, dtype=np.float64)
    columns: List[np.ndarray] = []
    names: List[str] = []
    margin = max(2.0, config.frames / (config.fluorophores * 2.0))
    centers = np.linspace(margin, config.frames - 1 - margin, config.fluorophores)
    width = max(2.6, config.frames / (config.fluorophores * 2.25))
    for index, center in enumerate(centers):
        primary = np.exp(-0.5 * ((bands - center) / width) ** 2)
        shoulder_center = center + (width * 2.4 if index % 2 == 0 else -width * 2.4)
        shoulder = 0.12 * np.exp(-0.5 * ((bands - shoulder_center) / (width * 1.8)) ** 2)
        spectrum = primary + shoulder
        spectrum /= float(np.max(spectrum))
        columns.append(config.endmember_scale_dn * spectrum)
        names.append(f"fluorophore_{index}")
    return np.stack(columns, axis=1), names


def _alloc_centers(config: CubeConfig) -> Tuple[Tuple[float, float], ...]:
    if config.fluorophores == 4:
        return resolve_centers(config.height, config.width)
    columns = 2
    rows = int(np.ceil(config.fluorophores / columns))
    centers: List[Tuple[float, float]] = []
    for k in range(config.fluorophores):
        row = k // columns
        column = k % columns
        x = config.width * (column + 1) / (columns + 1)
        y = config.height * (row + 1) / (rows + 1)
        centers.append((float(x), float(y)))
    return tuple(centers)


def _peak_concentrations(count: int) -> Tuple[float, ...]:
    pattern = (0.90, 1.20, 0.70, 1.40, 1.05, 0.80, 1.30, 0.95)
    return tuple(pattern[k % len(pattern)] for k in range(count))


def _spatial_grid(height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    y = np.arange(height, dtype=np.float64)[:, None]
    x = np.arange(width, dtype=np.float64)[None, :]
    return x, y


def _concentration_maps(
    config: CubeConfig,
    centers: Sequence[Tuple[float, float]],
    peaks: Sequence[float],
) -> np.ndarray:
    x, y = _spatial_grid(config.height, config.width)
    maps = np.zeros((config.fluorophores, config.height, config.width), dtype=np.float64)
    for k, ((cx, cy), peak) in enumerate(zip(centers, peaks)):
        radius2 = (x - cx) ** 2 + (y - cy) ** 2
        maps[k] = peak * np.exp(-0.5 * radius2 / (config.blob_sigma_px**2))
    return maps


def _make_masks(maps: np.ndarray) -> np.ndarray:
    masks = np.zeros(maps.shape, dtype=np.uint8)
    for k in range(maps.shape[0]):
        peak = float(np.max(maps[k]))
        if peak > 0:
            masks[k] = maps[k] >= (0.25 * peak)
    return masks


def _dark_frame(config: CubeConfig, rng: np.random.Generator) -> np.ndarray:
    x, y = _spatial_grid(config.height, config.width)
    ramp = 0.035 * x + 0.022 * y
    ripple = 0.45 * np.sin(x / 11.0) * np.cos(y / 13.0)
    values = config.dark_level_dn + ramp + ripple + rng.normal(0.0, 0.35, (config.height, config.width))
    return np.clip(np.rint(values), 0, 65535).astype(np.uint16)


def _flat_gain(config: CubeConfig, band: int) -> np.ndarray:
    x, y = _spatial_grid(config.height, config.width)
    xn = (x - (config.width - 1) / 2.0) / max(1.0, config.width / 2.0)
    yn = (y - (config.height - 1) / 2.0) / max(1.0, config.height / 2.0)
    vignetting = 1.0 - 0.055 * (xn**2 + yn**2)
    gradient = 0.018 * xn - 0.012 * yn
    ripple = 0.004 * np.sin((band + 1) * 0.17 + x / 19.0) * np.cos(y / 23.0)
    return vignetting + gradient + ripple


def _write_u16(path: Path, array: np.ndarray) -> str:
    prepared = np.ascontiguousarray(array, dtype="<u2")
    raw = prepared.tobytes(order="C")
    write_bytes(path, raw)
    return sha256_bytes(raw)


def generate_cube(config: CubeConfig, injection: InjectionConfig, output_dir: str | Path) -> GeneratedCube:
    """Generate all raw frames and truth artifacts using a fixed PCG64 stream."""
    config.validate()
    injection.validate(config.frames, config.fluorophores, config.height, config.width)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("raw", "dark", "flat", "truth"):
        (root / relative).mkdir(parents=True, exist_ok=True)

    endmembers, names = _endmember_spectra(config)
    if injection.degenerate_endmembers is not None:
        item = injection.degenerate_endmembers
        source = endmembers[:, item.source].copy()
        endmembers[:, item.target] = item.blend * source + (1.0 - item.blend) * endmembers[:, item.target]
    endmember_bytes = np.ascontiguousarray(endmembers, dtype="<f8").tobytes(order="C")
    endmember_hash = sha256_bytes(endmember_bytes)
    write_bytes(root / "endmembers.f64", endmember_bytes)
    write_json(
        root / "endmembers.json",
        {
            "shape": [config.frames, config.fluorophores],
            "dtype": "float64",
            "sha256": endmember_hash,
            "path": "endmembers.f64",
            "unit": "corrected_DN_per_concentration_unit",
            "fluorophore_names": names,
        },
    )

    reference_centers = _alloc_centers(config)
    peaks = _peak_concentrations(config.fluorophores)
    rng = np.random.Generator(np.random.PCG64(config.seed))
    frame_records: List[Dict[str, Any]] = []
    frame_stats: List[Dict[str, Any]] = []
    bias = injection.concentration_bias
    shift_injection = injection.center_shift

    for frame in range(config.frames):
        dx = config.drift_dx_per_frame * frame
        dy = config.drift_dy_per_frame * frame
        if shift_injection is not None and shift_injection.frame_start <= frame < shift_injection.frame_end:
            dx += shift_injection.dx_px
            dy += shift_injection.dy_px
        centers = tuple((cx + dx, cy + dy) for cx, cy in reference_centers)

        truth_maps = _concentration_maps(config, centers, peaks)
        masks = _make_masks(truth_maps)
        measured_maps = truth_maps.copy()
        injected_bias_record: Dict[str, Any] = {}
        if bias is not None:
            measured_maps[bias.fluorophore, bias.y0 : bias.y0 + bias.height, bias.x0 : bias.x0 + bias.width] += (
                bias.delta
            )
            injected_bias_record = {
                "fluorophore": bias.fluorophore,
                "delta": bias.delta,
                "bbox": [bias.y0, bias.x0, bias.height, bias.width],
                "anchor_frame": bias.frame,
                "applied_to_all_spectral_bands": True,
                "truth_unchanged": True,
            }

        dark = _dark_frame(config, rng)
        if injection.dark_inconsistency is not None:
            item = injection.dark_inconsistency
            if item.band_start <= frame < item.band_end:
                dark = np.clip(dark.astype(np.float64) + item.offset_dn, 0, 65535).astype(np.uint16)
        flat_gain = _flat_gain(config, frame)
        # The calibrator normalizes (flat-dark) by its mean. Apply the same
        # normalized gain to the observed signal so absolute concentration units
        # round-trip through calibration.
        normalized_gain = flat_gain / float(np.mean(flat_gain))
        flat_noise = rng.normal(0.0, 0.45, (config.height, config.width))
        flat = np.clip(
            np.rint(dark.astype(np.float64) + config.flat_mean_dn * flat_gain + flat_noise),
            0,
            65535,
        ).astype(np.uint16)

        spectral_signal = np.tensordot(endmembers[frame], measured_maps, axes=(0, 0))
        spectral_signal += config.background_dn
        scene_noise = rng.normal(0.0, config.noise_sigma_dn, (config.height, config.width))
        observed = dark.astype(np.float64) + normalized_gain * (spectral_signal + scene_noise)
        raw = np.clip(np.rint(observed), 0, 65535).astype(np.uint16)
        if injection.saturation_overload is not None and frame == injection.saturation_overload.frame:
            item = injection.saturation_overload
            raw[item.y0 : item.y0 + item.height, item.x0 : item.x0 + item.width] = 65535

        raw_path = f"raw/raw_{frame:04d}.bin"
        dark_path = f"dark/dark_{frame:04d}.bin"
        flat_path = f"flat/flat_{frame:04d}.bin"
        raw_hash = _write_u16(root / raw_path, raw)
        dark_hash = _write_u16(root / dark_path, dark)
        flat_hash = _write_u16(root / flat_path, flat)

        masks_path = f"truth/masks_{frame:04d}.bin"
        mask_bytes = np.ascontiguousarray(masks, dtype="u1").tobytes(order="C")
        write_bytes(root / masks_path, mask_bytes)
        mask_hash = sha256_bytes(mask_bytes)
        blobs: List[Dict[str, Any]] = []
        for k, ((cx, cy), (rx, ry)) in enumerate(zip(centers, reference_centers)):
            ys, xs = np.nonzero(masks[k])
            bbox = None
            if xs.size:
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            blobs.append(
                {
                    "fluorophore": k,
                    "channel_index": k,
                    "center": [float(cx), float(cy)],
                    "reference_center": [float(rx), float(ry)],
                    "true_concentration": float(peaks[k]),
                    "mask_pixels": int(xs.size),
                    "mask_bbox_xyxy": bbox,
                }
            )
        truth_record = {
            "frame": frame,
            "band_index": frame,
            "reference_centers": [list(value) for value in reference_centers],
            "centers": [list(value) for value in centers],
            "drift_xy": [float(dx), float(dy)],
            "blobs": blobs,
            "masks": {"path": masks_path, "shape": [config.fluorophores, config.height, config.width], "sha256": mask_hash},
            "injected_bias": injected_bias_record,
        }
        truth_path = f"truth/frame_{frame:04d}.json"
        write_json(root / truth_path, truth_record)
        frame_records.append(
            {
                "frame": frame,
                "raw": {"path": raw_path, "sha256": raw_hash},
                "dark": {"path": dark_path, "sha256": dark_hash},
                "flat": {"path": flat_path, "sha256": flat_hash},
                "truth": {"path": truth_path},
            }
        )
        frame_stats.append(
            {
                "frame": frame,
                "raw_peak": int(raw.max()),
                "raw_mean": float(raw.mean(dtype=np.float64)),
                "saturated_pixels": int(np.count_nonzero(raw == 65535)),
                "dark_mean": float(dark.mean(dtype=np.float64)),
                "flat_mean": float(flat.mean(dtype=np.float64)),
            }
        )

    write_json(root / "frame_stats.json", {"frames": frame_stats})
    write_json(root / "injections.json", injection.to_dict())
    content_payload = {
        "format_version": 1,
        "cube": config.to_dict(),
        "shape": [config.frames, config.height, config.width],
        "dtype": "uint16-le",
        "endmembers": {"path": "endmembers.f64", "shape": [config.frames, config.fluorophores], "sha256": endmember_hash},
        "reference_centers": [list(value) for value in reference_centers],
        "fluorophore_names": names,
        "injections": injection.to_dict(),
        "frames": frame_records,
        "frame_stats": frame_stats,
    }
    content_hash = key_for(content_payload)
    manifest = dict(content_payload)
    manifest["content_hash"] = content_hash
    write_json(root / "manifest.json", manifest)
    # The tree hash is an independent audit hook; it intentionally excludes the manifest itself.
    audit_payload = {**content_payload, "tree_sha256_excluding_manifest": _tree_hash_excluding_manifest(root)}
    write_json(root / "audit.json", audit_payload)
    return GeneratedCube(root=root, manifest_path=root / "manifest.json", content_hash=content_hash, frame_count=config.frames)


def _tree_hash_excluding_manifest(root: Path) -> str:
    digest_payload: List[Dict[str, str]] = []
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in {"manifest.json", "audit.json"}):
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        digest_payload.append({"path": relative, "sha256": sha256_bytes(raw)})
    for item in digest_payload:
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_manifest(root: str | Path, verify_hashes: bool = True) -> Dict[str, Any]:
    cube_root = Path(root)
    manifest_path = cube_root / "manifest.json"
    if not manifest_path.is_file():
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "manifest.json is missing", {"root": str(cube_root)})
    manifest = read_json(manifest_path)
    try:
        shape = tuple(int(v) for v in manifest["shape"])
        dtype = manifest["dtype"]
        endmember_meta = manifest["endmembers"]
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "manifest structure is invalid", {"error": str(exc)}) from exc
    if len(shape) != 3 or any(v <= 0 for v in shape) or dtype != "uint16-le":
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "cube shape/dtype declaration is invalid", {"shape": shape, "dtype": dtype})
    frames, height, width = shape
    endmembers_path = cube_root / endmember_meta["path"]
    if not endmembers_path.is_file():
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "endmember matrix is missing", {"path": str(endmembers_path)})
    raw = endmembers_path.read_bytes()
    em_shape = tuple(int(v) for v in endmember_meta["shape"])
    if len(em_shape) != 2 or em_shape[0] != frames or len(raw) != em_shape[0] * em_shape[1] * 8:
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "endmember matrix shape/size mismatch", {"shape": em_shape})
    if verify_hashes and sha256_bytes(raw) != endmember_meta["sha256"]:
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "endmember matrix hash mismatch")
    frames_meta = manifest.get("frames", [])
    if len(frames_meta) != frames:
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "frame manifest count mismatch", {"declared": frames, "actual": len(frames_meta)})
    expected = height * width * 2
    for record in frames_meta:
        for kind in ("raw", "dark", "flat"):
            path = cube_root / record[kind]["path"]
            if not path.is_file() or path.stat().st_size != expected:
                raise WorkbenchError(
                    ErrorCode.E_CUBE_FORMAT,
                    f"{kind} frame size mismatch",
                    {"frame": record.get("frame"), "path": str(path)},
                )
            if verify_hashes and sha256_bytes(path.read_bytes()) != record[kind]["sha256"]:
                raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, f"{kind} frame hash mismatch", {"frame": record.get("frame")})
    return manifest


def load_endmembers(root: str | Path, manifest: Dict[str, Any]) -> np.ndarray:
    path = Path(root) / manifest["endmembers"]["path"]
    shape = tuple(int(v) for v in manifest["endmembers"]["shape"])
    raw = path.read_bytes()
    return np.frombuffer(raw, dtype="<f8").reshape(shape).copy()


def load_frame_array(root: str | Path, record: Dict[str, Any], kind: str, shape: Tuple[int, int]) -> np.ndarray:
    raw = (Path(root) / record[kind]["path"]).read_bytes()
    return np.frombuffer(raw, dtype="<u2").reshape(shape).copy()


def load_truth(root: str | Path, frame: int) -> Dict[str, Any]:
    return read_json(Path(root) / f"truth/frame_{frame:04d}.json")


def load_masks(root: str | Path, truth: Dict[str, Any]) -> np.ndarray:
    meta = truth["masks"]
    shape = tuple(int(v) for v in meta["shape"])
    raw = (Path(root) / meta["path"]).read_bytes()
    expected = int(np.prod(shape))
    if len(raw) != expected:
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "truth mask size mismatch", {"frame": truth.get("frame")})
    if sha256_bytes(raw) != meta["sha256"]:
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "truth mask hash mismatch", {"frame": truth.get("frame")})
    return np.frombuffer(raw, dtype="u1").reshape(shape).copy()


def verify_generator_reproducibility(config: CubeConfig, injection: InjectionConfig, first: str | Path, second: str | Path) -> bool:
    first_manifest = read_json(Path(first) / "manifest.json")
    second_manifest = read_json(Path(second) / "manifest.json")
    return first_manifest.get("content_hash") == second_manifest.get("content_hash") and hash_tree(first) == hash_tree(second)
