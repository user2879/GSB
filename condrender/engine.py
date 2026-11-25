from __future__ import annotations

import base64
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import SNAPSHOT_VERSION, SNAPSHOT_SCHEMA
from .conditions import (
    check_noise_for_frame,
    encode_image,
    encode_text,
    fuse_frame,
    injection_for,
    validate_and_encode_motion,
)
from .errors import Codes, Failure, StageFailure
from .fixedpoint import (
    BASIS,
    decimal_to_q,
    q_scale,
    round_div_array,
    validate_quant_config,
)
from .hashing import array_fingerprint, full_fingerprint, sha256_hex, canonical_json
from .pngcodec import encode_png_rgb


def tile_bounds(height: int, tile_rows: int) -> list[tuple[int, int]]:
    if tile_rows <= 0:
        raise ValueError("tile_rows must be positive")
    return [(i * height // tile_rows, (i + 1) * height // tile_rows) for i in range(tile_rows)]


def tile_order(tile_rows: int, policy: str = "top_down") -> list[int]:
    if policy == "zigzag":
        order: list[int] = []
        low, high = 0, tile_rows - 1
        while low <= high:
            order.append(low)
            if high != low:
                order.append(high)
            low += 1
            high -= 1
        return order
    return list(range(tile_rows))


def decode_tile_slice(latent_slice: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """Decode one contiguous horizontal slice. It intentionally receives no outside pixels."""
    frac_bits, rounding_mode = validate_quant_config(config)
    scale = q_scale(frac_bits)
    decoder = config["decoder"]
    matrix = np.asarray(decoder["matrix_bp"], dtype=np.int64)
    bias = np.asarray(decoder.get("bias_q", [0, 0, 0]), dtype=np.int64)
    latent = np.asarray(latent_slice, dtype=np.int64)
    if latent.ndim != 3 or latent.shape[2] != 3:
        raise ValueError("latent slice must have shape (tile_height, width, 3)")

    projected_channels: list[np.ndarray] = []
    residual_fingerprints: list[str] = []
    for channel in range(3):
        numerator = (
            matrix[channel, 0] * latent[:, :, 0]
            + matrix[channel, 1] * latent[:, :, 1]
            + matrix[channel, 2] * latent[:, :, 2]
        )
        projected, _, residual = round_div_array(numerator, BASIS, rounding_mode)
        projected = projected + int(bias[channel])
        projected_channels.append(projected)
        residual_fingerprints.append(array_fingerprint(residual))
    projected_q = np.stack(projected_channels, axis=2)

    gain_q = decimal_to_q(decoder.get("pixel_gain", "48.0"), frac_bits, "decoder.pixel_gain")
    scaled_q, _, scale_residual = round_div_array(
        projected_q * gain_q, scale, rounding_mode
    )
    center_q = int(decoder.get("center_q_fraction", 128)) * scale
    pixel_q = center_q + scaled_q
    clipped_q = np.clip(pixel_q, 0, 255 * scale)
    pixels, pixel_direction, pixel_residual = round_div_array(clipped_q, scale, rounding_mode)
    rgb = np.clip(pixels, 0, 255).astype(np.uint8)
    return {
        "rgb": rgb.tolist(),
        "png_b64": base64.b64encode(encode_png_rgb(rgb)).decode("ascii"),
        "projection_residual_fps": residual_fingerprints,
        "scale_residual_fp": array_fingerprint(scale_residual),
        "pixel_residual_fp": array_fingerprint(pixel_residual),
        "pixel_direction_fp": array_fingerprint(pixel_direction),
    }


class RenderEngine:
    def __init__(self, state: dict[str, Any], injections: list[dict[str, Any]] | None = None,
                 workers: int | None = None, task_delay_ms: float = 0.0,
                 output_dir: str | Path | None = None,
                 product_checkpoints: bool = False):
        self.state = state
        self.injections = injections or []
        self.config = state["config"]
        self.frac_bits, self.rounding_mode = validate_quant_config(self.config)
        self.profile = self.config["fixed_point"]["quant_profile"]
        self.config_fp = full_fingerprint(self.config)
        self.cache: dict[str, dict[str, Any]] = {}
        self.ledger: list[dict[str, Any]] = []
        self.current_entries: list[dict[str, Any]] = []
        self.current_failures: list[dict[str, Any]] = []
        self.consumed_points = 0
        self.last_summary: dict[str, Any] | None = None
        configured_workers = max(1, int(self.config.get("max_workers", 1)))
        requested = configured_workers if workers is None else max(1, int(workers))
        self.workers = min(requested, configured_workers)
        self.task_delay_ms = float(task_delay_ms)
        self.product_checkpoints = bool(product_checkpoints)
        self.output_dir = Path(output_dir) if output_dir else None
        self._concurrency_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self.active_tasks = 0
        self.max_observed_concurrency = 0
        self.frame_results: dict[int, dict[str, Any]] = {}
        self.completed_frames: list[int] = []
        self.command_log: list[dict[str, Any]] = []
        self.snapshot_failures: list[dict[str, Any]] = []

    def make_key(self, kind: str, upstream: dict[str, Any], params: dict[str, Any]) -> str:
        return full_fingerprint({
            "snapshot_schema": SNAPSHOT_SCHEMA,
            "snapshot_version": SNAPSHOT_VERSION,
            "stage": kind,
            "upstream": upstream,
            "params": params,
            "quant": {
                "profile": self.profile,
                "frac_bits": self.frac_bits,
                "rounding": self.rounding_mode,
            },
        })

    def _add_ledger(self, status: str, kind: str, key: str, frame: int | None,
                    index: int | None, upstream: dict[str, Any], cost: int = 0,
                    reason: str | None = None) -> dict[str, Any]:
        entry = {
            "status": status,
            "stage": kind,
            "frame": frame,
            "index": index,
            "key": key,
            "cost_points": cost if status == "RECOMPUTE" else 0,
            "reason": reason or ("content_key_miss" if status == "RECOMPUTE" else "content_key_match"),
            "basis": {
                "snapshot_version": SNAPSHOT_VERSION,
                "snapshot_schema": SNAPSHOT_SCHEMA,
                "upstream": upstream,
                "quant_profile": self.profile,
            },
        }
        self.ledger.append(entry)
        self.current_entries.append(entry)
        return entry

    def get_or_build(self, kind: str, key: str, frame: int | None, index: int | None,
                     upstream: dict[str, Any], builder: Callable[[], dict[str, Any]],
                     cost: int | None = None) -> dict[str, Any]:
        if key in self.cache:
            self._add_ledger("HIT", kind, key, frame, index, upstream)
            return self.cache[key]["data"]
        data = builder()
        record = {
            "kind": kind,
            "key": key,
            "fingerprint": full_fingerprint(data),
            "frame": frame,
            "index": index,
            "data": data,
        }
        self.cache[key] = record
        stage_costs = self.config.get("stage_costs", {})
        points = int(cost if cost is not None else stage_costs.get(kind, 1))
        self.consumed_points += points
        self._add_ledger("RECOMPUTE", kind, key, frame, index, upstream, points)
        return data

    def _track(self, fn: Callable[[], Any]) -> Any:
        with self._concurrency_lock:
            self.active_tasks += 1
            self.max_observed_concurrency = max(self.max_observed_concurrency, self.active_tasks)
        try:
            if self.task_delay_ms > 0:
                time.sleep(self.task_delay_ms / 1000.0)
            return fn()
        finally:
            with self._concurrency_lock:
                self.active_tasks -= 1

    def _parallel_stage(self, specs: list[dict[str, Any]],
                        worker: Callable[[dict[str, Any]], dict[str, Any]],
                        kind: str, frame: int,
                        checkpoint: Callable[["RenderEngine"], None] | None = None
                        ) -> list[dict[str, Any]]:
        if not specs:
            return []
        worker_count = max(1, min(self.workers, len(specs)))

        def one(spec: dict[str, Any]) -> tuple[dict[str, Any], bool]:
            key = spec["key"]
            if key in self.cache:
                return self.cache[key]["data"], True
            data = self._track(lambda: worker(spec))
            return data, False

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_to_index = {
                pool.submit(one, spec): index for index, spec in enumerate(specs)
            }
            results_by_index: dict[int, tuple[dict[str, Any], bool]] = {}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                spec = specs[index]
                data, was_hit = future.result()
                if not was_hit:
                    key = spec["key"]
                    with self._state_lock:
                        if key not in self.cache:
                            self.cache[key] = {
                                "kind": kind, "key": key,
                                "fingerprint": full_fingerprint(data),
                                "frame": frame, "index": spec.get("index"),
                                "data": data,
                            }
                            points = int(spec.get(
                                "cost", self.config.get("stage_costs", {}).get(kind, 1)
                            ))
                            self.consumed_points += points
                results_by_index[index] = (data, was_hit)
                if checkpoint is not None and self.product_checkpoints:
                    checkpoint(self)

        output: list[dict[str, Any]] = []
        stage_costs = self.config.get("stage_costs", {})
        for index, spec in enumerate(specs):
            data, was_hit = results_by_index[index]
            key = spec["key"]
            upstream = spec.get("upstream", {})
            if was_hit:
                self._add_ledger("HIT", kind, key, frame, spec.get("index"), upstream)
            else:
                points = int(spec.get("cost", stage_costs.get(kind, 1)))
                self._add_ledger("RECOMPUTE", kind, key, frame, spec.get("index"),
                                 upstream, points)
            output.append(data)
        return output

    def _injection_fingerprint(self, frame: int, stages: list[str]) -> str:
        selected = [
            {"id": inj.get("id"), "type": inj.get("type"), "stage": inj.get("stage")}
            for inj in self.injections
            if int(inj.get("target_frame", -1)) == frame and inj.get("stage") in stages
        ]
        return full_fingerprint(selected)

    def _text_stage(self, frame: int) -> tuple[str, dict[str, Any]]:
        prompt = self.state["prompts"][frame]
        upstream = {
            "prompt_fp": full_fingerprint(prompt),
            "terms_fp": full_fingerprint(self.state["terms"]),
            "vocab_fp": full_fingerprint(self.state.get("vocab", {})),
        }
        key = self.make_key("text_encode", upstream, {"frame": frame})
        data = self.get_or_build(
            "text", key, frame, None, upstream, lambda: encode_text(frame, self.state)
        )
        return key, data

    def _image_stage(self, frame: int) -> tuple[str, dict[str, Any]]:
        ref = self.state["image_refs"][frame]
        source_fp = "absent"
        if ref is not None:
            source_fp = sha256_hex(self.state["image_bytes"][ref])
        upstream = {"source_fp": source_fp, "ref": ref}
        params = {
            "frame": frame,
            "image_config": self.config["condition_image"],
            "injection_fp": self._injection_fingerprint(
                frame, ["image_decode", "image_encode"]
            ),
        }
        key = self.make_key("image_encode", upstream, params)
        data = self.get_or_build(
            "image", key, frame, None, upstream,
            lambda: encode_image(frame, self.state, self.injections),
            cost=int(self.config.get("stage_costs", {}).get("image", 2)),
        )
        return key, data

    def _motion_stage(self, frame: int) -> tuple[str, dict[str, Any]]:
        motion_slice = self.state["motion"][frame]
        upstream = {
            "motion_frame_fp": array_fingerprint(np.asarray(motion_slice)),
            "motion_grid_fp": full_fingerprint(self.config["motion_grid"]),
        }
        params = {
            "frame": frame,
            "injection_fp": self._injection_fingerprint(frame, ["motion_validate"]),
        }
        key = self.make_key("motion_encode", upstream, params)
        injection = injection_for(self.injections, frame, "motion_validate")
        data = self.get_or_build(
            "motion", key, frame, None, upstream,
            lambda: validate_and_encode_motion(frame, self.state, injection),
        )
        return key, data

    def _fusion_stage(self, frame: int, text_key: str, text_data: dict[str, Any],
                      image_key: str, image_data: dict[str, Any],
                      motion_key: str, motion_data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        weights_fp = full_fingerprint(self.config["fusion"]["weights"])
        upstream = {"text_key": text_key, "image_key": image_key, "motion_key": motion_key}
        params = {"frame": frame, "weights_fp": weights_fp}
        key = self.make_key("fusion", upstream, params)
        data = self.get_or_build(
            "fusion", key, frame, None, upstream,
            lambda: fuse_frame(frame, self.state, text_data, image_data, motion_data),
        )
        return key, data

    def _initial_stage(self, frame: int, fusion_key: str,
                       fusion_data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        height, width = int(self.config["height"]), int(self.config["width"])
        latent_config_fp = full_fingerprint({
            "seed": int(self.config["seed"]),
            "width": width,
            "height": height,
            "latent_channels": 3,
            "condition_dims": self.config["condition_dims"],
            "latent_layout": "HWC_C_ORDER",
        })
        seed_material = {
            "snapshot_version": SNAPSHOT_VERSION,
            "fusion_digest": full_fingerprint(fusion_data["rows"]),
            "fusion_key": fusion_key,
            "seed": int(self.config["seed"]),
            "latent_config_fingerprint": latent_config_fp,
            "frame": frame,
        }
        seed_hex = sha256_hex(canonical_json(seed_material))
        seed_int = int(seed_hex[:16], 16)
        upstream = {"fusion_key": fusion_key, "latent_config_fp": latent_config_fp}
        params = {"frame": frame, "seed_material": seed_material, "shape": [height, width, 3]}
        key = self.make_key("initial_latent", upstream, params)

        def build() -> dict[str, Any]:
            rng = np.random.Generator(np.random.PCG64(seed_int))
            span = max(1, q_scale(self.frac_bits) // 8)
            latent = rng.integers(-span, span + 1, size=(height, width, 3), dtype=np.int64)
            return {
                "latent": latent.tolist(),
                "seed_int": seed_int,
                "seed_material": seed_material,
                "latent_fp": array_fingerprint(latent),
            }

        data = self.get_or_build("initial", key, frame, None, upstream, build)
        return key, data

    def _build_step(self, spec: dict[str, Any]) -> dict[str, Any]:
        step = int(spec["step"])
        initial_data = self.cache[spec["initial_key"]]["data"]
        fusion_data = self.cache[spec["fusion_key"]]["data"]
        config = self.config
        height, width = int(config["height"]), int(config["width"])
        scale = q_scale(self.frac_bits)
        latent_size = height * width * 3

        noise = self.state["noise"]
        noise_scale = q_scale(int(noise.get("frac_bits_hint", self.frac_bits)))
        entry = np.asarray(noise["entries"][step], dtype=np.int64).reshape(-1)[:latent_size]
        noise_q, _, noise_residual = round_div_array(entry * scale, noise_scale, self.rounding_mode)

        condition_pool = np.asarray(fusion_data["flat"], dtype=np.int64)
        condition = np.asarray(
            [int(condition_pool[i % condition_pool.size]) for i in range(latent_size)],
            dtype=np.int64,
        )
        guidance_q = decimal_to_q(config["guidance_scale"], self.frac_bits, "guidance_scale")
        guided_q, guidance_direction, guidance_residual = round_div_array(
            condition * guidance_q, scale, self.rounding_mode
        )
        combined = noise_q + guided_q
        alpha_bp = int(config["step_alpha_bp"][min(step, len(config["step_alpha_bp"]) - 1)])
        delta, update_direction, update_residual = round_div_array(
            alpha_bp * combined, BASIS, self.rounding_mode
        )
        delta = delta.reshape(height, width, 3)
        return {
            "step": step,
            "delta": delta.tolist(),
            "noise_entry_fp": array_fingerprint(entry),
            "noise_transcode_residual_fp": array_fingerprint(noise_residual),
            "guidance_direction": guidance_direction.tolist(),
            "guidance_residual": guidance_residual.tolist(),
            "guidance_residual_fp": array_fingerprint(guidance_residual),
            "update_direction": update_direction.tolist(),
            "update_residual": update_residual.tolist(),
            "residual_fingerprint": array_fingerprint(update_residual),
            "alpha_bp": alpha_bp,
            "guidance_q": guidance_q,
        }

    def _denoise_stage(self, frame: int, initial_key: str, initial_data: dict[str, Any],
                       fusion_key: str, fusion_data: dict[str, Any],
                       checkpoint: Callable[["RenderEngine"], None] | None = None
                       ) -> dict[str, Any]:
        check_noise_for_frame(
            frame, self.state, injection_for(self.injections, frame, "denoise_step")
        )
        steps = int(self.config["denoise_steps"])
        noise_fp = full_fingerprint(self.state["noise"].get("version"))
        specs: list[dict[str, Any]] = []
        step_keys: list[str] = []
        for step in range(steps):
            entry_fp = array_fingerprint(np.asarray(self.state["noise"]["entries"][step]))
            upstream = {
                "initial_key": initial_key,
                "fusion_key": fusion_key,
                "noise_version_fp": noise_fp,
                "noise_entry_fp": entry_fp,
            }
            params = {
                "step": step,
                "steps_total_param_independent": True,
                "guidance": str(self.config["guidance_scale"]),
                "alpha_bp": self.config["step_alpha_bp"][
                    min(step, len(self.config["step_alpha_bp"]) - 1)
                ],
            }
            key = self.make_key("denoise_step", upstream, params)
            step_keys.append(key)
            specs.append({
                "key": key, "index": step, "step": step,
                "initial_key": initial_key, "fusion_key": fusion_key,
                "upstream": upstream,
                "cost": int(self.config.get("stage_costs", {}).get("step", 3)),
            })
        step_data = self._parallel_stage(
            specs, self._build_step, "denoise_step", frame, checkpoint=checkpoint
        )

        latent = np.asarray(initial_data["latent"], dtype=np.int64).copy()
        cumulative_fingerprints: list[str] = []
        for data in step_data:
            latent += np.asarray(data["delta"], dtype=np.int64)
            cumulative_fingerprints.append(array_fingerprint(latent))
        upstream = {"initial_key": initial_key, "step_keys": step_keys}
        frame_key = self.make_key("denoise_frame", upstream, {"steps": steps})
        frame_data = self.get_or_build(
            "denoise_frame", frame_key, frame, None, upstream,
            lambda: {
                "latent": latent.tolist(),
                "latent_fp": array_fingerprint(latent),
                "step_keys": step_keys,
                "step_residual_fps": [item["residual_fingerprint"] for item in step_data],
                "cumulative_latent_fps": cumulative_fingerprints,
            },
            cost=0,
        )
        return {"key": frame_key, "data": frame_data, "step_keys": step_keys,
                "step_data": step_data}

    def _tile_stage(self, frame: int, denoise: dict[str, Any],
                    checkpoint: Callable[["RenderEngine"], None] | None = None
                    ) -> dict[str, Any]:
        rows = int(self.config["tile_rows"])
        height = int(self.config["height"])
        width = int(self.config["width"])
        bounds = tile_bounds(height, rows)
        order = tile_order(rows, str(self.config.get("tile_scheduling", "top_down")))
        latent = np.asarray(denoise["data"]["latent"], dtype=np.int64)
        decoder_fp = full_fingerprint(self.config["decoder"])
        specs: list[dict[str, Any]] = []
        for tile_index in order:
            y0, y1 = bounds[tile_index]
            latent_slice = latent[y0:y1, :, :].copy()
            slice_fp = array_fingerprint(latent_slice)
            upstream = {
                "denoise_frame_key": denoise["key"],
                "slice_fp": slice_fp,
                "decoder_fp": decoder_fp,
            }
            params = {
                "tile_index": tile_index, "y0": y0, "y1": y1,
                "width": width, "tile_rows": rows,
            }
            key = self.make_key("tile_decode", upstream, params)
            specs.append({
                "key": key, "index": tile_index, "tile_index": tile_index,
                "y0": y0, "y1": y1, "latent_slice": latent_slice,
                "upstream": upstream,
                "cost": int(self.config.get("stage_costs", {}).get("tile", 2)),
            })

        def worker(spec: dict[str, Any]) -> dict[str, Any]:
            result = decode_tile_slice(spec["latent_slice"], self.config)
            result.update({"tile_index": spec["tile_index"], "y0": spec["y0"], "y1": spec["y1"]})
            return result

        ordered_data = self._parallel_stage(
            specs, worker, "tile_decode", frame, checkpoint=checkpoint
        )
        tiles_by_index = {
            int(spec["tile_index"]): data for spec, data in zip(specs, ordered_data)
        }

        tile_keys_row_order: list[str] = []
        tile_records: list[dict[str, Any]] = []
        for tile_index in range(rows):
            y0, y1 = bounds[tile_index]
            upstream = {
                "denoise_frame_key": denoise["key"],
                "slice_fp": array_fingerprint(latent[y0:y1, :, :]),
                "decoder_fp": decoder_fp,
            }
            key = self.make_key("tile_decode", upstream, {
                "tile_index": tile_index, "y0": y0, "y1": y1,
                "width": width, "tile_rows": rows,
            })
            tile_keys_row_order.append(key)
            tile_records.append({"index": tile_index, "key": key, "y0": y0, "y1": y1,
                                 "data": tiles_by_index[tile_index]})

        compose_upstream = {"tile_keys": tile_keys_row_order}
        compose_key = self.make_key("frame_compose", compose_upstream, {
            "width": width, "height": height, "png": "rgb8",
        })

        def compose() -> dict[str, Any]:
            rgb = np.concatenate([
                np.asarray(tiles_by_index[i]["rgb"], dtype=np.uint8) for i in range(rows)
            ], axis=0)
            png = encode_png_rgb(rgb)
            return {
                "rgb": rgb.tolist(),
                "png_b64": base64.b64encode(png).decode("ascii"),
                "png_sha256": sha256_hex(png),
                "shape": [height, width, 3],
            }

        compose_data = self.get_or_build(
            "frame_compose", compose_key, frame, None, compose_upstream, compose,
            cost=int(self.config.get("stage_costs", {}).get("frame_compose", 1)),
        )
        return {
            "key": compose_key,
            "data": compose_data,
            "bounds": bounds,
            "order": order,
            "tile_records": tile_records,
        }

    def _write_outputs(self, frame: int, tiles: dict[str, Any]) -> None:
        if self.output_dir is None:
            return
        frame_dir = self.output_dir / "frames"
        tile_dir = frame_dir / f"frame_{frame:02d}_tiles"
        frame_dir.mkdir(parents=True, exist_ok=True)
        tile_dir.mkdir(parents=True, exist_ok=True)
        frame_png = base64.b64decode(tiles["data"]["png_b64"])
        (frame_dir / f"frame_{frame:02d}.png").write_bytes(frame_png)
        for record in tiles["tile_records"]:
            data = record["data"]
            tile_png = base64.b64decode(data["png_b64"])
            (tile_dir / f"tile_{record['index']:02d}_y{record['y0']}-{record['y1']}.png").write_bytes(
                tile_png
            )

    def render_frame(self, frame: int,
                     checkpoint: Callable[["RenderEngine"], None] | None = None
                     ) -> dict[str, Any]:
        text_key, text_data = self._text_stage(frame)
        image_key, image_data = self._image_stage(frame)
        motion_key, motion_data = self._motion_stage(frame)
        fusion_key, fusion_data = self._fusion_stage(
            frame, text_key, text_data, image_key, image_data, motion_key, motion_data
        )
        initial_key, initial_data = self._initial_stage(frame, fusion_key, fusion_data)
        denoise = self._denoise_stage(
            frame, initial_key, initial_data, fusion_key, fusion_data, checkpoint
        )
        tiles = self._tile_stage(frame, denoise, checkpoint)
        self._write_outputs(frame, tiles)
        result = {
            "frame": frame,
            "keys": {
                "text": text_key,
                "image": image_key,
                "motion": motion_key,
                "fusion": fusion_key,
                "initial": initial_key,
                "denoise_frame": denoise["key"],
                "frame_compose": tiles["key"],
            },
            "condition_fingerprint": full_fingerprint(fusion_data["rows"]),
            "tile_order": tiles["order"],
            "tile_bounds": tiles["bounds"],
            "tile_keys": [item["key"] for item in tiles["tile_records"]],
            "step_keys": denoise["step_keys"],
            "step_residual_fps": [item["residual_fingerprint"] for item in denoise["step_data"]],
            "png_sha256": tiles["data"]["png_sha256"],
        }
        self.frame_results[frame] = result
        return result

    def empty_summary(self) -> dict[str, Any]:
        total = int(self.config["frames"])
        return {
            "fresh": False,
            "config_version": self.config.get("config_version"),
            "global_version": self.state.get("global_version"),
            "config_hash": self.config_fp,
            "condition_fingerprint": None,
            "completed_frames": [],
            "completed_frame_count": 0,
            "total_frames": total,
            "affected_frame_range": None,
            "tile_rows": {},
            "reused_entries": 0,
            "recomputed_entries": 0,
            "progress": 0.0,
            "consumed_points": self.consumed_points,
            "point_budget": int(self.config.get("recompute_point_budget", 100000)),
        }

    def build_summary(self, completed: list[int], affected_range: list[int] | None) -> dict[str, Any]:
        tile_summary: dict[str, Any] = {}
        condition_fps: dict[str, str] = {}
        for frame in completed:
            result = self.frame_results[frame]
            rows = int(self.config["tile_rows"])
            tile_summary[str(frame)] = {
                "total": rows,
                "completed": list(range(rows)),
                "order": result["tile_order"],
                "bounds": result["tile_bounds"],
            }
            condition_fps[str(frame)] = result["condition_fingerprint"]
        reused = sum(1 for entry in self.current_entries if entry["status"] == "HIT")
        recomputed = sum(1 for entry in self.current_entries if entry["status"] == "RECOMPUTE")
        aggregate_condition_fp = full_fingerprint(condition_fps) if condition_fps else None
        return {
            "fresh": True,
            "config_version": self.config.get("config_version"),
            "global_version": self.state.get("global_version"),
            "config_hash": self.config_fp,
            "condition_fingerprint": aggregate_condition_fp,
            "condition_fingerprints": condition_fps,
            "completed_frames": completed,
            "completed_frame_count": len(completed),
            "total_frames": int(self.config["frames"]),
            "affected_frame_range": affected_range,
            "tile_rows": tile_summary,
            "reused_entries": reused,
            "recomputed_entries": recomputed,
            "progress": len(completed) / max(1, int(self.config["frames"])),
            "consumed_points": self.consumed_points,
            "point_budget": int(self.config.get("recompute_point_budget", 100000)),
            "max_observed_concurrency": self.max_observed_concurrency,
            "worker_limit": self.workers,
        }

    def render_all(self, budget_ms: float | None = None,
                   affected_range: list[int] | None = None,
                   checkpoint: Callable[["RenderEngine"], None] | None = None) -> dict[str, Any]:
        self.current_entries = []
        self.current_failures = []
        self.frame_results = {}
        self.completed_frames = []
        deadline = None if budget_ms is None else time.monotonic() + budget_ms / 1000.0
        completed: list[int] = []

        for frame in range(int(self.config["frames"])):
            if deadline is not None and time.monotonic() >= deadline:
                stale = dict(self.last_summary or self.empty_summary())
                stale["fresh"] = False
                stale["code"] = Codes.STALE_PREVIEW
                stale["affected_frame_range"] = affected_range
                return {
                    "status": "stale",
                    "code": Codes.STALE_PREVIEW,
                    "summary": stale,
                    "entries": list(self.current_entries),
                    "failures": list(self.current_failures),
                    "completed_frames": list(completed),
                }
            try:
                result = self.render_frame(frame, checkpoint=checkpoint)
                completed.append(frame)
                self.completed_frames = list(completed)
                if checkpoint is not None:
                    checkpoint(self)
            except StageFailure as exc:
                failure = exc.failure.to_dict()
                self.current_failures.append(failure)

        summary = self.build_summary(completed, affected_range)
        if len(self.current_failures) == 0 or completed:
            self.last_summary = summary
        if checkpoint is not None:
            checkpoint(self)
        return {
            "status": "complete",
            "summary": summary,
            "entries": list(self.current_entries),
            "failures": list(self.current_failures),
            "completed_frames": completed,
        }

    def cache_fingerprint(self) -> str:
        return full_fingerprint({key: record["fingerprint"] for key, record in self.cache.items()})
