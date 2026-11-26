"""Incremental fixed-point render pipeline with exact content reuse.

The pipeline keeps a mutable PipelineState (prompts, image per frame, motion
table, config). Edit commands mutate the state and invalidate only cache
entries whose key material actually changed. A from-scratch full render with
the final state produces byte-identical frames and tile digests.
"""

from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .cachestore import CacheStore, LedgerEntry
from .codes import BizCode
from .config import RenderConfig
from .encoding import (encode_image, encode_motion, encode_text,
                       fuse_conditions)
from .fingerprint import fingerprint
from .fixedpoint import FpSpec
from .formats import NOISE_CODE_TO_SCHEME
from .kernel import (build_noise_table, decode_tile_row, denoise_step,
                     init_latent)
from .policy import Policy, select_policy
from .scheduler import Scheduler, WorkItem, build_decode_items


@dataclass
class PipelineState:
    vocabulary: list[str]
    prompts: list[list[str]]
    images: dict[int, np.ndarray]
    motion: np.ndarray
    cfg: RenderConfig
    version: int = 1

    @property
    def frame_count(self) -> int:
        return len(self.prompts)

    def clone(self) -> "PipelineState":
        return PipelineState(
            vocabulary=list(self.vocabulary),
            prompts=[list(p) for p in self.prompts],
            images={k: v.copy() for k, v in self.images.items()},
            motion=self.motion.copy(),
            cfg=copy.deepcopy(self.cfg),
            version=self.version + 1)

    def state_fp(self) -> str:
        material = {
            "vocabulary": self.vocabulary,
            "prompts": self.prompts,
            "image_frames": sorted(self.images),
            "image_fp": {str(k): fingerprint(v.tobytes(), "state-image")
                         for k, v in sorted(self.images.items())},
            "motion": self.motion.tolist(),
            "cfg": self.cfg.config_fingerprint_material(),
        }
        return fingerprint(material, "pipeline-state")

    def config_hash(self) -> str:
        return fingerprint(self.cfg.config_fingerprint_material(), "config")


def apply_command(state: PipelineState, command: dict) -> dict:
    """Apply one edit command, returning a structured change description."""
    op = command["op"]
    change = {"op": op, "frames": [], "global": False, "scope": []}
    cfg = state.cfg
    if op == "replace_prompt":
        frame_id = int(command["frame_id"])
        terms = list(command["terms"])
        vocab = set(state.vocabulary)
        oov = [t for t in terms if t not in vocab]
        if oov:
            change["error"] = f"{BizCode.E_PROMPT_OOV.value}:{oov}"
            return change
        state.prompts[frame_id] = terms
        change["frames"] = [frame_id]
        change["scope"] = ["text", "fused", "denoise", "frame", "tile"]
    elif op == "replace_image":
        frame_id = int(command["frame_id"])
        pixels = np.asarray(command["pixels"], dtype=np.uint8)
        if pixels.shape != (cfg.image_height, cfg.image_width, 3):
            change["error"] = BizCode.E_DIM_MISMATCH.value
            return change
        state.images[frame_id] = pixels
        change["frames"] = [frame_id]
        change["scope"] = ["image", "fused", "denoise", "frame", "tile"]
    elif op == "delete_image":
        frame_id = int(command["frame_id"])
        state.images.pop(frame_id, None)
        change["frames"] = [frame_id]
        change["scope"] = ["image", "fused", "denoise", "frame", "tile"]
    elif op == "adjust_motion":
        tr = int(command["tile_row"])
        tc = int(command["tile_col"])
        state.motion[tr, tc, 0] = int(command["dx"])
        state.motion[tr, tc, 1] = int(command["dy"])
        change["global"] = True
        change["frames"] = list(range(state.frame_count))
        change["scope"] = ["motion", "fused", "denoise", "frame", "tile"]
    elif op == "set_weight":
        name = str(command["condition"])
        weights = dict(cfg.weights)
        weights[name] = float(command["value"])
        object.__setattr__(cfg, "weights", weights)
        change["global"] = True
        change["frames"] = list(range(state.frame_count))
        change["scope"] = ["fused", "denoise", "frame", "tile"]
    elif op == "set_denoise_steps":
        value = int(command["value"])
        if value <= 0:
            change["error"] = BizCode.E_POSITIVE_INT.value
            return change
        object.__setattr__(cfg, "denoise_steps", value)
        change["global"] = True
        change["frames"] = list(range(state.frame_count))
        change["scope"] = ["denoise", "frame"]
    elif op == "set_tile_rows":
        value = int(command["value"])
        if value <= 0 or cfg.image_height % value != 0:
            change["error"] = BizCode.E_POSITIVE_INT.value
            return change
        object.__setattr__(cfg, "tile_rows", value)
        change["global"] = True
        change["frames"] = list(range(state.frame_count))
        change["scope"] = ["tile", "frame"]
    elif op == "set_quant":
        width = int(command["width"])
        tier = str(command["tier"])
        try:
            FpSpec(width, tier)  # rejects illegal enum combinations
        except ValueError as exc:
            change["error"] = str(exc).split(":", 1)[0]
            return change
        object.__setattr__(cfg, "fp_width", width)
        object.__setattr__(cfg, "fp_tier", tier)
        change["global"] = True
        change["frames"] = list(range(state.frame_count))
        change["scope"] = ["all"]
    elif op == "set_exchange_version":
        value = int(command["value"])
        if value not in (1, 2):
            change["error"] = BizCode.E_PARSE.value
            return change
        object.__setattr__(cfg, "exchange_version", value)
        change["global"] = True
        change["frames"] = list(range(state.frame_count))
        change["scope"] = ["frame", "tile"]
    else:
        change["error"] = f"{BizCode.E_BAD_COMMAND.value}:{op}"
    state.version += 1
    return change


class RenderEngine:
    def __init__(self, state: PipelineState, cache_root: str | Path,
                 max_workers: int | None = None) -> None:
        self.state = state
        self.cfg = state.cfg
        self.cache = CacheStore(cache_root)
        self.max_workers = max_workers or self.cfg.max_workers
        self.policy: Policy = select_policy(state.frame_count, self.cfg)
        self.failures: list[dict] = []
        self._frame_context: dict[int, dict] = {}

    # ---- key material --------------------------------------------------
    def _quant(self) -> str:
        return self.cfg.fp_spec.key_material()

    def _text_fp(self, frame_id: int) -> str:
        terms = self.state.prompts[frame_id]
        return fingerprint({"frame": frame_id, "terms": terms,
                            "vocabulary": self.state.vocabulary,
                            "quant": self._quant()}, "text-fp")

    def _image_fp(self, frame_id: int) -> str:
        image = self.state.images.get(frame_id)
        if image is None:
            return fingerprint({"frame": frame_id, "absent": True,
                                "quant": self._quant()}, "image-fp")
        return fingerprint({"frame": frame_id, "pixels": image.tobytes(),
                            "quant": self._quant()}, "image-fp")

    def _motion_fp(self) -> str:
        return fingerprint({"motion": self.state.motion.tolist(),
                            "quant": self._quant()}, "motion-fp")

    def _fusion_param_fp(self) -> str:
        return fingerprint({"weights": self.cfg.weights,
                            "quant": self._quant()}, "fusion-param")

    def _fused_fp(self, frame_id: int) -> str:
        return fingerprint({
            "text": self._text_fp(frame_id),
            "image": self._image_fp(frame_id),
            "motion": self._motion_fp(),
            "param": self._fusion_param_fp()}, "fused-fp")

    def _config_denoise_fp(self) -> str:
        return fingerprint({
            "steps": self.cfg.denoise_steps,
            "latent_dim": self.cfg.latent_dim,
            "seed": self.cfg.seed,
            "noise_code": self.cfg.noise_code,
            "guidance": self.cfg.guidance_permille,
            "quant": self._quant(),
            "policy": self.policy.as_dict()}, "denoise-param")

    # ---- invalidation --------------------------------------------------
    _CONDITION_STAGE = {"text": "text", "image": "image", "motion": "motion"}
    _DOWNSTREAM = {
        "text": ("fused", "denoise", "frame", "tile"),
        "image": ("fused", "denoise", "frame", "tile"),
        "motion": ("fused", "denoise", "frame", "tile"),
        "fused": ("denoise", "frame", "tile"),
        "denoise": ("frame", "tile"),
        "frame": ("tile",),
        "tile": (),
    }

    def invalidate_for_change(self, change: dict) -> list[str]:
        scope = set(change.get("scope", []))
        frames = set(change.get("frames", []))
        global_change = bool(change.get("global"))
        if "all" in scope:
            target_stages = {"text", "image", "motion", "fused",
                             "denoise", "frame", "tile"}
        else:
            target_stages: set[str] = set()
            for name in scope:
                if name in self._CONDITION_STAGE:
                    target_stages.add(self._CONDITION_STAGE[name])
                    target_stages.update(self._DOWNSTREAM[name])
                else:
                    target_stages.add(name)
                    target_stages.update(self._DOWNSTREAM.get(name, ()))

        def predicate(entry: LedgerEntry) -> bool:
            if entry.stage not in target_stages:
                return False
            return global_change or entry.frame_id in frames

        reason = f"edit:{change['op']}"
        return sorted(set(self.cache.invalidate(predicate, reason)))

    # ---- per-stage computation ----------------------------------------
    def _ensure_condition_encodings(self, frame_id: int) -> dict:
        cfg = self.cfg
        text_encoded = encode_text(self.state.prompts[frame_id],
                                   self.state.vocabulary, cfg)
        image = self.state.images.get(frame_id)
        if image is None:
            blank = np.zeros((cfg.image_height, cfg.image_width, 3),
                             dtype=np.uint8)
            image_encoded = encode_image(blank, cfg)
        else:
            image_encoded = encode_image(image, cfg)
        motion_encoded = encode_motion(self.state.motion, cfg)
        return {"text": text_encoded, "image": image_encoded,
                "motion": motion_encoded}

    def _ensure_fused(self, frame_id: int):
        key = self.cache.make_key(
            "fused", self._fused_fp(frame_id), self._fusion_param_fp(),
            self._quant(), self.cfg.exchange_version)
        if self.cache.has_valid(key):
            payload = self.cache.load_artifact(key)
            self.cache.record(LedgerEntry(
                key=key, stage="fused", frame_id=frame_id, step=-1,
                tile_row=-1, upstream_fp=self._fused_fp(frame_id),
                param_fp=self._fusion_param_fp(), quant=self._quant(),
                exchange_version=self.cfg.exchange_version,
                sort_key=["fused", frame_id, -1, -1],
                artifact_path=self.cache.entries[key].artifact_path), True)
            return payload
        encodings = self._ensure_condition_encodings(frame_id)
        fused = fuse_conditions(list(encodings.values()), self.cfg)
        payload = {"vector": fused.vector,
                   "trace_order": fused.trace.order,
                   "residuals": fused.trace.residuals,
                   "content_fp": fused.trace.content_fp,
                   "weights_fp": fused.weights_fp}
        path = self.cache.save_artifact(key, payload)
        self.cache.record(LedgerEntry(
            key=key, stage="fused", frame_id=frame_id, step=-1,
            tile_row=-1, upstream_fp=self._fused_fp(frame_id),
            param_fp=self._fusion_param_fp(), quant=self._quant(),
            exchange_version=self.cfg.exchange_version,
            sort_key=["fused", frame_id, -1, -1],
            artifact_path=path), False)
        return payload

    # ---- denoise -------------------------------------------------------
    def _denoise_key(self, frame_id: int, step: int) -> str:
        # Upstream is pure content (fused fingerprint + step index); all
        # parameterization lives in the single param fingerprint so the key
        # cannot drift based on edit history.
        upstream = fingerprint(
            {"fused": self._fused_fp(frame_id), "step": step},
            "denoise-up")
        return self.cache.make_key(
            "denoise", upstream, self._config_denoise_fp(),
            self._quant(), self.cfg.exchange_version)

    def _run_denoise_item(self, item: WorkItem) -> dict:
        frame_id = item.frame_id
        step = item.step
        key = self._denoise_key(frame_id, step)
        if self.cache.has_valid(key):
            entry = self.cache.entries[key]
            self.cache.record(LedgerEntry(
                key=key, stage="denoise", frame_id=frame_id, step=step,
                tile_row=-1, upstream_fp=entry.upstream_fp,
                param_fp=entry.param_fp, quant=self._quant(),
                exchange_version=self.cfg.exchange_version,
                sort_key=["denoise", frame_id, step, -1],
                refs=entry.refs, artifact_path=entry.artifact_path), True)
            return {"item": item, "key": key, "reused": True}
        fused_payload = self._ensure_fused(frame_id)
        from .encoding import FusedCondition, FusionTrace
        fused = FusedCondition(
            fused_payload["vector"],
            FusionTrace(order=fused_payload["trace_order"],
                        content_fp=fused_payload["content_fp"]),
            fused_payload["weights_fp"], self._fusion_param_fp())
        if step == 0:
            latent = init_latent(self.cfg, frame_id, fused)
        else:
            prev_key = self._denoise_key(frame_id, step - 1)
            if not self.cache.has_valid(prev_key):
                self._run_denoise_item(WorkItem(
                    "denoise", frame_id, step - 1, -1, item.group_id,
                    item.merged, item.cost))
            prev = self.cache.load_artifact(prev_key)
            latent = list(prev["latent_fp"])
        noise_table = build_noise_table(self.cfg, frame_id,
                                        fused.trace.content_fp)
        result = denoise_step(self.cfg, frame_id, step, latent, fused,
                              noise_table[step], merged=item.merged)
        payload = {
            "latent_fp": result.latent,
            "delta_fp": result.delta,
            "guidance_permille": result.guidance_permille,
            "guidance_float": result.guidance_permille / 1000.0,
            "noise_code": self.cfg.noise_code,
            "noise_scheme": NOISE_CODE_TO_SCHEME[self.cfg.noise_code],
            "noise_scale_fp": result.noise_scale_fp,
            "condition_scale_fp": result.condition_scale_fp,
            "cost": result.cost,
            "merged": result.merged,
            "step": step,
        }
        path = self.cache.save_artifact(key, payload)
        self.cache.record(LedgerEntry(
            key=key, stage="denoise", frame_id=frame_id, step=step,
            tile_row=-1,
            upstream_fp=fingerprint({"fused": fused.trace.content_fp,
                                     "step": step}, "denoise-up-saved"),
            param_fp=self._config_denoise_fp(), quant=self._quant(),
            exchange_version=self.cfg.exchange_version,
            sort_key=["denoise", frame_id, step, -1],
            refs=list(range(self.cfg.latent_dim)),
            artifact_path=path), False)
        return {"item": item, "key": key, "reused": False}

    # ---- tile decoding -------------------------------------------------
    def _tile_key(self, frame_id: int, tile_row: int) -> str:
        final_step = self.cfg.denoise_steps - 1
        upstream = fingerprint({
            "denoise": self._denoise_key(frame_id, final_step),
            "tile_row": tile_row,
            "tile_rows": self.cfg.tile_rows}, "tile-up")
        param = fingerprint({
            "tile_rows": self.cfg.tile_rows,
            "image_width": self.cfg.image_width,
            "image_height": self.cfg.image_height,
            "seed": self.cfg.seed,
            "policy": self.policy.as_dict()}, "tile-param")
        return self.cache.make_key("tile", upstream, param, self._quant(),
                                   self.cfg.exchange_version)

    def _run_tile_item(self, item: WorkItem) -> dict:
        frame_id = item.frame_id
        tile_row = item.tile_row
        key = self._tile_key(frame_id, tile_row)
        if self.cache.has_valid(key):
            entry = self.cache.entries[key]
            self.cache.record(LedgerEntry(
                key=key, stage="tile", frame_id=frame_id, step=-1,
                tile_row=tile_row, upstream_fp=entry.upstream_fp,
                param_fp=entry.param_fp, quant=self._quant(),
                exchange_version=self.cfg.exchange_version,
                sort_key=["tile", frame_id, -1, tile_row],
                refs=entry.refs, artifact_path=entry.artifact_path), True)
            return {"item": item, "key": key, "reused": True}
        final_step = self.cfg.denoise_steps - 1
        denoise_key = self._denoise_key(frame_id, final_step)
        if not self.cache.has_valid(denoise_key):
            raise RuntimeError(f"missing denoise artifact frame={frame_id}")
        latent = list(self.cache.load_artifact(denoise_key)["latent_fp"])
        png_bytes, digest, reads = decode_tile_row(
            self.cfg, frame_id, tile_row, latent)
        payload = {"png_b64": _b64(png_bytes), "digest": digest,
                   "reads": reads, "tile_row": tile_row}
        path = self.cache.save_artifact(key, payload)
        self.cache.record(LedgerEntry(
            key=key, stage="tile", frame_id=frame_id, step=-1,
            tile_row=tile_row,
            upstream_fp=fingerprint({"latent": latent,
                                     "tile_row": tile_row}, "tile-up-saved"),
            param_fp=fingerprint({"tile_rows": self.cfg.tile_rows},
                                 "tile-param-saved"),
            quant=self._quant(),
            exchange_version=self.cfg.exchange_version,
            sort_key=["tile", frame_id, -1, tile_row], refs=reads,
            artifact_path=path), False)
        return {"item": item, "key": key, "reused": False}

    def _ensure_frame_record(self, frame_id: int) -> str:
        final_step = self.cfg.denoise_steps - 1
        denoise_key = self._denoise_key(frame_id, final_step)
        tile_keys = [self._tile_key(frame_id, tr)
                     for tr in range(self.cfg.tile_rows)]
        upstream = fingerprint(
            {"denoise": denoise_key, "tiles": tile_keys,
             "exchange": self.cfg.exchange_version}, "frame-up")
        key = self.cache.make_key(
            "frame", upstream,
            fingerprint({"exchange": self.cfg.exchange_version},
                        "frame-param"),
            self._quant(), self.cfg.exchange_version)
        if self.cache.has_valid(key):
            entry = self.cache.entries[key]
            self.cache.record(LedgerEntry(
                key=key, stage="frame", frame_id=frame_id, step=final_step,
                tile_row=-1, upstream_fp=entry.upstream_fp,
                param_fp=entry.param_fp, quant=self._quant(),
                exchange_version=self.cfg.exchange_version,
                sort_key=["frame", frame_id, final_step, -1],
                refs=tile_keys, artifact_path=entry.artifact_path), True)
            return key
        denoise_payload = self.cache.load_artifact(denoise_key)
        digests = []
        total_cost = 0
        merged_any = False
        for step in range(self.cfg.denoise_steps):
            step_payload = self.cache.load_artifact(
                self._denoise_key(frame_id, step))
            total_cost += int(step_payload["cost"])
            merged_any = merged_any or bool(step_payload["merged"])
        for tr in range(self.cfg.tile_rows):
            digests.append(
                self.cache.load_artifact(tile_keys[tr])["digest"])
        payload = {
            "frame_id": frame_id,
            "final_step": final_step,
            "latent_fp": denoise_payload["latent_fp"],
            "last_delta_fp": denoise_payload["delta_fp"],
            "guidance_permille": denoise_payload["guidance_permille"],
            "guidance_float": denoise_payload["guidance_float"],
            "noise_code": denoise_payload["noise_code"],
            "noise_scheme": denoise_payload["noise_scheme"],
            "noise_scale_fp": denoise_payload["noise_scale_fp"],
            "condition_scale_fp": denoise_payload["condition_scale_fp"],
            "pixel_digest": fingerprint(digests, "frame-digest"),
            "tile_digests": digests,
            "total_cost": total_cost,
            "merged_any": merged_any,
            "missing": [],
        }
        path = self.cache.save_artifact(key, payload)
        self.cache.record(LedgerEntry(
            key=key, stage="frame", frame_id=frame_id, step=final_step,
            tile_row=-1, upstream_fp=upstream,
            param_fp=fingerprint({"exchange": self.cfg.exchange_version},
                                 "frame-param"),
            quant=self._quant(),
            exchange_version=self.cfg.exchange_version,
            sort_key=["frame", frame_id, final_step, -1],
            refs=tile_keys, artifact_path=path), False)
        return key


def _b64(raw: bytes) -> str:
    import base64
    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes:
    import base64
    return base64.b64decode(text.encode("ascii"))


class RenderResult:
    def __init__(self) -> None:
        self.processed: list[dict] = []
        self.deductions: list[dict] = []
        self.pending_events: list[dict] = []
        self.pending_items: list[dict] = []
        self.failures: list[dict] = []
        self.completed_frames: list[int] = []
        self.completed_tiles: dict[int, list[int]] = {}
        self.consumed_budget = 0

    def as_dict(self) -> dict:
        return {
            "processed": self.processed,
            "deductions": self.deductions,
            "pending_events": self.pending_events,
            "pending_items": self.pending_items,
            "failures": self.failures,
            "completed_frames": self.completed_frames,
            "completed_tiles": self.completed_tiles,
            "consumed_budget": self.consumed_budget,
        }


def render_all(engine: RenderEngine, max_batches: int | None = None
               ) -> RenderResult:
    """Run budget batches for denoise, then decode tile rows and summarize."""
    result = RenderResult()
    scheduler = Scheduler(engine.cfg, engine.policy,
                          engine.state.frame_count)
    workers = max(1, min(engine.max_workers, engine.cfg.max_workers))
    batch_index = 0
    # Phase 1: denoise work under the single-batch point budget. Retained
    # todo items are revisited first on the next batch without double charging.
    while any(item.phase == "denoise" for item in scheduler.todo):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = scheduler.run_batch()
        denoise_items = sorted(
            (item for item in batch.processed if item.phase == "denoise"),
            key=lambda item: item.sort_key)
        if denoise_items:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(engine._run_denoise_item, item): item
                           for item in denoise_items}
                done = list(futures)
            done.sort(key=lambda fut: futures[fut].sort_key)
            for future in done:
                item = futures[future]
                try:
                    outcome = future.result()
                    result.processed.append(
                        {"key": list(item.sort_key),
                         "reused": outcome["reused"]})
                except Exception as exc:
                    result.failures.append(
                        {"key": list(item.sort_key), "error": str(exc)})
        result.deductions.extend(batch.deductions)
        result.pending_events.extend(batch.pending_events)
        if batch.pending:
            result.pending_items = [asdict(item) for item in batch.pending]
        batch_index += 1

    # Phase 2: tile-row decoding only after every denoise step is available.
    if not any(item.phase == "denoise" for item in scheduler.todo):
        decode_plan = sorted(
            build_decode_items(engine.state.frame_count,
                                         engine.cfg, engine.policy),
            key=lambda item: item.sort_key)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(engine._run_tile_item, item): item
                       for item in decode_plan}
            done = list(futures)
        done.sort(key=lambda fut: futures[fut].sort_key)
        for future in done:
            item = futures[future]
            try:
                outcome = future.result()
                result.processed.append(
                    {"key": list(item.sort_key),
                     "reused": outcome["reused"]})
                result.completed_tiles.setdefault(item.frame_id, [])
                if item.tile_row not in result.completed_tiles[item.frame_id]:
                    result.completed_tiles[item.frame_id].append(item.tile_row)
            except Exception as exc:
                result.failures.append(
                    {"key": list(item.sort_key), "error": str(exc)})

        for frame_id in range(engine.state.frame_count):
            if len(result.completed_tiles.get(frame_id, [])) == engine.cfg.tile_rows:
                engine._ensure_frame_record(frame_id)
                result.completed_frames.append(frame_id)

    # Preserve any still-pending denoise items for snapshot/todo recovery.
    if any(item.phase == "denoise" for item in scheduler.todo):
        result.pending_items = [asdict(item) for item in scheduler.todo
                                if item.phase == "denoise"]
    result.consumed_budget = scheduler.total_consumed
    result.failures.sort(key=lambda row: row["key"])
    engine.cache.persist_index()
    engine.failures = result.failures
    engine._last_scheduler_state = scheduler.state()
    return result


def render_incremental(engine: RenderEngine, command_or_change: dict,
                       max_batches: int | None = None
                       ) -> tuple[list[str], RenderResult]:
    """Apply an edit command, invalidate the changed subgraph, then reuse.

    Accepts either a raw edit command (contains ``op``) or an already-applied
    change descriptor. Raw commands are applied first so callers cannot forget
    the state mutation that invalidation is derived from.
    """
    if "op" in command_or_change and "scope" not in command_or_change:
        change = apply_command(engine.state, command_or_change)
        if "error" in change:
            raise ValueError(change["error"])
    else:
        change = command_or_change
    invalidated = engine.invalidate_for_change(change)
    engine.policy = select_policy(engine.state.frame_count, engine.cfg)
    result = render_all(engine, max_batches=max_batches)
    return invalidated, result
