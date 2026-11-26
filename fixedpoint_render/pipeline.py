"""End-to-end deterministic rendering pipeline (business rules 6-8).

The pipeline is content-addressed: each stage product gets a ledger key derived
from upstream fingerprints, parameters and the quantization mode.  Independent
denoise chains and tile decodes run in a bounded thread pool, but every task
carries a stable sort key and all ledger/state mutation happens on the main
thread after results are gathered in deterministic order, so concurrency cannot
change any output.
"""
from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .budget import allocate
from .config import CONDITION_TYPES
from .decode import (
    TileBoundaryGuard, assemble_frame, decode_tile_row, projection_matrix,
)
from .encoding import (
    EncodedCondition, encode_image, encode_motion, encode_text,
)

from .fingerprints import fingerprint
from .fusion import fuse_conditions
from .latent import GENESIS, run_frame
from .ledger import Ledger
from .snapshot import STAGES, SnapshotStore
from .state import RenderState


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, sort_keys=True, indent=1, ensure_ascii=False)


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sort_key(stage: str, frame: int, index: int) -> tuple:
    return (stage, frame, index)


# Ledger stage name -> snapshot stage directory.
STAGE_DIR = {
    "encode": STAGES[1],
    "fuse": STAGES[2],
    "budget": STAGES[3],
    "denoise": STAGES[4],
    "decode": STAGES[5],
    "compose": STAGES[6],
}


@dataclass
class RenderOutcome:
    version: int
    frames: list
    reused: int
    recomputed: int
    report: dict
    state_path: str
    output_dir: str


class Renderer:
    def __init__(self, workdir: str):
        self.workdir = workdir
        self.snapshots = SnapshotStore(os.path.join(workdir, "snapshots"))
        self.output_dir = os.path.join(workdir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.error_counts: dict[str, int] = {}

    # -- helpers -----------------------------------------------------------
    def _artifact(self, temp_dir: str, stage: str, name: str) -> str:
        # Artifacts live in the persistent version directory so ledger entries
        # remain resolvable on later runs.  The atomic commit only swaps the
        # manifest; stage dirs are created up front and appended to in place.
        stage_dir = STAGE_DIR.get(stage, stage)
        target = os.path.join(
            self.snapshots.root, f"v{self._version}", stage_dir)
        os.makedirs(target, exist_ok=True)
        return os.path.join(target, name)

    def _get_or_compute(self, ledger: Ledger, key: str, stage: str,
                        frame: int, index: int, upstream: list[str],
                        param_obj, qmode: str, frac_bits: int,
                        artifact_relpath: str, compute, temp_dir: str,
                        to_serializable, from_serializable):
        existing = ledger.lookup(key)
        if existing is not None:
            ledger.mark_reused(existing)
            return from_serializable(_read_json(existing.artifact_path)), True, existing
        result = compute()
        artifact_path = self._artifact(temp_dir, stage, artifact_relpath)
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        _write_json(artifact_path, to_serializable(result))
        entry = ledger.record(key, stage, frame, index, upstream, param_obj,
                              qmode, frac_bits, self._version, artifact_path)
        return result, False, entry

    # -- stage 1: validation snapshot -------------------------------------
    def _commit_validate(self, state: RenderState, temp_dir: str,
                         version: int) -> None:
        payload = {
            "config_hash": self.snapshots.config_hash(state.config),
            "frame_count": state.frame_count,
            "vocab": list(state.config.vocab),
            "image_present": state.image_present,
        }
        _write_json(self._artifact(temp_dir, STAGES[0], "validation.json"),
                    payload)
        self.snapshots.commit(temp_dir, version, state.config, 0,
                              {"stage": STAGES[0], "frames_done": []})

    # -- stage 2: encoding -------------------------------------------------
    def _encode(self, state: RenderState, ledger: Ledger,
                temp_dir: str) -> dict[tuple, EncodedCondition]:
        config = state.config
        encoded: dict[tuple, EncodedCondition] = {}
        tasks = []
        for frame in range(state.frame_count):
            tasks.append(("text", frame))
            if state.image_present[frame]:
                tasks.append(("image", frame))
            tasks.append(("motion", frame))
        for condition, frame in sorted(tasks):
            if condition == "text":
                upstream = [fingerprint(state.prompts[frame],
                                        f"prompt-f{frame}")]
                compute = lambda f=frame: encode_text(
                    f, state.prompts[f], config)
            elif condition == "image":
                image = state.image(frame)
                upstream = [fingerprint(state.image_payloads[frame],
                                        f"image-payload-f{frame}")]
                compute = lambda f=frame, img=image: encode_image(f, img, config)
            else:
                upstream = [fingerprint(state.motions[frame],
                                        f"motion-f{frame}")]
                compute = lambda f=frame: encode_motion(
                    f, state.motions[f], config)
            param = {"condition": condition, "dim": config.condition_dim,
                     "frac_bits": config.frac_bits}
            key = Ledger.make_key(self._version, "encode", frame,
                                  CONDITION_TYPES.index(condition),
                                  upstream, param, config.qmode,
                                  config.frac_bits)
            rel = f"frame_{frame:02d}/{condition}.json"
            result, _, _ = self._get_or_compute(
                ledger, key, "encode", frame,
                CONDITION_TYPES.index(condition), upstream, param,
                config.qmode, config.frac_bits, rel, compute, temp_dir,
                lambda r: r.to_dict(),
                lambda d: self._rehydrate_encoded(condition, frame, d))
            if condition == "image" and not state.image_present[frame]:
                continue
            encoded[(condition, frame)] = result
        return encoded

    @staticmethod
    def _rehydrate_encoded(condition: str, frame: int, data: dict):
        return EncodedCondition(
            condition=condition, frame=frame, vector=list(data["vector"]),
            coverage=data["coverage"], traces=[])

    @staticmethod
    def _rehydrate_fusion(data: dict):
        from .fusion import FusionResult
        return FusionResult(
            frame=data["frame"],
            weights_raw={k: int(v) for k, v in data["weights_raw"].items()},
            weights_norm={k: int(v) for k, v in data["weights_norm"].items()},
            vector=list(data["vector"]), traces=[])

    # -- stage 3: fusion ---------------------------------------------------
    def _fuse(self, state: RenderState, ledger: Ledger,
              encoded: dict, temp_dir: str) -> dict:
        config = state.config
        fusions = {}
        for frame in range(state.frame_count):
            frame_encoded = {c: encoded[(c, frame)] for c in CONDITION_TYPES
                             if (c, frame) in encoded}
            if "image" not in frame_encoded:
                # Deleted image: substitute an all-zero encoder output so the
                # fusion still has three typed inputs; the key reflects absence.
                frame_encoded["image"] = EncodedCondition(
                    "image", frame, [0] * config.condition_dim, 0, [])
            upstream = [fingerprint(
                {"condition": c, "vector": frame_encoded[c].vector,
                 "coverage": frame_encoded[c].coverage,
                 "present": state.image_present[frame] if c == "image" else True},
                f"enc-{c}-f{frame}") for c in sorted(CONDITION_TYPES)]
            param = {"weights": config.weights, "guidance": config.guidance,
                     "qmode": config.qmode, "frac_bits": config.frac_bits}
            key = Ledger.make_key(self._version, "fuse", frame, -1,
                                  upstream, param, config.qmode,
                                  config.frac_bits)
            compute = lambda f=frame, fe=frame_encoded: fuse_conditions(
                f, fe, config)
            result, _, _ = self._get_or_compute(
                ledger, key, "fuse", frame, -1, upstream, param,
                config.qmode, config.frac_bits,
                f"frame_{frame:02d}/fusion.json", compute, temp_dir,
                lambda r: r.to_dict(), self._rehydrate_fusion)
            fusions[frame] = result
        return fusions

    # -- stage 4: budget arbitration --------------------------------------
    def _budget(self, state: RenderState, ledger: Ledger,
                encoded: dict, fusions: dict, temp_dir: str):
        config = state.config
        motion_enc = {f: encoded[("motion", f)]
                      for f in range(state.frame_count)}
        param = {"per_frame_budget": config.per_frame_budget,
                 "global_step_budget": config.global_step_budget,
                 "frame_budgets": state.frame_budgets}
        upstream = [fusions[f].content_id for f in range(state.frame_count)]
        upstream += [fingerprint(
            {"vector": motion_enc[f].vector,
             "coverage": motion_enc[f].coverage},
            f"motion-budget-f{f}") for f in range(state.frame_count)]
        key = Ledger.make_key(self._version, "budget", -1, -1, upstream,
                              param, config.qmode, config.frac_bits)

        def compute():
            return allocate(state.frame_count, motion_enc, fusions, config,
                            state.frame_budgets)
        plan, _, _ = self._get_or_compute(
            ledger, key, "budget", -1, -1, upstream, param, config.qmode,
            config.frac_bits, "budget_plan.json", compute, temp_dir,
            lambda r: r.to_dict(), self._rehydrate_plan)
        return plan

    @staticmethod
    def _rehydrate_plan(data: dict):
        from .budget import BudgetPlan, FrameAllocation
        entries = [FrameAllocation(**e) for e in data["entries"]]
        return BudgetPlan(
            per_frame_budget=data["per_frame_budget"],
            global_budget=data["global_budget"],
            total_demand=data["total_demand"],
            total_allocated=data["total_allocated"],
            truncated_total=data["truncated_total"],
            frame_budgets={int(k): v for k, v in data["frame_budgets"].items()},
            entries=entries)

    # -- stage 5: denoise (concurrent frames, sequential steps) -----------
    def _denoise(self, state: RenderState, ledger: Ledger,
                 fusions: dict, plan, temp_dir: str) -> dict:
        config = state.config
        alloc_by_frame = {e.frame: e for e in plan.entries}
        matrix = projection_matrix(config)

        def work(frame: int):
            prev_fp = (GENESIS if frame == 0 else
                       fingerprint(fusions[frame - 1].vector,
                                   f"fusion-vector-f{frame - 1}"))
            return run_frame(frame, fusions[frame], alloc_by_frame[frame],
                             prev_fp, config)

        # Cache check first so unaffected frames are not recomputed at all.
        frame_keys: dict[int, str] = {}
        for frame in range(state.frame_count):
            prev_fp = (GENESIS if frame == 0 else
                       fingerprint(fusions[frame - 1].vector,
                                   f"fusion-vector-f{frame - 1}"))
            upstream = [fusions[frame].content_id,
                        fingerprint({
                            "frame": frame,
                            "allocated": alloc_by_frame[frame].allocated,
                            "demand": alloc_by_frame[frame].demand,
                            "truncated": alloc_by_frame[frame].truncated,
                            "degraded": alloc_by_frame[frame].degraded,
                        }, f"alloc-f{frame}"), prev_fp]
            param = {"noise_seed": config.noise_seed,
                     "guidance": config.guidance,
                     "latent_dim": config.latent_dim}
            frame_keys[frame] = Ledger.make_key(
                self._version, "denoise", frame, -1, upstream, param,
                config.qmode, config.frac_bits)

        cached: dict[int, object] = {}
        to_run: list[int] = []
        for frame in range(state.frame_count):
            existing = ledger.lookup(frame_keys[frame])
            if existing is not None:
                ledger.mark_reused(existing)
                cached[frame] = self._rehydrate_latent(
                    _read_json(existing.artifact_path))
            else:
                to_run.append(frame)

        computed: dict[int, object] = {}
        max_workers = min(config.max_workers, max(1, len(to_run)))
        if to_run:
            if max_workers == 1:
                for frame in sorted(to_run):
                    computed[frame] = work(frame)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {pool.submit(work, f): f for f in sorted(to_run)}
                    for future in futures:
                        frame = futures[future]
                        computed[frame] = future.result()

        for frame in sorted(computed):
            frame_latent = computed[frame]
            path = self._artifact(temp_dir, STAGES[4],
                                  f"frame_{frame:02d}/latent.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _write_json(path, frame_latent.to_dict())
            entry_upstream = [
                fusions[frame].content_id,
                fingerprint({
                    "frame": frame,
                    "allocated": alloc_by_frame[frame].allocated,
                    "demand": alloc_by_frame[frame].demand,
                    "truncated": alloc_by_frame[frame].truncated,
                    "degraded": alloc_by_frame[frame].degraded,
                }, f"alloc-f{frame}"),
                GENESIS if frame == 0 else fingerprint(
                    fusions[frame - 1].vector, f"fusion-vector-f{frame - 1}"),
            ]
            ledger.record(frame_keys[frame], "denoise", frame, -1,
                          entry_upstream,
                          {"noise_seed": config.noise_seed,
                           "guidance": config.guidance,
                           "latent_dim": config.latent_dim},
                          config.qmode, config.frac_bits,
                          self._version, path)
        out = dict(cached)
        out.update(computed)
        return out

    @staticmethod
    def _rehydrate_latent(data: dict):
        from .latent import FrameLatent, StepRecord
        steps = [StepRecord(frame=s["frame"], step=s["step"], kind=s["kind"],
                            input_fp=s["input_fp"], output=s["output"],
                            noise_entry=s["noise_entry"],
                            guidance_q=s["guidance_q"], trace=[])
                 for s in data["steps"]]
        return FrameLatent(frame=data["frame"],
                           init_material=data["init_material"],
                           initial=data["initial"], steps=steps,
                           final=data["final"], fill_basis=data["fill_basis"])

    # -- stage 6/7: tile decode and frame composition ---------------------
    def _decode_and_compose(self, state: RenderState, ledger: Ledger,
                            latents: dict, plan, temp_dir: str) -> dict:
        config = state.config
        matrix = projection_matrix(config)
        tile_outcomes: dict[tuple, object] = {}
        tasks = [(frame, tile) for frame in range(state.frame_count)
                 for tile in range(config.tile_rows)]

        def decode(frame: int, tile: int):
            guard = TileBoundaryGuard(frame, config.width, config.height,
                                      config.tile_rows)
            return decode_tile_row(frame, tile, latents[frame], matrix,
                                   config, guard)

        # Resolve cache per tile before scheduling work.
        tile_keys: dict[tuple, str] = {}
        cached: dict[tuple, object] = {}
        to_run: list[tuple] = []
        for frame, tile in tasks:
            upstream = [fingerprint(latents[frame].final,
                                    f"latent-final-f{frame}"),
                        fingerprint(matrix, f"proj-f{frame}")]
            param = {"tile": tile, "tile_rows": config.tile_rows,
                     "width": config.width, "height": config.height,
                     "pixel_mode": config.pixel_mode}
            key = Ledger.make_key(self._version, "decode", frame, tile,
                                  upstream, param, config.qmode,
                                  config.frac_bits)
            tile_keys[(frame, tile)] = key
            existing = ledger.lookup(key)
            if existing is not None:
                ledger.mark_reused(existing)
                base = os.path.dirname(existing.artifact_path)
                with open(os.path.join(base, "tile.png"), "rb") as fh:
                    png_bytes = fh.read()
                tile_result = self._rehydrate_tile(
                    _read_json(existing.artifact_path))
                tile_result.png_bytes = png_bytes
                cached[(frame, tile)] = tile_result
            else:
                to_run.append((frame, tile))

        computed: dict[tuple, object] = {}
        max_workers = min(config.max_workers, max(1, len(to_run)))
        if to_run:
            if max_workers == 1:
                for frame, tile in sorted(to_run, key=lambda t: sort_key("decode", *t)):
                    computed[(frame, tile)] = decode(frame, tile)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {pool.submit(decode, f, t): (f, t)
                               for f, t in to_run}
                    for future in futures:
                        frame, tile = futures[future]
                        computed[(frame, tile)] = future.result()

        for (frame, tile), result in sorted(computed.items(),
                                            key=lambda kv: sort_key("decode", *kv[0])):
            base = self._artifact(temp_dir, STAGES[5],
                                  f"frame_{frame:02d}/tile_{tile:02d}")
            os.makedirs(base, exist_ok=True)
            with open(os.path.join(base, "tile.png"), "wb") as fh:
                fh.write(result.png_bytes)
            _write_json(os.path.join(base, "meta.json"), result.to_dict())
            ledger.record(tile_keys[(frame, tile)], "decode", frame, tile,
                          [fingerprint(latents[frame].final,
                                       f"latent-final-f{frame}"),
                           fingerprint(matrix, f"proj-f{frame}")],
                          {"tile": tile, "tile_rows": config.tile_rows},
                          config.qmode, config.frac_bits, self._version,
                          os.path.join(base, "meta.json"))
        tile_outcomes.update(cached)
        tile_outcomes.update(computed)

        frames = {}
        for frame in range(state.frame_count):
            tiles = [tile_outcomes[(frame, t)]
                     for t in range(config.tile_rows)]
            png = assemble_frame(tiles, config)
            out_path = os.path.join(self.output_dir,
                                    f"frame_{frame:02d}.png")
            with open(out_path, "wb") as fh:
                fh.write(png)
            stage_path = self._artifact(temp_dir, STAGES[6],
                                        f"frame_{frame:02d}.png")
            os.makedirs(os.path.dirname(stage_path), exist_ok=True)
            with open(stage_path, "wb") as fh:
                fh.write(png)
            frames[frame] = {"png": png, "path": out_path}
        return frames

    @staticmethod
    def _rehydrate_tile(data: dict):
        from .decode import TileRowResult
        return TileRowResult(frame=data["frame"], tile=data["tile"],
                             row_start=data["row_start"],
                             row_end=data["row_end"], png_bytes=b"",
                             pixels=[], rounding_traces=data["rounding_traces"])

    # -- main entry --------------------------------------------------------
    def run(self, state: RenderState, resume: bool = False,
            force_crash_after: str | None = None) -> RenderOutcome:
        recovered_info = None
        if resume:
            recovered_info = self.snapshots.recover()
        version, temp_dir, previous = self.snapshots.begin_version(
            state.config, resume and recovered_info is not None)
        self._version = version
        ledger_path = os.path.join(self.snapshots.root, "ledger.json")
        ledger = Ledger(ledger_path)
        ledger.reset_counters()
        ledger.run_reused_keys = set()
        ledger.run_recomputed_keys = set()

        self._commit_validate(state, temp_dir, version)
        if force_crash_after == STAGES[0]:
            self._simulate_crash(temp_dir)

        encoded = self._encode(state, ledger, temp_dir)
        self.snapshots.commit(temp_dir, version, state.config, 1,
                              {"stage": STAGES[1]})
        if force_crash_after == STAGES[1]:
            self._simulate_crash(temp_dir)

        fusions = self._fuse(state, ledger, encoded, temp_dir)
        self.snapshots.commit(temp_dir, version, state.config, 2,
                              {"stage": STAGES[2]})
        if force_crash_after == STAGES[2]:
            self._simulate_crash(temp_dir)

        plan = self._budget(state, ledger, encoded, fusions, temp_dir)
        self.snapshots.commit(temp_dir, version, state.config, 3,
                              {"stage": STAGES[3],
                               "consumed_budget": plan.total_allocated})
        if force_crash_after == STAGES[3]:
            self._simulate_crash(temp_dir)

        latents = self._denoise(state, ledger, fusions, plan, temp_dir)
        self.snapshots.commit(temp_dir, version, state.config, 4,
                              {"stage": STAGES[4],
                               "frames_done": list(range(state.frame_count))})
        if force_crash_after == STAGES[4]:
            self._simulate_crash(temp_dir)

        frames = self._decode_and_compose(state, ledger, latents, plan,
                                          temp_dir)
        self.snapshots.commit(temp_dir, version, state.config, 5,
                              {"stage": STAGES[5]})
        self.snapshots.commit(temp_dir, version, state.config, 6,
                              {"stage": STAGES[6], "complete": True})
        ledger.save()

        reused_by_stage, recomputed_by_stage = self._stage_counts(ledger)
        from .report import build_report, write_report
        report = build_report(self.workdir, state, plan, latents, ledger,
                              version, recovered_info is not None,
                              reused_by_stage, recomputed_by_stage,
                              self.error_counts)
        report_path = write_report(self.workdir, report)
        state_path = os.path.join(self.workdir, "state.json")
        state.save(state_path)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return RenderOutcome(version=version, frames=frames,
                             reused=ledger.reused_count,
                             recomputed=ledger.recomputed_count,
                             report=report, state_path=state_path,
                             output_dir=self.output_dir)

    @staticmethod
    def _stage_counts(ledger: Ledger):
        reused: dict[str, int] = {}
        recomputed: dict[str, int] = {}
        reused_keys = getattr(ledger, "run_reused_keys", set())
        recomputed_keys = getattr(ledger, "run_recomputed_keys", set())
        for key in reused_keys:
            stage = ledger.entries[key].stage
            reused[stage] = reused.get(stage, 0) + 1
        for key in recomputed_keys:
            stage = ledger.entries[key].stage
            recomputed[stage] = recomputed.get(stage, 0) + 1
        return reused, recomputed

    @staticmethod
    def _simulate_crash(temp_dir: str) -> None:
        """Leave the temp directory in place and abort like SIGKILL."""
        raise RuntimeError(f"SIMULATED_FORCED_TERMINATION@{temp_dir}")
