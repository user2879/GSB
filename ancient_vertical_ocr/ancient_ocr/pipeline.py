"""主流水线：阶段编排、缓存对账、预算待办、快照提交与命令流增量执行。"""

from __future__ import annotations
import json
from pathlib import Path

from .cache import CacheEntry, StageCache
from .components import CompareLedger, run_fine_stage
from .config import Config
from .hashing import digest
from .loader import LoadResult, load_corpus
from .ordering import (AdjudicationLedger, build_intra_paths, merge_columns)
from .sample_gen import build_default_corpus, generate_sample, inject_bad_samples
from .scheduler import Scheduler, TodoUnit
from .snapshot import SnapshotStore, STAGE_ORDER
from .windows import (ScanLedger, coarse_cache_key, enumerate_windows,
                      merge_candidates, run_coarse_scan, score_window,
                      _window_pixels)
from .fixedpoint import FixedPoint


class Pipeline:
    def __init__(self, samples_root: str | Path, cfg: Config,
                 snapshot_root: str | Path) -> None:
        self.samples_root = Path(samples_root)
        self.cfg = cfg
        self.cache = StageCache()
        self.scheduler = Scheduler(cfg)
        self.snapshots = SnapshotStore(snapshot_root)
        self.global_version = 0
        self.last_load: LoadResult | None = None

    # ---------- 阶段 0：加载校验 ----------
    def load(self) -> LoadResult:
        result = load_corpus(self.samples_root)
        self.last_load = result
        for sample_id in sorted(result.samples):
            sample = result.samples[sample_id]
            entry = CacheEntry(
                stage="load", sample_id=sample_id,
                cache_key=digest(["load", sample.image_fingerprint,
                                  digest(sample.truth), digest(sample.declaration),
                                  sample.active_keypoint_fingerprint]),
                upstream=[],
                payload={"image_fingerprint": sample.image_fingerprint,
                         "keypoint_fingerprint": sample.active_keypoint_fingerprint},
                fingerprints={"image": sample.image_fingerprint,
                              "keypoint": sample.active_keypoint_fingerprint})
            self.cache.put(entry)
        for issue in result.issues:
            self.scheduler.record_failure(issue.sample_id, "load", issue.code, issue.message)
        return result

    # ---------- 阶段 1：粗定位（线程池，预算耗尽进待扫队列） ----------
    def run_coarse(self, sample) -> dict:
        cached = self.cache.get(sample.sample_id, "coarse")
        wanted_key = coarse_cache_key(sample, self.cfg)
        if cached and cached.cache_key == wanted_key:
            return cached.payload
        units = enumerate_windows(sample.sample_id, self.cfg)
        ledger = ScanLedger()
        fp = FixedPoint(self.cfg.frac_bits)

        def worker(unit):
            pixels = _window_pixels(sample.image, unit, self.cfg.window_size)
            density, valley, gradient, score, fingerprint = score_window(pixels, fp)
            return unit, (density, valley, gradient, score, fingerprint)

        # 预算准入在主线程按稳定顺序判定；计算在线程池，并发数严格取配置。
        admissible: list = []
        for unit in sorted(units, key=lambda item: (item.row, item.col)):
            if ledger.scan_rows_consumed + self.cfg.window_size > self.cfg.scan_row_budget:
                self.scheduler.add_todo(TodoUnit(
                    unit.sort_key, sample.sample_id, "coarse",
                    f"r{unit.row}c{unit.col}", {"scan_row": 0}))
                continue
            ledger.scan_rows_consumed += self.cfg.window_size
            ledger.deductions.append({
                "stage": "coarse", "unit": unit.sort_key, "budget": "scan_row",
                "cost": self.cfg.window_size,
                "remaining": self.cfg.scan_row_budget - ledger.scan_rows_consumed})
            admissible.append(unit)
        computed = self.scheduler.run_pool(admissible, worker)
        scores = []
        from .windows import WindowScore
        for unit, (density, valley, gradient, score, fingerprint) in computed:
            unit.fingerprint, unit.done = fingerprint, True
            ledger.completed.append(unit)
            if score >= self.cfg.candidate_score_threshold:
                scores.append(WindowScore(unit.row, unit.col, unit.x, unit.y,
                                          fingerprint, density, valley, gradient, score))
        regions = merge_candidates(scores, self.cfg)
        payload = {"regions": regions, "scores": [s.to_dict() for s in scores],
                   "scan_ledger": {"consumed": ledger.scan_rows_consumed,
                                   "deductions": ledger.deductions},
                   "pending_scan": [u.sort_key for u in self.scheduler.todos
                                    if u.sample_id == sample.sample_id and u.stage == "coarse"]}
        entry = CacheEntry(
            stage="coarse", sample_id=sample.sample_id, cache_key=wanted_key,
            upstream=[f"load:{sample.image_fingerprint}"],
            payload=payload,
            fingerprints={"image": sample.image_fingerprint,
                          "params": self.cfg.param_fingerprint(),
                          "quant": str(self.cfg.frac_bits)})
        self.cache.put(entry)
        self.scheduler.budget.scan_rows_remaining -= ledger.scan_rows_consumed
        self.scheduler.update_preview(sample.sample_id, "coarse",
                                      {"regions": len(regions),
                                       "scan_consumed": ledger.scan_rows_consumed})
        return payload

    # ---------- 阶段 2：细定位 + 回归 + 字模比对 ----------
    def run_fine(self, sample, coarse_payload: dict) -> dict:
        wanted_key = digest(["fine", coarse_cache_key(sample, self.cfg),
                             sample.active_keypoint_fingerprint, self.cfg.frac_bits])
        cached = self.cache.get(sample.sample_id, "fine")
        if cached and cached.cache_key == wanted_key:
            return cached.payload
        ledger = CompareLedger()
        payload = run_fine_stage(sample.image, coarse_payload["regions"], self.cfg, ledger)
        payload["compare_ledger"] = {"consumed": ledger.compare_points_consumed,
                                     "deductions": ledger.deductions,
                                     "pending": ledger.pending}
        for item in ledger.pending:
            self.scheduler.add_todo(TodoUnit(
                f"{sample.sample_id}|fine|{item['component_id']}", sample.sample_id,
                "fine", item["component_id"], {"compare_point": 0}))
        entry = CacheEntry(
            stage="fine", sample_id=sample.sample_id, cache_key=wanted_key,
            upstream=[f"coarse:{coarse_cache_key(sample, self.cfg)}"],
            payload=payload,
            fingerprints={"coarse": coarse_cache_key(sample, self.cfg),
                          "keypoint": sample.active_keypoint_fingerprint,
                          "quant": str(self.cfg.frac_bits)})
        self.cache.put(entry)
        self.scheduler.budget.compare_points_remaining -= ledger.compare_points_consumed
        self.scheduler.update_preview(sample.sample_id, "fine",
                                      {"components": len(payload["components"]),
                                       "regressed": len(payload["regressions"]),
                                       "compare_consumed": ledger.compare_points_consumed})
        return payload

    # ---------- 阶段 3/4：栏内路径 + 跨栏合并 ----------
    def run_ordering(self, sample, fine_payload: dict) -> dict:
        active_points = sample.keypoint_versions["versions"][
            sample.keypoint_versions["active_version"]]["points"]
        regressions = fine_payload["regressions"]
        for reg, point in zip(regressions, active_points):
            reg["annotator_x"] = point["x"]
            reg["annotator_y"] = point["y"]
            reg["x"] = point["x"]
            reg["y"] = point["y"]
        intra = build_intra_paths(regressions, sample.declaration)
        fine_key = self.cache.get(sample.sample_id, "fine").cache_key
        intra_key = digest(["intra", self.cache.get(sample.sample_id, "fine").cache_key,
                            sample.active_keypoint_fingerprint])
        self.cache.put(CacheEntry(
            stage="intra", sample_id=sample.sample_id, cache_key=intra_key,
            upstream=[f"fine:{fine_key}"],
            payload=intra,
            fingerprints={"fine": self.cache.get(sample.sample_id, "fine").cache_key,
                          "keypoint": sample.active_keypoint_fingerprint}))
        ledger = AdjudicationLedger()
        merged = merge_columns(intra, sample.declaration, self.cfg, ledger)
        for col in ledger.pending_columns:
            self.scheduler.add_todo(TodoUnit(
                f"{sample.sample_id}|merge|col{col}", sample.sample_id,
                "merge", f"col{col}", {"adjudication_point": ledger.points_consumed}))
        merge_key = digest(["merge", intra_key, digest(sample.declaration),
                            self.cfg.adjudication_budget])
        payload = {"intra": intra, "merge": merged,
                   "adjudication_ledger": {"consumed": ledger.points_consumed,
                                           "deductions": ledger.deductions,
                                           "pending_columns": ledger.pending_columns}}
        self.cache.put(CacheEntry(
            stage="merge", sample_id=sample.sample_id, cache_key=merge_key,
            upstream=[f"intra:{intra_key}"], payload=payload,
            fingerprints={"intra": intra_key,
                          "declaration": digest(sample.declaration),
                          "adjudication_budget": str(self.cfg.adjudication_budget)}))
        self.scheduler.budget.adjudication_points_remaining -= ledger.points_consumed
        self.scheduler.update_preview(sample.sample_id, "merge",
                                      {"final_order": merged["final_order"],
                                       "pending_columns": merged["pending_columns"]})
        return payload

    # ---------- 全量/增量统一入口 ----------
    def execute(self, persist: bool = True) -> dict:
        previous_entries = {
            (entry.sample_id, entry.stage): entry
            for entry in self.cache.all_entries()}
        load_result = self.load()
        sample_ids = sorted(load_result.samples)
        expected: dict[tuple[str, str], CacheEntry] = {}
        results: dict[str, dict] = {}
        for sample_id in sample_ids:
            sample = load_result.samples[sample_id]
            coarse = self.run_coarse(sample)
            fine = self.run_fine(sample, coarse)
            ordering = self.run_ordering(sample, fine)
            results[sample_id] = {"coarse": coarse, "fine": fine, "ordering": ordering}
        for entry in self.cache.all_entries():
            expected[(entry.sample_id, entry.stage)] = entry
        report = self.cache.reconcile(previous_entries, expected)
        state = self._state_dict(load_result, results, report)
        if persist:
            self.global_version += 1
            self.snapshots.commit(state, self.global_version, self.cfg.fingerprint())
        return state

    def _state_dict(self, load_result: LoadResult, results: dict, report) -> dict:
        sample_state: dict[str, dict] = {}
        for sample_id in sorted(results):
            sample_state[sample_id] = {
                "completed_units": [
                    {"unit_key": f"{sample_id}|{stage}|done", "stage": stage}
                    for stage in STAGE_ORDER if stage != "load"
                ],
                "artifacts": results[sample_id],
            }
        return {
            "stage": "merge",
            "global_version": self.global_version + 1,
            "config_hash": self.cfg.fingerprint(),
            "samples": sample_state,
            "issues": [issue.to_dict() for issue in load_result.issues],
            "invalidation": {"invalidated": report.invalidated,
                             "reused": report.reused, "reasons": report.reasons},
            "scheduler": self.scheduler.to_dict(),
            "cache_state_digest": self.cache.state_digest(),
        }

    # ---------- 命令流 ----------
    def apply_command(self, command: dict) -> None:
        op = command["op"]
        if op == "replace_image":
            generate_sample(self.samples_root, command["sample_id"],
                            command.get("seed", 7777), self.cfg.frac_bits)
        elif op == "add_sample":
            generate_sample(self.samples_root, command["sample_id"],
                            command["seed"], self.cfg.frac_bits)
        elif op == "remove_sample":
            target = self.samples_root / command["sample_id"]
            if target.exists():
                import shutil
                shutil.rmtree(target)
            self.cache.remove_sample(command["sample_id"])
            self.scheduler.todos = [
                item for item in self.scheduler.todos
                if item.sample_id != command["sample_id"]]
            self.scheduler.failures = [
                item for item in self.scheduler.failures
                if item["sample_id"] != command["sample_id"]]
        elif op == "replace_keypoint_version":
            path = self.samples_root / command["sample_id"] / "keypoint_versions.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["active_version"] = command["version"]
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        elif op in ("set_window", "set_quant", "set_budget"):
            raise RuntimeError("配置类命令由 CLI 重建 Config 后执行")
        elif op in ("crash", "recover", "run"):
            return
        else:  # pragma: no cover
            raise ValueError(f"unsupported op {op}")


def bootstrap_demo_workspace(root: Path, cfg: Config) -> list[str]:
    samples = root / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    ids = build_default_corpus(samples, cfg.frac_bits)
    ids_bad = inject_bad_samples(samples)
    return ids + ids_bad
