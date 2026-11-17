"""管道运行编排：解析、转换、编辑、增量执行、快照与报告。"""
from __future__ import annotations

import json
from pathlib import Path

from . import formats
from .cache import StepCache
from .converters import ConvertContext, convert_ir
from .edits import apply_edit, check_expressible
from .errors import EditRejected, PipelineError, failure_from_exc
from .inject import Injector
from .ir import IR, FeatureSet, fingerprint
from .snapshot import SnapshotManager
from .steps import PipelineResult, execute_pipeline

SAMPLE_FILES = {
    "v1": "sample.v1.txt",
    "v2": "sample.v2.json",
    "v3": "sample.v3.bin",
}


class PipelineRunner:
    def __init__(self, root: Path, inject_config: Path | None = None,
                 use_cache: bool = True, out_dir: Path | None = None):
        self.root = Path(root)
        self.data_dir = self.root / "data"
        self.out_dir = Path(out_dir) if out_dir else self.root / "output"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        config_dir = self.root / "config"
        self.name_map: dict[str, int] = json.loads(
            (config_dir / "step_name_map.json").read_text(encoding="utf-8"))
        self.name_map_hash = fingerprint(self.name_map)
        inject_text = (Path(inject_config).read_text(encoding="utf-8")
                       if inject_config and Path(inject_config).exists() else "")
        self.config_hash = fingerprint({
            "name_map": self.name_map, "inject": inject_text})
        self.injector = Injector(inject_config)
        self.cache = StepCache(self.out_dir / "cache") if use_cache else None
        self.snapshot = SnapshotManager(self.out_dir / "snapshot")
        self.ctx = ConvertContext(self.name_map, self.out_dir / "sidecar")
        self.ctx.load_sidecar()
        self.failures: list = []
        self.decisions: list = []
        self.ledger: list[dict] = self.ctx.ledger
        self.ir: IR | None = None
        self.parsed: dict[str, IR] = {}
        self.target_version = "v2"
        self.global_version = 0
        self.last_result: PipelineResult | None = None
        self.last_counter: dict = {}
        self.snapshot_status = "cold"

    def start(self) -> None:
        """启动：校验快照；命中则整体恢复，否则输出 E_SNAPSHOT_STALE 并重建。"""
        ok, failures = self.snapshot.verify()
        if ok:
            files = self.snapshot.load_files()
            self.ir = IR.from_json_dict(json.loads(files["ir.json"]))
            self.target_version = self.ir.source_version
            self.ctx.sidecar = json.loads(files["sidecar.json"])
            self.ledger.extend(json.loads(files["ledger.json"]))
            self.global_version = self.snapshot.read_manifest()["global_version"]
            self.snapshot_status = "hit"
            return
        self.failures.extend(failures)
        self.snapshot_status = "stale_rebuild"
        self.load_sources()
        self.commit_snapshot()

    def load_sources(self) -> None:
        """解析三版本样本（带异常注入与单实体失败隔离），记录版本差异。"""
        parsed: dict[str, IR] = {}
        raw_v1 = (self.data_dir / SAMPLE_FILES["v1"]).read_text(encoding="utf-8")
        raw_v1 = self.injector.apply_parse_v1(raw_v1)
        raw_v2 = (self.data_dir / SAMPLE_FILES["v2"]).read_text(encoding="utf-8")
        raw_v2 = self.injector.apply_parse_v2(raw_v2)
        raw_v3 = (self.data_dir / SAMPLE_FILES["v3"]).read_bytes()
        raw_v3 = self.injector.apply_parse_v3(raw_v3)
        for version, raw in (("v1", raw_v1), ("v2", raw_v2), ("v3", raw_v3)):
            try:
                if version == "v1":
                    parsed[version] = formats.parse_v1(raw, self.failures)
                elif version == "v2":
                    parsed[version] = formats.parse_v2(raw, self.failures)
                else:
                    parsed[version] = formats.parse_v3(raw, self.name_map, self.failures)
            except PipelineError as exc:
                self.failures.append(failure_from_exc(exc, f"parse_{version}"))
                parsed[version] = IR(source_version=version)
        counts = {v: len(ir.steps) for v, ir in parsed.items()}
        if len(set(counts.values())) > 1:
            note = f"步骤列表长度不一致: {counts}"
            for ir in parsed.values():
                ir.version_diffs.append(note)
        self.parsed = parsed
        base = parsed.get("v2") or parsed.get("v1") or parsed.get("v3")
        self.ir = base.clone()
        self.ir.version_diffs = sorted({d for ir in parsed.values()
                                        for d in ir.version_diffs})
        self.target_version = self.ir.source_version or "v2"

    def convert(self, target_version: str) -> None:
        assert self.ir is not None
        self.ir = convert_ir(self.ir, target_version, self.ctx)
        self.target_version = target_version
        self.failures.extend(self.ctx.failures)
        self.ctx.failures.clear()
        self.ctx.flush_sidecar()
        self.global_version += 1
        self.commit_snapshot()

    def apply_edit(self, edit: dict) -> bool:
        assert self.ir is not None
        try:
            new_ir = apply_edit(self.ir, edit, self.failures, self.decisions)
            check_expressible(new_ir, self.target_version, self.name_map,
                              old_ir=self.ir)
        except EditRejected as exc:
            self.failures.append(failure_from_exc(exc, "edit"))
            return False
        self.ir = new_ir
        self.ledger.append({"seq": len(self.ledger), "kind": "edit",
                            "edit": edit, "target_version": self.target_version})
        self.ctx.sidecar.setdefault("edit_history", []).append(edit)
        self.ctx.flush_sidecar()
        self.global_version += 1
        self.commit_snapshot()
        return True

    def _run_ir_with_execute_injection(self) -> IR:
        """execute 阶段注入：把目标实体特征值强制设为分桶边界（边界相等故障）。"""
        assert self.ir is not None
        run_ir = self.ir
        for item in self.injector.faults_for("execute"):
            if item.get("fault") != "bucket_boundary_equal":
                continue
            if run_ir is self.ir:
                run_ir = self.ir.clone()
            entity = item["entity"]
            for step in run_ir.steps:
                if step.name != "bucket":
                    continue
                feature = step.params["feature"]
                bounds = step.params.get("boundaries", [])
                if bounds and entity in run_ir.features.values:
                    run_ir.features.values[entity][feature] = float(bounds[0])
                    run_ir.features.missing.get(entity, set()).discard(feature)
                    self.injector.hit(item)
                    break
        return run_ir

    def run(self, incremental: bool = True) -> PipelineResult:
        assert self.ir is not None
        run_ir = self._run_ir_with_execute_injection()
        cache = self.cache if incremental else None
        self.last_counter = {}
        self.last_result = execute_pipeline(
            run_ir, cache, self.target_version, counter=self.last_counter)
        for result in self.last_result.step_results:
            self.decisions.extend(result.decisions)
        return self.last_result

    def apply_injected_edits(self) -> None:
        for item in self.injector.faults_for("edit"):
            if item.get("fault") != "merge_id_conflict":
                continue
            applied = self.apply_edit({
                "op": "merge_entities",
                "source": item["entity"],
                "target": item.get("target", "ent_000"),
            })
            if applied:
                self.injector.hit(item)

    def commit_snapshot(self) -> None:
        assert self.ir is not None
        files = {
            "ir.json": json.dumps(self.ir.to_json_dict(), ensure_ascii=False,
                                  sort_keys=True, indent=1),
            "sidecar.json": json.dumps(self.ctx.sidecar, ensure_ascii=False,
                                       sort_keys=True, indent=1),
            "ledger.json": json.dumps(self.ledger, ensure_ascii=False,
                                      sort_keys=True, indent=1),
        }
        self.snapshot.commit(self.global_version, self.config_hash,
                             self.name_map_hash, files)

    def state_digest(self) -> dict:
        """用于增量 vs 全量一致性逐字段比较的规范摘要。"""
        res = self.last_result
        return {
            "final_features": res.final_features_float() if res else None,
            "final_missing": ({e: sorted(m) for e, m in res.final_missing.items()}
                              if res else None),
            "step_summaries": ([{"index": s.index, "name": s.name,
                                 "summary": s.summary, "decisions": s.decisions}
                                for s in res.step_results] if res else []),
            "ledger": self.ledger,
            "failures": [f.to_dict() for f in self.failures],
        }
