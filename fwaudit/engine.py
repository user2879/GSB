"""基镜像增量换代引擎：代次水位、失效传播、并发预算与检查点衔接。"""

from concurrent.futures import ThreadPoolExecutor
import contextlib
import os

from .errors import AuditError, E_PATCH_HEADER, E_STALE_BASE, E_UNKNOWN
from .image import parse_image
from .patch import parse_patch, audit_patch, BudgetConfig
from .storage import (
    file_lock, save_checkpoint, load_checkpoint, reset_checkpoint,
    temp_cleanup,
)


class AuditEngine:
    def __init__(self, workdir, budget=None):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.budget = budget or BudgetConfig()
        self.generation_watermark = 0
        self.images = []          # 每代镜像摘要
        self.patch_records = []   # 补丁导入与审计记录
        self._restored = False

    # ---------- 检查点 ----------
    def restore(self):
        temp_cleanup(self.workdir)
        state = load_checkpoint(self.workdir, self.budget)
        if state is None:
            reset_checkpoint(self.workdir)
            self.generation_watermark = 0
            self.images = []
            self.patch_records = []
            self._restored = True
            return False
        self.generation_watermark = state["generation_watermark"]
        self.images = state["images"]
        self.patch_records = state["patch_records"]
        self._restored = True
        return True

    def _snapshot(self):
        return {
            "generation_watermark": self.generation_watermark,
            "images": self.images,
            "patch_records": self.patch_records,
        }

    def checkpoint(self):
        save_checkpoint(self.workdir, self._snapshot(), self.budget)

    # ---------- 导入 ----------
    def import_image(self, data, path="<memory>"):
        info = parse_image(data, path=path)
        self.generation_watermark += 1
        record = {
            "generation": self.generation_watermark,
            "path": path,
            "version": info.version,
            "base_image_id": info.base_image_id,
            "base_slot": info.base_slot,
            "part_count": info.part_count,
            "declared_length": info.declared_length,
            "table_region": [info.table_offset, info.table_end],
            "partitions": [p.as_dict() for p in info.partitions],
            "findings": list(info.findings),
        }
        self.images.append(record)
        return info, record

    def import_patch(self, data, path="<memory>"):
        patch = parse_patch(data, path=path)
        record = {
            "path": path,
            "base_image_id": patch.base_image_id,
            "base_version": self.images[-1]["version"] if self.images else None,
            "op_count": len(patch.operations),
            "import_order": len(self.patch_records),
            "result": None,
        }
        self.patch_records.append(record)
        return patch, record

    # ---------- 失效传播 ----------
    def is_stale(self, patch_base_id):
        """依据镜像代次水位计算失效集合：与最新代次标识不一致即失效。"""
        if not self.images:
            return False
        return self.images[-1]["base_image_id"] != patch_base_id

    def is_stale_record(self, record):
        """基镜像标识或格式版本相对最新代次变化均导致旧补丁失效。"""
        if record["base_image_id"] is None:
            return False
        latest = self.images[-1]
        return (
            latest["base_image_id"] != record["base_image_id"]
            or latest["version"] != record["base_version"]
        )

    # ---------- 批量审计 ----------
    def run(self, image_inputs, patch_inputs, lock=True):
        """image_inputs/patch_inputs 为 (path, bytes) 列表，可任意交错。

        为保证“任意交错导入最终失效集合一致”，引擎先按代次水位收集全部
        镜像，再对全部补丁做结论；导入顺序仅记录在 import_order 字段。
        """
        with (file_lock(self.workdir) if lock else contextlib.nullcontext()):
            if not self._restored:
                self.restore()

            parsed_images = []
            parsed_patches = []
            parse_errors = []
            first_order = len(self.patch_records)

            for kind, (path, data) in self._ordered_inputs(
                    image_inputs, patch_inputs):
                try:
                    if kind == "image":
                        info, _record = self.import_image(data, path)
                        parsed_images.append(info)
                    else:
                        patch, _record = self.import_patch(data, path)
                        parsed_patches.append(patch)
                except AuditError as exc:
                    parse_errors.append({
                        "path": path,
                        "kind": kind,
                        "error": exc.as_dict(),
                    })
                    if kind == "patch":
                        self.patch_records.append({
                            "path": path,
                            "base_image_id": None,
                            "op_count": None,
                            "import_order": len(self.patch_records),
                            "result": {
                                "path": path,
                                "status": "error",
                                "codes": [exc.code],
                                "findings": [exc.as_dict()],
                                "operations": [],
                                "op_count": None,
                                "coverage_ratio": 0.0,
                                "uncovered_ops": [],
                                "degraded": False,
                                "stale": False,
                                "write_chains": [],
                            },
                        })

            latest_info = parsed_images[-1] if parsed_images else None
            results = self._audit_all(parsed_patches, latest_info)

            pending = [
                record for record in self.patch_records
                if record["result"] is None
            ]
            assert len(pending) == len(results)
            for record, result in zip(pending, results):
                record["result"] = result

            self.checkpoint()
            run_results = [
                record["result"]
                for record in self.patch_records
                if record["import_order"] >= first_order
            ]
            return {
                "parse_errors": parse_errors,
                "image_generations": list(self.images),
                "results": run_results,
                "budget": self.budget.as_dict(),
            }

    def _ordered_inputs(self, image_inputs, patch_inputs):
        ordered = []
        images = list(image_inputs)
        patches = list(patch_inputs)
        while images or patches:
            if images:
                ordered.append(("image", images.pop(0)))
            if patches:
                ordered.append(("patch", patches.pop(0)))
        return ordered

    def _audit_all(self, patches, latest_info):
        if not patches:
            return []
        workers = max(1, min(self.budget.concurrency, len(patches)))

        def work(patch):
            if latest_info is None:
                return self._no_base_result(patch)
            record = next(
                (r for r in self.patch_records
                 if r["path"] == patch.path and r["result"] is None),
                None,
            )
            if record is not None and self.is_stale_record(record):
                return self._stale_result(patch, latest_info)
            return audit_patch(patch, latest_info, self.budget)

        if workers == 1:
            return [work(patch) for patch in patches]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(work, patches))

    def _stale_result(self, patch, image):
        return {
            "path": patch.path,
            "base_image_id": patch.base_image_id,
            "status": "fail",
            "codes": [E_STALE_BASE],
            "op_count": len(patch.operations),
            "operations": [
                {**op.as_dict(), "status": "skipped_stale"}
                for op in patch.operations
            ],
            "degraded": False,
            "coverage_ratio": 0.0,
            "uncovered_ops": [op.seq for op in patch.operations],
            "chunk_size": self.budget.chunk_size,
            "elapsed_seconds": 0.0,
            "findings": [{
                "code": E_STALE_BASE,
                "message": "基镜像已换代至标识 %d，补丁基于旧标识 %d，未重新校验"
                           % (image.base_image_id, patch.base_image_id),
                "offset": 12,
                "current_base_image_id": image.base_image_id,
                "generation_watermark": self.generation_watermark,
            }],
            "write_chains": [],
            "stale": True,
        }

    def _no_base_result(self, patch):
        return {
            "path": patch.path,
            "base_image_id": patch.base_image_id,
            "status": "error",
            "codes": [E_UNKNOWN],
            "op_count": len(patch.operations),
            "operations": [],
            "degraded": False,
            "coverage_ratio": 0.0,
            "uncovered_ops": [op.seq for op in patch.operations],
            "chunk_size": self.budget.chunk_size,
            "elapsed_seconds": 0.0,
            "findings": [{
                "code": E_UNKNOWN,
                "message": "无可用基镜像，补丁无法校验",
                "offset": None,
            }],
            "write_chains": [],
            "stale": False,
        }
