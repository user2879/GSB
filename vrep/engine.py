"""增量表征编辑引擎（业务要求 1-8 的编排核心）。

阶段流水线：加载校验 -> 分块特征 -> 图像嵌入 -> 检索索引。
每个阶段产物以“内容指纹 + 配置指纹 + 量化档位”派生缓存键并写入不可变存储；
账本记录引用关系与复用/重算事件，实现精确失效判定。
"""

from __future__ import annotations

import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from .config import Config, ConfigError
from .errors import (
    E_CONFIG_INVALID,
    E_POINT_BUDGET_PENDING,
    E_QUANT_LEVEL,
    Failure,
)
from .features import (
    aggregate_embedding,
    chunk_cache_key,
    embedding_cache_key,
    embedding_matrix,
    enumerate_tiles,
    extract_chunk,
    feature_bundle_key,
    index_entry_cache_key,
    projection_matrix,
    quantize_matrix,
)
from .hashing import (
    short_hash,
    config_fingerprint,
    content_fingerprint,
    embed_config_fingerprint,
    feature_config_fingerprint,
    index_config_fingerprint,
)
from .index import build_index
from .injection import InjectionManager
from .jsonio import read_json, write_json
from .loader import PNG_SIGNATURE, validate_truth
from .scheduler import BudgetScheduler
from .storage import Store, new_ledger, sort_failures

LIBRARY_DIR = "library"
MANIFEST_NAME = "library_manifest.json"


class Engine:
    def __init__(
        self,
        workspace: str,
        injection_manager: InjectionManager | None = None,
        stop_event: threading.Event | None = None,
    ):
        self.workspace = os.path.abspath(workspace)
        os.makedirs(self.workspace, exist_ok=True)
        self.library_dir = os.path.join(self.workspace, LIBRARY_DIR)
        os.makedirs(self.library_dir, exist_ok=True)
        self.store = Store(os.path.join(self.workspace, "store"))
        self.store.cleanup_tmp()
        self.injections = injection_manager or InjectionManager(None)
        self.stop_event = stop_event or threading.Event()
        self.ledger = self.store.load_ledger()
        self._lock = threading.Lock()
        self._recovered_at = None
        self._current_config = None
        self._current_batch = 0
        self._partial_images: set[str] = set()
        self._last_pending_items: list[dict] = []
        self._config_failure_log: list[Failure] = []

    # ------------------------------------------------------------------
    # 快照恢复
    # ------------------------------------------------------------------
    def recover(self) -> dict | None:
        ledger, manifest = self.store.recover()
        if ledger is None:
            self._recovered_at = None
            return None
        self.ledger = ledger
        self._recovered_at = manifest
        self.ledger["last_recovery"] = {
            "global_version": manifest["global_version"],
            "config_hash": manifest["config_hash"],
            "completed_images": manifest["completed_images"],
            "stage_versions": manifest["stage_versions"],
        }
        self.store.save_ledger(self.ledger)
        return manifest

    # ------------------------------------------------------------------
    # 工作库（样本目录的私有副本，所有编辑只作用于副本）
    # ------------------------------------------------------------------
    @property
    def manifest_path(self) -> str:
        return os.path.join(self.library_dir, MANIFEST_NAME)

    def _load_manifest(self) -> dict:
        if os.path.exists(self.manifest_path):
            return read_json(self.manifest_path)
        return {"images": []}

    def _save_manifest(self, manifest: dict) -> None:
        manifest = {"images": sorted(manifest["images"])}
        write_json(self.manifest_path, manifest)

    def _library_paths(self, name: str) -> tuple[str, str]:
        return (
            os.path.join(self.library_dir, f"{name}.png"),
            os.path.join(self.library_dir, f"{name}_truth.json"),
        )

    def seed_library(self, sample_dir: str) -> list[str]:
        names = []
        for filename in sorted(os.listdir(sample_dir)):
            if not filename.lower().endswith(".png"):
                continue
            name = filename[:-4]
            png_src = os.path.join(sample_dir, filename)
            truth_src = os.path.join(sample_dir, f"{name}_truth.json")
            png_dst, truth_dst = self._library_paths(name)
            with open(png_src, "rb") as handle:
                data = handle.read()
            with open(png_dst, "wb") as handle:
                handle.write(data)
            with open(truth_src, "r", encoding="utf-8") as handle:
                truth = json.load(handle)
            write_json(truth_dst, truth)
            names.append(name)
        manifest = {"images": names}
        self._save_manifest(manifest)
        return sorted(names)

    def apply_command(self, command: dict, source_dir: str) -> None:
        op = command["op"]
        manifest = self._load_manifest()
        images = set(manifest["images"])
        name = command.get("image")
        if op in ("add", "replace"):
            self._copy_from_source(
                name, source_dir, source_png=command.get("source")
            )
            images.add(name)
        elif op == "delete":
            png_dst, truth_dst = self._library_paths(name)
            for path in (png_dst, truth_dst):
                if os.path.exists(path):
                    os.unlink(path)
            images.discard(name)
        elif op == "config":
            pass
        else:
            raise ValueError(f"未知编辑命令: {op}")
        self._save_manifest({"images": sorted(images)})

    def _copy_from_source(self, name: str, source_dir: str,
                          source_png: str | None = None) -> None:
        if source_png:
            src_png = (
                source_png
                if os.path.isabs(source_png)
                else os.path.join(source_dir, source_png)
            )
            stem = os.path.splitext(os.path.basename(src_png))[0]
            src_truth = os.path.join(os.path.dirname(src_png), f"{stem}_truth.json")
        else:
            src_png = os.path.join(source_dir, f"{name}.png")
            src_truth = os.path.join(source_dir, f"{name}_truth.json")
        dst_png, dst_truth = self._library_paths(name)
        with open(src_png, "rb") as handle:
            data = handle.read()
        with open(dst_png, "wb") as handle:
            handle.write(data)
        with open(src_truth, "r", encoding="utf-8") as handle:
            truth = json.load(handle)
        write_json(dst_truth, truth)

    # ------------------------------------------------------------------
    # 配置加载（含量化档位越界注入）
    # ------------------------------------------------------------------
    def load_config(
        self,
        config_path: str,
        image_size: int | None = None,
        allow_quant_injection: bool = False,
    ):
        with open(config_path, "r", encoding="utf-8") as handle:
            raw_config = json.load(handle)
        if allow_quant_injection:
            raw_config, injection_failure = self.injections.maybe_config_quant_fault(
                raw_config
            )
        else:
            injection_failure = None
        try:
            config = Config.from_dict(raw_config, image_size=image_size)
        except ConfigError as exc:
            injection_id = None
            if injection_failure is not None:
                # 注入命中在常规校验失败时一并登记，业务码映射到 E_QUANT_LEVEL。
                self.injections._record_hit(
                    next(
                        item
                        for item in self.injections.entries
                        if item["type"] == "quant_level_out_of_range"
                    ),
                    E_QUANT_LEVEL,
                    injection_failure.detail,
                )
                injection_id = injection_failure.injection_id
            failure = Failure(
                "<config>",
                "config",
                E_CONFIG_INVALID,
                f"{exc.field}: {exc.reason}",
                injection_id,
            )
            return None, failure
        return config, None

    def record_config_failure(self, failure: Failure, batch_no: int) -> None:
        """配置类失败累积保留，最终失败清单必须包含全部非法配置事件。"""
        if failure.to_dict() not in [item.to_dict() for item in self._config_failure_log]:
            self._config_failure_log.append(failure)

    # ------------------------------------------------------------------
    # 单图加载与校验（含截断/真值注入）

    # ------------------------------------------------------------------
    def _load_image(self, name: str):
        png_path, truth_path = self._library_paths(name)
        try:
            with open(png_path, "rb") as handle:
                raw = handle.read()
        except OSError as exc:
            return None, Failure(name, "load", "E_IMAGE_FORMAT", f"无法读取: {exc}")

        raw, truncate_failure = self.injections.maybe_truncate_png(name, raw)
        if truncate_failure is not None:
            return None, truncate_failure
        if len(raw) < 8 or raw[:8] != PNG_SIGNATURE:
            return None, Failure(
                name, "load", "E_IMAGE_FORMAT", "PNG 容器头签名不正确或文件被截断"
            )
        try:
            image = Image.open(io.BytesIO(raw))
            image.load()
        except Exception as exc:
            return None, Failure(name, "load", "E_IMAGE_FORMAT", f"PNG 解码失败: {exc}")
        if image.mode != "RGB":
            return None, Failure(
                name, "load", "E_IMAGE_FORMAT", f"像素模式必须为 RGB，实际为 {image.mode}"
            )
        rgb = np.asarray(image, dtype=np.uint8)

        try:
            with open(truth_path, "r", encoding="utf-8") as handle:
                truth_data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            return None, Failure(name, "truth", "E_TRUTH_INVALID", f"真值读取失败: {exc}")
        height, width = rgb.shape[0], rgb.shape[1]
        truth_data, truth_injection = self.injections.maybe_corrupt_truth(
            name, truth_data, width, height
        )
        if truth_injection is not None:
            return None, truth_injection
        truth_failure = validate_truth(truth_data, width, height, name)
        if truth_failure is not None:
            return None, truth_failure
        return {"name": name, "rgb": rgb, "regions": truth_data["regions"],
                "width": width, "height": height}, None

    # ------------------------------------------------------------------
    # 特征阶段（含预算预订、并发执行、非有限值注入）
    # ------------------------------------------------------------------
    @property
    def partial_dir(self) -> str:
        path = os.path.join(self.store.root, "partials")
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _partial_key(bundle_key: str, chunk_seqs) -> str:
        seq_text = "-".join(map(str, sorted(chunk_seqs))) or "none"
        return f"partial::{bundle_key}::seqs={seq_text}"

    def _partial_path(self, partial_key: str) -> str:
        safe = partial_key.replace(":", "_").replace(",", "-")
        return os.path.join(self.partial_dir, f"{safe}.json")

    def _find_latest_partial(self, bundle_key: str):
        prefix = f"partial::{bundle_key}::"
        candidates = []
        for filename in os.listdir(self.partial_dir):
            if not filename.endswith(".json"):
                continue
            payload = read_json(os.path.join(self.partial_dir, filename))
            if payload.get("bundle_basis") == bundle_key:
                candidates.append(payload)
        if not candidates:
            return None
        return max(candidates, key=lambda payload: (len(payload["chunk_seqs"]),
                                                    filename))

    def _put_partial(self, partial_key: str, payload: dict) -> None:
        write_json(self._partial_path(partial_key), payload)

    def _delete_partial(self, partial_key: str) -> None:
        path = self._partial_path(partial_key)
        if os.path.exists(path):
            os.unlink(path)

    def _compute_feature_bundle(
        self,
        image_payload: dict,
        config: Config,
        feature_cfg_fp: str,
        scheduler: BudgetScheduler,
        batch_no: int,
    ):
        name = image_payload["name"]
        rgb = image_payload["rgb"]
        content_fp = content_fingerprint(rgb)
        tiles, truncated = enumerate_tiles(
            rgb.shape[0], rgb.shape[1], config.tile_size, config.max_tiles_per_image
        )
        bundle_key = feature_bundle_key(content_fp, feature_cfg_fp, config.quant_level)
        complete_key = bundle_key
        partial_key = None

        if self.store.has_blob(complete_key):
            self._record_reuse("feature", name, complete_key, batch_no)
            return self.store.get_blob(complete_key), None

        done_seqs: set[int] = set()
        chunks: list[dict] = []
        partial_key = self._partial_key(bundle_key, ())
        prior_partial = self._find_latest_partial(bundle_key)
        if prior_partial is not None:
            partial_key = self._partial_key(bundle_key, prior_partial["chunk_seqs"])
            done_seqs = set(prior_partial["chunk_seqs"])
            chunks = list(prior_partial["chunks"])
            self._record_reuse("feature", name, partial_key, batch_no)

        project = projection_matrix(config.feature_dim, feature_cfg_fp)
        nonfinite_targets = self.injections.nonfinite_targets()
        tasks = []
        for tile in tiles:
            if tile.seq in done_seqs:
                continue
            if not scheduler.reserve(name, tile.seq, "feature", batch_no):
                continue
            tasks.append(
                {
                    "sort_key": tile.sort_key(name, "feature"),
                    "tile": tile,
                    "project": project,
                }
            )

        hit_seq = sorted(
            seq
            for seq in nonfinite_targets.get(name, set())
            if any(task["tile"].seq == seq for task in tasks)
        )
        if hit_seq:
            failure = self.injections.record_nonfinite_hit(name, hit_seq[0])
            return None, failure

        def worker(task):
            return extract_chunk(
                rgb,
                task["tile"],
                config.tile_size,
                task["project"],
                config.quant_level,
                config.fixed_bits,
            )

        tasks.sort(key=lambda item: item["sort_key"])
        worker_count = min(config.worker_count, max(1, len(tasks)))
        new_chunks: list[dict] = []
        if tasks:
            if worker_count == 1:
                for task in tasks:
                    new_chunks.append(worker(task))
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as pool:
                    new_chunks.extend(pool.map(worker, tasks))

        for chunk in new_chunks:
            tile = next(tile for tile in tiles if tile.seq == chunk["seq"])
            self.store.put_blob(
                chunk_cache_key(content_fp, feature_cfg_fp, config.quant_level, tile),
                chunk,
            )
        chunks.extend(new_chunks)
        chunks.sort(key=lambda chunk: chunk["seq"])
        done_seqs = {chunk["seq"] for chunk in chunks}

        base_bundle = {
            "schema": "vrep-feature-bundle-v1",
            "bundle_basis": bundle_key,
            "image": name,
            "content_fingerprint": content_fp,
            "feature_config_fingerprint": feature_cfg_fp,
            "quant_level": config.quant_level,
            "fixed_bits": config.fixed_bits,
            "truncated_seqs": truncated,
            "chunk_seqs": [chunk["seq"] for chunk in chunks],
            "chunks": chunks,
            "chunk_keys": [
                chunk_cache_key(
                    content_fp,
                    feature_cfg_fp,
                    config.quant_level,
                    next(tile for tile in tiles if tile.seq == chunk["seq"]),
                )
                for chunk in chunks
            ],
        }
        if len(done_seqs) == len(tiles):
            self.store.put_blob(complete_key, base_bundle)
            stale = self._find_latest_partial(bundle_key)
            if stale is not None:
                self._delete_partial(
                    self._partial_key(bundle_key, stale["chunk_seqs"])
                )
            self._record_recompute("feature", name, complete_key, batch_no)
            return base_bundle, None

        # 预算耗尽：持久化部分特征束（即使本批次一个新块都没算到），
        # 未完成分块保留在待办队列。
        partial_key = self._partial_key(bundle_key, sorted(done_seqs))
        self._put_partial(partial_key, base_bundle)
        self._record_recompute("feature", name, partial_key, batch_no)
        self._partial_images.add(name)
        self._all_partial_images.add(name)
        return None, None

    # ------------------------------------------------------------------
    # 嵌入阶段（定点整数加权聚合，线程池并发）
    # ------------------------------------------------------------------
    def _compute_embedding(
        self,
        name: str,
        bundle: dict,
        config: Config,
        embed_cfg_fp: str,
        batch_no: int,
        pending_seqs: set[int],
    ):
        if pending_seqs:
            return None, Failure(
                name,
                "embedding",
                E_POINT_BUDGET_PENDING,
                f"存在 {len(pending_seqs)} 个待办分块，嵌入聚合暂缓",
            )
        emb_key = embedding_cache_key(
            feature_bundle_key(
                bundle["content_fingerprint"],
                bundle["feature_config_fingerprint"],
                config.quant_level,
            ),
            embed_cfg_fp,
            config.quant_level,
        )
        if self.store.has_blob(emb_key):
            self._record_reuse("embedding", name, emb_key, batch_no)
            return self.store.get_blob(emb_key), None

        matrix = embedding_matrix(
            config.embed_dim, config.feature_dim, embed_cfg_fp
        )
        matrix_int = quantize_matrix(matrix, config.quant_level, config.fixed_bits)
        pairs = [(chunk["seq"], chunk["vector"]) for chunk in bundle["chunks"]]
        vector = aggregate_embedding(
            pairs, matrix_int, config.fixed_bits, config.quant_level
        )
        payload = {
            "schema": "vrep-embedding-v1",
            "image": name,
            "embedding_key": emb_key,
            "bundle_key": feature_bundle_key(
                bundle["content_fingerprint"],
                bundle["feature_config_fingerprint"],
                config.quant_level,
            ),
            "vector": vector,
            "chunk_keys": list(bundle["chunk_keys"]),
            "embed_config_fingerprint": embed_cfg_fp,
            "quant_level": config.quant_level,
        }
        self.store.put_blob(emb_key, payload)
        self._record_recompute("embedding", name, emb_key, batch_no)
        return payload, None

    # ------------------------------------------------------------------
    # 索引阶段（仅在嵌入集合或阈值变化时整体重建，条目级缓存仍写账本）
    # ------------------------------------------------------------------
    def _build_index(
        self,
        embeddings: dict,
        config: Config,
        index_cfg_fp: str,
        batch_no: int,
    ) -> dict:
        seen_index_events: set[tuple[str, str]] = set()
        for name, payload in sorted(embeddings.items()):
            entry_key = index_entry_cache_key(payload["embedding_key"], index_cfg_fp)
            entry_payload = {
                "schema": "vrep-index-entry-v1",
                "image": name,
                "entry_key": entry_key,
                "embedding_key": payload["embedding_key"],
                "vector": payload["vector"],
                "chunk_keys": list(payload["chunk_keys"]),
                "index_config_fingerprint": index_cfg_fp,
            }
            event_key = (name, entry_key)
            if event_key in seen_index_events:
                continue
            seen_index_events.add(event_key)
            if not self.store.has_blob(entry_key):
                self.store.put_blob(entry_key, entry_payload)
                self._record_recompute("index", name, entry_key, batch_no)
            else:
                self._record_reuse("index", name, entry_key, batch_no)

        index_doc = build_index(embeddings, config.similarity_threshold)
        index_doc["schema"] = "vrep-index-v1"
        index_doc["index_config_fingerprint"] = index_cfg_fp
        index_doc["index_key"] = self._index_key(embeddings, index_cfg_fp)
        return index_doc

    @staticmethod
    def _index_key(embeddings: dict, index_cfg_fp: str) -> str:
        from .hashing import short_hash

        basis = [index_cfg_fp]
        for name in sorted(embeddings):
            basis.append(f"{name}:{embeddings[name]['embedding_key']}")
        return short_hash("index|" + "|".join(basis))

    # ------------------------------------------------------------------
    # 账本事件
    # ------------------------------------------------------------------
    def _record_reuse(self, stage: str, image: str, key: str, batch_no: int) -> None:
        self.ledger["runtime"]["reuse"].append(
            {"batch": batch_no, "stage": stage, "image": image, "cache_key": key}
        )

    def _record_recompute(self, stage: str, image: str, key: str, batch_no: int) -> None:
        self.ledger["runtime"]["recompute"].append(
            {"batch": batch_no, "stage": stage, "image": image, "cache_key": key}
        )

    # ------------------------------------------------------------------
    # 待办队列（由部分特征束反推，保证跨批次/跨重启一致）
    # ------------------------------------------------------------------
    def _order_with_pending_first(self, names: list[str]) -> list[str]:
        prior = self.ledger.get("runtime", {}).get("pending", [])
        pending_images = sorted({item["image"] for item in prior if item["image"] in names})
        rest = [name for name in names if name not in pending_images]
        return pending_images + rest

    def _scan_partial_images(self, names: set[str], feature_cfg_fp: str) -> set[str]:
        result = set()
        for filename in os.listdir(self.partial_dir):
            if not filename.endswith(".json"):
                continue
            payload = read_json(os.path.join(self.partial_dir, filename))
            if (
                payload.get("image") in names
                and payload.get("feature_config_fingerprint") == feature_cfg_fp
            ):
                result.add(payload["image"])
        return result

    def _cleanup_stale_partials(self, names: set[str], feature_cfg_fp: str) -> None:
        """删除已不在库中或配置已变更的遗留部分特征束。"""
        for filename in os.listdir(self.partial_dir):
            full_path = os.path.join(self.partial_dir, filename)
            if not filename.endswith(".json"):
                continue
            payload = read_json(full_path)
            if payload.get("image") not in names or (
                payload.get("feature_config_fingerprint") != feature_cfg_fp
            ):
                os.unlink(full_path)

    def _rebuild_pending(self, names: list[str], bundles: dict,
                         partial_images: set[str]):
        pending_by_image: dict[str, set[int]] = {}
        pending_items: list[dict] = []
        config = self._current_config
        feature_cfg_fp = feature_config_fingerprint(config)
        for name in names:
            if name in bundles or name not in partial_images:
                continue
            image_payload, load_failure = self._load_image(name)
            if load_failure is not None:
                continue
            content_fp = content_fingerprint(image_payload["rgb"])
            basis = feature_bundle_key(content_fp, feature_cfg_fp, config.quant_level)
            partial = self._find_latest_partial(basis)
            if partial is None:
                continue
            all_tiles, _ = enumerate_tiles(
                image_payload["height"],
                image_payload["width"],
                config.tile_size,
                config.max_tiles_per_image,
            )
            done = set(partial["chunk_seqs"])
            missing = sorted(tile.seq for tile in all_tiles if tile.seq not in done)
            if not missing:
                continue
            pending_by_image[name] = set(missing)
            for seq in missing:
                pending_items.append(
                    {
                        "image": name,
                        "seq": seq,
                        "stage": "feature",
                        "batch_deferred": self._current_batch,
                    }
                )
        pending_items.sort(key=lambda item: (item["image"], item["seq"], item["stage"]))
        return pending_by_image, pending_items

    def _library_grid(self, name: str, config: Config) -> int:
        image_payload, failure = self._load_image(name)
        if failure is not None:
            return 0
        tiles, _ = enumerate_tiles(
            image_payload["height"], image_payload["width"],
            config.tile_size, config.max_tiles_per_image,
        )
        return len(tiles)

    @staticmethod
    def _pending_failures(pending_items: list[dict]) -> list[Failure]:
        result = []
        for item in pending_items:
            result.append(
                Failure(
                    item["image"],
                    "feature",
                    E_POINT_BUDGET_PENDING,
                    f"分块 seq={item['seq']} 等待特征点预算，"
                    f"自批次 {item['batch_deferred']} 起挂起",
                )
            )
        return result

    # ------------------------------------------------------------------
    # 单批次执行
    # ------------------------------------------------------------------
    def run_batch(self, config: Config, batch_no: int) -> dict:
        self._current_config = config
        self._current_batch = batch_no
        self._partial_images = set()
        feature_cfg_fp = feature_config_fingerprint(config)
        embed_cfg_fp = embed_config_fingerprint(config)
        index_cfg_fp = index_config_fingerprint(config)
        cfg_fp = config_fingerprint(config.to_dict())

        manifest = self._load_manifest()
        names = sorted(manifest["images"])
        self._cleanup_stale_partials(names, feature_cfg_fp)
        self._partial_images = self._scan_partial_images(names, feature_cfg_fp)
        self._all_partial_images = set(self._partial_images)
        # 每个批次只记录本批复用/重算事件，批次结束时与历史事件合并。
        self.ledger["runtime"]["reuse"] = []
        self.ledger["runtime"]["recompute"] = []
        scheduler = BudgetScheduler(config.point_budget_per_batch)
        scheduler.reset_batch(batch_no)

        failures: list[Failure] = []
        failures.extend(self._config_failure_log)
        bundles: dict[str, dict] = {}
        pending_by_image: dict[str, set[int]] = {}

        # 历史待办优先：按图像名 + 分块 seq 的稳定顺序排在新任务之前消耗预算。
        ordered_names = self._order_with_pending_first(names)

        # 阶段 1+2：加载校验 -> 分块特征（按图像名顺序预订预算，线程池内并发计算）
        # 停止信号在当前图像处理完成后才生效，保证已开始的图像产物完整提交。
        for name in ordered_names:
            image_payload, load_failure = self._load_image(name)
            if load_failure is not None:
                failures.append(load_failure)
                if self.stop_event.is_set():
                    break
                continue
            bundle, feature_failure = self._compute_feature_bundle(
                image_payload, config, feature_cfg_fp, scheduler, batch_no
            )
            if feature_failure is not None:
                failures.append(feature_failure)
            if bundle is not None:
                bundles[name] = bundle
            if self.stop_event.is_set():
                break
        all_partial_images = set(self._partial_images)

        # 待办队列以持久化的部分特征束为唯一事实来源（见 _rebuild_pending），
        # scheduler.pending 仅负责本批次预算审计。
        pending_by_image, pending_items = self._rebuild_pending(
            names, bundles, all_partial_images
        )
        self._last_pending_items = pending_items
        failures.extend(self._pending_failures(pending_items))

        # 阶段 3：图像嵌入（仅处理无待办分块的图像；线程池并发聚合）
        embeddings: dict[str, dict] = {}
        embed_tasks = [(name, bundles[name]) for name in sorted(bundles)]
        embed_tasks.sort(key=lambda item: f"embedding|{item[0]}")

        def embed_worker(item):
            name, bundle = item
            return name, *self._compute_embedding(
                name,
                bundle,
                config,
                embed_cfg_fp,
                batch_no,
                pending_by_image.get(name, set()),
            )

        worker_count = min(config.worker_count, max(1, len(embed_tasks)))
        if worker_count <= 1:
            results = [embed_worker(item) for item in embed_tasks]
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                results = list(pool.map(embed_worker, embed_tasks))
        for name, payload, embedding_failure in sorted(results, key=lambda r: r[0]):
            if embedding_failure is not None:
                failures.append(embedding_failure)
            else:
                embeddings[name] = payload

        # 阶段 4：检索索引（未完成/待办图像不在 embeddings 中，自然不进索引）
        index_doc = self._build_index(embeddings, config, index_cfg_fp, batch_no)

        self._finalize_ledger(
            config=config,
            cfg_fp=cfg_fp,
            feature_cfg_fp=feature_cfg_fp,
            embed_cfg_fp=embed_cfg_fp,
            index_cfg_fp=index_cfg_fp,
            names=names,
            bundles=bundles,
            embeddings=embeddings,
            index_doc=index_doc,
            failures=failures,
            scheduler=scheduler,
            batch_no=batch_no,
            pending_by_image=pending_by_image,
        )
        return self.ledger


    def _batch_events(self) -> dict:
        return {
            "reuse": list(self.ledger["runtime"]["reuse"]),
            "recompute": list(self.ledger["runtime"]["recompute"]),
        }

    def _finalize_ledger(
        self,
        config: Config,
        cfg_fp: str,
        feature_cfg_fp: str,
        embed_cfg_fp: str,
        index_cfg_fp: str,
        names: list[str],
        bundles: dict,
        embeddings: dict,
        index_doc: dict,
        failures: list[Failure],
        scheduler: BudgetScheduler,
        batch_no: int,
        pending_by_image: dict | None = None,
    ) -> None:
        images_section = {}
        for name in names:
            entry = {
                "present": name in bundles or any(f.image == name for f in failures),
                "feature": None,
                "embedding": None,
                "pending_seqs": sorted(
                    item["seq"]
                    for item in scheduler.state()["pending"]
                    if item["image"] == name
                ),
                "pending": name in pending_by_image,
            }
            if name in bundles:
                bundle = bundles[name]
                entry["feature"] = {
                    "bundle_key": feature_bundle_key(
                        bundle["content_fingerprint"],
                        feature_cfg_fp,
                        config.quant_level,
                    ),
                    "chunk_keys": list(bundle["chunk_keys"]),
                    "truncated_seqs": list(bundle["truncated_seqs"]),
                    "content_fingerprint": bundle["content_fingerprint"],
                }
            if name in embeddings:
                entry["embedding"] = {
                    "embedding_key": embeddings[name]["embedding_key"],
                    "vector": embeddings[name]["vector"],
                    "chunk_keys": list(embeddings[name]["chunk_keys"]),
                }
            images_section[name] = entry

        failure_dicts = sort_failures([f.to_dict() for f in failures])
        prior_runtime = self.ledger["runtime"]
        next_ledger = new_ledger()
        next_ledger["global_version"] = int(self.ledger["global_version"]) + 1
        next_ledger["config_hash"] = cfg_fp
        next_ledger["quant_level"] = config.quant_level
        next_ledger["images"] = images_section
        next_ledger["index"] = {
            "index_key": index_doc["index_key"],
            "index_config_fingerprint": index_cfg_fp,
            "threshold": index_doc["threshold"],
            "images": index_doc["images"],
            "entry_keys": index_doc["entry_keys"],
            "boundary_set": index_doc["boundary_set"],
            "views": index_doc["views"],
        }
        next_ledger["failures"] = failure_dicts
        next_ledger["runtime"] = {
            "reuse": prior_runtime["reuse"] + self._batch_events()["reuse"],
            "recompute": prior_runtime["recompute"] + self._batch_events()["recompute"],
            "budget_audit": prior_runtime["budget_audit"] + scheduler.audit,
            "pending": [
                {
                    "image": item["image"],
                    "seq": item["seq"],
                    "stage": item["stage"],
                    "batch_deferred": item["batch_deferred"],
                }
                for item in self._last_pending_items
            ],
        }
        next_ledger["snapshots"] = list(self.ledger.get("snapshots", []))
        next_ledger["last_recovery"] = self.ledger.get("last_recovery")
        self.ledger = next_ledger
        self.store.save_ledger(self.ledger)

        stage_versions = {
            "feature": feature_cfg_fp,
            "embedding": embed_cfg_fp,
            "index": index_doc["index_key"],
        }
        if (
            next_ledger["global_version"] % max(1, config.snapshot_commit_interval)
            == 0
        ):
            manifest = self.store.commit_snapshot(next_ledger, stage_versions)
            next_ledger["snapshots"].append(
                {
                    "version": manifest["global_version"],
                    "config_hash": manifest["config_hash"],
                    "completed_images": manifest["completed_images"],
                    "stage_versions": manifest["stage_versions"],
                }
            )
            self.store.save_ledger(next_ledger)
        self.ledger = next_ledger

    @property
    def recovery_point(self):
        return self._recovered_at
