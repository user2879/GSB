"""工作区引导：创建目录、生成样例清单与确定性对象内容。"""
from __future__ import annotations

import json
from pathlib import Path

from .content import expected_digest
from .engine import TierEngine
from .ledger import Ledger
from .manifest import Manifest, load_manifest
from .applog import StructuredLogger

SEED_FILE = "seed.json"
TIERS_FILE = "tiers.json"
OBJECTS_FILE = "objects.json"


def workspace_paths(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    return {
        "root": root,
        "seed": root / SEED_FILE,
        "tiers": root / TIERS_FILE,
        "objects": root / OBJECTS_FILE,
        "ledger": root / "ledger.db",
        "logs": root / "logs" / "events.jsonl",
        "tiers_dir": root / "tiers",
    }


def write_seed_file(path: str | Path, seed: int, backend: str = "python") -> None:
    Path(path).write_text(json.dumps(
        {"seed": int(seed), "backend": backend}, indent=2), encoding="utf-8")


def write_sample_manifests(root: str | Path, seed: int = 20261122,
                           backend: str = "python",
                           hot_capacity: int | None = None) -> Manifest:
    """生成一套确定性样例清单（热层偏小以触发水位迁出）。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    objects_seed = [
        ("obj-alpha", 2048, 1_000_000, 9),
        ("obj-bravo", 4096, 900_000, 3),
        ("obj-charlie", 1024, 500_000, 12),
        ("obj-delta", 3072, 200_000, 1),
        ("obj-echo", 5120, 100_000, 7),
        ("obj-foxtrot", 1536, 800_000, 20),
        ("obj-golf", 2560, 50_000, 2),
        ("obj-hotel", 768, 950_000, 15),
    ]
    total = sum(size for _, size, _, _ in objects_seed)
    capacity = hot_capacity if hot_capacity is not None else total
    tiers_doc = {
        "spec_version": 1,
        "tiers": [
            {"tier_id": "hot", "role": "hot", "capacity_bytes": capacity,
             "unit_cost": 5, "rw_weight": 2, "eviction_priority": 10},
            {"tier_id": "cold", "role": "cold",
             "capacity_bytes": capacity * 4, "unit_cost": 1,
             "rw_weight": 8, "eviction_priority": 1},
        ],
        "rules": {
            "hot_watermark_permille": 600,
            "cold_reclaim_permille": 200,
            "batch_limit": 2,
            "heat_weight_recency": 1,
            "heat_weight_frequency": 1000,
            "on_capacity": "reject",
        },
    }
    objects_doc = {"objects": []}
    for key, size, ts, count in objects_seed:
        objects_doc["objects"].append({
            "key": key, "size_bytes": size, "last_access_ms": ts,
            "access_count": count,
            "content_digest": expected_digest(seed, key, size, backend),
        })
    (root / TIERS_FILE).write_text(
        json.dumps(tiers_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / OBJECTS_FILE).write_text(
        json.dumps(objects_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    write_seed_file(root / SEED_FILE, seed, backend)
    return load_manifest(root / TIERS_FILE, root / OBJECTS_FILE,
                         root / SEED_FILE)


def bootstrap_workspace(root: str | Path, *, seed: int = 20261122,
                        backend: str = "python",
                        hot_capacity: int | None = None,
                        logger: StructuredLogger | None = None) -> tuple[
                            TierEngine, Ledger, Manifest]:
    """完整引导：样例清单 -> 强校验 -> 建账本 -> 生成对象文件 -> 初始登记。"""
    root = Path(root)
    manifest = write_sample_manifests(root, seed=seed, backend=backend,
                                      hot_capacity=hot_capacity)
    engine = TierEngine(root, logger=logger)
    engine.prepare_dirs(manifest)
    ledger = Ledger(engine.db_path)
    if ledger.object_count() == 0:
        ledger.register_spec(manifest, _now_ms())
        engine.seed_objects(manifest)
        ledger.initialize_objects(manifest, _now_ms())
        engine.reconcile_bytes(manifest, ledger)
    return engine, ledger, manifest


def _now_ms() -> int:
    import time
    return int(time.time() * 1000)
