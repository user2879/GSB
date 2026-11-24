"""阶段化快照：临时目录暂存 + 版本变化整体替换 + 原子清单提交。"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np

from .hashing import canonical_hash, fingerprint_array


def _atomic_write_text(path: str, text: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _atomic_replace_dir(src: str, dst: str):
    """整体替换目录：先写新版本目录，再原子 rename 覆盖。"""
    if os.path.exists(dst):
        old = dst + ".old"
        if os.path.exists(old):
            shutil.rmtree(old)
        os.rename(dst, old)
        os.rename(src, dst)
        shutil.rmtree(old)
    else:
        os.rename(src, dst)


class SnapshotStore:
    def __init__(self, output_dir: str):
        self.output_dir = os.path.abspath(output_dir)
        self.snap_dir = os.path.join(self.output_dir, "snapshot")
        self.products_dir = os.path.join(self.snap_dir, "products")
        self.manifest_path = os.path.join(self.snap_dir, "manifest.json")
        os.makedirs(self.products_dir, exist_ok=True)
        self._staging: dict[str, str] = {}

    # ---- 清单 ----
    def load_manifest(self) -> dict | None:
        if not os.path.exists(self.manifest_path):
            return None
        with open(self.manifest_path, encoding="utf-8") as f:
            return json.load(f)

    def _write_manifest(self, manifest: dict):
        _atomic_write_text(
            self.manifest_path, json.dumps(manifest, ensure_ascii=False,
                                           indent=2, sort_keys=True))

    # ---- 产品读写 ----
    def product_path(self, node_id: str) -> str:
        return os.path.join(self.products_dir, f"{node_id}.npz")

    def save_product_staging(self, node_id: str, product: dict,
                             version: int) -> str:
        """写入暂存目录，返回暂存路径；提交时整体替换。"""
        stage_root = os.path.join(self.output_dir, "_stage_products")
        os.makedirs(stage_root, exist_ok=True)
        path = os.path.join(stage_root, f"{node_id}.npz")
        payload = {
            "output": np.asarray(product["output"]),
            "q": np.asarray(product["quant"]["q"]),
            "dequant": np.asarray(product["quant"]["dequant"]),
            "residual": np.asarray(product["quant"]["residual"]),
            "order_ids": np.asarray(product["order_ids"]),
            "round_counts": np.asarray(product["round_counts"]),
        }
        if isinstance(product["quant"]["step"], np.ndarray):
            payload["step"] = np.asarray(product["quant"]["step"])
        else:
            payload["step_scalar"] = np.asarray(product["quant"]["step"])
        np.savez(path, **payload)
        meta = {
            "node_id": node_id,
            "version": version,
            "cache_key": product["cache_key"],
            "output_fingerprint": fingerprint_array(product["output"]),
            "quant_meta": {
                "bit_width": product["quant"]["bit_width"],
                "mode": product["quant"]["mode"],
                "qmax": product["quant"]["qmax"],
            },
            "meta": product.get("meta", {}),
            "fusion": product.get("fusion"),
        }
        _atomic_write_text(path + ".json",
                           json.dumps(meta, ensure_ascii=False, indent=2,
                                      sort_keys=True, default=_json_default))
        self._staging[node_id] = path
        return path

    def load_product(self, node_id: str) -> dict | None:
        path = self.product_path(node_id)
        meta_path = path + ".json"
        if not (os.path.exists(path) and os.path.exists(meta_path)):
            return None
        z = np.load(path, allow_pickle=False)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        step = z["step"] if "step" in z else float(z["step_scalar"])
        return {
            "output": z["output"], "q": z["q"], "dequant": z["dequant"],
            "residual": z["residual"], "order_ids": z["order_ids"],
            "round_counts": z["round_counts"], "step": step, "meta": meta,
        }

    def commit(self, manifest: dict):
        """把暂存产品整体替换进 products 目录，再原子写清单。"""
        if self._staging:
            stage_root = os.path.join(self.output_dir, "_stage_products")
            new_products = os.path.join(self.output_dir, "_new_products")
            if os.path.exists(new_products):
                shutil.rmtree(new_products)
            shutil.copytree(self.products_dir, new_products)
            for node_id, src in self._staging.items():
                shutil.copy2(src, os.path.join(new_products,
                                                os.path.basename(src)))
                shutil.copy2(src + ".json",
                             os.path.join(new_products,
                                          os.path.basename(src) + ".json"))
            _atomic_replace_dir(new_products, self.products_dir)
            shutil.rmtree(stage_root, ignore_errors=True)
            self._staging.clear()
        self._write_manifest(manifest)

    def reset(self):
        if os.path.exists(self.snap_dir):
            shutil.rmtree(self.snap_dir)
        os.makedirs(self.products_dir, exist_ok=True)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.generic,)):
        return o.item()
    raise TypeError(type(o).__name__)


def build_config_hash(chain, inputs_desc, cfg) -> str:
    """影响计算结果的配置哈希（运行期预算/并发/快照间隔不纳入）。"""
    material = {
        "chain": chain,
        "inputs": inputs_desc,
        "precision": cfg["precision"],
        "bit_width": cfg["bit_width"],
        "quant_mode": cfg["quant_mode"],
        "tolerance": cfg["tolerance"],
        "enable_fusion": cfg["enable_fusion"],
    }
    return canonical_hash(material)
