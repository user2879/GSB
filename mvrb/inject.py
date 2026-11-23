"""异常注入：对指定视图/阶段施加真实输入故障，并保留注入项编号。"""
from __future__ import annotations

import os
import shutil

from .errors import INJECTION_CODES
from .utils import ensure_dir, read_json, write_json

VALID_FAULTS = set(INJECTION_CODES)


def load_injection_config(path: str) -> list:
    cfg = read_json(path)
    items = cfg["injections"] if isinstance(cfg, dict) else cfg
    for item in items:
        if item["fault"] not in VALID_FAULTS:
            raise ValueError(f"unsupported injection fault {item['fault']}")
    return items


def apply_injections(sample_dir: str, work_dir: str, items: list) -> dict:
    """把样本复制到工作目录并施加故障；返回注入项 -> 实际路径/参数映射。"""
    if os.path.abspath(sample_dir) != os.path.abspath(work_dir):
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        shutil.copytree(sample_dir, work_dir)
    manifest = {"work_dir": work_dir, "hits": []}
    for item in items:
        inj_id = item["id"]
        fault = item["fault"]
        view = item.get("view")
        hit = {"id": inj_id, "fault": fault, "code": INJECTION_CODES[fault],
               "view": view, "stage": item.get("stage"), "applied": False}
        try:
            if fault == "truncated_png":
                img = os.path.join(work_dir, "images", f"view_{view:02d}.png")
                _truncate_file(img, item.get("keep_bytes", 32))
                hit["applied"] = True
            elif fault == "missing_camera":
                cams = os.path.join(work_dir, "cameras.json")
                data = read_json(cams)
                data.pop(str(view), None)
                write_json(cams, data)
                hit["applied"] = True
            elif fault == "disparity_overflow":
                cfg_path = os.path.join(work_dir, "recon_config.json")
                cfg = read_json(cfg_path)
                # 最小视差抬到无法满足，制造真实视差越界
                cfg["min_disparity"] = item.get("forced_min_disparity", 1e6)
                write_json(cfg_path, cfg)
                hit["applied"] = True
            elif fault == "non_finite_feature":
                # 在交换文件中注入非有限特征，触发加载校验 E_NON_FINITE
                v2_path = os.path.join(work_dir, "result_v2.json")
                data = read_json(v2_path)
                for f in data.get("features", []):
                    if f["view"] == view:
                        f["x"] = float("nan")
                        break
                write_json(v2_path, data)
                hit["applied"] = True
            elif fault == "zero_voxel_size":
                cfg_path = os.path.join(work_dir, "recon_config.json")
                cfg = read_json(cfg_path)
                cfg["voxel_size"] = 0.0
                write_json(cfg_path, cfg)
                hit["applied"] = True
        except FileNotFoundError as exc:
            hit["error"] = str(exc)
        manifest["hits"].append(hit)
    write_json(os.path.join(ensure_dir(work_dir), "injection_manifest.json"), manifest)
    return manifest


def _truncate_file(path: str, keep_bytes: int) -> None:
    with open(path, "rb") as fh:
        blob = fh.read()
    with open(path, "wb") as fh:
        fh.write(blob[: max(1, int(keep_bytes))])

