"""清单目录的版本归一与中断续做。

打开目录时，低于当前版本（v3）的清单逐条记录升级：
  1. 每条记录升级后立即原子更新检查点文件（tmp + os.replace）；
  2. 全部完成后以当前版本原子重写清单文件；
  3. 旧清单内容保留为只读备份 <name>.<ver>.bak；
  4. 强制终止后重启只依据检查点继续，已升级记录不重复处理。
"""

import json
import os
import stat

from .canon import canon
from .errors import FormatError
from .manifest import (CURRENT_FORMAT, FORMAT_V1, Entry, Manifest, Strategy,
                       decode_manifest, detect_format, encode_manifest,
                       upgrade_entry)

UPGRADE_DELAY_ENV = "CONFIGDIST_UPGRADE_DELAY_MS"


def _atomic_write(path, text):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _ckpt_path(manifest_path):
    return manifest_path + ".upgchk.json"


def normalize_file(path, hook=None):
    """归一单个清单文件，返回报告 dict；已是当前版本则跳过。"""
    with open(path, "r", encoding="utf-8") as fh:
        original_text = fh.read()
    source_version, raw = detect_format(original_text)
    if source_version == CURRENT_FORMAT:
        decode_manifest(original_text)  # 仍做强校验
        return {"file": os.path.basename(path), "source_version": source_version,
                "target_version": CURRENT_FORMAT, "records": 0,
                "resumed_from": None, "status": "already-current"}

    # 先整体强校验源清单，失败则 E_FORMAT 整体拒绝，不产生任何副作用
    decode_manifest(original_text)
    entries_raw = raw["entries"]
    if source_version == FORMAT_V1:
        manifest_version = 1
    else:
        manifest_version = raw["manifest_version"]

    ckpt_file = _ckpt_path(path)
    resumed_from = None
    if os.path.exists(ckpt_file):
        with open(ckpt_file, "r", encoding="utf-8") as fh:
            ckpt = json.load(fh)
        if ckpt.get("source_version") != source_version:
            raise FormatError("检查点来源版本与清单不一致",
                              path=ckpt_file, source_version=source_version)
        records = ckpt["records"]
        next_index = ckpt["next_index"]
        resumed_from = next_index
    else:
        records = []
        next_index = 0

    delay_ms = int(os.environ.get(UPGRADE_DELAY_ENV, "0"))
    for i in range(next_index, len(entries_raw)):
        records.append(upgrade_entry(entries_raw[i], source_version))
        ckpt = {"source_version": source_version,
                "target_version": CURRENT_FORMAT,
                "manifest_version": manifest_version,
                "next_index": i + 1,
                "records": records}
        _atomic_write(ckpt_file, canon(ckpt))
        if hook is not None:
            hook(i + 1, len(entries_raw))
        if delay_ms:
            import time
            time.sleep(delay_ms / 1000.0)

    upgraded = Manifest(
        format=CURRENT_FORMAT,
        manifest_version=manifest_version,
        entries=[Entry(key=r["key"], value=r["value"], targets=r["targets"],
                       priority=r["priority"]) for r in records],
        derived=[],
        strategy=Strategy(),
    )
    # 旧清单保留为只读备份，再原子重写为当前版本
    backup_path = f"{path}.{source_version}.bak"
    _atomic_write(backup_path, original_text)
    os.chmod(backup_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    _atomic_write(path, encode_manifest(upgraded))
    os.remove(ckpt_file)
    return {"file": os.path.basename(path), "source_version": source_version,
            "target_version": CURRENT_FORMAT, "records": len(records),
            "resumed_from": resumed_from, "status": "upgraded"}


def normalize_directory(manifest_dir, hook=None):
    """归一目录下全部 *.json 清单，返回稳定排序的升级报告。"""
    reports = []
    for name in sorted(os.listdir(manifest_dir)):
        if not name.endswith(".json") or name.endswith(".upgchk.json"):
            continue
        reports.append(normalize_file(os.path.join(manifest_dir, name), hook=hook))
    return reports
