"""内容指纹缓存与窗口级复用账本。

缓存键由上游内容指纹、参数指纹与交换版本号分层派生：
window -> train(model) -> predict(model) -> metric(window)。
账本记录每个键的状态（reused/recomputed/invalidated）以及误差指标引用的
窗口与模型集合；未命中才计算，命中直接复用且不触发重算。
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .util import fingerprint, derive_key


@dataclass
class LedgerEntry:
    key: str
    stage: str
    series_id: str
    window_index: int
    model: str | None
    status: str                       # reused / recomputed / invalidated / failed
    refs: list[str] = field(default_factory=list)
    sort_key: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class CacheStore:
    """线程安全的内容寻址缓存，整体 JSON 落盘并以临时文件原子替换。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> Any | None:
        with self._lock:
            blob = self._data.get(key)
            return None if blob is None else json.loads(blob)

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = json.dumps(value, sort_keys=True, ensure_ascii=False)

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def flush(self, temp_dir: str | Path | None = None) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if temp_dir is not None:
                td = Path(temp_dir); td.mkdir(parents=True, exist_ok=True)
                tmp = td / (self.path.name + ".tmp")
            else:
                tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._data, sort_keys=True, indent=2),
                           encoding="utf-8")
            tmp.replace(self.path)

    def keys(self) -> set[str]:
        with self._lock:
            return set(self._data.keys())


class Ledger:
    """记录本次运行每个窗口/模型/阶段的复用与失效情况。"""

    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []
        self._index: dict[str, LedgerEntry] = {}

    def record(self, stage: str, series_id: str, window_index: int, model: str | None,
               status: str, refs: list[str] | None = None, sort_key: str = "") -> LedgerEntry:
        entry = LedgerEntry(
            key=sort_key or f"{series_id}|{window_index}|{stage}|{model or '-'}",
            stage=stage, series_id=series_id, window_index=window_index,
            model=model, status=status, refs=refs or [], sort_key=sort_key,
        )
        self.entries.append(entry)
        self._index[entry.key] = entry
        return entry

    def canonical(self) -> list[dict[str, Any]]:
        """跨运行可比的规范账本：按稳定排序键去重排序，仅保留最终状态。"""
        best: dict[str, LedgerEntry] = {}
        for entry in self.entries:
            best[entry.sort_key or entry.key] = entry
        ordered = sorted(best.values(), key=lambda e: (e.series_id, e.window_index,
                                                       e.stage, e.model or ""))
        return [e.as_dict() for e in ordered]


# ---- 分层缓存键派生 ----
def window_key(win_content_fp: str, config_hash: str, ex_version: str) -> str:
    return derive_key(["window", win_content_fp, config_hash, ex_version])


def train_key(window_k: str, model_spec: dict, ex_version: str) -> str:
    return derive_key(["train", window_k, fingerprint(model_spec), ex_version])


def predict_key(train_k: str, horizon: int, ex_version: str) -> str:
    return derive_key(["predict", train_k, horizon, ex_version])


def metric_key(predict_keys: list[str], quant: dict, ex_version: str) -> str:
    return derive_key(["metric", sorted(predict_keys), fingerprint(quant), ex_version])
