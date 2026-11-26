"""Content-fingerprint cache, exact invalidation and a reviewable ledger."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .fingerprint import fingerprint, cache_key
from .formats import FrameRecord
from .policy import Policy


@dataclass
class LedgerEntry:
    key: str
    stage: str
    frame_id: int
    step: int
    tile_row: int
    upstream_fp: str
    param_fp: str
    quant: str
    exchange_version: int
    status: str = ""  # computed | reused | invalidated
    reason: str = ""
    sort_key: list = field(default_factory=list)
    refs: list = field(default_factory=list)
    artifact_path: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class CacheStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.artifact_dir = self.root / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.entries: dict[str, LedgerEntry] = {}
        self.ledger: list[LedgerEntry] = []
        self._lock = threading.RLock()
        index_path = self.root / "index.json"
        if index_path.exists():
            self._load_index(index_path)

    def _load_index(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        for raw in data["entries"]:
            entry = LedgerEntry(**raw)
            self.entries[entry.key] = entry

    def persist_index(self) -> None:
        payload = {"entries": [entry.as_dict()
                               for entry in self.entries.values()]}
        tmp = self.root / "index.json.tmp"
        tmp.write_text(json.dumps(payload, sort_keys=True, indent=1),
                       encoding="utf-8")
        tmp.replace(self.root / "index.json")

    def make_key(self, stage: str, upstream_fp: str, param_fp: str,
                 quant: str, exchange_version: int) -> str:
        return cache_key(stage, upstream_fp, param_fp, quant, exchange_version)

    def has_valid(self, key: str) -> bool:
        with self._lock:
            entry = self.entries.get(key)
            return entry is not None and entry.status in ("computed", "reused") \
                and Path(entry.artifact_path).exists()

    def _artifact_path(self, key: str) -> Path:
        safe = key.replace(":", "_").replace("/", "_")
        return self.artifact_dir / f"{safe}.json"

    def save_artifact(self, key: str, payload: dict) -> str:
        path = self._artifact_path(key)
        with self._lock:
            tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
            tmp.write_text(json.dumps(payload, sort_keys=True, default=_default),
                           encoding="utf-8")
            tmp.replace(path)
        return str(path)

    def load_artifact(self, key: str) -> dict:
        return json.loads(Path(self.entries[key].artifact_path).read_text(
            encoding="utf-8"))

    def record(self, entry: LedgerEntry, existed: bool) -> None:
        with self._lock:
            entry.status = "reused" if existed else "computed"
            self.entries[entry.key] = entry
            self.ledger.append(entry)

    def invalidate(self, predicate, reason: str) -> list[str]:
        changed: list[str] = []
        for key, entry in self.entries.items():
            if entry.status in ("computed", "reused") and predicate(entry):
                entry.status = "invalidated"
                entry.reason = reason
                changed.append(key)
                self.ledger.append(LedgerEntry(
                    key=key, stage=entry.stage, frame_id=entry.frame_id,
                    step=entry.step, tile_row=entry.tile_row,
                    upstream_fp=entry.upstream_fp, param_fp=entry.param_fp,
                    quant=entry.quant,
                    exchange_version=entry.exchange_version,
                    status="invalidated", reason=reason,
                    sort_key=entry.sort_key, refs=entry.refs,
                    artifact_path=entry.artifact_path))
        return changed

    def canonical_ledger(self) -> list[dict]:
        rows = [entry.as_dict() for entry in self.entries.values()]
        rows.sort(key=lambda row: (row["stage"], row["frame_id"], row["step"],
                                   row["tile_row"], row["key"]))
        return rows

    def current_valid_view(self, active_exchange_version: int | None = None) -> dict:
        """Latest live artifact per coordinate (history excluded).

        Keyed by (stage, frame_id, step, tile_row); invalidated superseded
        entries are omitted so an incremental run is directly comparable to a
        from-scratch render at the same final state. When several exchange
        versions coexist for one coordinate, the active version wins.
        """
        view: dict[tuple, LedgerEntry] = {}
        for entry in self.cache_entries_ordered():
            if entry.status not in ("computed", "reused"):
                continue
            coord = (entry.stage, entry.frame_id, entry.step,
                     entry.tile_row)
            incumbent = view.get(coord)
            if incumbent is None:
                view[coord] = entry
                continue
            if (active_exchange_version is not None
                    and entry.exchange_version == active_exchange_version
                    and incumbent.exchange_version != active_exchange_version):
                view[coord] = entry
        return {",".join(map(str, coord)): entry.key
                for coord, entry in sorted(view.items())}

    def cache_entries_ordered(self) -> list[LedgerEntry]:
        return sorted(self.entries.values(),
                      key=lambda entry: (entry.stage, entry.frame_id,
                                         entry.step, entry.tile_row))

    def frame_records(self) -> list[FrameRecord]:
        records: list[FrameRecord] = []
        for entry in self.entries.values():
            if entry.stage != "frame" or entry.status == "invalidated":
                continue
            payload = self.load_artifact(entry.key)
            records.append(FrameRecord(
                frame_id=entry.frame_id, step=payload["final_step"],
                tile_row=-1,
                guidance_permille=payload["guidance_permille"],
                guidance_float=payload["guidance_float"],
                noise_code=payload["noise_code"],
                noise_scheme=payload["noise_scheme"],
                latent_fp=payload["latent_fp"],
                step_delta_fp=payload["last_delta_fp"],
                pixel_digest=payload["pixel_digest"],
                step_cost=payload["total_cost"],
                merged=payload["merged_any"],
                cache_key=entry.key,
                noise_scale_fp=payload["noise_scale_fp"],
                condition_scale_fp=payload["condition_scale_fp"],
                source_version=2,
                missing=payload.get("missing", [])))
        records.sort(key=lambda record: record.frame_id)
        return records


def _default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    return str(value)
