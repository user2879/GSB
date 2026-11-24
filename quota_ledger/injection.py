"""Command-line fault injection helpers used by the repeatable demo."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .errors import BusinessError
from .ledger import Ledger
from .util import atomic_write_json, atomic_write_text


def prepare_injected_sample(sample_path: str | Path, workdir: str | Path, kind: str) -> Path:
    source = Path(sample_path)
    target_root = Path(workdir) / "injected_sample"
    target_root.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        for filename in ("nodes.json", "shards.json", "revisions.json", "events.json", "manifest.json"):
            src = source / filename
            if src.exists():
                shutil.copyfile(src, target_root / filename)
        if kind == "corrupt-nodes":
            atomic_write_text(target_root / "nodes.json", '{"format_version":1,"global_budget":4,"nodes":[{"id":"broken"\n')
            return target_root
        if kind == "cycle":
            shards_path = target_root / "shards.json"
            shards_doc = json.loads(shards_path.read_text(encoding="utf-8"))
            rows = shards_doc.get("shards")
            if not isinstance(rows, list) or len(rows) < 2:
                raise BusinessError("E_SPEC", "cycle injection requires at least two shards", position=str(shards_path), field="shards")
            first = rows[0]
            last = rows[-1]
            first_deps = list(first.get("dependencies", []))
            if last.get("id") not in first_deps:
                first_deps.append(last.get("id"))
            first["dependencies"] = first_deps
            last_deps = list(last.get("dependencies", []))
            if first.get("id") not in last_deps:
                last_deps.append(first.get("id"))
            last["dependencies"] = last_deps
            atomic_write_json(shards_path, shards_doc)
            return target_root
        raise BusinessError("E_SPEC", f"unsupported injection in directory mode: {kind}", field="inject")
    if not source.is_file():
        raise BusinessError("E_SPEC", "sample path does not exist", position=str(source), field="sample")
    combined = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(combined, dict):
        raise BusinessError("E_SPEC", "combined sample must be an object", position=str(source), field="$")
    if kind == "corrupt-nodes":
        atomic_write_text(target_root / "spec.json", '{"nodes":{"nodes":[\n')
        return target_root / "spec.json"
    if kind == "cycle":
        shards_doc = combined.get("shards", {})
        rows = shards_doc.get("shards") if isinstance(shards_doc, dict) else None
        if not isinstance(rows, list) or len(rows) < 2:
            raise BusinessError("E_SPEC", "cycle injection requires at least two shards", field="shards")
        first_deps = list(rows[0].get("dependencies", []))
        if rows[-1].get("id") not in first_deps:
            first_deps.append(rows[-1].get("id"))
        rows[0]["dependencies"] = first_deps
        last_deps = list(rows[-1].get("dependencies", []))
        if rows[0].get("id") not in last_deps:
            last_deps.append(rows[0].get("id"))
        rows[-1]["dependencies"] = last_deps
        atomic_write_json(target_root / "spec.json", combined)
        return target_root / "spec.json"
    raise BusinessError("E_SPEC", f"unsupported injection: {kind}", field="inject")


def inject_snapshot_corruption(workdir: str | Path) -> Path:
    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)
    db_path = workdir_path / "ledger.sqlite3"
    ledger = Ledger(db_path)
    ledger.inject_digest_corruption()
    ledger.close()
    return db_path
