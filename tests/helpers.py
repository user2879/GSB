from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from pagecache.aggregate import FixedPointAggregator
from pagecache.cache import PageCache
from pagecache.fsutil import remove_tree
from pagecache.manifest import read_manifest
from pagecache.session import run_prefix, run_reference
from pagecache.snapshot import make_snapshot, save_snapshot
from samples.generate import generate_samples


def make_samples(root: Path) -> tuple[Path, dict[str, Any]]:
    data = root / "samples"
    meta = generate_samples(data)
    return data, meta


def version_paths(data: Path, version: int) -> tuple[Path, Path, Path]:
    base = data / "versions" / f"v{version}"
    return base / "manifest.pcm", data / "pages", base / "overflow"


def copy_overflow(overflow_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(overflow_dir.glob("*.seg"), key=lambda item: item.name):
        shutil.copy2(path, target_dir / path.name)


def run_pipeline(
    *,
    data: Path,
    meta: dict[str, Any],
    overflow_dir: Path,
    budget_bytes: int,
) -> tuple[FixedPointAggregator, PageCache, dict[str, Any]]:
    manifest_path, pages_dir, sample_overflow = version_paths(data, 3)
    copy_overflow(sample_overflow, overflow_dir)
    cache = PageCache(
        manifest=read_manifest(manifest_path),
        pages_dir=pages_dir,
        overflow_dir=overflow_dir,
        budget_bytes=budget_bytes,
        page_size=int(meta["page_size"]),
    )
    aggregator = FixedPointAggregator(str(meta["group_column"]), str(meta["value_column"]))
    reference = run_reference(cache=cache, aggregator=aggregator, scan_order=list(meta["scan_order"]))
    return aggregator, cache, reference


def build_snapshot_run(
    *,
    data: Path,
    meta: dict[str, Any],
    run_dir: Path,
    position: int,
    budget_bytes: int | None = None,
) -> dict[str, Any]:
    manifest_path, pages_dir, sample_overflow = version_paths(data, 3)
    overflow_dir = run_dir / "overflow"
    copy_overflow(sample_overflow, overflow_dir)
    budget = int(meta["small_budget_bytes"]) if budget_bytes is None else budget_bytes
    page_size = int(meta["page_size"])

    reference_cache = PageCache(
        manifest=read_manifest(manifest_path),
        pages_dir=pages_dir,
        overflow_dir=overflow_dir,
        budget_bytes=budget,
        page_size=page_size,
    )
    reference_aggregator = FixedPointAggregator(str(meta["group_column"]), str(meta["value_column"]))
    reference = run_reference(cache=reference_cache, aggregator=reference_aggregator, scan_order=list(meta["scan_order"]))

    cache = PageCache(
        manifest=read_manifest(manifest_path),
        pages_dir=pages_dir,
        overflow_dir=overflow_dir,
        budget_bytes=budget,
        page_size=page_size,
    )
    aggregator = FixedPointAggregator(str(meta["group_column"]), str(meta["value_column"]))
    run_prefix(cache=cache, aggregator=aggregator, scan_order=list(meta["scan_order"]), position=position)
    snapshot = make_snapshot(
        cache=cache,
        aggregator=aggregator,
        scan_order=list(meta["scan_order"]),
        position=position,
        reference_trace=reference["trace"],
        reference_final_output=reference["final_output"],
    )
    snapshot_path = run_dir / "snapshot.json"
    save_snapshot(snapshot_path, snapshot)
    for page_id in list(meta["scan_order"])[position:]:
        aggregator.consume_page(cache.get_page(page_id))
    output_path = run_dir / "output.json"
    output_bytes = aggregator.write_output(output_path)
    write_json(run_dir / "manifest.json", meta)
    return {
        "snapshot_path": snapshot_path,
        "snapshot": snapshot,
        "output_path": output_path,
        "output_bytes": output_bytes,
        "output": aggregator.output(),
        "overflow_dir": overflow_dir,
        "pages_dir": pages_dir,
        "manifest_path": manifest_path,
        "budget_bytes": budget,
        "page_size": page_size,
        "reference": reference,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clear_tree(path: Path) -> None:
    remove_tree(path)
