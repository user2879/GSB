"""Pipeline orchestration: load -> detect -> ledger -> arbitrate.

Concurrency uses a thread pool capped by config.max_concurrency; every task
carries a stable sort key derived from (stage, sample_id, category) so the
final ordering equals a serial run ordered by sample id.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import arbitrate as arbitrate_mod
from . import detect as detect_mod
from . import ledger as ledger_mod
from . import loader
from .cache import CacheStore, category_fingerprints, category_key, detection_key
from .config import Config
from .errors import E_SNAPSHOT_STALE, failure_sort_key
from .fingerprint import sha256_bytes
from .snapshot import SnapshotStore, StaleCommit


def source_hashes(data_dir):
    root = Path(data_dir)
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            hashes[str(path.relative_to(root))] = sha256_bytes(path.read_bytes())
    return hashes


def _run_pool(task_keys, fn, max_workers):
    task_keys = sorted(task_keys)
    if max_workers <= 1 or len(task_keys) <= 1:
        return {key: fn(key) for key in task_keys}
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, key): key for key in task_keys}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def _catalog(samples):
    return {
        sid: {
            "has_image": sample.get("image") is not None,
            "description": sample.get("description", []),
            "label_a": sample.get("label_a", []),
            "label_b": sample.get("label_b", []),
        }
        for sid, sample in samples.items()
    }


def _compute_detection(samples, sample_ids, config, cache, incremental, max_workers):
    def task(sort_key):
        sid = sort_key.split(":", 1)[1]
        arr = samples[sid].get("image")
        if arr is None:
            return None
        image_fp = sha256_bytes(arr.tobytes())
        key = detection_key(sid, image_fp, config)
        if incremental and cache is not None:
            hit = cache.get_detection(key)
            if hit is not None:
                return ("reused", key, hit)
        artifact = detect_mod.detect_sample(sid, arr, config)
        return ("computed", key, artifact)

    keys = [f"detect:{sid}" for sid in sample_ids]
    raw = _run_pool(keys, task, max_workers)
    detection, reused, recomputed = {}, [], []
    for sort_key in sorted(raw):
        sid = sort_key.split(":", 1)[1]
        result = raw[sort_key]
        if result is None:
            continue
        kind, key, artifact = result
        detection[sid] = artifact
        (reused if kind == "reused" else recomputed).append(sid)
        if kind == "computed" and cache is not None:
            cache.put_detection(key, artifact)
    return detection, sorted(reused), sorted(recomputed)


def _compute_categories(samples, detection, sample_ids, config, cache,
                        incremental, max_workers):
    def task(sort_key):
        _, sid, category = sort_key.split(":", 2)
        sample = samples[sid]
        det_instances = detection.get(sid, {}).get("instances", [])
        fps = category_fingerprints(sample, category)
        image_fp = detection.get(sid, {}).get("image_fingerprint")
        key = category_key(sid, category, image_fp, fps, config)
        if incremental and cache is not None:
            hit = cache.get_category(key)
            if hit is not None:
                return ("reused", key, hit)
        payload = ledger_mod.build_category_payload(
            sid,
            category,
            det_instances,
            sample.get("description", []),
            sample.get("label_a", []),
            sample.get("label_b", []),
            config,
        )
        counts_by_color = {}
        for entry in payload["ledger"]:
            counts_by_color.setdefault(entry["color"], {})[entry["source"]] = entry["count"]
        payload["arbitration"] = [
            arbitrate_mod.arbitrate(sid, category, color, counts, config)
            for color, counts in sorted(counts_by_color.items())
        ]
        return ("computed", key, payload)

    keys = [f"ledger:{sid}:{cat}" for sid in sample_ids for cat in config.categories]
    raw = _run_pool(keys, task, max_workers)
    payloads, reused, recomputed = {}, [], []
    for sort_key in sorted(raw):
        _, sid, category = sort_key.split(":", 2)
        kind, key, payload = raw[sort_key]
        payloads[(sid, category)] = payload
        item = {"sample_id": sid, "category": category}
        (reused if kind == "reused" else recomputed).append(item)
        if kind == "computed" and cache is not None:
            cache.put_category(key, payload)
    key_of = lambda item: (item["sample_id"], item["category"])
    return payloads, sorted(reused, key=key_of), sorted(recomputed, key=key_of)


def _assemble(sample_ids, config, detection, payloads):
    ledger_entries = []
    arbitration_records = []
    for (sid, category) in sorted(payloads):
        payload = payloads[(sid, category)]
        ledger_entries.extend(payload["ledger"])
        arbitration_records.extend(payload["arbitration"])
    ledger_entries.sort(
        key=lambda e: (e["sample_id"], e["category"], e["color"], e["source"])
    )
    arbitration_records.sort(
        key=lambda r: (r["sample_id"], r["category"], r["color"])
    )
    deadlocks = [
        {
            "sample_id": r["sample_id"],
            "category": r["category"],
            "color": r["color"],
            "code": r["code"],
        }
        for r in arbitration_records
        if r["status"] == "deadlock"
    ]
    return ledger_entries, arbitration_records, deadlocks


def run_pipeline(config_path, data_dir, out_dir, *, cache_path=None,
                 snapshot_dir=None, incremental=False, concurrency=None,
                 verify_serial=False):
    config = Config.load(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_workers = max(1, min(concurrency or config.max_concurrency,
                             config.max_concurrency))

    events = []
    store = SnapshotStore(snapshot_dir) if snapshot_dir else None
    hashes = source_hashes(data_dir)
    observed = None
    if store is not None:
        manifest = store.read_manifest()
        if manifest and (
            manifest.get("version") != config.version
            or manifest.get("config_hash") != config.fingerprint()
            or manifest.get("source_hashes") != hashes
        ):
            events.append({"code": E_SNAPSHOT_STALE, "action": "reset_and_recompute"})
            store.reset()
        observed = store.read_manifest()

    def commit(stage, artifact, cursor):
        nonlocal observed
        if store is None:
            return
        try:
            store.commit(
                stage, artifact, observed,
                version=config.version, config_hash=config.fingerprint(),
                source_hashes=hashes, cursor=cursor,
            )
            observed = store.read_manifest()
        except StaleCommit:
            events.append({"code": E_SNAPSHOT_STALE, "action": f"commit_aborted:{stage}"})

    # Stage 1: load + validate (always re-validated; failures are never cached)
    samples, failures = loader.load_inputs(data_dir, config)
    failures = sorted(failures, key=failure_sort_key)
    sample_ids = sorted(samples)
    commit("load", {"failures": failures, "samples": _catalog(samples)},
           sample_ids[-1] if sample_ids else None)

    cache = CacheStore(cache_path) if cache_path else None

    # Stage 2: detection (threaded, cache-aware)
    detection, det_reused, det_recomputed = _compute_detection(
        samples, sample_ids, config, cache, incremental, max_workers
    )
    commit("detect", detection, sample_ids[-1] if sample_ids else None)

    # Stage 3+4: ledger aggregation + arbitration per (sample, category)
    payloads, cat_reused, cat_recomputed = _compute_categories(
        samples, detection, sample_ids, config, cache, incremental, max_workers
    )
    ledger_entries, arbitration_records, deadlocks = _assemble(
        sample_ids, config, detection, payloads
    )
    commit("ledger", ledger_entries, sample_ids[-1] if sample_ids else None)
    commit("arbitrate", arbitration_records, sample_ids[-1] if sample_ids else None)

    if cache is not None:
        cache.save()

    serial_consistent = None
    if verify_serial:
        det_s, _, _ = _compute_detection(samples, sample_ids, config, None, False, 1)
        pay_s, _, _ = _compute_categories(samples, det_s, sample_ids, config,
                                          None, False, 1)
        led_s, arb_s, dead_s = _assemble(sample_ids, config, det_s, pay_s)
        serial_consistent = (
            det_s == detection and led_s == ledger_entries
            and arb_s == arbitration_records and dead_s == deadlocks
        )

    report = {
        "version": config.version,
        "config_hash": config.fingerprint(),
        "samples": sample_ids,
        "detection": detection,
        "ledger": ledger_entries,
        "arbitration": arbitration_records,
        "deadlocks": deadlocks,
        "failures": failures,
        "concurrency": {
            "max_workers": max_workers,
            "task_ordering": "lexicographic(stage:sample_id[:category])",
            "serial_consistent": serial_consistent,
        },
        "incremental": {
            "enabled": bool(incremental),
            "detection": {"reused": det_reused, "recomputed": det_recomputed},
            "categories": {"reused": cat_reused, "recomputed": cat_recomputed},
        },
        "snapshot": {"stale": bool(events), "events": events},
        "injections": None,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return report
