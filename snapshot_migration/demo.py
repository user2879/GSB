"""Deterministic, repeatable demonstration of all six required scenarios."""

import json
import os
import tempfile
from typing import List

from .chain import load_chain
from .diagnostics import cursor_from_row, repack_range
from .engine import MigrationEngine
from .errors import BoundError, CleanError, CursorError, InheritError
from .recovery import (inject_bound_fault, inject_clean_fault,
                       inject_inherit_fault, replay, verify_integrity)


class Recorder:
    def __init__(self):
        self.lines: List[str] = []

    def add(self, text=""):
        self.lines.append(text)

    def block(self, title):
        self.add("")
        self.add("=" * 72)
        self.add(title)
        self.add("=" * 72)


def _fresh_paths():
    work = tempfile.mkdtemp(prefix="snap_demo_")
    return (work, os.path.join(work, "store.sqlite3"),
            os.path.join(work, "snapshot.json"),
            os.path.join(work, "reopened.sqlite3"))


def run_demo(chain_path) -> str:
    rec = Recorder()
    chain = load_chain(chain_path)
    work, db_path, snapshot_path, reopened_path = _fresh_paths()
    rec.add(f"workdir: {work}")
    rec.add(f"chain: {os.path.basename(chain_path)} | limits=({chain.limits.lower},"
            f"{chain.limits.upper}) | generations={len(chain.generations)}")

    engine = MigrationEngine(db_path, chain)

    # Scenario 1: three (here four) ordered migrations with split & merge
    rec.block("SCENARIO 1 - ordered migrations, split and merge")
    for gen in chain.generations:
        report = engine.migrate_generation(gen.generation_id)
        rec.add(f"[migrate] {gen.generation_id}")
        for item in report["ranges"]:
            rec.add(f"  - {item['range_id']:<3} {item['disposition']:<8} "
                    f"fragments={item['affected_fragments']}")
    for gen_id in ("G0", "G1", "G3"):
        for frag in engine.repo.active_fragments(gen_id):
            rec.add(f"  frag {frag.fragment_id} [{frag.lower},{frag.upper}) "
                    f"keys={frag.key_count} v{frag.version} "
                    f"inherit={frag.inherited_from or '-'}")

    # Scenario 2: merge/split trace + inheritance evidence
    rec.block("SCENARIO 2 - split/merge trace and inheritance mapping")
    for event in engine.repo.fragment_events():
        rec.add(f"  event {event['event']:<20} {event['fragment_id']} "
                f"detail={json.dumps(event['detail'], sort_keys=True)}")
    manifest = engine.build_manifest()
    for frag_id, info in sorted(manifest["inheritance"].items()):
        rec.add(f"  inherit {frag_id} <- {info['from_generation']} "
                f"source={info['source_fragment_id']} digest={info['digest'][:12]}")

    # Scenario 3: concurrent read view blocks then allows cleanup
    rec.block("SCENARIO 3 - cleanup coexisting with a parent read view")
    view = engine.open_read_view("demo-view", "G0")
    deferred = engine.run_cleanup()
    rec.add(f"  active refs while view open: {engine.active_read_views()}")
    for item in deferred:
        rec.add(f"  cleanup {item['action']:<16} {item['range_id']} "
                f"blockers={item.get('blockers', [])}")
    view.close()
    cleaned = engine.run_cleanup()
    for item in cleaned:
        rec.add(f"  cleanup {item['action']:<16} "
                f"{item.get('fragment_id', item['range_id'])}")
    rerun = engine.run_cleanup()
    rec.add(f"  rerun produces only idempotent states: "
            f"{sorted({item['action'] for item in rerun})}")

    # Scenario 4: two-dimensional cursor continuation + fault
    rec.block("SCENARIO 4 - two-dimensional cursor continuation")
    with engine.open_read_view("cursor-view", "G3") as read_view:
        first = read_view.scan(limit=4)
        cursor = cursor_from_row(first["rows"][-1])
        second = read_view.scan(limit=4, cursor=cursor)
        rec.add("  page1=" + json.dumps([r["key"] for r in first["rows"]]))
        rec.add("  page2=" + json.dumps([r["key"] for r in second["rows"]]))
        stable_before = read_view.iterate_all(4)
        repack_range(engine, "G3", "R1", destructive=False)
        benign_cursor = cursor_from_row(second["rows"][-1])
        benign_page = read_view.scan(limit=4, cursor=benign_cursor)
        rec.add("  after benign split/merge next=" +
                json.dumps(benign_page["rows"][0]))
        repack_range(engine, "G3", "R1", destructive=True)
        try:
            read_view.scan(limit=4, cursor=cursor_from_row(stable_before[0]))
        except CursorError as exc:
            rec.add(f"  E_CURSOR {json.dumps(exc.location, sort_keys=True)}")

    # Scenario 5: snapshot save, reopen, whole replacement and diff
    rec.block("SCENARIO 5 - snapshot save, reopen and whole replacement")
    saved = engine.save_snapshot(snapshot_path)
    rec.add(f"  saved format={saved['format_version']} "
            f"digest={saved['content_digest'][:16]} "
            f"fragments={len(saved['fragments'])}")
    engine.close()

    reopened = MigrationEngine(reopened_path, chain)
    with open(snapshot_path, "r", encoding="utf-8") as handle:
        previous = json.load(handle)
    diff = reopened.replace_from_chain(previous)
    rec.add(f"  replacement added={len(diff['added'])} "
            f"reused={len(diff['inherited_reused'])} "
            f"rewritten={len(diff['rewritten'])} "
            f"ownership_changed={len(diff['ownership_changed'])} "
            f"removed={len(diff['removed'])}")
    replay_result = replay(reopened)
    rec.add(f"  replay ok={replay_result.ok} "
            f"first_mismatch={replay_result.first_mismatch}")

    modified_path = os.path.join(work, "modified.sqlite3")
    modified_chain = _modified_chain(chain)
    modified_engine = MigrationEngine(modified_path, modified_chain)
    modified_diff = modified_engine.replace_from_chain(previous)
    rec.add("  modified-chain replacement classification:")
    rec.add(f"    added={modified_diff['added']}")
    rec.add(f"    inherited_reused={modified_diff['inherited_reused']}")
    rec.add(f"    rewritten={modified_diff['rewritten']}")
    rec.add(f"    ownership_changed={modified_diff['ownership_changed']}")
    rec.add(f"    removed={modified_diff['removed']}")
    modified_engine.close()

    # Scenario 6: three fault injections and recovery replay verification
    rec.block("SCENARIO 6 - fault injection and recovery replay")
    fault_specs = (
        ("E_BOUND", "frag:G0:R2:000",
         lambda eng: inject_bound_fault(eng, "G0", "frag:G0:R2:000"),
         BoundError),
        ("E_INHERIT", "frag:G1:R1:000",
         lambda eng: inject_inherit_fault(eng, "G1", "frag:G1:R1:000"),
         InheritError),
        ("E_CLEAN", "G-uncommitted",
         lambda eng: inject_clean_fault(eng, "G-UNCOMMITTED"), CleanError),
    )
    for label, target, injection, expected_error in fault_specs:
        fault_path = os.path.join(work, f"fault_{label}.sqlite3")
        if os.path.exists(fault_path):
            os.unlink(fault_path)
        fault_engine = MigrationEngine(fault_path, chain)
        committed_gens = (chain.generations[:-1]
                          if label == "E_CLEAN" else chain.generations)
        for gen in committed_gens:
            fault_engine.migrate_generation(gen.generation_id)
        if label == "E_CLEAN":
            pending = chain.generations[-1]
            fault_engine.repo.upsert_generation(
                "G-UNCOMMITTED", pending.seq, pending.parent_id,
                fault_engine._generation_digest(pending), committed=False)
        _fault_scenario(rec, fault_engine, label,
                        lambda injection=injection: injection(fault_engine),
                        expected_error)
        fault_engine.close()
    reopened.close()

    rec.add("")
    rec.add("DEMO COMPLETE - deterministic output end marker")
    return "\n".join(rec.lines)


def _fault_scenario(rec, engine, label, injection, expected_error):
    injection()
    try:
        verify_integrity(engine)
        rec.add(f"  {label}: FAILED to detect")
    except expected_error as exc:
        rec.add(f"  {label} detected location="
                f"{json.dumps(exc.location, sort_keys=True)}")
    try:
        result = replay(engine)
        rec.add(f"  {label} replay ok={result.ok} "
                f"first_mismatch="
                f"{json.dumps(result.first_mismatch, sort_keys=True)}")
    except (BoundError, InheritError, CleanError) as exc:
        rec.add(f"  {label} recovery correctly refused replay: {exc.code} "
                f"{json.dumps(exc.location, sort_keys=True)}")


def _modified_chain(chain):
    """A declaration change that forces one rewrite while the rest is reused."""
    import dataclasses
    from .model import KeyRange, Generation, SnapshotChain
    generations = []
    for gen in chain.generations:
        if gen.generation_id != "G3":
            generations.append(gen)
            continue
        ranges = []
        for item in gen.ranges:
            if item.range_id == "R4":
                keys = list(item.keys)
                keys[0] = "k0300039"
                keys.sort()
                ranges.append(KeyRange(item.range_id, item.lower, item.upper,
                                       tuple(keys), item.action, item.tombstone))
            else:
                ranges.append(item)
        generations.append(Generation(
            gen.generation_id, gen.seq, gen.parent_id, tuple(ranges),
            gen.rules))
    return SnapshotChain(chain.title, chain.limits, tuple(generations),
                         chain.range_universe)
