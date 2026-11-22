"""Crash recovery, fault injection and field-level replay verification."""

import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from .engine import MigrationEngine
from .errors import BoundError, CleanError, InheritError
from .fragments import fragment_digest

COMPARED_FIELDS = ("lower", "upper", "key_count", "digest", "generation_id",
                   "inherited_from", "source_fragment_id", "status")


@dataclass
class ReplayResult:
    rebuilt_db: str
    first_mismatch: Optional[dict]
    fragment_differences: list
    cleanup_differences: list
    ok: bool
    first_uncommitted: Optional[str] = None


def verify_integrity(engine: MigrationEngine):
    """E_BOUND / E_INHERIT / E_CLEAN checks over the committed store."""
    repo, chain = engine.repo, engine.chain
    for gen in chain.generations:
        if not repo.is_committed(gen.generation_id):
            continue
        fragments = repo.active_fragments(gen.generation_id)
        for frag in fragments:
            if frag.key_count != len(frag.keys):
                raise BoundError(
                    "injected/stored key_count disagrees with key list",
                    generation_id=gen.generation_id,
                    fragment_id=frag.fragment_id, field="key_count")
            if frag.upper < frag.lower:
                raise BoundError("fragment upper bound below lower bound",
                                 generation_id=gen.generation_id,
                                 fragment_id=frag.fragment_id, field="upper")
            if frag.digest != fragment_digest(frag.keys):
                raise BoundError("fragment digest disagrees with key bytes",
                                 generation_id=gen.generation_id,
                                 fragment_id=frag.fragment_id, field="digest")
            if frag.inherited_from:
                if chain.generation(frag.inherited_from) is None:
                    raise InheritError(
                        "inheritance source generation is missing",
                        generation_id=gen.generation_id,
                        fragment_id=frag.fragment_id,
                        field="inherited_from")
                source = repo.fragment_versions(
                    frag.source_fragment_id or "")
                if not source:
                    raise InheritError(
                        "inheritance source fragment is missing",
                        generation_id=gen.generation_id,
                        fragment_id=frag.fragment_id,
                        field="source_fragment_id")

    committed = {row["generation_id"] for row in repo.committed_generations()}
    for entry in repo.cleanup_ledger():
        if entry["generation_id"] not in committed:
            raise CleanError(
                "cleanup ledger references an uncommitted generation pointer",
                generation_id=entry["generation_id"],
                fragment_id=entry["fragment_id"], field="cleanup_ledger")
        if entry["range_id"] and chain.generation(entry["generation_id"]):
            pass
    active_refs = {row["generation_id"] for row in repo.active_refs()}
    for entry in repo.cleanup_ledger():
        if entry["generation_id"] in active_refs:
            raise CleanError(
                "cleaned fragment still has an active read view reference",
                generation_id=entry["generation_id"],
                fragment_id=entry["fragment_id"], field="read_refs")


# ------------------------------------------------------------ fault injection
def inject_bound_fault(engine: MigrationEngine, generation_id, fragment_id):
    """Corrupt key_count so the next integrity check raises E_BOUND."""
    versions = engine.repo.fragment_versions(fragment_id)
    target = next((item for item in versions if item.status == "active"), None)
    if target is None:
        raise BoundError("unknown fragment for injection",
                         generation_id=generation_id, fragment_id=fragment_id)
    engine.repo.conn.execute(
        "UPDATE fragments SET key_count=key_count+1 WHERE fragment_id=? AND version=?",
        (fragment_id, target.version))
    engine.repo.conn.commit()
    return {"injected": "E_BOUND", "fragment_id": fragment_id,
            "generation_id": generation_id}


def inject_inherit_fault(engine: MigrationEngine, generation_id, fragment_id):
    engine.repo.conn.execute(
        "UPDATE fragments SET inherited_from='__missing_generation__' "
        "WHERE fragment_id=? AND status='active'", (fragment_id,))
    engine.repo.conn.commit()
    return {"injected": "E_INHERIT", "fragment_id": fragment_id,
            "generation_id": generation_id}


def inject_clean_fault(engine: MigrationEngine, generation_id):
    """Insert a cleanup row pointing at an uncommitted generation pointer."""
    with engine.repo.transaction():
        engine.repo.add_cleanup_entry(
            engine._next_seq(), generation_id, "__fault_range__",
            "frag:fault:000", "injected cleanup pointer conflict",
            {"injected": True})
    return {"injected": "E_CLEAN", "generation_id": generation_id}


# ------------------------------------------------------------ replay
def replay(engine: MigrationEngine) -> ReplayResult:
    verify_integrity(engine)
    fd, rebuilt_path = tempfile.mkstemp(prefix="replay_", suffix=".sqlite3")
    os.close(fd)
    os.unlink(rebuilt_path)
    rebuilt = MigrationEngine(rebuilt_path, engine.chain)
    first_uncommitted = None
    for gen in engine.chain.generations:
        if not engine.repo.is_committed(gen.generation_id):
            first_uncommitted = gen.generation_id
            break
    for gen in engine.chain.generations:
        rebuilt.migrate_generation(gen.generation_id)
    _replay_cleanup(engine, rebuilt)

    differences, first = _compare_fragments(engine, rebuilt)
    cleanup_differences = _compare_cleanup(engine, rebuilt)
    if first is None and cleanup_differences:
        first = cleanup_differences[0]
    rebuilt.close()
    return ReplayResult(
        rebuilt_db=rebuilt_path,
        first_mismatch=first,
        fragment_differences=differences,
        cleanup_differences=cleanup_differences,
        ok=not differences and not cleanup_differences,
        first_uncommitted=first_uncommitted)


def _replay_cleanup(engine: MigrationEngine, rebuilt: MigrationEngine):
    """Re-apply the durable cleanup ledger onto the rebuilt store."""
    for entry in engine.repo.cleanup_ledger():
        versions = rebuilt.repo.fragment_versions(entry["fragment_id"])
        for version_row in versions:
            if version_row.status == "active":
                with rebuilt.repo.transaction():
                    rebuilt.repo.mark_cleaned(entry["fragment_id"],
                                              version_row.version)
                    rebuilt.repo.add_cleanup_entry(
                        rebuilt._next_seq(), entry["generation_id"],
                        entry["range_id"], entry["fragment_id"],
                        entry["reason"], entry["evidence"])


def _active_index(engine):
    index = {}
    for frag in engine.repo.all_active_fragments():
        index[(frag.generation_id, frag.range_id, frag.ordinal)] = frag
    return index


def _compare_fragments(engine: MigrationEngine, rebuilt: MigrationEngine):
    left, right = _active_index(engine), _active_index(rebuilt)
    differences = []
    for identity in sorted(set(left) | set(right)):
        a, b = left.get(identity), right.get(identity)
        if a is None or b is None:
            differences.append({
                "generation_id": identity[0], "range_id": identity[1],
                "fragment_id": (a or b).fragment_id,
                "field": "existence", "expected": bool(a), "actual": bool(b)})
            continue
        for field in COMPARED_FIELDS:
            if getattr(a, field) != getattr(b, field):
                differences.append({
                    "generation_id": identity[0], "range_id": identity[1],
                    "fragment_id": a.fragment_id, "field": field,
                    "expected": getattr(a, field), "actual": getattr(b, field)})
                break
    first = differences[0] if differences else None
    return differences, first


def _compare_cleanup(engine: MigrationEngine, rebuilt: MigrationEngine):
    def signature(rows):
        return sorted((row["generation_id"], row["range_id"],
                       row["fragment_id"], row["reason"]) for row in rows)
    left = signature(engine.repo.cleanup_ledger())
    right = signature(rebuilt.repo.cleanup_ledger())
    if left == right:
        return []
    return [{"field": "cleanup_ledger", "expected": left, "actual": right}]
