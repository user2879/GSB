"""Diagnostic-only fragment repacking to exercise cursor stability.

Production migrations never mutate a committed generation; these hooks exist to
demonstrate what happens to a two-dimensional cursor when the pinned generation
undergoes a split/merge.  ``benign`` keeps every key in deterministic order so
the cursor re-anchors safely; ``destructive`` drops the anchor key so the next
scan must raise E_CURSOR.
"""

from .fragments import (fragment_digest, make_fragment_id,
                        verify_fragment_invariants)
from .readview import Cursor


def repack_range(engine, generation_id, range_id, *, destructive=False):
    repo, chain = engine.repo, engine.chain
    current = [f for f in repo.active_fragments(generation_id)
               if f.range_id == range_id]
    current.sort(key=lambda item: item.ordinal)
    keys = [key for frag in current for key in frag.keys]
    if destructive and keys:
        removed = keys[0]
        keys = keys[1:]
    else:
        removed = None

    events = []
    with repo.transaction():
        for frag in current:
            repo.supersede_fragment(frag.fragment_id, frag.version)
            repo.add_event(engine._next_seq(), generation_id, range_id,
                           frag.fragment_id, "diagnostic_supersede",
                           {"old_version": frag.version,
                            "destructive": destructive})

        from .fragments import pack_range
        spec = chain.generation(generation_id).range(range_id)
        next_version = max((frag.version for frag in current), default=0) + 1
        packed = pack_range(
            generation_id, range_id, spec.lower, spec.upper, tuple(keys),
            chain.limits.lower, chain.limits.upper, version=next_version)
        for frag in packed:
            repo.insert_fragment(frag)
            repo.add_event(engine._next_seq(), generation_id, range_id,
                           frag.fragment_id,
                           "diagnostic_split" if len(packed) > 1
                           else "diagnostic_merge",
                           {"new_version": frag.version,
                            "key_count": frag.key_count})
            events.append(frag.fragment_id)
        verify_fragment_invariants(
            repo.active_fragments(generation_id),
            chain.limits.lower, chain.limits.upper)
    return {"generation_id": generation_id, "range_id": range_id,
            "destructive": destructive, "removed_key": removed,
            "new_fragments": events}


def cursor_from_row(row):
    return Cursor(row["range_id"], row["key"], row["fragment_id"],
                  row["fragment_version"])
