"""Fragment packing, canonical bytes and explicit split / merge rules.

Rules (declared explicitly and applied deterministically):

R1 (no crossing)   A fragment never crosses a key-range boundary.
R2 (capacity)      Every committed fragment holds ``lower <= key_count <= upper``
                   keys; a single-key range is the documented ``lower=1`` case.
R3 (over-size)     A range with more than ``upper`` keys is split at explicit
                   split points into the fewest fragments possible: choose the
                   smallest fragment count ``k`` whose balanced split keeps every
                   fragment within capacity, then split at the derived boundary
                   keys. A split point is the first key of the right fragment.
R4 (under-size)    After splitting, trailing fragments below ``lower`` are merged
                   into their predecessor; when that exceeds ``upper`` the keys
                   are rebalanced using the R3 explicit split points instead.
R5 (contiguity)    Bounds are half-open [lower, upper): the first fragment lower
                   bound equals the range lower bound, the last upper bound
                   equals the range upper bound, and fragment ``n`` upper bound
                   equals fragment ``n+1`` lower bound (its first key).

A key range whose length cannot be partitioned under R2 raises ``E_BOUND``.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .errors import BoundError


def canonical_bytes(keys: Tuple[str, ...]) -> bytes:
    """Byte-identical serialization reused for inherited fragments."""
    return json.dumps(list(keys), ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def fragment_digest(keys: Tuple[str, ...]) -> str:
    return hashlib.sha256(canonical_bytes(keys)).hexdigest()


@dataclass
class Fragment:
    fragment_id: str
    generation_id: str
    range_id: str
    ordinal: int
    lower: str
    upper: str
    keys: Tuple[str, ...]
    key_count: int
    digest: str
    version: int
    inherited_from: Optional[str]   # generation id whose bytes are reused
    source_fragment_id: Optional[str]
    status: str = "active"          # active | superseded | cleaned

    def to_manifest(self):
        return {
            "fragment_id": self.fragment_id,
            "generation_id": self.generation_id,
            "range_id": self.range_id,
            "ordinal": self.ordinal,
            "lower": self.lower,
            "upper": self.upper,
            "key_count": self.key_count,
            "digest": self.digest,
            "version": self.version,
            "inherited_from": self.inherited_from,
            "source_fragment_id": self.source_fragment_id,
            "status": self.status,
        }


def explicit_split_sizes(total: int, lower: int, upper: int) -> List[int]:
    """Return fragment sizes for ``total`` keys honoring R2-R4.

    Uses the fewest fragments possible and keeps sizes as balanced as possible,
    which makes the explicit split points stable for identical inputs.
    """
    if total <= 0:
        return []
    min_fragments = (total + upper - 1) // upper
    max_fragments = total // lower if lower else total
    if min_fragments > max_fragments:
        raise BoundError(
            f"key range of {total} key(s) cannot be partitioned into fragments "
            f"within [{lower}, {upper}]", field="key_count")
    count = min_fragments
    base, extra = divmod(total, count)
    sizes = [base + (1 if index < extra else 0) for index in range(count)]
    if any(not (lower <= size <= upper) for size in sizes):
        raise BoundError(
            f"derived split sizes {sizes} violate capacity [{lower}, {upper}]",
            field="key_count")
    return sizes


def make_fragment_id(generation_id: str, range_id: str, ordinal: int) -> str:
    return f"frag:{generation_id}:{range_id}:{ordinal:03d}"


def pack_range(generation_id: str, range_id: str, range_lower: str,
               range_upper: str, keys: Tuple[str, ...], lower: int, upper: int,
               *, inherited_from: Optional[str] = None,
               source_fragment_id: Optional[str] = None,
               version: int = 1) -> List[Fragment]:
    sizes = explicit_split_sizes(len(keys), lower, upper)
    fragments: List[Fragment] = []
    cursor = 0
    for ordinal, size in enumerate(sizes):
        part = tuple(keys[cursor:cursor + size])
        cursor += size
        frag_lower = range_lower if ordinal == 0 else part[0]
        is_last = ordinal == len(sizes) - 1
        frag_upper = range_upper if is_last else keys[cursor]
        fragments.append(Fragment(
            fragment_id=make_fragment_id(generation_id, range_id, ordinal),
            generation_id=generation_id,
            range_id=range_id,
            ordinal=ordinal,
            lower=frag_lower,
            upper=frag_upper,
            keys=part,
            key_count=len(part),
            digest=fragment_digest(part),
            version=version,
            inherited_from=inherited_from,
            source_fragment_id=source_fragment_id,
        ))
    _assert_contiguous(generation_id, range_id, range_lower, range_upper, fragments)
    return fragments


def _assert_contiguous(generation_id, range_id, range_lower, range_upper, fragments):
    if not fragments:
        return
    if fragments[0].lower != range_lower:
        raise BoundError("first fragment lower bound != range lower bound",
                         generation_id=generation_id, range_id=range_id,
                         fragment_id=fragments[0].fragment_id, field="lower")
    if fragments[-1].upper != range_upper:
        raise BoundError("last fragment upper bound != range upper bound",
                         generation_id=generation_id, range_id=range_id,
                         fragment_id=fragments[-1].fragment_id, field="upper")
    for left, right in zip(fragments, fragments[1:]):
        if left.upper != right.lower:
            raise BoundError(
                f"fragments {left.fragment_id} and {right.fragment_id} are not "
                "contiguous: upper bound must meet next lower bound",
                generation_id=generation_id, range_id=range_id,
                fragment_id=right.fragment_id, field="lower")
        if left.upper > right.lower:
            raise BoundError("fragments overlap",
                             generation_id=generation_id, range_id=range_id,
                             fragment_id=right.fragment_id, field="lower")


def verify_fragment_invariants(fragments: List[Fragment], lower: int, upper: int):
    """Used by integrity checks and replay; raises E_BOUND on first violation."""
    by_range: dict = {}
    for frag in fragments:
        if frag.status != "active":
            continue
        by_range.setdefault(frag.range_id, []).append(frag)
    for range_id, items in by_range.items():
        items.sort(key=lambda item: item.ordinal)
        for frag in items:
            if frag.key_count != len(frag.keys):
                raise BoundError(
                    "stored key_count does not match number of keys",
                    generation_id=frag.generation_id, fragment_id=frag.fragment_id,
                    field="key_count")
            if not (lower <= frag.key_count <= upper):
                raise BoundError(
                    f"fragment key_count {frag.key_count} outside [{lower}, {upper}]",
                    generation_id=frag.generation_id, fragment_id=frag.fragment_id,
                    field="key_count")
            if frag.digest != fragment_digest(frag.keys):
                raise BoundError("fragment digest mismatch",
                                 generation_id=frag.generation_id,
                                 fragment_id=frag.fragment_id, field="digest")
        for left, right in zip(items, items[1:]):
            if left.upper != right.lower or left.upper > right.lower:
                raise BoundError(
                    "fragment coverage is not contiguous and non-overlapping",
                    generation_id=right.generation_id,
                    fragment_id=right.fragment_id, field="lower")
