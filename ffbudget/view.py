"""Stage 3: effective-time availability view with stale fallback.

At a ranking timestamp a slot group's usable revision is the highest-ranked
authoritative revision whose effective_ms <= rank_ms. When none is effective
yet the view falls back to the most recent revision inside the retained
history and reports an integer staleness. Revisions older than the configured
staleness cap make the slot unavailable and a fixed default vector is used.
"""
from __future__ import annotations

from .fingerprint import cache_key, content_fingerprint
from .fixedpoint import WIDTH_TIER
from .models import AuthorityRecord, Receipt, SlotResolution, SlotSpec

STATE_FRESH = "fresh"
STATE_STALE = "stale"
STATE_FUTURE_FALLBACK = "future_fallback"
STATE_UNAVAILABLE = "unavailable"


def staleness_fingerprint(config: dict) -> str:
    return content_fingerprint([config["staleness"], WIDTH_TIER])


def build_history_index(authorities: dict[str, AuthorityRecord],
                        receipts: list[Receipt]) -> dict[str, list[Receipt]]:
    retained = {group: set(record.history)
                for group, record in authorities.items()}
    index: dict[str, list[Receipt]] = {}
    for receipt in receipts:
        if receipt.digest in retained.get(receipt.group_key, set()):
            index.setdefault(receipt.group_key, []).append(receipt)
    for group in index:
        index[group].sort(key=lambda r: (-r.rev_seq, -r.effective_ms,
                                         -r.batch_seq, r.digest))
    return index


def resolve_slot(rank_ms: int, slot: SlotSpec,
                 authority: AuthorityRecord | None,
                 history_receipts: list[Receipt]) -> SlotResolution:
    if authority is None or not history_receipts:
        return _default_resolution(rank_ms, slot,
                                   authority.group_key if authority else None)

    effective = [r for r in history_receipts if r.effective_ms <= rank_ms]
    if effective:
        chosen = max(effective, key=lambda r: (r.effective_ms, r.rev_seq,
                                               r.batch_seq))
        staleness_ms = rank_ms - chosen.effective_ms
        return SlotResolution(
            slot_key=slot.key, group_key=authority.group_key,
            item_id=chosen.item_id, rank_ms=rank_ms,
            used_digest=chosen.digest, used_rev_seq=chosen.rev_seq,
            used_batch_seq=chosen.batch_seq,
            effective_ms=chosen.effective_ms, staleness_ms=staleness_ms,
            state=STATE_FRESH if staleness_ms == 0 else STATE_STALE,
            vector=chosen.vector,
            vector_source="backfill:" + chosen.digest)

    chosen = min(history_receipts,
                 key=lambda r: (r.effective_ms, -r.rev_seq, -r.batch_seq))
    staleness_ms = rank_ms - chosen.effective_ms
    return SlotResolution(
        slot_key=slot.key, group_key=authority.group_key,
        item_id=chosen.item_id, rank_ms=rank_ms,
        used_digest=chosen.digest, used_rev_seq=chosen.rev_seq,
        used_batch_seq=chosen.batch_seq,
        effective_ms=chosen.effective_ms, staleness_ms=staleness_ms,
        state=STATE_FUTURE_FALLBACK,
        vector=chosen.vector, vector_source="backfill:" + chosen.digest)


def apply_staleness_cap(resolution: SlotResolution, slot: SlotSpec,
                        max_staleness: int) -> SlotResolution:
    if resolution.state == STATE_UNAVAILABLE:
        return resolution
    if resolution.staleness_ms is not None and \
            resolution.staleness_ms > max_staleness:
        return SlotResolution(
            slot_key=resolution.slot_key, group_key=resolution.group_key,
            item_id=resolution.item_id, rank_ms=resolution.rank_ms,
            used_digest=resolution.used_digest,
            used_rev_seq=resolution.used_rev_seq,
            used_batch_seq=resolution.used_batch_seq,
            effective_ms=resolution.effective_ms,
            staleness_ms=resolution.staleness_ms,
            state=STATE_UNAVAILABLE, vector=slot.default_vector,
            vector_source="default:" + slot.default_vector_source)
    return resolution


def _default_resolution(rank_ms, slot: SlotSpec,
                        group_key: str | None) -> SlotResolution:
    group = group_key if group_key is not None else slot.key
    return SlotResolution(
        slot_key=slot.key, group_key=group, item_id=None, rank_ms=rank_ms,
        used_digest=None, used_rev_seq=None, used_batch_seq=None,
        effective_ms=None, staleness_ms=None, state=STATE_UNAVAILABLE,
        vector=slot.default_vector,
        vector_source="default:" + slot.default_vector_source)


def view_entry(resolution: SlotResolution, staleness_fp: str) -> dict:
    content_fp = content_fingerprint(list(resolution.vector))
    return {
        "cache_key": cache_key(["view", resolution.group_key,
                                str(resolution.rank_ms), content_fp,
                                staleness_fp, WIDTH_TIER]),
        "content_fingerprint": content_fp,
        "staleness_fingerprint": staleness_fp,
        "width_tier": WIDTH_TIER,
        "state": resolution.state,
        "staleness_ms": resolution.staleness_ms,
        "used_digest": resolution.used_digest,
        "vector_source": resolution.vector_source,
    }


def build_view(config: dict, slots: dict, authorities: dict,
               requests: list[dict],
               history_index: dict[str, list[Receipt]]):
    staleness_fp = staleness_fingerprint(config)
    max_staleness = config["staleness"]["max_staleness_ms"]
    resolutions: dict[tuple, SlotResolution] = {}
    view_cache: dict[str, dict] = {}
    for request in sorted(requests, key=lambda r: r["request_id"]):
        rank_ms = request["rank_ms"]
        for slot_key in config["_slot_order"]:
            slot = slots[slot_key]
            if slot.layer == "request":
                group_keys = [(None, slot_key)]
            else:
                group_keys = [(item_id, f"{slot_key}#{item_id}")
                              for item_id in sorted(request["candidates"])]
            for item_id, group_key in group_keys:
                authority = authorities.get(group_key)
                history = history_index.get(group_key, [])
                resolution = resolve_slot(rank_ms, slot, authority, history)
                resolution = apply_staleness_cap(resolution, slot,
                                                  max_staleness)
                key = cache_key(["view", request["request_id"], group_key,
                                 WIDTH_TIER])
                view_cache[key] = view_entry(resolution, staleness_fp)
                resolutions[(request["request_id"], item_id,
                             slot_key)] = resolution
    return resolutions, view_cache, staleness_fp
