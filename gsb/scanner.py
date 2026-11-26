from __future__ import annotations

from typing import Any


def _scope_sort_key(scope: dict[str, Any]) -> tuple[int, int, int]:
    # Ascending start, descending end (encoded as negative end), ascending trigger order.
    return (scope["start"], -scope["end"], scope["trigger_order"])


def build_scopes(ticket: dict[str, Any], bundle: dict[str, Any]) -> list[dict[str, Any]]:
    tokens = ticket["tokens"]
    vocab = bundle["vocab"]
    triggers = bundle["triggers"]
    max_tokens = int(bundle["config"]["scope_max_tokens"])
    scopes: list[dict[str, Any]] = []

    for start, token in enumerate(tokens):
        trigger = triggers.get(token)
        if trigger is None:
            continue
        end = start + 1
        while end < len(tokens) and end - start < max_tokens:
            if vocab[tokens[end]].get("kind") == "punctuation":
                break
            end += 1
        scope_id = f"{ticket['ticket_id']}:S{start:03d}"
        scopes.append({
            "scope_id": scope_id,
            "ticket_id": ticket["ticket_id"],
            "trigger_id": trigger["trigger_id"],
            "trigger_term": token,
            "trigger_order": int(trigger["order"]),
            "start": start,
            "end": end,
            "inclusive_end": end - 1,
            "length": end - start,
            "token_positions": list(range(start, end)),
            "geometric_hit_ids": [],
            "attributed_hit_ids": [],
        })
    return sorted(scopes, key=lambda item: (item["start"], item["scope_id"]))


def scan_ticket(ticket: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    ticket_id = ticket["ticket_id"]
    tokens = ticket["tokens"]
    vocab = bundle["vocab"]
    scopes = build_scopes(ticket, bundle)
    hits: list[dict[str, Any]] = []

    scan_order = 0
    for position, token in enumerate(tokens):
        entry = vocab[token]
        if entry.get("kind") != "sentiment":
            continue
        hit_id = f"{ticket_id}:H{position:03d}"
        candidates = [scope for scope in scopes if scope["start"] <= position < scope["end"]]
        ordered = sorted(candidates, key=_scope_sort_key)
        effective = ordered[0]["scope_id"] if ordered else None
        hit = {
            "hit_id": hit_id,
            "ticket_id": ticket_id,
            "position": position,
            "term": token,
            "term_order": int(entry["order"]),
            "polarity": entry["polarity"],
            "base_evidence_points": float(entry["evidence_points"]),
            "scan_order": scan_order,
            "candidate_scope_ids": [scope["scope_id"] for scope in ordered],
            "effective_scope_id": effective,
        }
        hits.append(hit)
        scan_order += 1
        for scope in candidates:
            scope["geometric_hit_ids"].append(hit_id)
        if effective is not None:
            next(scope for scope in scopes if scope["scope_id"] == effective)["attributed_hit_ids"].append(hit_id)

    suppressions: list[dict[str, Any]] = []
    scope_by_id = {scope["scope_id"]: scope for scope in scopes}
    for hit in hits:
        effective_id = hit["effective_scope_id"]
        for scope_id in hit["candidate_scope_ids"]:
            if scope_id == effective_id:
                continue
            suppressed = scope_by_id[scope_id]
            selected = scope_by_id[effective_id] if effective_id else None
            suppressions.append({
                "ticket_id": ticket_id,
                "hit_id": hit["hit_id"],
                "hit_position": hit["position"],
                "suppressed_scope_id": scope_id,
                "selected_scope_id": effective_id,
                "basis": "scope_key ordering: start ASC, end DESC, trigger_order ASC",
                "suppressed_key": _scope_sort_key(suppressed),
                "selected_key": _scope_sort_key(selected) if selected else None,
            })

    for scope in scopes:
        scope["geometric_hit_ids"].sort(key=lambda hit_id: int(hit_id.rsplit("H", 1)[1]))
        scope["attributed_hit_ids"].sort(key=lambda hit_id: int(hit_id.rsplit("H", 1)[1]))

    return {
        "ticket_id": ticket_id,
        "token_count": len(tokens),
        "hits": hits,
        "scopes": scopes,
        "suppressions": suppressions,
        "scan_order_hit_ids": [hit["hit_id"] for hit in hits],
    }


def scan_corpus(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {ticket["ticket_id"]: scan_ticket(ticket, bundle) for ticket in bundle["tickets"]}
