"""Loading and validation layer for corpus, revisions, config and snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .errors import (
    FailureCollector,
    E_BAD_CONCURRENCY,
    E_BAD_FRACTION_BITS,
    E_BAD_GEAR,
    E_BAD_REVISION_PARENT,
    E_BAD_THRESHOLD,
    E_DUPLICATE_REVISION_ID,
    E_DUPLICATE_TEXT_ID,
    E_EMPTY_TEXT,
    E_REVISION_CYCLE,
    E_BAD_BUDGET,
    E_BAD_WEIGHT,
    E_TOKEN_OUT_OF_VOCAB,
)
from .fixedpoint import ALLOWED_FRACTION_BITS, ALLOWED_GEARS
from .models import (
    Corpus,
    Revision,
    RevisionChain,
    RoutingConfig,
    TextItem,
    TriggerOp,
)


def read_yaml(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_corpus(payload: dict, failures: FailureCollector) -> Corpus:
    vocab = tuple(str(t) for t in payload.get("vocab", []))
    vocab_set = set(vocab)
    texts: list[TextItem] = []
    seen: set[str] = set()
    for raw in payload.get("texts", []):
        text_id = str(raw["text_id"])
        if text_id in seen:
            failures.add(
                E_DUPLICATE_TEXT_ID,
                "duplicate text id",
                "text",
                text_id,
            )
            continue
        tokens = tuple(str(t) for t in raw.get("tokens", []))
        if not tokens:
            failures.add(E_EMPTY_TEXT, "empty token sequence", "text", text_id)
            continue
        out_of_vocab = sorted({t for t in tokens if t not in vocab_set})
        if out_of_vocab:
            failures.add(
                E_TOKEN_OUT_OF_VOCAB,
                "tokens outside controlled vocab",
                "text",
                text_id,
                tokens=out_of_vocab,
            )
            continue
        seen.add(text_id)
        texts.append(
            TextItem(
                text_id=text_id,
                tokens=tokens,
                channel=str(raw["channel"]),
                gold_label=str(raw["gold_label"]),
            )
        )
    return Corpus(texts=tuple(texts), vocab=vocab)


def load_revision_chain(
    payload: dict, failures: FailureCollector
) -> RevisionChain | None:
    root_id = str(payload["root_id"])
    head_id = str(payload["head_id"])
    revisions: list[Revision] = []
    seen_ids: set[str] = set()
    for raw in payload.get("revisions", []):
        rev_id = str(raw["revision_id"])
        if rev_id in seen_ids:
            failures.add(
                E_DUPLICATE_REVISION_ID,
                "duplicate revision id",
                "revision",
                rev_id,
            )
            continue
        seen_ids.add(rev_id)
        ops = tuple(
            TriggerOp(
                op=str(op["op"]),
                entry_id=str(op["entry_id"]),
                term=tuple(str(t) for t in op.get("term", [])),
                weight=int(op.get("weight", 0)),
                intent=str(op.get("intent", "")),
            )
            for op in raw.get("ops", [])
        )
        # Weight validation at load time as required.
        for op in ops:
            if op.op == "upsert" and (
                not isinstance(op.weight, int)
                or isinstance(op.weight, bool)
                or op.weight < 0
            ):
                failures.add(
                    E_BAD_WEIGHT,
                    "weight must be non-negative finite integer",
                    "trigger_op",
                    op.entry_id,
                    weight=repr(op.weight),
                )
        revisions.append(
            Revision(
                revision_id=rev_id,
                parent_id=str(raw["parent_id"]),
                sequence=int(raw["sequence"]),
                ops=ops,
            )
        )

    by_id = {rev.revision_id: rev for rev in revisions}
    # Parent existence (root's own id is the terminal parent and needs no entry).
    for rev in revisions:
        if rev.parent_id != root_id and rev.parent_id not in by_id:
            failures.add(
                E_BAD_REVISION_PARENT,
                "revision references missing parent",
                "revision",
                rev.revision_id,
                parent_id=rev.parent_id,
            )
            return None

    chain = RevisionChain(
        root_id=root_id, head_id=head_id, revisions=tuple(revisions)
    )
    # Acyclic walk from head.
    cursor = head_id
    visited: set[str] = set()
    while cursor != root_id:
        if cursor in visited:
            failures.add(
                E_REVISION_CYCLE,
                "revision chain contains a cycle",
                "revision",
                cursor,
            )
            return None
        if cursor not in by_id:
            failures.add(
                E_BAD_REVISION_PARENT,
                "head reaches unknown revision",
                "revision",
                cursor,
            )
            return None
        visited.add(cursor)
        cursor = by_id[cursor].parent_id
    return chain


def load_config(payload: dict, failures: FailureCollector) -> RoutingConfig | None:
    fraction_bits = payload.get("fraction_bits")
    if fraction_bits not in ALLOWED_FRACTION_BITS:
        failures.add(
            E_BAD_FRACTION_BITS,
            "fraction_bits outside allowed enum",
            "config",
            "fraction_bits",
            allowed=list(ALLOWED_FRACTION_BITS),
            got=fraction_bits,
        )
    gear = payload.get("gear")
    if gear not in ALLOWED_GEARS:
        failures.add(
            E_BAD_GEAR,
            "gear outside allowed enum",
            "config",
            "gear",
            allowed=list(ALLOWED_GEARS),
            got=gear,
        )
    max_workers = payload.get("max_workers")
    if not isinstance(max_workers, int) or max_workers <= 0:
        failures.add(
            E_BAD_CONCURRENCY,
            "max_workers must be positive integer",
            "config",
            "max_workers",
            got=max_workers,
        )
    for key in (
        "global_match_budget",
        "per_batch_recompute_limit",
    ):
        value = payload.get(key)
        if not isinstance(value, int) or value <= 0:
            failures.add(
                E_BAD_BUDGET,
                f"{key} must be positive integer",
                "config",
                key,
                got=value,
            )
    for key, value in payload.get("channel_quotas", {}).items():
        if not isinstance(value, int) or value <= 0:
            failures.add(
                E_BAD_BUDGET,
                "channel quota must be positive integer",
                "config",
                key,
                got=value,
            )
    threshold = payload.get("dedup_threshold_permille")
    if not isinstance(threshold, int) or not 0 <= threshold <= 1000:
        failures.add(
            E_BAD_THRESHOLD,
            "dedup threshold must be integer in [0,1000]",
            "config",
            "dedup_threshold_permille",
            got=threshold,
        )
    if failures.items and any(
        f.code in {
            E_BAD_FRACTION_BITS,
            E_BAD_GEAR,
            E_BAD_CONCURRENCY,
            E_BAD_BUDGET,
            E_BAD_THRESHOLD,
        }
        for f in failures.items
    ):
        return None
    return RoutingConfig(
        fraction_bits=int(fraction_bits),
        gear=int(gear),
        intents=tuple(str(i) for i in payload["intents"]),
        intent_thresholds={
            str(k): int(v) for k, v in payload["intent_thresholds"].items()
        },
        dedup_threshold_permille=int(threshold),
        max_workers=int(max_workers),
        global_match_budget=int(payload["global_match_budget"]),
        channel_quotas={
            str(k): int(v) for k, v in payload["channel_quotas"].items()
        },
        per_batch_recompute_limit=int(payload["per_batch_recompute_limit"]),
        fallback_intent=str(payload["fallback_intent"]),
    )


def load_inputs(
    corpus_path: str | Path,
    revisions_path: str | Path,
    config_path: str | Path,
) -> tuple[Corpus | None, RevisionChain | None, RoutingConfig | None, FailureCollector]:
    failures = FailureCollector()
    corpus = load_corpus(read_yaml(corpus_path), failures)
    chain = load_revision_chain(read_yaml(revisions_path), failures)
    config = load_config(read_yaml(config_path), failures)
    return corpus, chain, config, failures
