"""Self-contained deterministic demonstration of every routing capability."""

from __future__ import annotations

import json
from dataclasses import replace

from intent_router.errors import FailureCollector
from intent_router.attribution import attribute_flip
from intent_router.dedup import jaccard_permille, subword_set
from intent_router.engine import IncrementalEngine
from intent_router.fixedpoint import QuantGear
from intent_router.library import match_text, synthesize_library
from intent_router.models import (
    Revision,
    RevisionChain,
    RoutingConfig,
    TextItem,
    TriggerOp,
)
from intent_router.scheduler import BudgetScheduler
from intent_router.scoring import score_text


def config(**over) -> RoutingConfig:
    base = dict(
        fraction_bits=8,
        gear=4,
        intents=("IA", "IB"),
        intent_thresholds={"IA": 1, "IB": 1},
        dedup_threshold_permille=500,
        max_workers=4,
        global_match_budget=100,
        channel_quotas={"C1": 100},
        per_batch_recompute_limit=100,
        fallback_intent="FB",
    )
    base.update(over)
    return RoutingConfig(**base)


def chain() -> RevisionChain:
    ops = [
        TriggerOp("upsert", "SHORT", ("a", "b"), 4, "IA"),
        TriggerOp("upsert", "LONG", ("a", "b", "c", "d"), 10, "IA"),
    ]
    rev = Revision("R1", "ROOT", 1, tuple(ops))
    return RevisionChain("ROOT", "R1", (rev,))


def main() -> None:
    cfg = config(gear=2)
    gear = QuantGear(cfg.fraction_bits, cfg.gear)
    library = synthesize_library(chain(), FailureCollector())

    # 1. Longest-span match + suppression.
    text = TextItem("D1", ("a", "b", "c", "d", "x"), "C1", "IA")
    match = match_text(text, library)
    print("[match] accepted:", [c.entry_id for c in match.accepted])
    print("[match] suppressed:", [(s.entry_id, s.swallowed_by) for s in match.suppressed])

    # 2. Fixed-point scoring and half-step boundary.
    score = score_text("D1", len(text.tokens), match, cfg, gear)
    print("[score] decision:", score.decided_intent, "boundary:", score.boundary, score.boundary_reasons)
    boundary_text = TextItem("B1", ("a",), "C1", "IA")
    boundary_match = match_text(boundary_text, library)
    boundary_score = score_text("B1", 1, boundary_match, cfg, gear)
    print("[boundary] half-step flag:", boundary_score.boundary, boundary_score.boundary_reasons)

    # 3. Fixed-point Jaccard dedup.
    sim, exact = jaccard_permille(subword_set(("a", "b", "c")), subword_set(("a", "b", "z")))
    print("[dedup] jaccard permille:", sim, "exact:", exact)

    # 4. Budget scheduling with a tiny global budget.
    tight_cfg = replace(cfg, global_match_budget=6, channel_quotas={"C1": 100})
    scheduler = BudgetScheduler(tight_cfg)
    texts = [TextItem(f"T{i}", tuple("w" * 4), "C1", "IA") for i in range(3)]
    plan = scheduler.reserve(texts)
    print("[budget] accepted:", [t.text_id for t in plan.accepted], "pending:", [p.text_id for p in plan.pending])

    # 5. Flip attribution numeric evidence.
    from intent_router.scoring import IntentPath, ScoreResult
    before = ScoreResult("D1", "FB")
    before.paths = {"IA": IntentPath("IA", 1000, 1000, 1, [], 1500)}
    before.eligible = []
    after = ScoreResult("D1", "IA")
    after.paths = {"IA": IntentPath("IA", 1000, 1000, 1, [], 500)}
    after.eligible = ["IA"]
    flip = attribute_flip("D1", before, after, None, None)
    print("[flip] source:", flip.source, "evidence:", json.dumps(flip.evidence, sort_keys=True))

    # 6. Full incremental engine run (reuse ledger path).
    corpus = {t.text_id: t for t in texts + [text]}
    engine = IncrementalEngine(cfg)
    result = engine.run(corpus, chain(), BudgetScheduler(cfg))
    print("[engine] concluded:", len(result.scores), "clusters:", len(result.clusters))


if __name__ == "__main__":
    main()
