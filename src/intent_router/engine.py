"""Incremental deterministic intent-routing engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import dedup as dedup_mod
from .attribution import FlipEvidence, attribute_flip
from .errors import FailureCollector
from .fingerprints import (
    chain_fingerprint,
    digest,
    text_fingerprint,
)
from .fixedpoint import QuantGear
from .ledger import CacheEntry, Ledger
from .library import (
    MatchResult,
    library_entry_fingerprints,
    match_text,
    synthesize_library,
)
from .models import (
    EffectiveEntry,
    RevisionChain,
    RoutingConfig,
    TextItem,
)
from .scheduler import BudgetScheduler, PendingItem
from .scoring import ScoreResult, score_text


@dataclass
class EngineResult:
    texts: dict[str, TextItem]
    library: dict[str, EffectiveEntry]
    matches: dict[str, MatchResult]
    scores: dict[str, ScoreResult]
    clusters: list[dedup_mod.DedupCluster]
    removed: dict[str, dedup_mod.DedupMember]
    flips: list[FlipEvidence] = field(default_factory=list)
    pending: list[PendingItem] = field(default_factory=list)
    consumed: dict[str, Any] = field(default_factory=dict)
    failures: FailureCollector = field(default_factory=FailureCollector)
    ledger: Ledger = field(default_factory=Ledger)

    def final_intent(self, text_id: str) -> str:
        """Authoritative conclusion, accounting for dedup suppression."""
        member = self.removed.get(text_id)
        if member is not None:
            return self.scores[member.representative_id].decided_intent
        return self.scores[text_id].decided_intent


class IncrementalEngine:
    """Runs match -> score -> dedup with fingerprint-gated reuse.

    The engine is concurrency-agnostic: budget reservation happens in stable
    order on the main thread, and the bounded thread pool only runs pure
    per-text functions, so output is identical for any ``max_workers``.
    """

    def __init__(
        self,
        config: RoutingConfig,
        ledger: Ledger | None = None,
    ) -> None:
        self.config = config
        self.gear = QuantGear(config.fraction_bits, config.gear)
        self.ledger = ledger or Ledger()
        self.config_fp = digest({"kind": "routing_config", "config": config.to_dict()})

    # ------------------------------------------------------------------ lib
    def _library_state(
        self, chain: RevisionChain
    ) -> tuple[dict[str, EffectiveEntry], dict[str, str], str, str]:
        ordered = chain.ordered()
        chain_fp = chain_fingerprint(
            [r.revision_id for r in ordered], chain.head_id
        )
        library = synthesize_library(chain, FailureCollector())
        entry_fps = library_entry_fingerprints(library)
        lib_key = Ledger.library_key(chain_fp, entry_fps, self.gear.key)
        return library, entry_fps, chain_fp, lib_key

    def _text_fp(self, text: TextItem) -> str:
        return text_fingerprint(text.text_id, text.tokens, text.channel)

    # ------------------------------------------------------------- per-text
    def _process_text(
        self,
        text: TextItem,
        library: dict[str, EffectiveEntry],
        lib_key: str,
        chain_fp: str,
        entry_fps: dict[str, str],
    ) -> tuple[MatchResult, ScoreResult, str, str]:
        t_fp = self._text_fp(text)
        m_key = Ledger.match_key(t_fp, lib_key)
        cached_match = self.ledger.get(m_key)
        if cached_match is not None:
            match = MatchResult(text_id=text.text_id)
            payload = cached_match.payload
            from .library import MatchCandidate, SuppressedCandidate
            match.accepted = [
                MatchCandidate(
                    entry_id=c["entry_id"],
                    intent=c["intent"],
                    weight=c["weight"],
                    start=c["start"],
                    end=c["end"],
                    term=tuple(c["term"]),
                )
                for c in payload["accepted"]
            ]
            match.suppressed = [
                SuppressedCandidate(**s) for s in payload["suppressed"]
            ]
            self.ledger.record_reuse("match", m_key, text.text_id, "fingerprint_hit")
        else:
            match = match_text(text, library)
            referenced = {
                c.entry_id: entry_fps.get(c.entry_id, "")
                for c in match.accepted
            }
            self.ledger.put(
                CacheEntry(
                    key=m_key,
                    kind="match",
                    text_id=text.text_id,
                    referenced_entry_fp=referenced,
                    text_fp=t_fp,
                    chain_fp=chain_fp,
                    gear_key=self.gear.key,
                    payload=match.to_dict(),
                )
            )

        s_key = Ledger.score_key(m_key, self.gear.key, self.config_fp)
        cached_score = self.ledger.get(s_key)
        if cached_score is not None:
            score = self._restore_score(text.text_id, cached_score.payload)
            self.ledger.record_reuse("score", s_key, text.text_id, "fingerprint_hit")
        else:
            score = score_text(
                text.text_id, len(text.tokens), match, self.config, self.gear
            )
            referenced = {
                c.entry_id: entry_fps.get(c.entry_id, "")
                for c in match.accepted
            }
            self.ledger.put(
                CacheEntry(
                    key=s_key,
                    kind="score",
                    text_id=text.text_id,
                    referenced_entry_fp=referenced,
                    text_fp=t_fp,
                    chain_fp=chain_fp,
                    gear_key=self.gear.key,
                    payload=score.to_dict(self.gear),
                )
            )
        return match, score, m_key, s_key

    def _restore_score(self, text_id: str, payload: dict[str, Any]) -> ScoreResult:
        from .scoring import IntentPath
        paths: dict[str, IntentPath] = {}
        for raw_path in payload["paths"]:
            paths[raw_path["intent"]] = IntentPath(
                intent=raw_path["intent"],
                reference_raw=raw_path["reference_raw"],
                quantized_raw=raw_path["quantized_raw"],
                hit_count=raw_path["hit_count"],
                entry_ids=list(raw_path["entry_ids"]),
                threshold_raw=raw_path["threshold_raw"],
            )
        result = ScoreResult(
            text_id=text_id,
            decided_intent=payload["decided_intent"],
            paths=paths,
            boundary=payload["boundary"],
            boundary_reasons=list(payload["boundary_reasons"]),
            eligible=list(payload["eligible"]),
        )
        return result

    # --------------------------------------------------------------- run all
    def run(
        self,
        texts: dict[str, TextItem],
        chain: RevisionChain,
        scheduler: BudgetScheduler,
        previous: EngineResult | None = None,
    ) -> EngineResult:
        self.ledger.reset_trails()
        # Each batch gets fresh per-batch budgets; only the backlog carries over.
        scheduler.reset_budget()
        library, entry_fps, chain_fp, lib_key = self._library_state(chain)

        # Precise invalidation: drop cached products whose text disappeared or
        # whose referenced trigger-entry fingerprint changed.
        self._invalidate_stale(texts, library, entry_fps, chain_fp)

        # Already-concluded texts are served straight from the cache (recording
        # reuse hits); only new / backlog texts consume the match budget.
        already = set(previous.scores) if previous is not None else set()
        reusable = [texts[tid] for tid in sorted(texts) if tid in already]
        ordered = [texts[tid] for tid in sorted(texts) if tid not in already]
        carried = previous.pending if previous is not None else scheduler.backlog
        lookup = dict(texts)

        def worker(text: TextItem):
            return self._process_text(
                text, library, lib_key, chain_fp, entry_fps
            )

        # Verify cache reuse for every previously concluded live text.
        reused_results = [worker(text) for text in reusable]

        processed, plan = scheduler.run(
            ordered, worker, carried_backlog=carried, text_lookup=lookup
        )

        matches: dict[str, MatchResult] = (
            dict(previous.matches) if previous is not None else {}
        )
        scores: dict[str, ScoreResult] = (
            dict(previous.scores) if previous is not None else {}
        )
        # Drop conclusions for texts that were deleted since the last run.
        for gone_id in [tid for tid in scores if tid not in texts]:
            matches.pop(gone_id, None)
            scores.pop(gone_id, None)
        for match, score, _m_key, _s_key in (reused_results + processed):
            matches[match.text_id] = match
            scores[score.text_id] = score

        # Dedup runs over every concluded text; pending texts are excluded.
        processed_ids = sorted(scores)
        token_sets = {
            tid: dedup_mod.subword_set(texts[tid].tokens) for tid in processed_ids
        }
        clusters, removed = dedup_mod.cluster_texts(
            processed_ids,
            token_sets,
            scores,
            self.config.dedup_threshold_permille,
            self.gear,
        )

        flips = self._attribute(previous, scores, removed)

        return EngineResult(
            texts=dict(texts),
            library=library,
            matches=matches,
            scores=scores,
            clusters=clusters,
            removed=removed,
            flips=flips,
            pending=list(plan.pending),
            consumed={
                "global": plan.consumed_global,
                "channel": dict(plan.consumed_channel),
                "global_remaining": scheduler.global_remaining,
            },
            failures=FailureCollector(),
            ledger=self.ledger,
        )

    def _invalidate_stale(
        self,
        texts: dict[str, TextItem],
        library: dict[str, EffectiveEntry],
        entry_fps: dict[str, str],
        chain_fp: str,
    ) -> None:
        live_text_fps = {
            tid: self._text_fp(text) for tid, text in texts.items()
        }
        stale_keys: list[str] = []
        for key, entry in list(self.ledger.entries.items()):
            # Text deleted or replaced (fingerprint changed).
            current_fp = live_text_fps.get(entry.text_id)
            if current_fp is None:
                stale_keys.append((key, "text_deleted", entry.text_fp))
                continue
            if entry.kind in {"match", "score"} and current_fp != entry.text_fp:
                stale_keys.append((key, "text_replaced", current_fp))
                continue
            # Trigger entry referenced by this product changed.
            changed = [
                eid
                for eid, fp in entry.referenced_entry_fp.items()
                if entry_fps.get(eid) != fp
            ]
            if changed:
                stale_keys.append(
                    (key, "trigger_changed", digest(sorted(changed)))
                )
                continue
            if entry.chain_fp != chain_fp:
                stale_keys.append((key, "chain_changed", chain_fp))
            # Quantization gear / fraction width or routing config changed.
            if entry.gear_key != self.gear.key:
                stale_keys.append((key, "gear_changed", self.gear.key))
        for key, reason, fp in stale_keys:
            entry = self.ledger.entries.get(key)
            text_id = entry.text_id if entry else ""
            self.ledger.invalidate(
                entry.kind if entry else "unknown",
                key,
                text_id,
                reason,
                fp,
            )

    def _attribute(
        self,
        previous: EngineResult | None,
        scores: dict[str, ScoreResult],
        removed: dict[str, dedup_mod.DedupMember],
    ) -> list[FlipEvidence]:
        if previous is None:
            return []
        flips: list[FlipEvidence] = []
        for text_id in sorted(scores):
            prior = previous.scores.get(text_id)
            if prior is None:
                continue
            flips.append(
                attribute_flip(
                    text_id,
                    prior,
                    scores[text_id],
                    previous.removed.get(text_id),
                    removed.get(text_id),
                )
            )
        return flips
