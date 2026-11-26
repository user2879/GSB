"""End-to-end orchestration: batches, snapshots, full rebuild and replay."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .commands import Command, apply_text_commands
from .engine import EngineResult, IncrementalEngine
from .errors import FailureCollector
from .fixedpoint import QuantGear
from .models import RevisionChain, RoutingConfig, TextItem
from .report import build_report
from .scheduler import BudgetScheduler
from .scheduler import PendingItem
from .snapshot import SnapshotBundle, SnapshotStore, config_hash


class Pipeline:
    """Holds mutable corpus/chain/config state and drives incremental runs."""

    def __init__(
        self,
        config: RoutingConfig,
        chain: RevisionChain,
        texts: dict[str, TextItem],
        vocab: tuple[str, ...],
        store: SnapshotStore | None = None,
    ) -> None:
        self.config = config
        self.chain = chain
        self.texts = dict(texts)
        self.vocab = tuple(vocab)
        self.store = store
        self.version = 0
        self.ledger = store.load_ledger() if store else None
        self.previous: EngineResult | None = None
        self.failures = FailureCollector()

    @classmethod
    def resume_from_snapshot(
        cls,
        config: RoutingConfig,
        chain: RevisionChain,
        texts: dict[str, TextItem],
        vocab: tuple[str, ...],
        store: SnapshotStore,
    ) -> "Pipeline":
        """Reboot strictly from the last committed snapshot.

        No in-memory state is assumed: the ledger and pending queue are read
        back from disk, and a partial result is reconstructed so the scheduler
        drains the persisted backlog first.
        """
        pipeline = cls(config, chain, texts, vocab, store=store)
        bundle = store.load()
        if bundle is None:
            return pipeline
        pipeline.ledger = store.load_ledger()
        pipeline.version = bundle.version
        pending = [PendingItem(**item) for item in bundle.pending]
        # Faithfully reconstruct the prior result from committed products.
        from .engine import EngineResult
        from .library import MatchCandidate, MatchResult, SuppressedCandidate
        from .scoring import IntentPath, ScoreResult
        from .dedup import DedupCluster, DedupMember

        matches: dict[str, MatchResult] = {}
        for tid, payload in bundle.products.get("matches", {}).items():
            match = MatchResult(tid)
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
            matches[tid] = match

        scores: dict[str, ScoreResult] = {}
        for tid, payload in bundle.products.get("scores", {}).items():
            paths = {
                p["intent"]: IntentPath(
                    intent=p["intent"],
                    reference_raw=p["reference_raw"],
                    quantized_raw=p["quantized_raw"],
                    hit_count=p["hit_count"],
                    entry_ids=list(p["entry_ids"]),
                    threshold_raw=p["threshold_raw"],
                )
                for p in payload["paths"]
            }
            scores[tid] = ScoreResult(
                text_id=tid,
                decided_intent=payload["decided_intent"],
                paths=paths,
                boundary=payload["boundary"],
                boundary_reasons=list(payload["boundary_reasons"]),
                eligible=list(payload["eligible"]),
            )

        clusters = [
            DedupCluster(
                cluster_id=c["cluster_id"],
                representative_id=c["representative_id"],
                members=list(c["members"]),
                removed=[DedupMember(**m) for m in c["removed"]],
            )
            for c in bundle.products.get("clusters", [])
        ]
        removed = {
            tid: DedupMember(**m)
            for tid, m in bundle.products.get("removed", {}).items()
        }

        prior = EngineResult(
            texts=dict(texts),
            library={},
            matches=matches,
            scores=scores,
            clusters=clusters,
            removed=removed,
            pending=pending,
        )
        pipeline.previous = prior
        return pipeline

    def _engine(self) -> IncrementalEngine:
        return IncrementalEngine(self.config, self.ledger)

    def _scheduler(self) -> BudgetScheduler:
        return BudgetScheduler(self.config)

    def run_batch(self) -> EngineResult:
        engine = self._engine()
        scheduler = self._scheduler()
        # Carry pending items from the prior result.
        result = engine.run(
            self.texts, self.chain, scheduler, previous=self.previous
        )
        self.ledger = engine.ledger
        self.previous = result
        self.version += 1
        if self.store is not None:
            self._commit(result)
        return result

    def run_to_completion(self, max_rounds: int = 100) -> EngineResult:
        """Keep processing batches until the backlog is empty."""
        result: EngineResult | None = None
        for _ in range(max_rounds):
            result = self.run_batch()
            if not result.pending:
                break
        assert result is not None
        return result

    def _commit(self, result: EngineResult) -> None:
        bundle = SnapshotBundle(
            version=self.version,
            config_hash=config_hash(self.config.to_dict()),
            gear_key=QuantGear(
                self.config.fraction_bits, self.config.gear
            ).key,
            head_revision=self.chain.head_id,
            products={
                "scores": {
                    tid: result.scores[tid].to_dict(
                        QuantGear(self.config.fraction_bits, self.config.gear)
                    )
                    for tid in sorted(result.scores)
                },
                "matches": {
                    tid: result.matches[tid].to_dict()
                    for tid in sorted(result.matches)
                },
                "clusters": [c.to_dict() for c in result.clusters],
                "removed": {
                    tid: member.to_dict()
                    for tid, member in sorted(result.removed.items())
                },
            },
            pending=[item.to_dict() for item in result.pending],
            consumed=dict(result.consumed),
            ledger=self.ledger.to_snapshot() if self.ledger else {},
        )
        expected = self.version - 1 if self.version > 1 else None
        self.store.commit(bundle, expected_version=expected)

    def apply_commands(self, commands: list[Command]) -> None:
        self.texts = apply_text_commands(
            self.texts, commands, self.vocab, self.failures
        )
        for command in commands:
            if command.kind == "set_head_revision":
                self.chain = replace(
                    self.chain, head_id=str(command.payload["head_id"])
                )
            elif command.kind == "set_dedup_threshold":
                self.config = replace(
                    self.config,
                    dedup_threshold_permille=int(
                        command.payload["dedup_threshold_permille"]
                    ),
                )
            elif command.kind == "set_gear":
                self.config = replace(
                    self.config, gear=int(command.payload["gear"])
                )
            elif command.kind == "set_fraction_bits":
                self.config = replace(
                    self.config,
                    fraction_bits=int(command.payload["fraction_bits"]),
                )
            elif command.kind == "set_weight":
                self.chain = self._apply_weight(command)
            elif command.kind in {"upsert_trigger", "delete_trigger"}:
                self.chain = self._apply_trigger(command)

    def _apply_weight(self, command: Command) -> RevisionChain:
        # Weights live in revision ops; the engine always synthesizes from the
        # chain, so a weight edit appends a new revision on the current head.
        from .models import Revision, TriggerOp
        p = command.payload
        new_id = f"R-edit-{command.seq}"
        op = TriggerOp(
            op="upsert",
            entry_id=str(p["entry_id"]),
            term=tuple(p.get("term", [])),
            weight=int(p["weight"]),
            intent=str(p.get("intent", "")),
        )
        revision = Revision(
            revision_id=new_id,
            parent_id=self.chain.head_id,
            sequence=10_000 + command.seq,
            ops=(op,),
        )
        return replace(
            self.chain,
            head_id=new_id,
            revisions=self.chain.revisions + (revision,),
        )

    def _apply_trigger(self, command: Command) -> RevisionChain:
        from .models import Revision, TriggerOp
        p = command.payload
        new_id = f"R-edit-{command.seq}"
        op = TriggerOp(
            op="upsert" if command.kind == "upsert_trigger" else "delete",
            entry_id=str(p["entry_id"]),
            term=tuple(p.get("term", [])),
            weight=int(p.get("weight", 0)),
            intent=str(p.get("intent", "")),
        )
        revision = Revision(
            revision_id=new_id,
            parent_id=self.chain.head_id,
            sequence=10_000 + command.seq,
            ops=(op,),
        )
        return replace(
            self.chain,
            head_id=new_id,
            revisions=self.chain.revisions + (revision,),
        )

    def report(self) -> dict[str, Any]:
        gear = QuantGear(self.config.fraction_bits, self.config.gear)
        assert self.previous is not None
        return build_report(
            self.previous,
            self.failures,
            self.version,
            config_hash(self.config.to_dict()),
            gear,
            config_intents=self.config.intents + (self.config.fallback_intent,),
        )


def full_rebuild(
    config: RoutingConfig,
    chain: RevisionChain,
    texts: dict[str, TextItem],
    vocab: tuple[str, ...],
) -> EngineResult:
    """From-scratch build with an empty ledger and fresh budgets."""
    pipeline = Pipeline(config, chain, texts, vocab)
    return pipeline.run_to_completion()
