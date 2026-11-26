import unittest

from intent_router.errors import (
    E_BAD_CONCURRENCY,
    E_DUPLICATE_TEXT_ID,
    E_REVISION_CYCLE,
    E_TOKEN_OUT_OF_VOCAB,
    E_TEXT_BUDGET_EXCEEDED,
    E_CHANNEL_QUOTA_EXCEEDED,
    FailureCollector,
)
from intent_router.loader import load_config, load_corpus, load_revision_chain
from intent_router.models import RoutingConfig, TextItem
from intent_router.scheduler import BudgetScheduler


class LoaderTest(unittest.TestCase):
    def test_corpus_unique_and_vocab_validation(self):
        failures = FailureCollector()
        payload = {
            "vocab": ["a", "b"],
            "texts": [
                {"text_id": "T1", "tokens": ["a"], "channel": "C", "gold_label": "I"},
                {"text_id": "T1", "tokens": ["b"], "channel": "C", "gold_label": "I"},
                {"text_id": "T2", "tokens": ["z"], "channel": "C", "gold_label": "I"},
            ],
        }
        corpus = load_corpus(payload, failures)
        ids = [t.text_id for t in corpus.texts]
        self.assertEqual(ids, ["T1"])
        codes = {f.code for f in failures.items}
        self.assertIn(E_DUPLICATE_TEXT_ID, codes)
        self.assertIn(E_TOKEN_OUT_OF_VOCAB, codes)

    def test_revision_cycle_detected(self):
        failures = FailureCollector()
        payload = {
            "root_id": "ROOT",
            "head_id": "R2",
            "revisions": [
                {"revision_id": "R1", "parent_id": "R2", "sequence": 1, "ops": []},
                {"revision_id": "R2", "parent_id": "R1", "sequence": 2, "ops": []},
            ],
        }
        self.assertIsNone(load_revision_chain(payload, failures))
        self.assertTrue(
            any(f.code == E_REVISION_CYCLE for f in failures.items)
        )

    def test_config_enum_and_concurrency(self):
        failures = FailureCollector()
        cfg = load_config(
            {
                "fraction_bits": 99,
                "gear": 7,
                "max_workers": 0,
                "global_match_budget": 10,
                "per_batch_recompute_limit": 10,
                "channel_quotas": {"C": 5},
                "dedup_threshold_permille": 500,
                "intents": ["IA"],
                "intent_thresholds": {"IA": 1},
                "fallback_intent": "FB",
            },
            failures,
        )
        self.assertIsNone(cfg)
        self.assertTrue(
            any(f.code == E_BAD_CONCURRENCY for f in failures.items)
        )


class SchedulerTest(unittest.TestCase):
    def _config(self, **over):
        base = dict(
            fraction_bits=8,
            gear=1,
            intents=("IA",),
            intent_thresholds={"IA": 1},
            dedup_threshold_permille=500,
            max_workers=3,
            global_match_budget=10,
            channel_quotas={"C1": 6, "C2": 100},
            per_batch_recompute_limit=100,
            fallback_intent="FB",
        )
        base.update(over)
        return RoutingConfig(**base)

    def _text(self, tid, channel="C1", length=4):
        return TextItem(tid, tuple("x" * length), channel, "IA")

    def test_global_budget_defers_excess(self):
        scheduler = BudgetScheduler(
            self._config(channel_quotas={"C1": 1000, "C2": 1000})
        )
        texts = [self._text(f"T{i}") for i in range(4)]
        plan = scheduler.reserve(texts)
        # Each text costs length+1 = 5; budget 10 -> two accepted.
        self.assertEqual(len(plan.accepted), 2)
        self.assertTrue(
            all(p.code == E_TEXT_BUDGET_EXCEEDED for p in plan.pending)
        )

    def test_channel_quota_defers_excess(self):
        cfg = self._config(global_match_budget=1000, channel_quotas={"C1": 7})
        scheduler = BudgetScheduler(cfg)
        texts = [self._text(f"T{i}", "C1") for i in range(3)]
        plan = scheduler.reserve(texts)
        self.assertEqual(len(plan.accepted), 1)
        self.assertEqual(plan.pending[0].code, E_CHANNEL_QUOTA_EXCEEDED)

    def test_backlog_priority_on_next_batch(self):
        cfg = self._config(channel_quotas={"C1": 1000, "C2": 1000})
        scheduler = BudgetScheduler(cfg)
        texts = [self._text(f"T{i}") for i in range(4)]
        first = scheduler.reserve(texts)
        self.assertEqual(len(first.pending), 2)
        # New batch with fresh budgets: backlog drains first.
        scheduler.reset_budget()
        second = scheduler.reserve([], carried_backlog=first.pending,
                                   text_lookup={t.text_id: t for t in texts})
        backlog_ids = {p.text_id for p in first.pending}
        self.assertEqual(
            {t.text_id for t in second.accepted}, backlog_ids
        )
        self.assertEqual(second.pending, [])

    def test_stable_order_channel_then_text(self):
        scheduler = BudgetScheduler(
            self._config(global_match_budget=1000, channel_quotas={"C1": 1000, "C2": 1000})
        )
        texts = [
            self._text("B2", "C2"),
            self._text("A1", "C1"),
            self._text("A0", "C1"),
        ]
        plan = scheduler.reserve(texts)
        self.assertEqual(
            [t.text_id for t in plan.accepted], ["A0", "A1", "B2"]
        )


if __name__ == "__main__":
    unittest.main()
