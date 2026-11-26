from __future__ import annotations

import unittest
from dataclasses import replace

from intentlab.diagnostics import diagnose_query
from intentlab.errors import IssueCollector
from intentlab.fingerprints import digest
from intentlab.matching import match_query
from intentlab.merging import QueryScorePair, build_segment
from intentlab.models import (
    IntentScore,
    Query,
    RuleView,
    ScoreResult,
    Slot,
    TriggerEntry,
)
from intentlab.rules import compose_effective_rules
from intentlab.scoring import high_width_reference, score_query_fixed, score_query_float64
from intentlab.validation import load_inputs, parse_config

from tests.helpers import INPUT, read_json


def entry(
    trigger_id: str,
    terms: tuple[str, ...],
    intent: str,
    weight: int,
    order: int,
    revision: str = "r00",
    pattern: tuple[str, ...] = (),
) -> TriggerEntry:
    return TriggerEntry(
        trigger_id=trigger_id,
        terms=terms,
        intent_id=intent,
        weight=weight,
        slot_pattern=pattern,
        source_revision=revision,
        entry_order=order,
    )


def view(entries: list[TriggerEntry]) -> RuleView:
    return RuleView(
        entries=tuple(entries),
        active_revision_ids=("test",),
        lineage_revision_ids=("test",),
        chain_version=digest([item.to_dict() for item in entries]),
    )


def query(
    query_id: str,
    terms: tuple[str, ...],
    *,
    intent: str = "weather",
    candidates: tuple[str, ...] = ("weather",),
    turn: int = 0,
    slots: tuple[Slot, ...] = (),
) -> Query:
    return Query(
        query_id=query_id,
        terms=terms,
        session_id="session",
        turn=turn,
        source="source-a",
        ground_truth_intent=intent,
        candidates=candidates,
        slots=slots,
        text=" ".join(terms),
    )


def fake_score(query_id: str, intent: str, key: str) -> ScoreResult:
    return ScoreResult(
        query_id=query_id,
        predicted_intent=intent,
        scores=(IntentScore(intent, 1, 1, 1, True, False, 0),),
        contributions=(),
        referenced_trigger_ids=(),
        boundary=False,
        margin=0,
        fraction_bits=12,
        quantum=16,
        cache_key=key,
    )


class RuleCompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loaded, issues = load_inputs(
            config_path=INPUT / "config.json",
            vocabulary_path=INPUT / "vocabulary.json",
            queries_path=INPUT / "queries.json",
            rules_path=INPUT / "rules.json",
            edits_path=None,
        )
        self.assertEqual(len(issues), 0)

    def by_id(self, active: tuple[str, ...]) -> dict[str, TriggerEntry]:
        issues = IssueCollector()
        effective = compose_effective_rules(self.loaded.revisions, active, issues)
        self.assertEqual(len(issues), 0)
        return {item.trigger_id: item for item in effective.entries}

    def test_effective_view_records_latest_revision_and_removal(self) -> None:
        entries = self.by_id(("r06",))
        self.assertNotIn("t_music_play", entries)
        self.assertNotIn("t_nav_route", entries)
        self.assertNotIn("t_hotel_price", entries)
        self.assertEqual(entries["t_music_artist"].source_revision, "r04")
        self.assertEqual(entries["t_news_stock"].weight, 1200)
        self.assertEqual(entries["t_weather_city"].source_revision, "r01")
        city_revisions = [item["revision_id"] for item in entries["t_weather_city"].provenance]
        self.assertEqual(city_revisions, ["r00", "r01"])

    def test_rollback_invalidates_descendant_additions_but_restores_deleted_trigger(self) -> None:
        entries = self.by_id(("r03",))
        self.assertNotIn("t_music_artist", entries)
        self.assertIn("t_music_play", entries)
        self.assertNotIn("t_nav_route", entries)
        self.assertEqual(entries["t_weather_tomorrow"].weight, 900)

    def test_sibling_nonconflicting_trigger_remains_while_rollback_branch_is_active(self) -> None:
        entries = self.by_id(("r03", "r05"))
        self.assertIn("t_hotel_price", entries)
        self.assertNotIn("t_music_artist", entries)
        self.assertIn("t_music_play", entries)
        self.assertEqual(entries["t_weather_city"].source_revision, "r05")
        self.assertEqual(entries["t_weather_city"].weight, 1000)

    def test_two_active_sibling_tips_merge_without_losing_nonconflicting_entries(self) -> None:
        entries = self.by_id(("r06", "r05"))
        self.assertIn("t_hotel_price", entries)
        self.assertIn("t_music_artist", entries)
        self.assertNotIn("t_music_play", entries)
        self.assertEqual(entries["t_weather_city"].source_revision, "r05")


class MatchingAndScoringTests(unittest.TestCase):
    def config(self, **overrides):
        raw = read_json(INPUT / "config.json")
        issues = IssueCollector()
        config = parse_config(raw, issues)
        return replace(config, **overrides) if overrides else config

    def test_longest_match_then_lexical_then_start_order_with_suppression_ledger(self) -> None:
        entries = [
            entry("t_short", ("b", "c"), "weather", 800, 0),
            entry("t_long", ("b", "c", "d"), "weather", 900, 1),
            entry("t_ab", ("a", "b"), "weather", 700, 2),
        ]
        result = match_query(query("q-long", ("a", "b", "c", "d")), view(entries))
        self.assertEqual([(span.trigger_id, span.start, span.end) for span in result.selected], [("t_long", 1, 4)])
        self.assertEqual({item["swallowed_by"] for item in result.suppressed}, {"t_long"})
        self.assertEqual({item["trigger_id"] for item in result.suppressed}, {"t_short", "t_ab"})

        lexical = match_query(
            query("q-lex", ("a", "b", "c")),
            view([entry("t_bc", ("b", "c"), "weather", 100, 0), entry("t_ab", ("a", "b"), "weather", 100, 1)]),
        )
        self.assertEqual([span.trigger_id for span in lexical.selected], ["t_ab"])

        repeated = match_query(
            query("q-repeat", ("a", "a", "a")),
            view([entry("t_aa", ("a", "a"), "weather", 100, 0)]),
        )
        self.assertEqual([(span.start, span.end) for span in repeated.selected], [(0, 2)])
        self.assertEqual(repeated.suppressed[0]["start"], 1)

    def test_fixed_score_uses_hit_count_length_reduction_and_boundary_tie(self) -> None:
        config = self.config(fraction_bits=12, quantum=16, thresholds={"weather": 500})
        entries = [entry("t_city", ("北京", "天气"), "weather", 1000, 0)]
        query_obj = query("q-score", ("北京", "天气", "北京", "天气"))
        match = match_query(query_obj, view(entries))
        self.assertEqual(len(match.selected), 2)
        result = score_query_fixed(query_obj, match, entries, config)
        self.assertEqual(result.scores[0].raw_score, 2_048_000)
        self.assertEqual(result.scores[0].threshold, 2_048_000)
        self.assertTrue(result.boundary)
        self.assertEqual(result.predicted_intent, "weather")
        self.assertEqual(result.contributions[0].hit_count, 2)

    def test_quantization_flip_is_attributed_to_rounding_with_high_width_reference(self) -> None:
        raw_config = read_json(INPUT / "config.json")
        raw_config["fixed_point"] = {"fraction_bits": 8, "quantum": 64}
        raw_config["thresholds"]["intents"]["weather"] = 89
        raw_config["diagnostics"]["reference_extra_bits"] = 4
        config = parse_config(raw_config, IssueCollector())
        entries = [entry("t_city", ("北京", "天气"), "weather", 800, 0)]
        terms = ("北京", "天气", "f1", "f2", "f3", "f4", "f5", "f6", "f7")
        query_obj = query("q-flip", terms)
        match = match_query(query_obj, view(entries))
        fixed = score_query_fixed(query_obj, match, entries, config)
        float_reference = score_query_float64(query_obj, match, entries, config)
        high = high_width_reference(query_obj, match, entries, config)
        self.assertEqual(fixed.scores[0].raw_score, 22755)
        self.assertEqual(fixed.scores[0].score, 22784)
        self.assertEqual(fixed.scores[0].threshold, 22784)
        self.assertEqual(fixed.predicted_intent, "weather")
        self.assertEqual(float_reference.predicted_intent, "unknown")
        self.assertEqual(high.predicted_intent, "unknown")
        diagnostics = diagnose_query(
            query=query_obj,
            match=match,
            fixed=fixed,
            float_reference=float_reference,
            high_width=high,
            entries=entries,
            config=config,
            fixed_segment=None,
            reference_segment=None,
        )
        self.assertTrue(diagnostics["flip"]["causes"]["quantization_rounding"])
        self.assertEqual(diagnostics["flip"]["primary_reason"], "quantization_rounding")
        self.assertEqual(diagnostics["flip"]["high_width_bits"], 12)
        self.assertEqual(diagnostics["flip"]["high_width_intent"], "unknown")
        self.assertEqual(diagnostics["first_exceeding_stage"], "intent_score")

    def test_slot_alignment_maps_offsets_and_records_overlap_conflict(self) -> None:
        config = self.config()
        first = query(
            "q1",
            ("附近", "咖啡", "店"),
            intent="nearby",
            candidates=("nearby",),
            turn=0,
            slots=(Slot("poi", 0, 3), Slot("wide", 0, 2)),
        )
        second = query(
            "q2",
            ("附近", "咖啡", "店"),
            intent="nearby",
            candidates=("nearby",),
            turn=1,
            slots=(Slot("poi", 1, 3),),
        )
        group = (
            QueryScorePair(first, fake_score("q1", "nearby", "k1")),
            QueryScorePair(second, fake_score("q2", "nearby", "k2")),
        )
        segment = build_segment(group, config)
        self.assertEqual(segment.query_ids, ("q1", "q2"))
        self.assertEqual(len(segment.conflicts), 1)
        self.assertEqual(segment.terms[:5], ("附近", "咖啡", "店", "<gap>", "附近"))
        accepted = [slot for slot in segment.aligned_slots if slot.status == "accepted"]
        conflicted = [slot for slot in segment.aligned_slots if slot.status == "conflicted"]
        self.assertEqual([(slot.name, slot.start, slot.end) for slot in accepted], [("poi", 0, 3), ("poi", 5, 7)])
        self.assertEqual(conflicted[0].name, "wide")
        self.assertEqual(conflicted[0].conflict_with, "q1:poi")


if __name__ == "__main__":
    unittest.main()
