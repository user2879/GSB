from __future__ import annotations

from pathlib import Path

from audiofp.audio import decode_audio
from audiofp.config import DEFAULT_CONFIG
from audiofp.fingerprint import extract_fingerprints, match_fingerprint_sequences
from audiofp.library import Library


def test_cross_format_consistency_passes_declared_threshold(
    fixture_root: Path, manifest: dict
) -> None:
    for track in manifest["tracks"]:
        sequences = []
        for entry in track["formats"]:
            document = decode_audio(fixture_root / entry["relative_path"], DEFAULT_CONFIG.sample_rate)
            sequences.append(extract_fingerprints(document.samples, DEFAULT_CONFIG))
        for left_index in range(len(sequences)):
            for right_index in range(left_index + 1, len(sequences)):
                match = match_fingerprint_sequences(
                    sequences[left_index], sequences[right_index]
                )
                assert match.hit_rate >= DEFAULT_CONFIG.format_hit_threshold
                assert match.reason == "consistent"


def test_query_format_ranking_and_repeated_segment_alignment(
    fixture_root: Path, manifest: dict, main_index: Path
) -> None:
    query_entries = manifest["query"]["formats"]
    with Library(main_index, DEFAULT_CONFIG) as library:
        comparison = library.compare_query_rankings(
            [(entry["label"], fixture_root / entry["relative_path"]) for entry in query_entries],
            top_k=5,
        )
        assert comparison["ok"] is True
        for run in comparison["runs"]:
            top = run["results"][0]
            assert top["display_label"] == "Aurora Circuit"
            assert len(top["formats"]) == 3
            assert any(
                abs(top["alignment_start_seconds"] - start) <= 0.08
                for start in manifest["query"]["expected_alignment_starts_seconds"]
            )


def test_variant_queries_use_explicit_transform_hypotheses(
    fixture_root: Path, manifest: dict, main_index: Path
) -> None:
    with Library(main_index, DEFAULT_CONFIG) as library:
        for variant in manifest["variants"]:
            result = library.query_file(fixture_root / variant["relative_path"], top_k=3)
            assert result["results"][0]["display_label"] == "Aurora Circuit"
            assert result["variant_search"]["triggered"] is True
            assert result["results"][0]["transform"] is not None
