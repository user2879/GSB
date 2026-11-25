from __future__ import annotations

from pathlib import Path

from voice_diag.arbitration import arbitrate_confusion_pairs
from voice_diag.engine import DiagnosticEngine

from .conftest import update_config


def _run(demo: Path, name: str):
    result, _ = DiagnosticEngine(demo / "config.json", demo / name).run(mode="auto")
    assert result["failures"] == []
    return result


def test_rejection_boundary_and_overlap_candidates(demo: Path) -> None:
    result = _run(demo, "baseline")
    assert len(result["conclusions"]) == 28
    negatives = [item for item in result["conclusions"] if item["kind"] == "negative"]
    positives = [item for item in result["conclusions"] if item["kind"] == "positive"]
    assert all(item["rejected"] for item in negatives)
    assert all(item["reject_reason"] == "top_similarity_below_reject_threshold" for item in negatives)
    assert all(item["runner_up"] is not None for item in negatives)
    assert all(not item["rejected"] for item in positives)
    for conclusion in result["conclusions"]:
        assert len(conclusion["candidates"]) == min(3, len(result["representatives"]))
        ranking = [(item["similarity_permille"], item["command_word"]) for item in conclusion["candidates"]]
        assert ranking == sorted(ranking, key=lambda item: (-item[0], item[1]))
        if conclusion["boundary"]:
            assert conclusion["boundary_delta"]["within_half_step"] is True
            assert conclusion["boundary_delta"]["direction"] in {"top_ahead", "tie"}


def test_reject_threshold_change_pushes_accepted_fragments_back_to_rejection(demo: Path) -> None:
    baseline = _run(demo, "baseline")
    update_config(demo, lambda config: config["thresholds"].__setitem__("reject_permille", 1000))
    stricter = _run(demo, "strict")
    before = sum(1 for item in baseline["conclusions"] if item["rejected"])
    after = sum(1 for item in stricter["conclusions"] if item["rejected"])
    assert after > before
    assert stricter["confusion_matrix"]["totals"]["rejected_fragment_count"] == after
    assert stricter["confusion_matrix"]["consistency"]["consistent"] is True


def test_matrix_counts_match_conclusion_list_and_high_reference_is_present(demo: Path) -> None:
    result = _run(demo, "matrix")
    conclusions = result["conclusions"]
    matrix = result["confusion_matrix"]
    assert sum(cell["support_fragment_count"] for cell in matrix["cells"]) == len(conclusions)
    assert sum(cell["confusion_boundary_fragment_count"] for cell in matrix["cells"]) == sum(
        1 for item in conclusions if item["boundary"]
    )
    assert sum(cell["rejected_fragment_count"] for cell in matrix["cells"]) == sum(
        1 for item in conclusions if item["rejected"]
    )
    assert matrix["consistency"]["consistent"] is True
    boundary_cells = [
        cell
        for cell in matrix["cells"]
        if cell["support_fragment_count"] and cell["confusion_boundary_fragment_count"]
    ]
    assert boundary_cells
    assert all(cell["truncation_boundary"] for cell in boundary_cells)
    assert all(cell["high_reference"] is not None for cell in boundary_cells)
    for cell in boundary_cells:
        reference = cell["high_reference"]
        assert reference["reference_profile"] == "q16"
        assert reference["difference_direction_counts"]
        assert reference["reference_conclusion_counts"]


def test_arbitration_covers_all_candidates_and_uses_fixed_point_ties() -> None:
    representatives = {
        "A": {"template_key": "A:T1", "composite_score_permille": 900},
        "B": {"template_key": "B:T1", "composite_score_permille": 900},
    }
    candidate = {
        "pair_key": "A:T1|B:T1",
        "template_a": "A:T1",
        "template_b": "B:T1",
        "similarity_permille": 950,
        "candidate_threshold_permille": 850,
        "threshold_delta_permille": 100,
        "candidate_cache_key": "candidate-key",
    }
    results = arbitrate_confusion_pairs(
        [candidate], representatives, {"A": "A", "B": "B"}, cluster_threshold_permille=900
    )
    assert len(results) == 1
    item = results[0]
    assert item["decision"] == "unresolvable"
    assert item["tie_detected"] is True
    assert item["tie_break_basis"] == "template_pair_key_lexicographic"
    assert isinstance(item["divergence_score"], int)


def test_arbitration_merges_only_when_clusters_already_agree(demo: Path) -> None:
    result = _run(demo, "arb")
    assert len(result["confusion_arbitration"]) == len(result["confusion_candidates"])
    for item in result["confusion_arbitration"]:
        assert item["decision"] in {"retain_both", "merge_to_consensus", "unresolvable"}
        if item["decision"] == "merge_to_consensus":
            assert item["cluster_a"] == item["cluster_b"]
            assert item["consensus_command_word"] == item["cluster_a"]
