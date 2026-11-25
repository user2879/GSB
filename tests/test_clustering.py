from __future__ import annotations

import json
from pathlib import Path

from voice_diag.engine import DiagnosticEngine

from .conftest import update_config


def _run(demo: Path, name: str = "state"):
    result, report = DiagnosticEngine(demo / "config.json", demo / name).run(mode="auto")
    assert result["failures"] == []
    return result, report


def test_default_clusters_merge_only_controlled_confusion_pairs(demo: Path) -> None:
    result, _ = _run(demo)
    groups = [set(cluster["merged_command_words"]) for cluster in result["clusters"]["clusters"]]
    assert {"LIN", "LING"} in groups
    assert {"MA", "NA"} in groups
    assert {"SASA", "SHASHA"} in groups
    assert len(groups) == 3
    for representative in result["representatives"]:
        assert representative["candidate_scores"]
        assert representative["tie_break_basis"]


def test_cluster_threshold_change_changes_cluster_partition(demo: Path) -> None:
    baseline, _ = _run(demo, "s1")
    update_config(demo, lambda config: config["thresholds"].__setitem__("cluster_merge_permille", 999))
    changed, _ = _run(demo, "s2")
    groups = [set(cluster["merged_command_words"]) for cluster in changed["clusters"]["clusters"]]
    assert {"SASA", "SHASHA"} not in groups
    assert groups != [set(cluster["merged_command_words"]) for cluster in baseline["clusters"]["clusters"]]


def test_representative_rule_change_can_change_selected_template(demo: Path) -> None:
    engine = DiagnosticEngine(demo / "config.json", demo / "s1")
    baseline, _ = engine.run(mode="auto")
    update_config(
        demo,
        lambda config: config["registration"].__setitem__("representative_rule", "most_frames"),
    )
    changed, _ = engine.run(mode="auto")
    full, _ = DiagnosticEngine(demo / "config.json", demo / "full").run(mode="full")
    assert changed == full
    old = {item["command_word"]: item["template_key"] for item in baseline["representatives"]}
    new = {item["command_word"]: item["template_key"] for item in changed["representatives"]}
    assert old != new
    assert all(item["representative_rule"] == "most_frames" for item in changed["representatives"])


def test_single_template_word_is_degenerate_and_exempt(demo: Path) -> None:
    config = json.loads((demo / "config.json").read_text(encoding="utf-8"))
    config["registration"]["template_min"] = 1
    for word in config["words"]:
        if word["command_word"] == "NA":
            word["templates"] = word["templates"][-1:]
    (demo / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    result, _ = _run(demo)
    representative = next(item for item in result["representatives"] if item["command_word"] == "NA")
    assert representative["degenerate"] is True
    assert representative["skip_cross_word_merge"] is True
    trace = next(item for item in result["clusters"]["comparison_trace"] if item["command_word"] == "NA")
    assert trace["decision"] == "exempt_degenerate"
    assert trace["comparisons"] == []
