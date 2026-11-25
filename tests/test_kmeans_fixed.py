from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ins_diag.config import load_config, replace


def test_empty_clusters_take_farthest_windows_with_basis(tmp_path) -> None:
    from pathlib import Path

    from ins_diag.kmeans import run_fixed_kmeans

    config = load_config(Path("examples/config.yaml"))
    config = replace(config, families=3, max_iterations=3)
    observations = {
        index: [0.0, 0.0, 0.0, 0.0, 0.0]
        for index in range(6)
    }
    with ThreadPoolExecutor(max_workers=2) as executor:
        result = run_fixed_kmeans(observations, config, executor)
    assert result["rounds"]
    actions = [
        action
        for round_payload in result["rounds"]
        for action in round_payload["empty_cluster_actions"]
    ]
    assert len(actions) >= 2
    assert {action["basis"] for action in actions} == {
        "farthest_window_from_its_assigned_centroid"
    }
    assert all(action["tie_break"] == "lowest_window_index" for action in actions)
    assert all(round_payload["quantization_residual_units"] >= 0 for round_payload in result["rounds"])
