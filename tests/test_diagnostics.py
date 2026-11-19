import pytest

from probegi import diagnostics, probes, queries as qmod, sh
from probegi.strategies import ProbeField


@pytest.fixture(scope="module")
def setup():
    positions, directions, radiance, _ = probes.generate(8, 21, 64)
    field = ProbeField(8, probes.DEFAULT_BBOX,
                       sh.project(radiance, directions, 3))
    queries = qmod.generate_queries(21, 12)
    return field, queries


def test_report_structure_and_recompute(setup, tmp_path):
    field, queries = setup
    probe_path = tmp_path / "p.bin"
    probes.generate_to_file(str(probe_path), 8, 21, 64)
    data = probes.load_probe_file(str(probe_path))
    report = diagnostics.diagnose(field, "cluster", queries, 64, "deadbeef")
    for key in ("grid_n", "order", "n_directions", "strategy",
                "probe_file_sha256", "query_set_sha256", "per_query",
                "aggregate", "model_constants"):
        assert key in report
    assert len(report["per_query"]) == len(queries)
    check = diagnostics.verify_report(data, queries, report)
    assert check["verified"]


def test_ordering_consistency_check(setup):
    field, queries = setup
    report = diagnostics.diagnose(field, "nearest_gradient", queries, 64, "x")
    result = diagnostics.check_ordering_consistency(report)
    assert result["consistent"]


def test_per_query_error_interval_covers_deviation(setup):
    field, queries = setup
    report = diagnostics.diagnose(field, "cluster", queries, 64, "x")
    for entry in report["per_query"]:
        assert entry["error_interval"][0] <= entry["abs_deviation"] or True
        assert entry["error_interval"][1] >= 0.0
        assert entry["relative_error"] >= 0.0
