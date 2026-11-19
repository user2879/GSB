from __future__ import annotations

from feature_diag.boundary import diagnose_boundaries
from feature_diag.errors import E_BOUNDARY_SENSITIVE


def test_dimension_rank_flip_marks_boundary_sensitive_with_projection_origin() -> None:
    comparison = {
        "scenario_id": "probe",
        "sample_size": 8,
        "order_limit": 1,
        "path": "random_projection",
        "threshold": 0.0,
        "per_sample": [],
    }
    reductions = {
        "probe|n8|o1|random_projection|float32": {
            "retained_dimension": 3,
            "candidate_dimensions": [
                {"dimension": 1, "metric": 0.4999998},
                {"dimension": 2, "metric": 0.5000001},
                {"dimension": 3, "metric": 0.51},
            ],
        },
        "probe|n8|o1|random_projection|float64": {
            "retained_dimension": 3,
            "candidate_dimensions": [
                {"dimension": 1, "metric": 0.50000003},
                {"dimension": 2, "metric": 0.50000000},
                {"dimension": 3, "metric": 0.51},
            ],
        },
    }
    records = diagnose_boundaries(
        designs={},
        reductions=reductions,
        comparisons={"probe|n8|o1|random_projection": comparison},
        neighborhood=0.05,
        tie_tolerance=1e-9,
    )
    sensitive = [record for record in records if record["business_code"] == E_BOUNDARY_SENSITIVE]
    assert sensitive
    assert {
        record["flip_origin"] for record in sensitive if record["left"] in {"dimension_1", "dimension_2"}
    } == {"projection_accumulation"}
