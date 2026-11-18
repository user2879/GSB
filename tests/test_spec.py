from __future__ import annotations

import pytest

from graygate.errors import SpecError


@pytest.mark.parametrize(
    ("mutator", "field_path"),
    [
        (
            lambda spec: spec["models"][1].__setitem__(
                "version", spec["models"][0]["version"]
            ),
            "$.models[1].version",
        ),
        (
            lambda spec: spec["models"][1].__setitem__(
                "port", spec["models"][0]["port"]
            ),
            "$.models[1].port",
        ),
        (
            lambda spec: spec["models"][0].__setitem__("host", "localhost"),
            "$.models[0].host",
        ),
        (
            lambda spec: spec["models"][0].__setitem__("model_dim", 0),
            "$.models[0].model_dim",
        ),
        (
            lambda spec: spec["models"][0].__setitem__("hidden_dim", True),
            "$.models[0].hidden_dim",
        ),
        (
            lambda spec: spec["routes"][0]["stages"][1].__setitem__(
                "gray_weight_bp", 10001
            ),
            "$.routes[0].stages[1].gray_weight_bp",
        ),
        (
            lambda spec: spec["routes"][0]["stages"][1].__setitem__(
                "start_request", 122
            ),
            "$.routes[0].stages[1].start_request",
        ),
        (
            lambda spec: spec["routes"][0]["stages"][2].__setitem__(
                "gray_weight_bp", 1000
            ),
            "$.routes[0].stages[2].gray_weight_bp",
        ),
        (
            lambda spec: spec["gateway"].__setitem__("host", "0.0.0.0"),
            "$.gateway.host",
        ),
    ],
)
def test_invalid_spec_is_rejected_with_field_path(
    mutate_raw_spec, mutator, field_path
):
    with pytest.raises(SpecError) as exc_info:
        mutate_raw_spec(mutator)
    assert exc_info.value.code == "E_SPEC"
    assert exc_info.value.field_path == field_path


def test_sample_spec_loads_with_expected_routes(sample_spec):
    assert sample_spec.host == "127.0.0.1"
    assert len(sample_spec.models) == 4
    assert len(sample_spec.routes) == 2
    assert sample_spec.stage_count == 8
    assert len({model.port for model in sample_spec.models}) == 4

