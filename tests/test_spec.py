from __future__ import annotations

import copy

import pytest

from vscan.errors import SpecError
from vscan.spec import parse_spec


def _code_and_location(exc: pytest.ExceptionInfo[SpecError]) -> tuple[str, str, str]:
    return exc.value.code, exc.value.detail["segment_id"], exc.value.detail["field"]


def test_spec_normalizes_order(spec_dict):
    first = parse_spec(copy.deepcopy(spec_dict))
    reordered = copy.deepcopy(spec_dict)
    reordered["queries"][0]["project"].reverse()
    reordered["queries"][0]["predicate"]["args"].reverse()
    reordered["segments"][0]["columns"].reverse()
    second = parse_spec(reordered)
    assert first.digest == second.digest
    assert first.canonical_dict() == second.canonical_dict()


@pytest.mark.parametrize(
    ("mutate", "segment_id", "field_name"),
    [
        (lambda data: data["segments"][0].update(rows=0), "s1", "rows"),
        (lambda data: data["segments"][0]["columns"][0].update(encoding="zip"), "s1", "encoding"),
        (lambda data: data["execution"].update(batch_rows=0), "<root>", "batch_rows"),
        (lambda data: data["execution"].update(scale=3), "<root>", "scale"),
        (lambda data: data["queries"][0].update(group_by=["missing"]), "q1", "group_by"),
        (
            lambda data: data["queries"][0]["predicate"]["args"][0].update(column="missing"),
            "s1",
            "predicate",
        ),
    ],
)
def test_spec_reports_first_invalid_field(spec_dict, mutate, segment_id, field_name):
    data = copy.deepcopy(spec_dict)
    mutate(data)
    with pytest.raises(SpecError) as caught:
        parse_spec(data)
    code, actual_segment, actual_field = _code_and_location(caught)
    assert code == "E_SPEC"
    assert actual_segment == segment_id
    assert actual_field == field_name


def test_duplicate_segment_and_column_rejected(spec_dict):
    data = copy.deepcopy(spec_dict)
    data["segments"].append(copy.deepcopy(data["segments"][0]))
    with pytest.raises(SpecError) as caught:
        parse_spec(data)
    assert caught.value.detail == {"segment_id": "s1", "field": "id"}

    data = copy.deepcopy(spec_dict)
    data["segments"][0]["columns"].append(copy.deepcopy(data["segments"][0]["columns"][0]))
    with pytest.raises(SpecError) as caught:
        parse_spec(data)
    assert caught.value.detail["field"] == "name"
    assert "duplicate column" in caught.value.message


def test_group_column_must_be_projected(spec_dict):
    data = copy.deepcopy(spec_dict)
    data["queries"][0]["project"] = ["v", "r"]
    with pytest.raises(SpecError) as caught:
        parse_spec(data)
    assert caught.value.detail["field"] == "group_by"


def test_boolean_is_not_an_integer(spec_dict):
    data = copy.deepcopy(spec_dict)
    data["segments"][0]["rows"] = True
    with pytest.raises(SpecError) as caught:
        parse_spec(data)
    assert caught.value.detail["field"] == "rows"
