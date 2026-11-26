import pytest

from decchain.errors import BizCode
from decchain.config import DecodeConfig
from decchain.models import (
    CandidateSequence, Request, Step, validate_request, validate_requests)


def _step(index=1, logprobs=None):
    return Step(index=index,
                logprobs=logprobs or {1: -0.1, 2: -0.2})


def test_valid_request_passes(simple_request):
    assert validate_request(simple_request) == []


def test_duplicate_request(simple_request):
    errors = validate_requests([simple_request, simple_request])
    assert any(e.code == BizCode.DUPLICATE_REQUEST for e in errors)


def test_non_positive_sequence_id():
    req = Request("R", [CandidateSequence(0, 1, "SAMPLE", 1.0, [_step()])])
    errors = validate_request(req)
    assert any(e.code == BizCode.INVALID_SEQUENCE_ID for e in errors)


def test_step_hole_and_regression():
    seq = CandidateSequence(1, 1, "SAMPLE", 1.0,
                            [_step(1), _step(3)])
    errors = validate_request(Request("R", [seq]))
    assert any(e.code == BizCode.STEP_HOLE for e in errors)

    seq2 = CandidateSequence(1, 1, "SAMPLE", 1.0,
                             [_step(2), _step(1)])
    errors = validate_request(Request("R", [seq2]))
    assert any(e.code == BizCode.STEP_REGRESSION for e in errors)


def test_non_finite_and_positive_logprob():
    seq = CandidateSequence(
        1, 1, "SAMPLE", 1.0,
        [Step(1, {1: float("inf"), 2: float("nan"), 3: 0.5})])
    errors = validate_request(Request("R", [seq]))
    codes = {e.code for e in errors}
    assert BizCode.NON_FINITE_LOGPROB in codes
    assert BizCode.POSITIVE_LOGPROB in codes


def test_bad_temperature_topk_topp():
    seq = CandidateSequence(1, 1, "SAMPLE", 0.0, [_step()])
    errors = validate_request(Request("R", [seq]))
    assert any(e.code == BizCode.INVALID_TEMPERATURE for e in errors)

    cfg_errors = DecodeConfig(top_k=0).validate()
    assert any(e.code == BizCode.TOPK_OUT_OF_RANGE for e in cfg_errors)
    cfg_errors = DecodeConfig(top_p_mille=5000).validate()
    assert any(e.code == BizCode.TOPP_OUT_OF_RANGE for e in cfg_errors)
    cfg_errors = DecodeConfig(width=9).validate()
    assert any(e.code == BizCode.INVALID_WIDTH for e in cfg_errors)
    cfg_errors = DecodeConfig(tier="NOPE").validate()
    assert any(e.code == BizCode.INVALID_TIER for e in cfg_errors)


def test_collects_multiple_locations():
    seq = CandidateSequence(
        0, 1, "BAD", -1.0,
        [Step(1, {1: float("nan")}), Step(3, {1: 0.4})])
    errors = validate_request(Request("R", [seq]))
    locations = [e.location for e in errors]
    assert len(locations) == len(set(locations))
    assert len(errors) >= 4
