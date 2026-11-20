import pytest

from hdr_sdr_pipeline.config import STATE_SCHEMA_VERSION, ProcessingState
from hdr_sdr_pipeline.detector import assert_no_miss
from hdr_sdr_pipeline.errors import (
    ErrorCode, ClosureDivergeError, DetectMissError, FrameFormatError,
    HashMismatchError, InjectBoundsError, SnapshotStaleError,
    StateVersionError, TonemapDomainError,
)

EXPECTED_CODES = {
    "E_FRAME_FORMAT", "E_STATE_VERSION", "E_HASH_MISMATCH",
    "E_CLOSURE_DIVERGE", "E_INJECT_BOUNDS", "E_DETECT_MISS",
    "E_SNAPSHOT_STALE", "E_TONEMAP_DOMAIN",
}


def test_all_business_codes_defined():
    assert set(ErrorCode.ALL) == EXPECTED_CODES


def test_exception_codes_match_registry():
    assert FrameFormatError("x").code == ErrorCode.FRAME_FORMAT
    assert StateVersionError("x").code == ErrorCode.STATE_VERSION
    assert HashMismatchError("x").code == ErrorCode.HASH_MISMATCH
    assert ClosureDivergeError("x").code == ErrorCode.CLOSURE_DIVERGE
    assert InjectBoundsError("x").code == ErrorCode.INJECT_BOUNDS
    assert DetectMissError("x").code == ErrorCode.DETECT_MISS
    assert SnapshotStaleError("x").code == ErrorCode.SNAPSHOT_STALE
    assert TonemapDomainError("x").code == ErrorCode.TONEMAP_DOMAIN


def test_unknown_state_version_rejected():
    state = ProcessingState(schema_version=999)
    with pytest.raises(StateVersionError):
        state.validate()


def test_unknown_state_field_rejected():
    data = ProcessingState().to_dict()
    data["mystery_param"] = 1
    with pytest.raises(StateVersionError):
        ProcessingState.from_dict(data)


def test_assert_no_miss_raises_detect_miss():
    report = {"missed": [{"item_id": "x"}], "detected": [],
              "false_positive": []}
    with pytest.raises(DetectMissError):
        assert_no_miss(report)
