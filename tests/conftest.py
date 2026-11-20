from __future__ import annotations

import copy

import pytest

from trace_diag.spec import (
    PoliciesDoc,
    ServicesDoc,
    parse_policies_document,
    parse_services_document,
)


def field_declarations(
    *,
    duration_min_us: int = 1_000,
    duration_max_us: int = 120_000,
    key_space: int = 97,
) -> dict:
    return {
        "timestamp_ms": {"type": "int", "start": 1_700_000_000_000, "max_step_ms": 19},
        "magnitude": {"type": "int", "min": 0, "max": 8},
        "client_key": {"type": "string", "key_space": key_space},
        "duration_us": {
            "type": "int",
            "min_us": duration_min_us,
            "max_us": duration_max_us,
        },
        "trace_id": {"type": "string"},
        "sampled": {"type": "bool", "rate_bp": 2_500},
    }


@pytest.fixture
def service_doc() -> ServicesDoc:
    raw = {
        "version": 1,
        "services": [
            {
                "id": "svc-a",
                "seed": 101,
                "record_count": 96,
                "fields": field_declarations(),
                "description": "service a",
            },
            {
                "id": "svc-b",
                "seed": 202,
                "record_count": 96,
                "fields": field_declarations(
                    duration_min_us=1_100,
                    duration_max_us=121_000,
                    key_space=89,
                ),
                "description": "service b",
            },
        ],
    }
    return parse_services_document(raw)


@pytest.fixture
def policy_doc() -> PoliciesDoc:
    raw = {
        "version": 1,
        "policies": [
            {
                "id": "head-balanced",
                "head_ratio_bp": 4_500,
                "tail_duration_threshold_us": 40_000,
                "tail_ratio_bp": 1_800,
                "scale_power": 1,
                "bucket_count": 16,
                "description": "balanced",
            },
            {
                "id": "tail-first",
                "head_ratio_bp": 2_800,
                "tail_duration_threshold_us": 30_000,
                "tail_ratio_bp": 2_900,
                "scale_power": 1,
                "bucket_count": 16,
                "description": "tail",
            },
        ],
    }
    return parse_policies_document(raw)


@pytest.fixture
def service_raw() -> dict:
    return {
        "version": 1,
        "services": [
            {
                "id": "svc-a",
                "seed": 101,
                "record_count": 20,
                "fields": field_declarations(),
            }
        ],
    }


@pytest.fixture
def policy_raw() -> dict:
    return {
        "version": 1,
        "policies": [
            {
                "id": "p",
                "head_ratio_bp": 5_000,
                "tail_duration_threshold_us": 40_000,
                "tail_ratio_bp": 2_000,
                "scale_power": 1,
                "bucket_count": 8,
            }
        ],
    }


@pytest.fixture
def clone():
    return copy.deepcopy
