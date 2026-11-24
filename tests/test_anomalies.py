from __future__ import annotations

import copy
import json

import pytest

from tunnel_interop.errors import E_REF, E_SPEC, TunnelError
from tunnel_interop.workflow import inject_anomaly


def test_corrupt_spec_injection_reports_e_spec(tmp_path, spec_data):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec_data), encoding="utf-8")
    with pytest.raises(TunnelError) as captured:
        inject_anomaly(
            kind="corrupt-spec",
            spec_path=spec_path,
            output_dir=tmp_path / "out",
        )
    assert captured.value.code == E_SPEC
    assert captured.value.path == "$"


def test_dangling_route_injection_reports_e_ref(tmp_path, spec_data):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec_data), encoding="utf-8")
    with pytest.raises(TunnelError) as captured:
        inject_anomaly(
            kind="dangling-route",
            spec_path=spec_path,
            output_dir=tmp_path / "out",
        )
    assert captured.value.code == E_REF
    assert captured.value.path == "$.routes[0].next_hop"
