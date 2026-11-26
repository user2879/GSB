from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hopledger.errors import SpecError
from hopledger.sample import make_sample_configs
from hopledger.specs import load_bundle, parse_chain


def write_config(tmp_path: Path, **make_kwargs):
    return make_sample_configs(tmp_path / "config", **make_kwargs)


def test_valid_sample_loads_and_has_expected_chain(tmp_path: Path):
    bundle = write_config(tmp_path)
    assert bundle.chain.hop_ids == ["fwd1", "rev2"]
    assert bundle.chain.node_ids == ["client", "fwd1", "rev2", "origin"]
    assert len({hop.port for hop in bundle.chain.hops} | {bundle.origin.port}) == 3


@pytest.mark.parametrize(
    ("mutate", "hop_id", "field"),
    [
        (lambda c: c["hops"][0].update(hop_id="rev2"), "rev2", "hop_id"),
        (lambda c: c["hops"][0].update(port=c["hops"][1]["port"]), "fwd1", "port"),
        (lambda c: c["hops"][0].update(role="socks"), "fwd1", "role"),
        (lambda c: c["hops"][0].update(queue_ms=-1), "fwd1", "queue_ms"),
        (lambda c: c["hops"][0].update(process_ms=True), "fwd1", "process_ms"),
        (lambda c: c["hops"][0].update(scale_factor=3), "fwd1", "scale_factor"),
        (lambda c: c["hops"][0].update(scale_factor=12), "fwd1", "scale_factor"),
        (lambda c: c["hops"][0].update(config_version=0), "fwd1", "config_version"),
        (lambda c: c.update(hops=list(reversed(c["hops"]))), "rev2", "role"),
    ],
)
def test_invalid_chain_reports_first_hop_and_field(tmp_path, mutate, hop_id, field):
    bundle = write_config(tmp_path)
    chain_path = bundle.config_dir / "chain.json"
    data = json.loads(chain_path.read_text())
    mutate(data)
    chain_path.write_text(json.dumps(data))

    with pytest.raises(SpecError) as exc_info:
        load_bundle(bundle.config_dir)
    error = exc_info.value
    assert error.code == "E_SPEC"
    assert error.hop_id == hop_id
    assert error.field == field


def test_cycle_is_rejected_with_hop_and_field(tmp_path: Path):
    bundle = make_sample_configs(tmp_path / "cycle-config")
    data = json.loads((bundle.config_dir / "chain.json").read_text())
    data["hops"][0]["next_hop_id"] = "rev2"
    data["hops"][1]["next_hop_id"] = "fwd1"
    with pytest.raises(SpecError) as exc_info:
        parse_chain(data)
    assert exc_info.value.code == "E_SPEC"
    assert exc_info.value.field == "next_hop_id"
    assert exc_info.value.hop_id in {"fwd1", "rev2"}


def test_origin_id_mismatch_is_espec(tmp_path: Path):
    bundle = write_config(tmp_path)
    origin_path = bundle.config_dir / "origin.json"
    data = json.loads(origin_path.read_text())
    data["origin_id"] = "source-other"
    origin_path.write_text(json.dumps(data))
    with pytest.raises(SpecError) as exc_info:
        load_bundle(bundle.config_dir)
    assert exc_info.value.code == "E_SPEC"
    assert exc_info.value.field == "origin_id"
