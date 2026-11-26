import copy
import os

from decchain.commands import Command, CommandApplier
from decchain.config import DecodeConfig
from decchain.injection import load_injections, apply_injections
from decchain.io_utils import load_config
from decchain.ledger import Ledger, Snapshot
from decchain.errors import BizCode, BizError
from decchain.runner import Runner
from decchain.attribution import attribute_step


SCHED = {"batch_budget": 60, "per_request_quota": 600,
         "step_recompute_cap": 5, "max_workers": 1}


def _runner(tmp_path, config, workers=1):
    sched = dict(SCHED)
    sched["max_workers"] = workers
    return Runner(config, sched, "v2",
                  ledger_dir=str(tmp_path / f"ledger_{workers}"))


def _final_state(requests, config, commands):
    cloned = copy.deepcopy(requests)
    applier = CommandApplier(cloned, config, "v2")
    applier.apply_many([Command.from_dict(c) for c in commands])
    return list(applier.requests.values()), applier.config, \
        applier.exchange_version


def test_full_run_is_deterministic(sample_requests, config, tmp_path):
    a = _runner(tmp_path, config).run_full(copy.deepcopy(sample_requests))
    b = _runner(tmp_path, config).run_full(copy.deepcopy(sample_requests))
    assert a["conclusions"] == b["conclusions"]
    assert a["boundaries"] == b["boundaries"]


def test_incremental_matches_full_rebuild(sample_requests, config, tmp_path):
    commands = [
        {"command": "append_step", "request_id": "R001", "seq_id": 1,
         "step_index": 4,
         "logprobs": {"100": -0.4, "101": -1.1, "102": -2.2,
                      "103": -3.0, "104": -4.4, "105": -5.0}},
        {"command": "replace_step", "request_id": "R001", "seq_id": 1,
         "step_index": 2,
         "logprobs": {"100": -0.1, "101": -0.7, "102": -1.9,
                      "103": -2.6, "104": -3.8, "105": -4.5}},
        {"command": "set_top_p", "request_id": "R001", "top_p_mille": 800},
        {"command": "set_quantization", "request_id": "R001",
         "width": 5, "tier": "FLOOR"},
    ]
    inc = _runner(tmp_path, config)
    incremental = inc.run_incremental(copy.deepcopy(sample_requests),
                                      commands)
    final_req, final_cfg, exchange = _final_state(sample_requests, config,
                                                  commands)
    full = _runner(tmp_path, final_cfg).run_full(copy.deepcopy(final_req))

    keyf = lambda c: (c["request_id"], c["seq_id"], c["step_index"])
    assert sorted(incremental["conclusions"], key=keyf) == \
           sorted(full["conclusions"], key=keyf)
    assert sorted(incremental["boundaries"],
                  key=lambda b: (b["request_id"], b["seq_id"],
                                 b["step_index"], b["token"])) == \
           sorted(full["boundaries"],
                  key=lambda b: (b["request_id"], b["seq_id"],
                                 b["step_index"], b["token"]))


def test_concurrent_matches_serial(sample_requests, config, tmp_path):
    serial = _runner(tmp_path, config, workers=1).run_full(
        copy.deepcopy(sample_requests))
    parallel = _runner(tmp_path, config, workers=4).run_full(
        copy.deepcopy(sample_requests))
    keyf = lambda c: (c["request_id"], c["seq_id"], c["step_index"])
    assert sorted(serial["conclusions"], key=keyf) == \
           sorted(parallel["conclusions"], key=keyf)


def test_injection_isolation_and_markers(sample_requests, config, tmp_path):
    injections = [
        {"injection_id": "INJ-001", "kind": "NON_FINITE_PROB",
         "request_id": "R002", "seq_id": 1, "step_index": 1},
        {"injection_id": "INJ-003", "kind": "STEP_REGRESSION",
         "request_id": "R003", "seq_id": 1, "step_index": 2},
    ]
    runner = _runner(tmp_path, config)
    result = runner.run_full(copy.deepcopy(sample_requests),
                             injections=injections)
    markers = {h["injection_id"] for h in result["report"]["injection_hits"]}
    assert {"INJ-001", "INJ-003"} <= markers
    # The healthy request still produces conclusions.
    request_ids = {c["request_id"] for c in result["conclusions"]}
    assert "R001" in request_ids


def test_attribution_rows_cover_four_stages(simple_request, config):
    request = simple_request
    seq = request.sequences[0]
    row = attribute_step(request, seq, seq.steps[0], config)
    stages = [s.stage for s in row.stages]
    assert stages == ["temperature_scaling", "truncation_accumulation",
                      "sampling_decision", "version_conversion"]
    assert row.first_breach_stage in stages + [None]


def test_ledger_atomic_commit_and_recovery(config, tmp_path):
    ledger = Ledger(str(tmp_path / "ledger"))
    snapshot = Snapshot(
        version=1, config_fingerprint=config.fingerprint(),
        config={"width": config.width}, exchange_version="v2",
        conclusions=[{"x": 1}], boundaries=[], cache_events=[],
        failures=[], todos=[])
    digest = ledger.commit_snapshot(snapshot)
    assert not ledger.discard_temp_files()
    loaded = ledger.load_snapshot(1, config.fingerprint())
    assert loaded is not None and loaded.conclusions == [{"x": 1}]

    # Stale version / config hash must be rejected with business codes.
    try:
        ledger.load_snapshot(2, config.fingerprint())
        assert False
    except BizError as exc:
        assert exc.code == BizCode.STALE_SNAPSHOT_VERSION
    other_fp = "f" * 64
    try:
        ledger.load_snapshot(1, other_fp)
        assert False
    except BizError as exc:
        assert exc.code == BizCode.CONFIG_HASH_MISMATCH
    assert digest
