#!/usr/bin/env python3
"""End-to-end deterministic demo on local samples.

Outputs are written under ./ledger and ./artifacts:
  - full and incremental run reports
  - v1 / v2 exchange text and round-trip checks
  - budget-constrained multi-batch schedule
  - injection diagnostic report
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from decchain.io_utils import load_config, load_json_file, load_requests
from decchain.runner import Runner
from decchain.exchange import (
    render, parse, documents_equal, write_sidecar, load_sidecar,
)


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        if isinstance(obj, str):
            handle.write(obj)
        else:
            json.dump(obj, handle, sort_keys=True, indent=2,
                      ensure_ascii=False)


def main() -> int:
    config_data = load_json_file(os.path.join(ROOT, "config",
                                              "decode_config.json"))
    config, cfg_errors = load_config(config_data)
    if cfg_errors:
        for err in cfg_errors:
            print("CONFIG ERROR", err.to_dict())
        return 2

    payload = load_json_file(os.path.join(ROOT, "samples",
                                          "sample_requests.json"))
    requests, load_errors = load_requests(payload)
    if load_errors:
        for err in load_errors:
            print("LOAD ERROR", err.to_dict())
        return 2

    ledger_dir = os.path.join(ROOT, "ledger")
    runner = Runner(config, config_data["scheduler"],
                    config_data.get("exchange_version", "v2"),
                    ledger_dir=ledger_dir)

    # 1) Full rebuild from blank.
    full = runner.run_full([_clone(r) for r in requests])
    dump(os.path.join(ROOT, "artifacts", "full_run.json"),
         _strip_exchange(full))
    dump(os.path.join(ROOT, "artifacts", "exchange_v1.txt"),
         full["exchange"]["v1"])
    dump(os.path.join(ROOT, "artifacts", "exchange_v2.json"),
         full["exchange"]["v2"])

    # v1 <-> v2 round trip with sidecar.
    doc = parse(full["exchange"]["v2"], "v2")
    v1_text = render(doc, "v1")
    sidecar_path = os.path.join(ROOT, "artifacts", "v1_sidecar.json")
    write_sidecar(doc, sidecar_path)
    back_v1 = load_sidecar(parse(v1_text, "v1"), sidecar_path)
    back_v2 = parse(render(back_v1, "v2"), "v2")
    assert documents_equal(back_v2, doc), "v1/v2 round trip not equivalent"
    dump(os.path.join(ROOT, "artifacts", "roundtrip_ok.txt"),
         "v1<->v2 round trip equivalent\n")

    # 2) Incremental run driven by the edit command stream.
    inc_runner = Runner(config, config_data["scheduler"],
                        config_data.get("exchange_version", "v2"),
                        ledger_dir=os.path.join(ROOT, "ledger",
                                                "incremental"))
    commands = load_json_file(os.path.join(ROOT, "config",
                                           "commands.json"))["commands"]
    incremental = inc_runner.run_incremental(_clone_list(requests), commands)
    dump(os.path.join(ROOT, "artifacts", "incremental_run.json"),
         _strip_exchange(incremental))

    # 3) Full rebuild from the same final inputs for field-wise comparison.
    final_requests, final_config, final_exchange = _final_state(requests,
                                                                 config,
                                                                 commands)
    rebuild_runner = Runner(final_config, config_data["scheduler"],
                            final_exchange,
                            ledger_dir=os.path.join(ROOT, "ledger",
                                                    "rebuild"))
    rebuild = rebuild_runner.run_full(_clone_list(final_requests))
    dump(os.path.join(ROOT, "artifacts", "rebuild_run.json"),
         _strip_exchange(rebuild))

    consistent = _compare_conclusions(incremental, rebuild)
    equal_length = len(incremental["conclusions"]) == \
        len(rebuild["conclusions"])
    dump(os.path.join(ROOT, "artifacts", "incremental_vs_full.json"),
         {"conclusions_identical": consistent,
          "equal_conclusion_count": equal_length,
          "incremental_count": len(incremental["conclusions"]),
          "full_count": len(rebuild["conclusions"])})

    # 4) Budget-constrained multi-batch scheduling.
    tight = dict(config_data["scheduler"])
    max_task_points = max(
        len(step.logprobs) for request in final_requests
        for _seq_id, step in request.all_steps())
    tight["batch_budget"] = max_task_points + 4
    tight["per_request_quota"] = 600
    budget_runner = Runner(final_config, tight, final_exchange,
                           ledger_dir=os.path.join(ROOT, "ledger", "budget"))
    budget_result = budget_runner.run_full(_clone_list(final_requests))
    dump(os.path.join(ROOT, "artifacts", "budget_run.json"),
         _strip_exchange(budget_result))

    # 5) Injected fault diagnostic report.
    injections = load_json_file(os.path.join(ROOT, "config",
                                             "injections.json"))["injections"]
    fault_runner = Runner(config, config_data["scheduler"], "v2",
                          ledger_dir=os.path.join(ROOT, "ledger", "faults"))
    fault_result = fault_runner.run_full(_clone_list(requests),
                                         injections=injections)
    dump(os.path.join(ROOT, "artifacts", "injection_report.json"),
         _strip_exchange(fault_result))

    print(json.dumps({
        "baseline_full_conclusions": len(full["conclusions"]),
        "incremental_conclusions": len(incremental["conclusions"]),
        "final_full_rebuild_conclusions": len(rebuild["conclusions"]),
        "incremental_equals_final_full": consistent,
        "budget_remaining_todos": len(budget_result["todos"]),
        "budget_budget_per_batch": tight["batch_budget"],
        "injection_hits":
            fault_result["report"]["biz_code_counts"],
        "receipt_counts": full["report"]["receipt_counts"],
    }, indent=2, ensure_ascii=False))
    return 0


def _strip_exchange(result):
    return {k: v for k, v in result.items() if k != "exchange"}


def _clone(request):
    from decchain.models import Request, CandidateSequence, Step
    return Request(
        request_id=request.request_id,
        sequences=[CandidateSequence(
            seq_id=s.seq_id, seed_token=s.seed_token, channel=s.channel,
            temperature=s.temperature,
            steps=[Step(index=st.index, logprobs=dict(st.logprobs))
                   for st in s.steps]) for s in request.sequences])


def _clone_list(requests):
    return [_clone(r) for r in requests]


def _final_state(base_requests, base_config, commands):
    from decchain.commands import Command, CommandApplier
    cloned = _clone_list(base_requests)
    applier = CommandApplier(cloned, base_config,
                             base_config and "v2")
    applier.apply_many([Command.from_dict(c) for c in commands])
    return list(applier.requests.values()), applier.config, \
        applier.exchange_version


def _compare_conclusions(left, right):
    a = sorted(left["conclusions"], key=lambda c: (c["request_id"], c["seq_id"],
                                                   c["step_index"]))
    b = sorted(right["conclusions"], key=lambda c: (c["request_id"], c["seq_id"],
                                                    c["step_index"]))
    return a == b


if __name__ == "__main__":
    raise SystemExit(main())
