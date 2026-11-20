"""End-to-end verification harness; writes reports/verification.json."""
import json
import os
import platform
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "..", "data")
REPORTS = os.path.join(ROOT, "..", "reports")
sys.path.insert(0, ROOT)

from qcg.config import Config
from qcg.converter import downgrade_v2_to_v1, upgrade_v1_to_v2, write_sidecar
from qcg.editor import edit_bitwidth
from qcg.engine import Engine
from qcg.quant_pack import read_pack
from qcg.v1_codec import parse_v1, dump_v1
from qcg.v2_codec import parse_v2
from qcg.validator import validate


def main() -> None:
    result = {"python": sys.version, "platform": platform.platform(),
              "cwd": os.getcwd(), "checks": []}

    def check(name, ok, detail):
        result["checks"].append({"name": name, "ok": bool(ok),
                                 "detail": detail})

    v2_text = open(os.path.join(DATA, "graphs_v2",
                                "sample_graph.v2.json"), encoding="utf-8").read()
    graph = parse_v2(v2_text)
    validate(graph)
    v1_graph, sidecar, losses = downgrade_v2_to_v1(graph)
    sidecar_path = os.path.join(DATA, "sidecar", "verify.sidecar.json")
    write_sidecar(sidecar_path, sidecar)
    restored, restore_losses = upgrade_v1_to_v2(
        parse_v1(dump_v1(v1_graph)), sidecar)
    same = [a.__dict__ == b.__dict__
            for a, b in zip(graph.topo_order(), restored.topo_order())]
    check("version_roundtrip", all(same),
          {"downgrade_losses": len(losses),
           "restore_losses": len(restore_losses)})

    blocks = read_pack(os.path.join(DATA, "quant_pack",
                                    "sample_pack.le.bin"))
    pack_note = {"blocks": sorted(blocks),
                 "dtypes": {k: str(v.dtype) for k, v in blocks.items()}}
    check("quant_pack_le", all(v.dtype.byteorder in ("<", "|")
                               for v in blocks.values()), pack_note)

    snap_full = os.path.join(DATA, "snapshots", "verify_full")
    shutil.rmtree(snap_full, ignore_errors=True)
    cfg = Config(max_workers=1)
    full_report = Engine(cfg, snap_full).run(graph.clone(),
                                             os.path.join(DATA, "quant_pack",
                                                          "sample_pack.le.bin"))
    edited = graph.clone()
    edit_bitwidth(edited, "n0002", 16)
    inc_report = Engine(cfg, snap_full).run(
        edited, os.path.join(DATA, "quant_pack", "sample_pack.le.bin"))
    snap_parallel = os.path.join(DATA, "snapshots", "verify_parallel")
    shutil.rmtree(snap_parallel, ignore_errors=True)
    par_report = Engine(Config(max_workers=4), snap_parallel).run(
        graph.clone(),
        os.path.join(DATA, "quant_pack", "sample_pack.le.bin"))
    check("incremental_invalidation",
          inc_report["reused"] == ["n0000", "n0001"] and
          inc_report["recomputed"] == ["n0002", "n0003", "n0004"],
          {"reused": inc_report["reused"],
           "recomputed": inc_report["recomputed"]})
    check("serial_equals_parallel",
          full_report["sort_keys"] == par_report["sort_keys"] and
          full_report["budget_events"] == par_report["budget_events"],
          {"sort_keys": full_report["sort_keys"]})

    tight_root = os.path.join(DATA, "snapshots", "verify_tight")
    shutil.rmtree(tight_root, ignore_errors=True)
    tight = Config(max_workers=2, global_mac_budget=600,
                   batch_recompute_budget=600)
    tight_report = Engine(tight, tight_root).run(
        graph.clone(),
        os.path.join(DATA, "quant_pack", "sample_pack.le.bin"))
    check("budget_pending_queue",
          any(p["code"] in ("E_GLOBAL_BUDGET_PENDING",
                            "E_RECOMPUTE_PENDING")
              for p in tight_report["pending"]),
          {"pending": tight_report["pending"],
           "budget_events": len(tight_report["budget_events"])})

    cache_root = os.path.join(DATA, "snapshots", "verify_cache")
    shutil.rmtree(cache_root, ignore_errors=True)
    cache_cfg = Config(max_workers=1, cache_resident_bytes=1400)
    cache_report = Engine(cache_cfg, cache_root).run(
        graph.clone(),
        os.path.join(DATA, "quant_pack", "sample_pack.le.bin"))
    check("lru_eviction", len(cache_report["evictions"]) > 0,
          {"eviction_events": cache_report["evictions"][:2]})

    os.makedirs(REPORTS, exist_ok=True)
    out = os.path.join(REPORTS, "verification.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    assert all(c["ok"] for c in result["checks"]), result


if __name__ == "__main__":
    main()
