"""流水线编排：加载 -> 归一 -> 仲裁 -> 事件判定 -> 迁移复核 -> 归因 -> 账本。"""

import json

from .arbitrate import RULE_VERSION, arbitrate_volume
from .attribute import attribute_conflicts
from .events import evaluate_events, load_events, summarize_verdicts
from .loader import load_declaration
from .migrate import migration_review
from .models import digest
from .normalize import normalize_volume_sources


def load_rules(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_view(declaration, events, profile, rule_version=RULE_VERSION):
    """对全部卷执行仲裁、事件判定、迁移复核与归因，返回结论视图。"""
    volumes = declaration["volumes"]
    sources = declaration["sources"]
    normalized = {}
    arbitration = {}
    for vid in sorted(volumes):
        records = normalize_volume_sources(vid, [sources[s] for s in volumes[vid]])
        normalized[vid] = records
        arbitration[vid] = arbitrate_volume(vid, records, rule_version)
    verdict_rows = evaluate_events(events, arbitration)
    events_summary = summarize_verdicts(verdict_rows, arbitration)
    migration = {}
    attribution = {}
    for vid in sorted(volumes):
        mrows = migration_review(vid, arbitration[vid], normalized[vid],
                                 profile, rule_version)
        migration[vid] = mrows
        attribution[vid] = attribute_conflicts(vid, mrows, normalized[vid],
                                               profile, rule_version)
    return {
        "normalized": normalized,
        "arbitration": arbitration,
        "migration": migration,
        "attribution": attribution,
        "events_summary": events_summary,
        "verdict_rows": verdict_rows,
    }


def volume_batch_rows(view, volume_id):
    """单卷批次的全部结论行：[(domain, payload)]，用于同事务提交。"""
    rows = []
    for metric in sorted(view["arbitration"].get(volume_id, {})):
        rows.append(("arbitration", view["arbitration"][volume_id][metric]))
    for mrow in view["migration"].get(volume_id, []):
        rows.append(("migration", mrow))
    for arow in view["attribution"].get(volume_id, []):
        rows.append(("attribution", arow))
    for srow in view["events_summary"]:
        if srow["volume_id"] == volume_id:
            rows.append(("events", srow))
    norm_digest = digest([r.to_dict() for r in view["normalized"][volume_id]])
    rows.append(("events", {"volume_id": volume_id, "metric": "-",
                            "normalized_digest": norm_digest,
                            "kind": "normalized_digest"}))
    return rows


def all_rows(view):
    """视图的全部结论行：[(domain, volume_id, payload)]，用于整体替换与重放。"""
    rows = []
    for vid in sorted(view["arbitration"]):
        for domain, payload in volume_batch_rows(view, vid):
            rows.append((domain, vid, payload))
    return rows


def scheme_meta(declaration, rule_version, scheme_version):
    return {
        "sources_digest": digest(declaration["sources"]),
        "rule_version": rule_version,
        "scheme_version": scheme_version,
    }


def run_to_ledger(declaration_path, events_path, rules_path, ledger,
                  scheme_version=1, kill_after=None):
    """完整执行流水线并按卷批次提交账本。kill_after 用于强制终止恢复演练：
    在提交完第 kill_after 个卷批次后以 os._exit 结束进程。"""
    import os

    declaration = load_declaration(declaration_path)
    events = load_events(events_path)
    rules = load_rules(rules_path)
    profile = rules["target_profile"]
    rule_version = rules.get("rule_version", RULE_VERSION)
    meta = scheme_meta(declaration, rule_version, scheme_version)

    status = ledger.check_snapshot(meta["sources_digest"], rule_version)
    view = build_view(declaration, events, profile, rule_version)
    if status == "empty":
        committed = set()
    else:
        committed = set(ledger.committed_volumes())
    seq = len(committed)
    for vid in sorted(declaration["volumes"]):
        if vid in committed:
            continue
        seq += 1
        ledger.commit_batch(seq, vid, volume_batch_rows(view, vid), meta)
        if kill_after is not None and seq >= kill_after:
            ledger.conn.commit()
            os._exit(42)  # 模拟强制终止：批次已提交、后续批次未开始
    ledger.replay_verify(all_rows(view))
    return view, meta
