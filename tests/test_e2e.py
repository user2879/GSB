import json
import sqlite3

import pytest

from faultdiag.errors import DiagError
from faultdiag.ledger import Ledger
from faultdiag.replay import replay_verify
from faultdiag.runner import RunConfig, run
from faultdiag.spec import load_domains, load_events, load_instances

BASE = 27100


def _config(spec_files, tmp_path, **kw):
    kw.setdefault("base_port", BASE)
    return RunConfig(
        instances_path=str(spec_files / "instances.json"),
        domains_path=str(spec_files / "domains.json"),
        events_path=str(spec_files / "events.ndjson"),
        workdir=str(tmp_path / "work"), **kw)


def _load_specs(spec_files):
    domains = load_domains(spec_files / "domains.json")
    instances = load_instances(spec_files / "instances.json", domains)
    events = load_events(spec_files / "events.ndjson", instances)
    return domains, instances, events


def test_full_run_domain_decisions(spec_files, tmp_path):
    report = run(_config(spec_files, tmp_path))
    assert report["status"] == "ok"
    assert report["events_processed"] == 10
    assert report["cursor"] == 10
    assert report["max_concurrent"] <= 4
    # 两个域均完成 隔离->恢复 全周期
    assert report["domain_states"] == {"dom-a": "normal", "dom-b": "normal"}
    ledger = Ledger(tmp_path / "work" / "ledger.db", {})
    transitions = [(r["kind"], r["entity"], r["to_state"], r["event_seq"])
                   for r in ledger.transitions()]
    assert ("domain", "dom-a", "isolating", 1) in transitions
    assert ("domain", "dom-a", "isolated", 1) in transitions
    assert ("domain", "dom-a", "recovering", 4) in transitions
    assert ("domain", "dom-a", "normal", 4) in transitions
    assert ("domain", "dom-b", "isolating", 5) in transitions
    assert ("domain", "dom-b", "recovering", 9) in transitions
    # 冷却约束: dom-a 第二次隔离于 seq6(ts=600), seq7(ts=700)恢复时
    # 冷却 300ms 未满, 不得 recovering; 直到 seq10(ts=1000)才恢复
    recovering_seqs = [t[3] for t in transitions
                       if t[1] == "dom-a" and t[2] == "recovering"]
    assert recovering_seqs == [4, 10]
    # 指令回执已持久化且含排队等待时间
    commands = ledger.commands()
    assert {c["action"] for c in commands} == {"isolate", "recover"}
    assert all(c["status"] == "ok" for c in commands)
    assert all(c["start_mono_ms"] >= c["enqueue_mono_ms"] for c in commands)
    ledger.close()


def test_replay_after_restart(spec_files, tmp_path):
    run(_config(spec_files, tmp_path))
    domains, instances, events = _load_specs(spec_files)
    ledger = Ledger(tmp_path / "work" / "ledger.db", {})
    result = replay_verify(ledger, instances, domains, events)
    ledger.close()
    assert result["reproducible"] is True
    assert result["checked_events"] == 10


def test_replay_detects_tampering(spec_files, tmp_path):
    run(_config(spec_files, tmp_path))
    domains, instances, events = _load_specs(spec_files)
    conn = sqlite3.connect(tmp_path / "work" / "ledger.db")
    row = conn.execute(
        "SELECT instance_states FROM event_states WHERE event_seq=4").fetchone()
    tampered = json.loads(row[0])
    tampered["s-a1"] = "degraded"
    conn.execute("UPDATE event_states SET instance_states=? WHERE event_seq=4",
                 (json.dumps(tampered, sort_keys=True),))
    conn.commit()
    conn.close()
    ledger = Ledger(tmp_path / "work" / "ledger.db", {})
    with pytest.raises(DiagError) as exc:
        replay_verify(ledger, instances, domains, events)
    ledger.close()
    assert exc.value.code == "E_REPLAY"
    assert exc.value.details["event_seq"] == 4
    assert exc.value.details["instance"] == "s-a1"
    assert exc.value.details["field"] == "health"


def test_replay_non_reproducible_on_digest_mismatch(spec_files, tmp_path):
    run(_config(spec_files, tmp_path))
    domains, instances, events = _load_specs(spec_files)
    ledger = Ledger(tmp_path / "work" / "ledger.db", {})
    ledger.set_meta("events_digest", "0" * 64)
    ledger.conn.commit()
    result = replay_verify(ledger, instances, domains, events)
    ledger.close()
    assert result["reproducible"] is False
    assert "events_digest" in result["reason"]


def test_resume_from_cursor_after_interrupted_run(spec_files, tmp_path):
    # 首轮只跑前 4 个事件(通过截断事件文件模拟), 再补全事件流重跑
    cfg = _config(spec_files, tmp_path)
    events_path = spec_files / "events.ndjson"
    lines = events_path.read_text().splitlines()
    truncated = tmp_path / "events_part.ndjson"
    truncated.write_text("\n".join(lines[:4]) + "\n")
    cfg1 = RunConfig(cfg.instances_path, cfg.domains_path, str(truncated),
                     cfg.workdir, base_port=BASE + 20)
    report1 = run(cfg1)
    assert report1["cursor"] == 4
    # 第二轮: 同一工作目录, 完整事件流 -> 快照整体替换(事件流摘要变化)
    cfg2 = _config(spec_files, tmp_path, base_port=BASE + 20)
    cfg2.workdir = cfg.workdir
    report2 = run(cfg2)
    assert report2["cursor"] == 10
    assert report2["events_processed"] == 6  # 只推进剩余事件
    assert report2["snapshot_replace"] is not None
    assert report2["unknown_commands"] == []


def test_inject_event(spec_files, tmp_path):
    with pytest.raises(DiagError) as exc:
        run(_config(spec_files, tmp_path, inject="event", base_port=BASE + 40))
    assert exc.value.code == "E_EVENT"
    log = (tmp_path / "work" / "logs" / "faultdiag.jsonl").read_text()
    assert '"E_EVENT"' in log


def test_inject_level(spec_files, tmp_path):
    with pytest.raises(DiagError) as exc:
        run(_config(spec_files, tmp_path, inject="level", base_port=BASE + 50))
    assert exc.value.code == "E_LEVEL"


def test_inject_snapshot(spec_files, tmp_path):
    with pytest.raises(DiagError) as exc:
        run(_config(spec_files, tmp_path, inject="snapshot", base_port=BASE + 60))
    assert exc.value.code == "E_SNAPSHOT"
