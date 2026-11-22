import json
import os

import pytest

from faultdiag.errors import DiagError
from faultdiag.snapshot import (build_payload, list_snapshots, open_or_replace,
                                read_snapshot, write_snapshot)


def _payload(**kw):
    base = {"instances_version": "iv", "domains_version": "dv",
            "events_digest": "ed", "cursor": 3,
            "domain_states": {"dom-a": "isolated"},
            "instance_states": {"s-a1": "unhealthy"},
            "ledger_digest": "ld"}
    base.update(kw)
    return build_payload(**base)


def test_write_and_read(tmp_path):
    payload = _payload()
    path = write_snapshot(tmp_path, payload)
    assert read_snapshot(path)["digest"] == payload["digest"]


def test_digest_mismatch_raises_e_snapshot(tmp_path):
    path = write_snapshot(tmp_path, _payload())
    payload = json.loads(path.read_text())
    payload["cursor"] = 99  # 篡改内容但不更新摘要
    path.write_text(json.dumps(payload))
    with pytest.raises(DiagError) as exc:
        read_snapshot(path)
    assert exc.value.code == "E_SNAPSHOT"


def test_whole_replace_on_version_mismatch(tmp_path):
    old = _payload()
    write_snapshot(tmp_path, old)
    current = _payload(domains_version="dv2",
                       domain_states={"dom-a": "normal", "dom-b": "normal"},
                       instance_states={"s-a1": "healthy", "s-b1": "healthy"})
    payload, report = open_or_replace(tmp_path, current)
    assert report is not None
    assert report["replaced_domains"] == ["dom-a", "dom-b"]
    assert report["replaced_instances"] == ["s-a1", "s-b1"]
    backup = report["backup"]
    assert os.path.exists(backup)
    assert not os.access(backup, os.W_OK)  # 只读备份
    # 替换后再次打开应直接命中
    _, report2 = open_or_replace(tmp_path, current)
    assert report2 is None


def test_no_replace_when_consistent(tmp_path):
    payload = _payload()
    write_snapshot(tmp_path, payload)
    _, report = open_or_replace(tmp_path, payload)
    assert report is None
    snaps = list_snapshots(tmp_path)
    assert snaps[0]["name"] == "snapshot.json"
