import json
import socket

import pytest

from cfgdist.derivation import node_digest
from cfgdist.errors import BizError, ErrorCode
from cfgdist.planner import build_plan, expected_digest_map, plan_fingerprint
from cfgdist.runtime import check_ports_free, load_node_declaration


def test_plan_is_deterministic_and_broadcast_reaches_all_nodes(
    v3_manifest, node_specs
):
    node_ids = [spec["id"] for spec in node_specs]
    first = build_plan(v3_manifest, node_ids)
    second = build_plan(json.loads(json.dumps(v3_manifest)), list(reversed(node_ids)))
    assert plan_fingerprint(first) == plan_fingerprint(second)
    # Broadcast entry must be present on every node slice.
    for slice_payload in first["slices"]:
        assert any(item["key"] == "shared" for item in slice_payload["entries"])


def test_undeclared_target_node_rejected(v3_manifest, node_specs):
    v3_manifest["entries"][0]["node"] = "ghost-node"
    with pytest.raises(BizError) as exc_info:
        build_plan(v3_manifest, [spec["id"] for spec in node_specs])
    assert exc_info.value.code == ErrorCode.E_FORMAT
    assert "ghost-node" in exc_info.value.message


def test_node_digest_is_order_independent():
    entries_one = [
        {"key": "a", "value": 1},
        {"key": "b", "value": 3},
    ]
    entries_two = list(reversed(entries_one))
    derivations = [
        {"name": "d", "inputs": ["a", "b"], "op": "sum", "target_node": "n"}
    ]
    assert node_digest("n", entries_one, derivations) == node_digest(
        "n", entries_two, derivations
    )


def test_port_check_reports_e_port(node_specs):
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind((node_specs[0]["host"], node_specs[0]["port"]))
    holder.listen(1)
    try:
        with pytest.raises(BizError) as exc_info:
            check_ports_free(node_specs)
        assert exc_info.value.code == ErrorCode.E_PORT
        assert exc_info.value.details["busy"][0]["node"] == "node-a"
    finally:
        holder.close()


def test_node_declaration_rejects_non_loopback(tmp_path):
    path = tmp_path / "nodes.json"
    path.write_text(
        json.dumps(
            {"nodes": [{"id": "x", "host": "8.8.8.8", "port": 19000, "faults": {}}]}
        )
    )
    with pytest.raises(BizError) as exc_info:
        load_node_declaration(path)
    assert exc_info.value.code == ErrorCode.E_FORMAT
    assert exc_info.value.details["field_path"].endswith(".host")
