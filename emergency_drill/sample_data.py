"""本地确定性样本生成：预案、注入事件及异常变体。"""

import json
import os


def valid_plan_dict():
    return {
        "plan_id": "drill-demo",
        "version": 1,
        "stages": [
            {
                "stage_id": "detect",
                "order": 1,
                "duration_ms": 1200,
                "prerequisites": [],
                "actions": [
                    {"action_id": "a_detect_node", "nominal": "ok"},
                    {"action_id": "a_detect_link", "nominal": "ok"},
                ],
                "criterion": {"mode": "all_of", "required_actions": ["a_detect_node", "a_detect_link"]},
            },
            {
                "stage_id": "isolate",
                "order": 2,
                "duration_ms": 900,
                "prerequisites": ["detect"],
                "actions": [
                    {"action_id": "a_block_traffic", "nominal": "ok"},
                    {"action_id": "a_switch", "nominal": "ok"},
                ],
                "criterion": {"mode": "all_of", "required_actions": ["a_block_traffic", "a_switch"]},
            },
            {
                "stage_id": "failover",
                "order": 3,
                "duration_ms": 1500,
                "prerequisites": ["isolate"],
                "actions": [
                    {"action_id": "a_promote_standby", "nominal": "ok"},
                    {"action_id": "a_route_check", "nominal": "ok"},
                ],
                "criterion": {"mode": "all_of", "required_actions": ["a_promote_standby", "a_route_check"]},
            },
            {
                "stage_id": "verify",
                "order": 4,
                "duration_ms": 800,
                "prerequisites": ["failover"],
                "actions": [{"action_id": "a_consistency_check", "nominal": "ok"}],
                "criterion": {"mode": "all_of", "required_actions": ["a_consistency_check"]},
            },
            {
                "stage_id": "recover",
                "order": 5,
                "duration_ms": 1000,
                "prerequisites": ["verify"],
                "actions": [{"action_id": "a_service_restore", "nominal": "ok"}],
                "criterion": {"mode": "all_of", "required_actions": ["a_service_restore"]},
            },
        ],
    }


def valid_events_dict():
    return {
        "events": [
            {"event_id": "evt-node-1", "kind": "node_lost", "offset_ms": 1500, "target_stage": "failover"},
            {"event_id": "evt-switch-1", "kind": "switch_fail", "offset_ms": 300, "target_stage": "isolate"},
            {"event_id": "evt-timeout-1", "kind": "check_timeout", "offset_ms": 100, "target_stage": "verify"},
        ]
    }


def cyclic_plan_dict():
    plan = valid_plan_dict()
    # detect <- isolate <- detect 形成环。
    for stage in plan["stages"]:
        if stage["stage_id"] == "detect":
            stage["prerequisites"] = ["isolate"]
    return plan


def bad_reference_plan_dict():
    plan = valid_plan_dict()
    for stage in plan["stages"]:
        if stage["stage_id"] == "isolate":
            stage["prerequisites"] = ["detect", "ghost-stage"]
    return plan


def noncontinuous_order_plan_dict():
    plan = valid_plan_dict()
    plan["stages"][-1]["order"] = 9
    return plan


def negative_duration_plan_dict():
    plan = valid_plan_dict()
    plan["stages"][0]["duration_ms"] = -1
    return plan


def bad_event_target_dict():
    events = valid_events_dict()
    events["events"][0]["target_stage"] = "ghost-stage"
    return events


def generate_samples(directory):
    os.makedirs(directory, exist_ok=True)
    files = {
        "plan_valid.json": valid_plan_dict(),
        "events_valid.json": valid_events_dict(),
        "plan_cycle.json": cyclic_plan_dict(),
        "plan_bad_reference.json": bad_reference_plan_dict(),
        "plan_order_gap.json": noncontinuous_order_plan_dict(),
        "plan_negative_duration.json": negative_duration_plan_dict(),
        "events_bad_target.json": bad_event_target_dict(),
    }
    for name, payload in files.items():
        with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(directory, "plan_corrupt.json"), "w", encoding="utf-8") as handle:
        handle.write('{"plan_id": "broken", "stages": [ { "stage_id": "x", ')
    return sorted(files) + ["plan_corrupt.json"]
