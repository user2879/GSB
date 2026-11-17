"""管道编辑操作集与版本语义约束。

五类编辑：调整步骤顺序 / 增删步骤 / 修改步骤参数 / 替换实体特征值 / 合并两个实体。

版本语义约束（编辑前校验，不满足即拒绝并输出 E_VERSION_UNSUPPORTED，不静默降级）：
- v1：仅支持内建步骤类型（normalize/cross/bucket），无法表达显式空值区分。
- v2：缺键与显式空值无法区分，IR 含显式空值标记时拒绝落到 v2。
- v3：步骤名必须存在于名称映射表（未知枚举占位步骤除外）。
每次成功编辑同步更新统一中间表示、旁路元数据与降级损失清单。
"""
from __future__ import annotations

from .errors import E_MERGE_CONFLICT, E_VERSION_UNSUPPORTED, EditRejected, Failure
from .ir import IR, Step

V1_STEP_TYPES = {"normalize", "cross", "bucket"}


def _explicit_null_pairs(ir: IR) -> set[tuple[str, str]]:
    return {(e, f) for e in ir.features.explicit_null
            for f in ir.features.explicit_null[e]}


def check_expressible(ir: IR, target_version: str, name_map: dict[str, int],
                      old_ir: IR | None = None) -> None:
    old_nulls = _explicit_null_pairs(old_ir) if old_ir is not None else set()
    added_nulls = _explicit_null_pairs(ir) - old_nulls
    if target_version == "v1":
        for step in ir.steps:
            if step.name not in V1_STEP_TYPES:
                raise EditRejected(
                    E_VERSION_UNSUPPORTED,
                    f"v1 不支持新增步骤类型 {step.name!r}")
        if added_nulls:
            raise EditRejected(
                E_VERSION_UNSUPPORTED, "v1 缺失位无法表达显式空值区分")
    elif target_version == "v2":
        if added_nulls:
            raise EditRejected(
                E_VERSION_UNSUPPORTED, "v2 缺键与显式空值无法区分")
    elif target_version == "v3":
        for step in ir.steps:
            if step.name not in name_map and not step.name.startswith("__unknown_enum_"):
                raise EditRejected(
                    E_VERSION_UNSUPPORTED,
                    f"v3 名称映射表缺步骤 {step.name!r}，编辑被拒绝")
    else:
        raise EditRejected(E_VERSION_UNSUPPORTED, f"未知目标版本 {target_version!r}")


def apply_edit(ir: IR, edit: dict, failures: list[Failure],
               decisions: list[dict]) -> IR:
    """在 IR 副本上施加编辑，返回新 IR；不校验版本约束（由调用方校验）。"""
    op = edit.get("op")
    out = ir.clone()
    if op == "reorder_steps":
        order = edit["order"]
        if sorted(order) != list(range(len(out.steps))):
            raise EditRejected("E_BAD_EDIT", f"重排索引非法: {order}")
        out.steps = [out.steps[i] for i in order]
    elif op == "add_step":
        step = Step(name=edit["step"]["name"],
                    params=dict(edit["step"].get("params", {})))
        index = edit.get("index", len(out.steps))
        out.steps.insert(index, step)
    elif op == "remove_step":
        del out.steps[edit["index"]]
    elif op == "modify_params":
        out.steps[edit["index"]].params.update(edit["params"])
    elif op == "replace_entity":
        _replace_entity(out, edit)
    elif op == "merge_entities":
        _merge_entities(out, edit, failures, decisions)
    else:
        raise EditRejected("E_BAD_EDIT", f"未知编辑操作 {op!r}")
    return out


def _replace_entity(ir: IR, edit: dict) -> None:
    entity = edit["entity"]
    feats = edit.get("features", {})
    ir.features.values[entity] = {}
    ir.features.missing[entity] = set()
    ir.features.explicit_null[entity] = set()
    ir.features.precision[entity] = {}
    for feature in sorted(feats):
        value = feats[feature]
        if value is None:
            ir.features.missing[entity].add(feature)
            if edit.get("explicit_null"):
                ir.features.explicit_null[entity].add(feature)
            continue
        ir.features.values[entity][feature] = float(value)
    for feature in edit.get("missing", []):
        ir.features.missing[entity].add(feature)
    for feature, label in edit.get("precision", {}).items():
        ir.features.precision[entity][feature] = label


def _merge_entities(ir: IR, edit: dict, failures: list[Failure],
                    decisions: list[dict]) -> None:
    source = edit["source"]
    target = edit["target"]
    src_values = ir.features.values.get(source, {})
    tgt_values = ir.features.values.setdefault(target, {})
    tgt_missing = ir.features.missing.setdefault(target, set())
    src_missing = ir.features.missing.get(source, set())
    for feature in sorted(set(src_values) | set(src_missing)):
        if feature in src_missing:
            tgt_missing.add(feature)
            continue
        if feature in tgt_missing:
            continue
        if feature in tgt_values and tgt_values[feature] != src_values[feature]:
            decisions.append({
                "type": "merge_id_conflict", "source": source, "target": target,
                "feature": feature, "kept": tgt_values[feature],
                "dropped": src_values[feature], "rule": "target_wins",
            })
            failures.append(Failure(
                entity_id=target, stage="edit", code=E_MERGE_CONFLICT,
                exc_type="MergeIdConflict",
                detail=f"合并 {source} 时特征 {feature} 冲突，按 target_wins 决断"))
            continue
        tgt_values[feature] = src_values[feature]
        label = ir.features.precision.get(source, {}).get(feature)
        if label:
            ir.features.precision.setdefault(target, {})[feature] = label
    for coll in ("values", "missing", "explicit_null", "precision"):
        getattr(ir.features, coll).pop(source, None)
