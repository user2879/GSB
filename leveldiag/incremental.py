"""增量重生成与重算闭包。

闭包以「工件文件 + 账本条目」为粒度声明。字段级影响表（单一规格数值/阈值
修改的最小闭包）：

  thresholds.* / resources.density_cap / resources.tolerance /
  connectivity[*].min_channel_width_ratio / connectivity[*].tolerance /
  terrain.weight_normalization_tolerance
      -> 仅 ledger.json + boundary_report.json + manifest.json，
         且只重算受影响的账本条目；网格/连续域/映射分片哈希保持不变。

  quantization.elevation_step / density_step / rounding
      -> mapping 全部分片 + grid + ledger + boundary + manifest
         （连续域分片不变）。

  seed / terrain.weights / grid.* / regions / points / terrain_rules /
  resources.count / density_anchor
      -> 全部工件（结构性变更，保守全量闭包）。

实际重算集合与声明闭包不一致 -> E_CLOSURE_MISMATCH。
"""

from __future__ import annotations

import copy
from pathlib import Path

from .errors import ErrorCode, LevelDiagError
from .generator import _write, run_pipeline, write_artifacts
from .ledger import boundary_report
from .spec import spec_from_dict
from .util import file_hash


# 规格内容一变，spec.canonical.json 必然重写，故所有部分闭包都包含它
LEDGER_ONLY_FILES = ["boundary_report.json", "ledger.json", "manifest.json",
                     "spec.canonical.json"]

# 仅影响账本判定的字段 -> 受影响账本条目 kind 集合；None 表示全部账本条目
_LEDGER_FIELD_TABLE = [
    ("thresholds.cell_density_max", {"cell_density_max"}),
    ("thresholds.cell_density_tolerance", {"cell_density_max"}),
    ("thresholds.elevation_pass", None),
    ("resources.density_cap", {"resource_density_cap"}),
    ("resources.tolerance", {"resource_density_cap"}),
    ("terrain.weight_normalization_tolerance", {"weight_normalization"}),
]
# 连通性数组内字段影响全部连通/通道条目
_CONNECTIVITY_PREFIXES = (
    "connectivity[",
)


def _walk_paths(obj, prefix: str = "") -> list[str]:
    paths = []
    if isinstance(obj, dict):
        for key in sorted(obj):
            paths.extend(_walk_paths(obj[key], f"{prefix}.{key}" if prefix else key))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            paths.extend(_walk_paths(item, f"{prefix}[{i}]"))
    else:
        paths.append(prefix)
    return paths


def changed_leaf_paths(old: dict, new: dict) -> list[str]:
    old_leaves = {p: _get(old, p) for p in _walk_paths(old)}
    new_leaves = {p: _get(new, p) for p in _walk_paths(new)}
    added_or_removed = set(old_leaves) ^ set(new_leaves)
    modified = {p for p in old_leaves.keys() & new_leaves.keys()
                if old_leaves[p] != new_leaves[p]}
    return sorted(added_or_removed | modified)


def _get(obj, dotted: str):
    cur = obj
    token = ""
    i = 0
    tokens = []
    while i < len(dotted):
        ch = dotted[i]
        if ch == ".":
            if token:
                tokens.append(token)
                token = ""
        elif ch == "[":
            if token:
                tokens.append(token)
                token = ""
            end = dotted.index("]", i)
            tokens.append(int(dotted[i + 1:end]))
            i = end
        else:
            token += ch
        i += 1
    if token:
        tokens.append(token)
    for t in tokens:
        cur = cur[t]
    return cur


def declared_closure(changed_paths: list[str]) -> dict:
    """根据变更叶子路径声明受影响文件与账本条目集合。"""
    files: set[str] = set()
    entry_kinds: set[str] = set()
    full_rebuild = False
    for path in changed_paths:
        head = path.split("[")[0].split(".")[0]
        if head == "quantization":
            files.update(["boundary_report.json", "grid.json", "ledger.json",
                          "manifest.json", "mapping/index.json",
                          "spec.canonical.json"])
            files.add("mapping/__shards__")  # 占位：调用处展开为 region 分片
            entry_kinds.clear()
            entry_kinds.add("*")
            continue

        if any(path.startswith(p) for p in _CONNECTIVITY_PREFIXES) and (
                "min_channel_width_ratio" in path or path.endswith(".tolerance")):
            files.update(LEDGER_ONLY_FILES)
            entry_kinds.clear()
            entry_kinds.add("*")
            continue

        matched = False
        for field, kinds in _LEDGER_FIELD_TABLE:
            if path == field or path.startswith(field + "."):
                files.update(LEDGER_ONLY_FILES)
                if kinds is None or "*" in entry_kinds:
                    entry_kinds.clear()
                    entry_kinds.add("*")
                else:
                    entry_kinds.update(kinds)
                matched = True
                break
        if matched:
            continue
        full_rebuild = True
    if full_rebuild:
        return {"scope": "full", "files": "*", "ledger_entries": "*"}
    return {"scope": "partial",
            "files": sorted(files),
            "ledger_entries": sorted(entry_kinds)}


def _hash_map(out: Path) -> dict[str, str]:
    return {str(p.relative_to(out)).replace("\\", "/"): file_hash(p)
            for p in out.rglob("*") if p.is_file()}


def incremental_regenerate(old_spec_dict: dict, new_spec_dict: dict,
                           out_dir: str | Path) -> dict:
    """执行增量重生成并核对实际重算集合 == 声明闭包。"""
    out = Path(out_dir)
    before = _hash_map(out)
    changes = changed_leaf_paths(old_spec_dict, new_spec_dict)
    closure = declared_closure(changes)

    old_spec = spec_from_dict(copy.deepcopy(old_spec_dict))
    new_spec = spec_from_dict(copy.deepcopy(new_spec_dict))
    old_result = run_pipeline(old_spec)
    new_result = run_pipeline(new_spec)

    if closure["scope"] == "full":
        write_artifacts(new_spec, new_result, out)
        rewritten = sorted(set(_hash_map(out)) | set(before))
        return {"changed_paths": changes, "declared_closure": closure,
                "rewritten_files": rewritten, "unchanged_files": []}

    # 部分闭包：连续域在所有部分场景下都必须保持不变
    if old_result["continuous"]["continuous_hash"] != \
            new_result["continuous"]["continuous_hash"]:
        raise LevelDiagError(
            ErrorCode.E_CLOSURE_MISMATCH,
            "声明为部分闭包，但连续域哈希发生变化",
            {"stage": "continuous"})

    quant_change = "mapping/__shards__" in closure["files"]
    declared_files = set(closure["files"])
    if quant_change:
        declared_files.discard("mapping/__shards__")
        for rid in new_result["mapping"]["region_shards"]:
            declared_files.add(f"mapping/region_{rid}.json")
        # 量化变更：重写 mapping 分片/索引与 grid，连续域文件保持不动
        for rid, shard in new_result["mapping"]["region_shards"].items():
            _write(out / "mapping" / f"region_{rid}.json", shard)
        _write(out / "mapping" / "index.json", {
            "spec_hash": new_result["mapping"]["spec_hash"],
            "continuous_hash": new_result["mapping"]["continuous_hash"],
            "mapping_hash": new_result["mapping"]["mapping_hash"],
            "rules": new_result["mapping"]["rules"],
            "observed_max_error": new_result["mapping"]["observed_max_error"],
            "region_shard_files": sorted(
                f"mapping/region_{rid}.json"
                for rid in new_result["mapping"]["region_shards"]),
        })
        _write(out / "grid.json", new_result["grid"])

    entry_filter = closure["ledger_entries"]
    old_by_id = {e["constraint_id"]: e for e in old_result["ledger"]["entries"]}
    new_entries = []
    rewritten_entries = []
    for entry in new_result["ledger"]["entries"]:
        kind = entry["kind"]
        affected = "*" in entry_filter or kind in entry_filter
        if affected:
            new_entries.append(entry)
            rewritten_entries.append(entry["constraint_id"])
        else:
            if entry != old_by_id[entry["constraint_id"]]:
                raise LevelDiagError(
                    ErrorCode.E_CLOSURE_MISMATCH,
                    f"非闭包条目被意外改变: {entry['constraint_id']}",
                    {"constraint_id": entry["constraint_id"]})
            new_entries.append(old_by_id[entry["constraint_id"]])

    partial_ledger = copy.deepcopy(new_result["ledger"])
    partial_ledger["entries"] = new_entries
    # 部分重算时 ledger_hash 必须变化（条目内容变了），且与全量结果自洽
    from .util import content_hash
    partial_ledger["ledger_hash"] = content_hash({
        "spec_hash": partial_ledger["spec_hash"],
        "continuous_hash": partial_ledger["continuous_hash"],
        "mapping_hash": partial_ledger["mapping_hash"],
        "grid_hash": partial_ledger["grid_hash"],
        "entries": new_entries})
    if partial_ledger["ledger_hash"] != new_result["ledger"]["ledger_hash"]:
        raise LevelDiagError(
            ErrorCode.E_CLOSURE_MISMATCH,
            "增量账本哈希与全量重算结果不一致", {})
    partial_ledger["summary"] = new_result["ledger"]["summary"]

    _write(out / "ledger.json", partial_ledger)
    _write(out / "boundary_report.json", boundary_report(partial_ledger))
    # 规格文件先于 manifest 刷新，保证 manifest 记录的是最终字节哈希
    _write(out / "spec.canonical.json", new_spec.raw)
    # manifest 仅刷新受影响文件哈希
    import json
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["spec_hash"] = new_spec.spec_hash
    manifest["continuous_hash"] = new_result["continuous"]["continuous_hash"]
    manifest["mapping_hash"] = new_result["mapping"]["mapping_hash"]
    manifest["grid_hash"] = new_result["grid"]["grid_hash"]
    manifest["ledger_hash"] = partial_ledger["ledger_hash"]
    for rel in list(manifest["files"]):
        manifest["files"][rel] = file_hash(out / rel)
    _write(out / "manifest.json", manifest)

    after = _hash_map(out)
    actual_rewritten = sorted(rel for rel in after
                              if before.get(rel) != after.get(rel))
    expected = sorted(declared_files)
    if sorted(actual_rewritten) != expected:
        raise LevelDiagError(
            ErrorCode.E_CLOSURE_MISMATCH,
            "实际重算文件集合与声明闭包不一致",
            {"declared": expected, "actual": actual_rewritten,
             "changed_paths": changes})
    unchanged = sorted(rel for rel in after if before.get(rel) == after[rel])
    return {"changed_paths": changes, "declared_closure": closure,
            "rewritten_files": actual_rewritten,
            "rewritten_ledger_entries": rewritten_entries,
            "unchanged_files": unchanged}
