# 实际验证记录

验证时间：2026-11-27（Asia/Shanghai）。

## 1. 环境与回环

- WSL/Ubuntu 工作目录，Python `3.11.0rc1`。
- 标准 `python3.11 -m venv .venv` 因系统缺少 `ensurepip`/`python3.11-venv` 失败；当前账号无免密 sudo，因此使用 `virtualenv -p python3.11 .venv` 创建等价环境。
- `.venv` 中实际安装：`numpy 2.4.6`、`pytest 9.2.0`。
- `python scripts/check_loopback.py` 通过，一次实际输出为 `127.0.0.1:45008 loopback ok`（端口由操作系统动态分配）。

## 2. 自动化测试

命令：

```bash
source .venv/bin/activate
pytest -q
```

结果：25 个测试函数全部通过，退出码 `0`。覆盖：

- 样本 Schema、坏 JSON、唯一 ID、词表、偏移、千分比、引用端点和修订链。
- 最长字符跨度、同长按词项字典序、重叠抑制和版本回滚。
- 定点位宽/量化档位、残差、舍入方向和半步边界。
- 必填缺失、互斥置信度/同分角色排序。
- 跨事件合并和去重阈值。
- 账本复用、角色权重精确失效、词表删除失效和 13 步编辑等价。
- 预算顺序、超限、待办、下一批优先、并发上限。
- 快照原子切换、临时目录恢复、五类注入隔离和注入后续跑。
- CLI 生成/核验/诊断和 NumPy 环境。

## 3. 主样本业务结果

`arg-review run --samples samples --state-dir artifacts` 的基线结果：

- 7 条主事件全部通过校验，修订链有效。
- 触发匹配 7 个，抑制 1 个短触发词。
- 论元候选 26 个。
- 裁决 7 个事件，其中 1 个因缺少必填 `host` 不完整。
- 互斥抑制 2 个 `guest` 候选。
- 引用边 6 条，接受 4 条，形成 3 个跨事件合并组。
- 预算打点 26/200，单事件配额 30，待办 0；配置并发 4，调度器记录的峰值不超过 4。
- 基线投影指纹：`12e73df6169d15b0`。

## 4. 13 步编辑序列的增量/全量等价

命令：

```bash
arg-review verify-edits \
  --samples samples \
  --state-dir artifacts/verification \
  --report artifacts/edit_equivalence_report.json
```

最终结果：`all_steps_equal = true`，`completed_steps = 13`。每一步暖账本增量运行均与相同最终状态的空白全量运行逐字段一致。

| 步骤 | 命令 | 复用 | 重算 |
|---:|---|---:|---:|
| 1 | `set_quantization` | 0 | 43 |
| 2 | `update_weights(host)` | 40 | 3 |
| 3 | `set_dedup_threshold` | 40 | 3 |
| 4 | `replace_event` | 41 | 2 |
| 5 | `add_events` | 43 | 5 |
| 6 | `add_revision(未激活草稿)` | 48 | 0 |
| 7 | `delete_revision(草稿)` | 48 | 0 |
| 8 | `update_role_rules(meeting 必填/互斥)` | 45 | 5 |
| 9 | `rollback_revision(rev-2)` | 30 | 20 |
| 10 | `rollback_revision(rev-4)` | 50 | 0 |
| 11 | `delete_event(E004)` | 45 | 0 |
| 12 | `set_quantization(coarse)` | 7 | 38 |
| 13 | `set_budget(max_workers=3)` | 45 | 0 |

步骤 2 只重算 `arg:E005:E005-A1`、`decision:E005` 和引用它的一个合并结论；未受影响触发与论元直接命中持久化账本。步骤 6/7 证明新增但不激活、随后删除的修订不会使现有产物失效。步骤 10 证明回滚到曾经计算过的 `rev-4` 时，50 个当前产物全部可按内容键复用且不残留旧版本字段。

## 5. 注入诊断

命令：

```bash
arg-review diagnose --samples samples --state-dir artifacts
```

结果：5/5 注入编号全部命中，汇总见 `artifacts/diagnostic_report.json`。

| 注入编号 | 故障 | 业务码 | 阶段码 |
|---|---|---|---|
| `INJ-OFFSET-001` | 越界偏移 | `E_SCHEMA_INVALID` | `STAGE_VALIDATION_REJECTED_ENTRY` |
| `INJ-CONF-002` | 非千分比置信度 | `E_SCHEMA_INVALID` | `STAGE_ARGUMENT_CONFIDENCE_INVALID` |
| `INJ-REL-003` | 引用不存在论元 | `E_SCHEMA_INVALID` | `STAGE_REFERENCE_ENDPOINT_MISSING` |
| `INJ-ROLE-004` | 角色不在集合 | `E_SCHEMA_INVALID` | `STAGE_ARGUMENT_ROLE_INVALID` |
| `INJ-CHAIN-005` | 修订链断链 | `E_SCHEMA_INVALID` | `STAGE_REVISION_CHAIN_INVALID` |

偏移、置信度和角色注入各只使目标事件无法形成最终裁决，其余 6 个事件结论保留；引用注入保留全部 7 个事件裁决，只隔离坏引用边；断链注入仍可用活动修订中的事件类型切片完成其余事件。持久化状态在移除注入后重跑可续跑，并与干净全量投影一致。

## 6. 一键复验

```bash
bash scripts/run_acceptance.sh
```

该脚本会检查回环、生成样本、运行 pytest、执行 `arg-review all`，并刷新：

- `artifacts/ledger.json`
- `artifacts/edit_equivalence_report.json`
- `artifacts/diagnostic_report.json`
- `artifacts/snapshots/current`
