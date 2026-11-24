# 实际验证记录

验证日期：2026-11-24。执行环境：工作区 Linux 容器/终端；实现只使用跨平台 Python/NumPy/SQLite 接口。Windows 命令见根目录 `README.md`。

## 环境

```text
Python 3.11.0rc1
NumPy 1.26.4
SQLite 3.37.2
```

`python -m colmv doctor` 返回 `python_ok=true`、`numpy_ok=true`、`sqlite_ok=true`。

## 自动化测试

命令：

```text
python -m unittest discover -s tests -v
```

结果：`Ran 31 tests ... OK`。

覆盖点：强校验与首错误定位、位图长度、非有限浮点、物理截断、零预算、定点截断/饱和、双路径字节一致、槽位严格互补、边界决胜、跨 `PYTHONHASHSEED` 事件序列一致、账本链式摘要、快照摘要损坏整体替换、未提交批次 unknown、崩溃恢复续跑、重放篡改 `E_REPLAY`、演进陈旧标记、模拟结果差异 `E_DIVERGE`。

## 双路径一致性

命令：

```text
python -m colmv verify-paths \
  --catalog samples/base/catalog.json \
  --views samples/views.json \
  --changes samples/changes.ndjson \
  --budget 1 --batch-plan 3,6,3,6,2
```

结果：

```text
equal_bytes = true
byte_length = 17017
materialized_digest = d7467813ac4d0180dca66d388090665088193cfce010e1f57103964fad902723
strict_slot_partition.valid = true
resident: hits=1, replacements=40, rescans=0
rescan:   hits=1, replacements=40, rescans=106
strict_slot_partition: replacement victims == all slot-denied groups
strict_slot_partition: rescan events cover every replacement victim
```

两条路径最终摘要相同；重扫路径对槽位未驻留分组逐页重扫，但不改变最终结果。

## 边界判定

单元测试固定 `C=4, P=2, V=3, B=1`：

```text
R = 4 / 1 = 4
7/2 <= R <= 9/2 => boundary_affected=true
resident_work = 4 * 3 = 12
rescan_work = 2 * 3 * 8 + 4 = 52
12 <= 52 => resident
```

对照用例 `C=3` 选择 `resident`，`C=5` 选择 `rescan`，两者均 `boundary_affected=false`。精确相等时稳定规则选择 `resident`。

## 自动路径切换与账本

命令：

```text
python -m colmv run \
  --catalog samples/base/catalog.json \
  --views samples/views.json \
  --changes samples/changes.ndjson \
  --ledger demo.sqlite \
  --budget 1 --batch-plan 3,6,3,6,2
```

路径和事件：

```text
batch 1: resident, changes=3, replacement_events=8,  rescan_events=0
batch 2: rescan,   changes=6, replacement_events=11, rescan_events=28
batch 3: resident, changes=3, replacement_events=7,  rescan_events=0
batch 4: rescan,   changes=6, replacement_events=8,  rescan_events=22
batch 5: resident, changes=2, replacement_events=6,  rescan_events=0
switch_history: resident→rescan→resident→rescan→resident
slot_summary: budget=1, hits=1, replacements=40, rescans=50, final_resident_groups=1
all batches: complete_for_affected=true, disjoint=true
```

账本：

```text
ledger_format_version=1
cursor=20
last_batch=5
content_chain.valid=true
snapshot.valid=true
materialized_digest=d7467813ac4d0180dca66d388090665088193cfce010e1f57103964fad902723
```

## 重放校验

```text
python -m colmv replay --catalog samples/base/catalog.json --ledger demo.sqlite --changes samples/changes.ndjson
```

结果：

```text
scheme_version=1
verified_batches=5
cursor=20
materialized_digest=d7467813ac4d0180dca66d388090665088193cfce010e1f57103964fad902723
stream_complete=true
content_chain.valid=true
```

测试中篡改第 2 批某分组 `row_count` 后，重放返回 `E_REPLAY`，`batch_no=2`，定位到首个不一致分组键和字段 `row_count`。

## 强制终止与恢复

在第 3 批写入意图后强杀（退出码 137）：

```text
crash_exit=137
recover: unknown_group_count=7, exit_report_available=false
resume_from_batch=2, resume_from_cursor=9
```

使用 `run --resume --batch-size 4` 续跑后：

```text
last_batch=5
cursor=20
content_chain.valid=true
materialized_digest=d7467813ac4d0180dca66d388090665088193cfce010e1f57103964fad902723
```

续跑结果与不中断运行的摘要完全一致。

## 快照整体替换

单元测试把 `snapshot.payload_json` 改为 `{}`：

```text
verify_snapshot: valid=false, replaced=true, replacement.available=true
再次 verify_snapshot: valid=true
```

快照从最后完整提交批次整体重建，而不是逐字段修补。

## 演进失效传播

```text
python -m colmv evolve \
  --catalog samples/base/catalog.json \
  --old-views samples/views.json \
  --new-views samples/views_v2.json \
  --old-budget 1 --new-budget 1 \
  --changes samples/changes.ndjson --ledger demo.sqlite
```

结果：

```text
old_scheme_version=1
new_scheme_version=2
monotonic_increment=true
old_traces_marked_stale=true
current_conclusions_visible=false
requires_recompute=true
snapshot_action=cleared_until_whole_recompute
affected_groups=26
affected_pages=4 (delta,p000,p001,p002)
```

变更字段包括 `sales_by_region.group_by` 和 `sales_region_product_live.filters`。每步引用依据写入 `steps`，旧方案批次/分组/事件整体 `is_stale=1`。

## 异常注入

```text
python -m colmv validate ... --inject truncate-page
exit=5, code=E_INTEGRITY, field=p000

python -m colmv run ... --budget 1 --inject zero-budget
exit=4, code=E_BUDGET, field=budget
```

零预算在打开/修改账本前即被拒绝。
