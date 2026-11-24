# 回执账本与资源归一组件说明

## 1. 运行方式

Windows PowerShell / CMD 下使用 Python 3.11：

```powershell
py -3.11 -m pip install numpy pytest
python baseline.py init
python baseline.py verify
python -m pytest -q
```

常用命令：

```powershell
python baseline.py ports --data data
python baseline.py validate --data data
python baseline.py run --data data --run runs\manual --reset
python baseline.py run --data data --run runs\budget --reset --max-commands 1
python baseline.py run --data data --run runs\budget
python baseline.py run --data data --run runs\injected --reset --injections
python baseline.py compare runs\reference\ledger.json runs\incremental-restart\ledger.json
```

`baseline.py init` 会生成固定种子 `234` 的本地样本、检查回环端口，并从空白重放完整命令流到 `runs/reference/ledger.json` 与 `runs/reference/report.json`。
`python baseline.py verify` 还会写出汇总验收结论的 `runs/verification.json` 和 `runs/diagnostic-report.json`。

## 2. 代码结构

- `baseline.py`：CLI 入口。
- `receipt_ledger/config.py`：工人、请求流、批次、规则和命令的 schema 校验。
- `receipt_ledger/receipts.py`：r1/r2/r3 定宽记录解析、排名补齐、双向转换和旁路元数据。
- `receipt_ledger/generator.py`：固定整数种子样本、异常注入配置和非法 schema 夹具。
- `receipt_ledger/normalizer.py`：归属裁决、冲突/不可判定处理、定点资源指标。
- `receipt_ledger/engine.py`：命令状态、依赖失效、复用、预算/配额和确定性并发调度。
- `receipt_ledger/snapshot.py`：临时目录、原子指针、版本/配置哈希识别和 JSON 字段对比。
- `receipt_ledger/workspace.py`：本地 JSON/回执加载与注入施加。
- `receipt_ledger/runner.py`：端到端运行、断点续跑和报告落盘。
- `receipt_ledger/cli.py`：`init/run/compare/ports/validate/verify` 命令。
- `tests/test_ledger.py`：15 个覆盖验收点的 pytest 用例。

## 3. 三版本回执格式

记录为带 `|` 分隔的定宽填充列，列宽和字段顺序由 `receipt_ledger/receipts.py` 中的 `WIDTHS` 唯一约定：

- r1：`batch_id`、`request_id`、`output_digest`。
- r2：追加 `param_version`、`latency_ms`、`worker_id`。
- r3：追加 `rank`、`overrides`、`param_digest`。

解析器同时校验列数和每列固定宽度；截断行进 `failures`，业务码为 `E_RECEIPT_TRUNCATED`，不会中断同文件或同批次其他记录。

r3 批次内顺序由稳定名次 `rank` 决定。r1/r2 没有名次时使用 `(request_id, record_version)` 字典序补齐，并写入：

- `rank_source=lexicographic_request_id`
- `rank_basis=missing stable rank; ordered by request_id lexicographic ascending`

## 4. 转换与旁路

- 低版本升 r3 时，缺失字段统一写为 `MISSING`，并生成 `precision_labels` 与 `completeness_labels`。
- r3 降 r1 时只保留三列权威线格式；`rank`、`overrides`、`param_digest` 等写入 `sidecars.json` 和快照中的 `sidecars.json`。
- 每个 sidecar 含 `truncated_fields`，账本顶层也汇总 `r3_to_r1_truncated_fields`。
- 归并只读回执权威字段，不把 sidecar 当作新的裁决来源。
- r3 记录的 `overrides_state=missing` 或覆盖引用指向不存在候选时，对应请求输出 `status=undecidable`、`undecidable=true` 和 `E_ATTRIBUTION_UNDECIDABLE`，保留全部候选，不静默通过。

## 5. 归属裁决

候选先按 `(batch_id, worker_id)` 聚合，再按 `request_id` 汇总跨批次候选。裁决顺序：

1. 可解析且能落到同请求候选的 r3 覆盖引用；唯一覆盖根成为权威回执。
2. 无覆盖引用或存在多个根时，按参数版本序号降序、批次标识字典序升序、工作进程标识升序、有限耗时升序、摘要和回执 UID 决断。
3. 若最高优先级候选的版本、批次、工人、耗时坐标完全相同但输出摘要不同，输出 `E_ATTRIBUTION_CONFLICT`，不选择权威回执，并保留全部候选。
4. r3 覆盖引用缺失或悬空时输出 `E_ATTRIBUTION_UNDECIDABLE`。

每条归属记录包含 `candidate_receipt_uids`、`covered_receipts`、`conflict`、`undecidable` 和有序 `decision_basis`。

## 6. 定点资源公式

定点倍数 `SCALE=1_000_000`，所有权威指标均为 JSON 整数：

```text
version_weight_q = SCALE + (param_version - 1) * version_weight_step_q
base_mac_per_request = mac_factor * model_dim * hidden_dim
mac = request_count * base_mac_per_request * version_weight_q // SCALE
batch_utilization_q = request_count * SCALE // batch_size
mac_per_ms_q = mac * SCALE // latency_ms       # latency 缺失时为 null
```

r1 批次没有耗时字段时，`latency_state=missing`，但乘加量和利用率仍可复核；转换标签明确记录精度损失。

## 7. 依赖失效与复用

依赖图按以下链维护：

```text
param_version:<worker>:<v> -> batch:<batch>:<worker> -> attribution:<request>
request:<id> -> attribution:<id> -> batch metric
rules:<hash> -> attribution / batch metric
```

- `set_param_version W3 2` 只传播到 W3/v2 的三个批次和八个请求归属。
- `replace_request REQ-0001` 只失效 `attribution:REQ-0001` 与 `batch:B101:W1`。
- `switch_rules` 失效引用规则的归属和资源条目；文本解析快照直接复用。
- 未变化且已提交的条目从持久化快照恢复，写入 `cycles[].reused` 与扁平 `reuse_hits`。

## 8. 预算与确定性并发

规则中的 `batch_budget_units` 限制单批每轮可提交条目数，`worker_quota_units` 限制单工作进程每轮配额。预算不足的条目进入 `pending_queue`，输出 `E_NORMALIZE_BUDGET_PENDING`；下一轮旧待办按首次入队序号优先。

实际计算可使用 `ThreadPoolExecutor`，但工作线程只返回不可变值；主线程始终按 `entry_id` 排序提交账本。因此 `--jobs 1` 与 `--jobs 4` 的账本逐字段一致，不依赖线程调度顺序。

## 9. 快照与恢复

每次命令推进后：

1. 在 `snapshots/.staging-*` 临时目录写 `state.json`、`ledger.json`、`01_parse.json`、`02_attribution.json`、`03_resources.json`、`04_summary.json`、`sidecars.json` 和 `manifest.json`。
2. `os.replace` 整体提升为不可变 release 目录。
3. 原子替换 `snapshots/current.json` 指针。

manifest 记录 `global_version`、`config_hash`、`completed_command_cursor` 和 `artifact_versions`。启动时若快照格式版本或配置哈希不同，不复用旧谱系，但保留旧 release；新快照使用含配置哈希前缀的目录名。SIGINT/SIGTERM 门在命令边界抛出，提交阶段异常会清理临时目录，当前指针仍指向上一完整快照。SIGKILL/断电等不可处理终止也只会留下未引用临时目录，不会产生半提交 current 指针。

## 10. 异常注入

`data/injections.json` 包含四个固定注入项：

| 注入编号 | 故障 | 业务码 |
| --- | --- | --- |
| `INJ-001-truncated-receipt` | 截断 r1 定宽行 | `E_RECEIPT_TRUNCATED` |
| `INJ-002-unknown-param-version` | 写入工人未知参数版本 999 | `E_VERSION_UNKNOWN` |
| `INJ-003-nonfinite-latency` | 写入 `NaN` 耗时 | `E_RECEIPT_NONFINITE_LATENCY` |
| `INJ-004-duplicate-request-id` | 请求流追加重复请求标识 | `E_SCHEMA_INVALID` |

报告中的 `injection_results` 会给出每个编号的 `applied`、`observed`、`business_code` 和匹配失败位置。单条回执解析失败只追加失败清单，不阻断整批。

## 11. 实测结果

实际验证环境为提供的 Linux/WSL 工作区：

- Python 3.11.0rc1
- numpy 1.26.4
- pytest 8.2.0

已执行：

```text
python baseline.py init
python baseline.py verify
python -m pytest -q
```

结果：

- `15 passed`。
- `incremental_restart_matches_full = true`。
- `budget_pending_demonstrated = true`，首次受限归一待办 `15` 条。
- `pending_then_complete_matches_full = true`。
- `injections_observed = true`，4/4 注入均映射到预期业务码。
- `threaded_matches_single_thread = true`。
- 参考账本回执计数为 r1=3、r2=9、r3=10、total=22。
- 最终参考账本有 18 条归属、8 条批次指标、1 个显式冲突、1 个不可判定项，最终待办为 0。

端到端 wall-clock 耗时写入各 `report.json` 的 `end_to_end_elapsed_wall_ms`；逐字段比对只比较确定性账本，不比较该本机耗时。回执中的整数耗时统计和顺序仍参与逐字段比较。
