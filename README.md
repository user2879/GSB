# 本地特征服务诊断工作台

这是一个面向推荐与风控场景的确定性本地工作台。它从本地样本出发，维护定点特征值、增量失效范围、融合打分、阶段化快照和可恢复进度，并用“增量结果 == 空白状态全量重建结果”作为核心验收条件。

工作台不访问线上服务，不使用 Docker/容器或远程执行。所有样本、状态和报告都在本地目录中生成。

## 交付内容

- `feature_workbench/`：加载校验、定点计算、拓扑调度、内容指纹、融合打分、快照和 CLI。
- `scripts/generate_samples.py`：固定种子 `188` 的样本生成器。
- `samples/`：主样本、低预算配置和独立异常注入样本。
- `tests/test_workbench.py`：13 个 pytest 用例，覆盖增量/全量一致、随机编辑序列、并发、预算、恢复、信号和异常隔离。
- `artifacts/diagnostic_report.json`：最近一次端到端诊断报告。
- `artifacts/pytest.txt`：最近一次 pytest 输出。

## 环境与安装

实际执行环境记录：

- 执行器：WSL2 Ubuntu 22.04，Linux 内核 `6.18.33.2-microsoft-standard-WSL2`。
- Python：`3.11.0rc1`。
- NumPy：`2.4.6`。
- pytest：`8.4.2`。
- PyYAML：`6.0.3`。
- 系统没有可用的原生 Windows Python 进程（WSL 互操作未启用），因此本次实际测试不是原生 Windows 会话。

代码使用 `pathlib`、相对路径和原子 `os.replace`，并对 Windows 不支持的目录 `fsync` 做了降级处理。Windows 原生环境可按下面的方式建立独立虚拟环境并运行：

```powershell
py -3.11 -m venv .venv-win
.\.venv-win\Scripts\python -m pip install -r requirements.txt
.\.venv-win\Scripts\python scripts\generate_samples.py --output-dir samples
.\.venv-win\Scripts\python -m feature_workbench diagnose --sample-dir samples --work-dir .workbench --report artifacts\diagnostic_report.json
```

Linux/WSL 环境：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/generate_samples.py --output-dir samples
.venv/bin/python -m feature_workbench diagnose --sample-dir samples --work-dir .workbench --report artifacts/diagnostic_report.json
.venv/bin/python -m pytest -q
```

依赖全部来自公开包源，版本范围见 `requirements.txt`。

## 输入格式

### 特征定义表 `definitions.json`

```json
{
  "features": [
    {
      "name": "user_amount_sum_10m",
      "side": "user",
      "source_field": "amount",
      "window_ms": 600000,
      "aggregate": "sum",
      "decimal_places": 4,
      "quantization_step": "0.0001",
      "upstream_features": []
    }
  ]
}
```

- `side`：`user`、`item` 或 `both`，默认 `both`。
- `aggregate`：`count`、`sum`、`mean`、`latest`。
- `decimal_places`：定点小数位宽，取值 `0..18`。
- `quantization_step`：量化档位，量化后的值必须在给定位宽上可精确表示。
- `source_field`：原始事件上下文中的数值字段；计数特征可为空。
- `upstream_features`：同侧依赖特征名，必须存在且依赖图无环。
- 同时配置 `source_field` 和上游特征时，特征值是原始聚合值与上游定点值之和；仅上游特征可作为派生特征。

### 事件流 `events.jsonl`

每行一个事件：

```json
{"event_id":"evt-000001","user_id":"u-001","item_id":"i-001","behavior":"view","timestamp_ms":1700000015000,"context":{"amount":23.5,"risk_score":0.21}}
```

事件 ID 必须唯一，用户/物品 ID 非空，`timestamp_ms` 按到达顺序严格递增，上下文每个值都必须是有限数值。行为类型来自配置的 `allowed_behaviors`。`_injection_id` 是仅用于诊断的扩展字段，不参与特征内容指纹。

### 打分配置 `scoring.yaml`

```yaml
weights:
  user_amount_sum_10m: 0.5
  item_risk_mean_10m: -0.25
fusion_temperature: 2.0
fusion_bias: 0.05
clamp_interval: {min: -50.0, max: 50.0}
concurrency: 4
point_budget_per_batch: 5000
snapshot_commit_interval: 1
score_decimal_places: 4
score_quantization_step: "0.0001"
allowed_behaviors: [view, click, favorite, cart, purchase, refund]
```

打分公式：

```text
score_raw = fusion_bias + sum(weight[f] * feature_value[f] / fusion_temperature)
score = quantize_half_even(clamp(score_raw, clamp_min, clamp_max), score_quantization_step)
```

特征求和使用特征名字典序；贡献分解也按同一顺序输出。排序键为：得分降序、参与事件数降序、实体 ID 字典序、实体类型字典序。

### 增量编辑计划 `edits.json`

```json
{
  "batches": [
    {
      "batch_id": "append",
      "appends": [],
      "backfills": [],
      "definition_changes": []
    }
  ]
}
```

- `appends`：新增事件，到达时间必须晚于当前事件游标。
- `backfills`：历史内容修正，使用 `event_id` 加 `patch`/`replacement`；事件 ID、用户、物品和时间戳不可改变。
- `backfills[].late_event`：回填迟到事件；时间戳早于或等于已完成游标时输出 `E_EVENT_INVALID`，由注入样本 `INJ-003` 验证。
- `definition_changes`：`{"op":"upsert","definition":{...}}` 或 `{"op":"remove","name":"..."}`。同一批的定义变更整体校验，出现非法定义或依赖环时整批定义变更被拒绝。

## 调度与确定性

### 特征计算

- 当前时间游标是已接受事件的最大时间戳；窗口为闭区间 `[cursor - window_ms, cursor]`。
- 事件先按时间戳升序、再按事件 ID 字典序处理。
- 用户侧和物品侧分别计算，实体集合来自当前已接受事件。
- `count` 计数窗口内事件；`sum`/`mean` 使用原始数值字段；`latest` 使用窗口内最后一个带有该字段的事件。
- 定点值使用十进制量化档位和 `ROUND_HALF_EVEN`，所有 Decimal 聚合使用 80 位上下文，避免宿主默认精度影响。
- 特征输出包含 `value_fixed`、`decimal_places`、`value_decimal`、参与事件 ID、事件窗口指纹、参数指纹、上游指纹、缓存键和内容指纹。

### 指纹与缓存

每个特征计算由四层指纹组成：

1. 定义参数指纹：名称、侧、来源字段、窗口、聚合、精度、量化档位、上游列表。
2. 事件窗口指纹：当前游标、窗口毫秒、窗口内事件内容指纹及实体。
3. 上游指纹：上游特征名和上游内容指纹。
4. 特征缓存键/内容指纹：由参数、事件窗口、上游指纹和定点结果派生。

融合得分缓存键由评分语义指纹与所有特征内容指纹派生；执行并发度、点预算和快照间隔不进入评分语义指纹，因此改变并发不会改变得分内容指纹。

### 失效范围

- 事件追加会移动游标，可能改变所有窗口，因此按全量候选进行指纹比对；键未变化的项直接复用。
- 历史回填只标记“覆盖该事件的窗口”对应的用户/物品侧特征，再闭包传播到全部下游特征；没有覆盖该事件的特征保留缓存。
- 定义变更只标记该特征及依赖它的下游特征；未失效特征不执行计算。
- 预算不足的项进入 `pending`，带有 `E_POINT_BUDGET_PENDING`，下一批优先继续；不会静默丢弃。

### 并发

特征计算和评分均通过 `ThreadPoolExecutor` 执行。特征任务稳定键包含拓扑位次、特征名、首个事件 ID、侧和实体 ID；结果按稳定键回收并排序。因此并发 `1/4/8` 的最终 JSON 投影一致。

## CLI

增量运行：

```bash
.venv/bin/python -m feature_workbench run \
  --definitions samples/definitions.json \
  --events samples/events.jsonl \
  --config samples/scoring.yaml \
  --edits samples/edits.json \
  --state-dir .workbench/incremental_state \
  --report artifacts/incremental_report.json
```

异常注入运行：

```bash
.venv/bin/python -m feature_workbench run \
  --definitions samples/injections/definitions.json \
  --events samples/injections/events.jsonl \
  --config samples/scoring.yaml \
  --edits samples/injections/edits.json \
  --injection-config samples/injections/injections.json \
  --state-dir .workbench/injection_state \
  --report artifacts/injection_report.json
```

端到端诊断（增量、空白全量重建、并发矩阵、异常注入）：

```bash
.venv/bin/python -m feature_workbench diagnose \
  --sample-dir samples \
  --work-dir .workbench \
  --report artifacts/diagnostic_report.json
```

退出码：`0` 成功或收到可处理信号并完成当前批；`1` 诊断不等价或有检查失败；`2` 配置无法加载；`3` 仍有预算挂起项（优雅停止仍返回 `0`）。

## 快照与恢复

状态目录结构：

```text
.workbench/incremental_state/
  manifest.json
  stages/v<global_version>-<uuid>/
    load.json
    features.json
    scores.json
    progress.json
```

提交过程先把全部阶段产物写入 `.tmp-*` 目录并 `fsync`，再把临时目录整体改名为不可变阶段目录，最后原子替换 `manifest.json`。清单包含：

- `global_version`；
- `config_hash`、基础定义哈希、当前定义哈希、基础事件哈希；
- 计划哈希、事件游标、已完成批次数；
- 各阶段产物版本、SHA-256 指纹和状态。

续跑会同时核对 schema、全局版本、配置哈希、基础定义/事件哈希、计划哈希、清单游标、阶段文件哈希，以及 `progress.json` 与清单的一致性。任一不一致输出 `E_CONFIG_MISMATCH`，拒绝复用旧快照并从空白状态重建。

`SIGINT`/`SIGTERM` 处理只设置停止标志；当前批计算和原子快照提交完成后以 `0` 退出。`SIGKILL` 后只可能遗留旧清单、完整阶段目录或临时目录；下一次启动只读取清单指向且哈希完整的一组产物。

## 业务码与注入项

| 注入编号 | 场景 | 业务码 |
|---|---|---|
| `INJ-001` | 非有限上下文数值 | `E_EVENT_INVALID` |
| `INJ-002` | 未知行为类型 | `E_EVENT_INVALID` |
| `INJ-003` | 回填时间戳早于已完成游标 | `E_EVENT_INVALID` |
| `INJ-004` | 特征依赖成环 | `E_DEFINITION_INVALID` |
| `INJ-005` | 量化档位越界/不可表示 | `E_DEFINITION_INVALID` |
| `INJ-006` | 单特征计算抛异常 | `E_FEATURE_COMPUTE` |
| `INJ-007` | 单实体评分抛异常 | `E_SCORE_COMPUTE` |

单个特征/实体异常不会中止其他项。失败记录包含阶段、特征名、实体、业务码和异常类型；下游不可用记录 `E_UPSTREAM_UNAVAILABLE`，评分不可用记录 `E_SCORE_UNAVAILABLE`。失败清单按稳定字段排序。

预算挂起项单独输出，示例：

```json
{
  "feature_name": "user_amount_sum_10m",
  "side": "user",
  "entity_id": "u-003",
  "event_ids": ["evt-000004", "evt-000011"],
  "stage": "feature",
  "code": "E_POINT_BUDGET_PENDING"
}
```

## 实际验证结果

生成与主诊断命令均以退出码 `0` 完成。最近一次结果：

- 增量与空白全量重建：`equal=true`，差异路径为空。
- 投影指纹：`518aa11133a754da4f1e65c9ebcda10570736b2fedbc8297afe47458512cc5ba`。
- 特征记录：增量 `56`，全量 `56`。
- 得分记录：增量 `14`，全量 `14`。
- 失败记录：增量 `0`，全量 `0`。
- 并发一致性：并发 `1/4/8` 均 `equal_to_full=true`。
- 定点位宽覆盖：主样本与定义变更阶段覆盖 `0/2/4/6/8` 等位宽档位；参数化诊断另覆盖 `+1/+2` 位宽偏移。
- 注入命中：`7/7`，全部映射到上表业务码。
- 回填精确失效：重算 `8` 个特征单元，复用 `48` 个；仅 `u-001`/`i-001` 的覆盖窗口及下游特征重算。
- 定义变更精确失效：重算 `8` 个 `user_value_composite_10m` 单元，复用 `48` 个未失效特征。
- 低预算：首批出现待算项，后续空批追平，最终 `pending=[]`，与无限预算全量结果一致。
- 快照恢复：同配置续跑 `resumed=true`；修改并发配置后续跑被拒绝并输出 `E_CONFIG_MISMATCH`。
- 优雅信号：完成当前批快照后退出码 `0`；强杀后仅从完整阶段目录和清单恢复。
- 随机合法编辑序列：3 组种子 × 并发 `1/4/8`，均与全量重建一致。
- pytest：`13 passed`（最近一次执行约 `14.09s`）。

详细逐字段报告见 `artifacts/diagnostic_report.json`，各运行报告在 `.workbench/`（运行产物目录，已被 `.gitignore` 排除）。
