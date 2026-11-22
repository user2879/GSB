# 确定性离线特征回填应用

这是一个只依赖本地文件和 Python 进程的领域数据编辑与回填应用。它在有限单批点数配额、实体租约、LRU 驻留预算和原子快照约束下，对确定性 JSONL 样本执行增量特征回填，并保证从已提交快照断点续跑的结果与最终样本集合的一次性全量回填逐字段一致。

## 交付内容

| 路径 | 说明 |
| --- | --- |
| `feature_backfill/` | 校验、窗口计算、内容指纹、租约、LRU 缓存、调度、快照和 CLI 实现 |
| `data/default/` | 固定种子生成的基准样本、特征定义、YAML 配置和 7 批计划 |
| `data/final/` | 计划执行后的最终样本与最终窗口参数，用于全量一致性基线 |
| `data/illegal/` | 可单独触发的坏行、坏 DAG、坏配置和坏计划夹具 |
| `injections/` | 四类运行时故障的独立注入清单及组合清单 |
| `tests/` | 28 项单元与端到端测试 |
| `results/acceptance/` | 实际运行报告、快照和 `summary.json` 验收摘要 |
| `docs/ARCHITECTURE.md` | 配额、租约、缓存、失效与快照设计 |
| `docs/INJECTIONS.md` | 非法输入、注入清单和单独触发方法 |
| `docs/VALIDATION_RESULTS.md` | 本机实际验证结果 |

## 环境

要求 Python 3.11，依赖仅 NumPy 与 PyYAML，锁定在项目内 `.venv`：

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.lock
```

如果系统 Python 缺少 `ensurepip`，可用官方引导脚本：

```bash
python3.11 -m venv --without-pip .venv
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py
.venv/bin/python -m pip install -r requirements.lock
```

当前锁定版本：`numpy==2.2.6`、`PyYAML==6.0.2`。应用运行时不访问网络、不使用 Docker、不连接外部服务。

重新生成全部确定性夹具：

```bash
.venv/bin/python scripts/generate_fixtures.py
```

生成器使用 NumPy PCG64 固定种子 `20261123`。

## 快速开始

校验默认输入：

```bash
.venv/bin/python -m feature_backfill validate \
  --samples data/default/samples.jsonl \
  --features data/default/features.json \
  --config data/default/config.yaml \
  --plan data/default/plan.json
```

执行 7 批增量计划：

```bash
.venv/bin/python -m feature_backfill run \
  --samples data/default/samples.jsonl \
  --features data/default/features.json \
  --config data/default/config.yaml \
  --plan data/default/plan.json \
  --snapshot-dir results/my-run/snapshots \
  --report results/my-run/report.json
```

对最终样本与最终窗口参数执行空计划全量回填：

```bash
.venv/bin/python -m feature_backfill run \
  --samples data/final/samples.jsonl \
  --features data/final/features.json \
  --config data/final/config.yaml \
  --plan data/final/plan.json \
  --snapshot-dir results/my-final/snapshots \
  --report results/my-final/report.json

.venv/bin/python -m feature_backfill compare \
  results/my-run/report.json results/my-final/report.json
```

`compare` 比较逐字段结果、实体汇总、跳过明细、待办队列、失败清单和租约版本。加入 `--execution` 还会比较实体分配顺序、任务排序键、配额扣减序列和完成批次。

## 一键验收

```bash
./scripts/run_acceptance.sh
```

脚本覆盖：合法/非法输入、真实租约过期、增量与全量一致、并行与串行一致、低配额待办、四类故障注入、真实 `os._exit(137)` 中断后续跑，以及 28 项测试。机器可读摘要：

```text
results/acceptance/summary.json
```

当前实测状态为 `passed`，摘要见 `docs/VALIDATION_RESULTS.md`。

## 输入格式

### 样本 JSONL

每行一个对象：

```json
{"row_id":"R-E001-001","entity_id":"E001","timestamp_ms":100,"amount":12.5,"qty":2,"score":0.31}
```

- `row_id` 全局唯一；`entity_id` 非空；`timestamp_ms` 非负整数；同一实体按文件顺序严格递增。
- 数值必须有限；`null`、配置的 `missing_marker`（默认 `__MISSING__`）或 `missing_fields` 中列出的字段表示缺失。
- 缺失值按窗口内固定规则跳过，并写入 `skips` 明细。

### 特征定义 JSON

```json
{
  "raw_fields": ["amount", "qty", "score"],
  "features": [
    {
      "name": "amount_sum_5000",
      "source_field": "amount",
      "window_ms": 5000,
      "aggregation": "sum",
      "depends_on": []
    },
    {
      "name": "amount_chain_5000",
      "source_feature": "amount_mean_5000",
      "window_ms": 5000,
      "aggregation": "mean",
      "depends_on": ["amount_mean_5000"]
    }
  ]
}
```

每个特征必须且只能声明 `source_field` 或 `source_feature`；`depends_on` 必须引用存在特征，引用图必须无环。支持 `count`、`sum`、`mean`、`latest`。

### 配置 YAML

```yaml
batch_rows: 1000
entity_shards: 4
batch_point_quota: 100000
lease_ttl_ms: 500
lease_renew_interval_ms: 100
thread_pool_concurrency: 4
snapshot_every_batches: 1
cache_resident_blocks: 16
clock_source: monotonic_ns
missing_marker: __MISSING__
lease_wait_timeout_ms: 5000
```

`clock_source` 只接受单调时钟来源：`monotonic_ns`、`process_monotonic`、`perf_counter_ns` 或 `performance_counter`。租约过期从不读取墙钟。

### 计划 JSON

计划由严格升序的批次组成，每批 `operations` 可为 `append`、`correct`、`window_adjust`、`rollback`。默认计划包含追加、单行修正、窗口调整、先追加后回滚到第 2 批，以及回滚后再次追加。

## 业务语义

- 窗口对目标时间戳闭区间 `[timestamp_ms - window_ms, timestamp_ms]` 生效，只读取不晚于目标行的样本。
- 累加使用 `math.fsum`，输出固定按实体 ID、时间戳、行 ID、特征名排序。
- 实体按 SHA-256 映射分片；单批按实体 ID 稳定排序分配整实体点数，累计不超 `batch_point_quota` 与 `batch_rows`。
- 租约通过配置的单调时钟记录签发与到期。冲突记录 `E_LEASE_CONFLICT` 并在条件变量上等待回收；续期只允许在到期点之前。
- 缓存块键由源行内容/上游块指纹、特征参数指纹和稳定租约版本派生。LRU 淘汰按访问序号，同序号以块键字典序兜底。
- 追加只新增该实体的新目标行块；单行修正只删除命中窗口及其依赖闭包；窗口调整令全部实体重新调度，并只失效被调特征及下游。
- 快照使用不可变代际目录，所有阶段文件写完后才原子替换 `CURRENT` 指针。未发布临时代不会被恢复逻辑读取。

详细设计见 `docs/ARCHITECTURE.md`。

## 退出码与报告

| 退出码 | 含义 |
| --- | --- |
| `0` | 完成且无失败 |
| `1` | `validate`/`compare` 发现非法项或不等价 |
| `2` | 配置、DAG、计划或快照兼容性拒绝运行 |
| `3` | 运行完成但存在失败实体或配额待办 |
| `137` | 仅崩溃测试钩子主动调用的 `os._exit(137)` |

完整报告包含 `results`、`summaries`、`skips`、`pending`、`failures`、`quota_allocations`、`quota_events`、`lease_events`、`lease_versions`、`invalidation_manifest`、缓存物理事件、复用/重算块清单和注入命中。

## 断点续跑

```bash
.venv/bin/python -m feature_backfill run ... --resume
```

恢复前同时校验：`CURRENT` 指针、清单自哈希、`state.json` 哈希、三个阶段的哈希、全局版本号、配置哈希、计划哈希、基础输入哈希、当前数据集/特征指纹、逐实体租约版本、已完成批次、待办队列和完成实体集合。任一不一致输出 `E_SNAPSHOT_STALE` 并拒绝在半成品上续跑。

崩溃测试钩子：

```bash
.venv/bin/python -m feature_backfill run ... --crash-before-commit-batch 3
```

它会在第 3 批窗口阶段写入未发布临时代后终止；下一次 `--resume` 只能看到第 2 批的完整提交。

## Windows 与临时目录

实现不对“替换已有非空目录”作平台相关假设。每个快照代先在根目录下的 `.tmp-*` 中完整落盘并 `fsync`，再重命名为 `generations/gen-N`，最后用 `os.replace` 原子替换小文件 `CURRENT`。这等价于整体替换，但规避 Windows 上非空目录替换的限制；崩溃遗留的 `.tmp-*` 永远没有指针，不会参与恢复。当前实测环境为 WSL2 Linux；该策略同时适用于 Windows 文件系统语义。

## 已知边界

- 低配额运行可能以退出码 `3` 结束并保留 `pending`；后续批次或再次续跑才会消费剩余实体。这是配额约束下的设计行为，不是静默完成。
- 缓存物理命中/淘汰轨迹属于执行审计。领域等价比较锁定结果、跳过、汇总、待办、失败与租约版本；缓存块指纹本身也只由上游内容、参数和租约版本决定。
- 本应用不实现分布式租约服务；租约是进程内线程安全实体所有权，进程终止后由已提交快照和重新签发恢复。
