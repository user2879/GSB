# FlowLedger：本地确定性流量采样与热点键账本

从零实现的 Go 本地管线，面向 Windows 工作区运行。它从 JSON 声明生成确定性事件流，执行键级稳定哈希头部采样，按固定整数分钟窗口聚合热点键，在全局并发上限与每客户端键配额下调度窗口任务，并把逐事件判定、聚合中间量、排名、误差界、任务时钟和进度游标提交到 SQLite。

运行时不需要 Docker、外部服务或网络。SQLite 使用纯 Go 驱动 `modernc.org/sqlite v1.34.5`，构建产物可直接在 Windows 上读写工作区数据库。

## 交付内容

- `cmd/flowledger`：`run`、`verify`、`replay`、`show`、`history` 命令。
- `internal/model`：规范加载、强校验、确定性生成、稳定采样、窗口聚合、误差界。
- `internal/ledger`：SQLite schema、单窗口原子提交、任务状态、摘要与恢复状态。
- `internal/engine`：双预算调度、崩溃恢复、版本演进、当前视图替换、验证与重放。
- `samples/`：固定的 v1/v2 流量与策略声明。
- `scripts/demo.sh`：Linux/WSL 演示脚本。
- `scripts/demo.ps1`：原生 Windows PowerShell 演示脚本。
- 自动化测试覆盖规范拒绝路径、确定性摘要、并发/乱序采样一致性、低采样阈值、崩溃恢复、并发预算、版本防护、重放篡改定位。

## 构建与 Windows 准备

要求 Go 1.22 或更高。当前实现已在 Go 1.23.6 上构建和测试。

首次准备依赖时联网执行一次：

```powershell
go env -w GOMODCACHE=C:\path\to\your\go-mod-cache
go mod download
```

之后构建与演示可断网执行：

```powershell
go build -o bin\flowledger.exe .\cmd\flowledger
powershell -ExecutionPolicy Bypass -File .\scripts\demo.ps1
```

Linux/WSL：

```bash
go mod download
./scripts/demo.sh
```

两个演示脚本都会设置 `GOTOOLCHAIN=local` 与 `GOPROXY=off`，验证后续运行不依赖网络。SQLite 文件可放在工作区内；正常使用会产生 `ledger.db`、`ledger.db-wal` 和 `ledger.db-shm`，强制终止后由 SQLite WAL 恢复已提交事务。

## 命令

```text
flowledger run     --spec FILE --policy FILE --ledger FILE [options]
flowledger verify  --ledger FILE [--output FILE]
flowledger replay  --ledger FILE [--output FILE]
flowledger show    --ledger FILE [--output FILE]
flowledger history --ledger FILE [--output FILE]
```

`run` 关键参数：

- `--report FILE`：成功后写完整报告；崩溃/失败不写。
- `--k`：每窗口热点键数量，必须为正。
- `--window-minutes`：固定窗口分钟数，必须为正。
- `--global-concurrency`：同时运行的窗口任务上限。
- `--per-key-concurrency`：任一客户端键同时参与运行的窗口任务上限。
- `--min-sample-events`：估计可用性阈值。
- `--task-delay-ms`：用于本地观察并发与排队的任务延迟。
- `--crash-after N`：崩溃注入。达到 N 个完成窗口后，在已持久化下一个 `running` 任务时立即退出，不写报告。由于在途任务会先排空，实际完成数可能略高于 N。

退出码：

- `0` 成功。
- `64` `E_SPEC` 规范/参数/版本错误。
- `65` `E_REPLAY` 重放不一致。
- `66` `E_VERIFY` 账本自洽或预算校验失败。
- `70` `E_LEDGER`/`E_IO` 本地账本或 I/O 错误。
- `75` `E_CRASH` 注入的强制终止。

## JSON 声明

流量文件示例：

```json
{
  "spec_version": 1,
  "sources": [
    {
      "source_id": "src-east",
      "seed": 101,
      "event_count": 90,
      "start_timestamp_ms": 1700000000000,
      "step_ms": 2000,
      "keys": ["hot", "hot", "hot", "warm", "cold"],
      "weight_min": 1,
      "weight_max": 9
    }
  ]
}
```

校验规则：

- `source_id` 唯一，字符集为 `A-Z a-z 0-9 . _ : -`，长度 1..64。
- `seed`、`event_count` 为正整数；`start_timestamp_ms` 非负；`step_ms` 为正。
- 客户端键字符集同上，不以 `-` 开头，长度 1..128；源内键列表非空。
- `weight_min > 0` 且 `weight_max >= weight_min`。
- 未知字段、非整数、数组/对象类型错误、尾随 JSON 均拒绝。
- 任一错误以 `E_SPEC` 整体拒绝，并输出 JSON Pointer 字段路径，如 `/sources/1/seed`。

策略文件示例：

```json
{
  "policy_version": 1,
  "salt": "flowledger-demo-v1",
  "strategies": [
    {"strategy_id": "east-head", "source_ids": ["src-east"], "ratio_bp": 5500}
  ]
}
```

`ratio_bp` 为 `0..10000` 的整数万分比。每个源必须且只能被一个策略覆盖。策略版本、盐、策略 ID 和源覆盖均强校验。

## 确定性事件流

- 每个源使用显式 SplitMix64 状态推进；同一 `seed`、键列表和参数在任意进程中得到同一事件序列。
- 时间戳为 `start_timestamp_ms + i * step_ms`，严格有序；事件 ID 为 `source_id#000000000001` 形式。
- 对源内事件按字段做长度前缀 SHA-256，得到源摘要；所有源摘要按源 ID 稳定组合为事件流摘要。
- 全程不使用 Go `math/rand` 的隐式版本行为，也不使用系统时间或随机数。

## 采样判定

系统使用 10001 个整数桶，编号 `0..10000`：

```text
B = 10001
T = floor(ratio_bp * B / 10000)                 # 加载时量化，截断方向为 floor
bucket = BE64(SHA256(
  "flowledger/head-sample/v1" || policy_version || salt || strategy_id || client_key
)) % B
sampled = bucket < T
```

因此比例为 `0` 时阈值为 0（不采样），比例为 `10000` 时阈值为 10001（全采样）。同一 `policy_version + salt + strategy_id + client_key` 的桶与判定固定，与事件顺序、窗口、调度顺序无关。同一客户端键在相同策略作用域内始终得到相同判定；不同策略作用域可因阈值不同而不同。

逐事件判定表记录来源、序号、窗口、时间戳、键、权重、策略、策略版本、桶、阈值和 `sampled`，并写 `decision_digest`。窗口分别计算：

- `decision_set_digest`：窗口内全部逐事件判定摘要。
- `sample_set_digest`：仅被采样事件判定摘要的稳定组合。
- 运行级 `sample_set_digest`：各已完成窗口样本集合摘要的稳定组合。

## 窗口聚合与排名

- 窗口起点为 `floor(timestamp_ms / window_ms) * window_ms`，只创建含事件的窗口。
- 对每个采样事件按键累加采样计数与采样权重。
- 中间量按客户端键字典序稳定存储。
- 排名主分数为 Horvitz–Thompson 估计计数。顺序为：
  1. `estimated_count_milli` 降序；
  2. `sample_weight` 降序；
  3. `sample_count` 降序；
  4. `client_key` 升序。
- 不存在依赖 map 遍历顺序的输出；乱序输入会先按 `(timestamp, source_id, event_seq)` 规范化。

## 估计与误差界

采样是键级聚类采样：某个键被选中时，该键在对应策略/源中的事件全部可见。报告同时保留两类信息，避免混淆：

- `sample_count` / `sample_weight`：样本中的精确观测值。
- `estimated_count_milli`：全量计数的 HT 缩放估计，单位 1/1000 计数。

单策略贡献的估计为 `C * B / T`。误差包络分别使用 `T+1` 和 `max(T-1,1)` 作为分母，覆盖两部分合成误差：

- 万分比到 10001 桶的 `floor` 截断误差；
- 稳定哈希分桶粒度 `1/10001`。

窗口报告同时给出 `ratio_truncation_error_numerator`、`hash_bucket_granularity_*`、`combined_error_*`、`envelope_*` 和整数化的 `quantization_error_bound_milli`，验证工具会逐项复算。

真实计数使用保守区间：

```text
observed_count_lower = sample_count
observed_count_upper = sample_count + unseen_event_count
```

`unseen_event_count = total_event_count - sampled_event_count`。该上界把所有未采样事件都视为可能属于同一竞争键，因此不会低估。

前 K 包含判定：按估计排名取 K 个候选后，若第 K 个候选的 `observed_count_lower` 严格大于“所有非候选键最大采样计数 + 未采样事件数”，则标记 `top_k_guaranteed=true`，因为任何未观测或观测较少的键都不可能超过它。严格不等号意味着边界并列不宣称保证。该判定是确定性的保守充分条件，不是概率置信区间。

若窗口采样事件数低于 `min_sample_events`：

- `estimate_applicable=false`；
- `ranking` 为空数组；
- 写入 `inapplicability_reason`；
- `top_k_guaranteed=false`。

采样中间量仍保留用于诊断，但不会冒充全量排名结论。

## 双预算调度

每个窗口任务需要：

- 1 个全局并发槽；
- 该窗口内每个不同客户端键各 1 个键配额。

调度器只从 `queued/unknown` 状态启动任务，启动前同时检查全局槽和所有键配额；不满足则保持排队。任务记录包含：

- `queue_start_mono_ms`、`start_mono_ms`、`end_mono_ms`、`queue_duration_ms`、`duration_ms`；
- `global_quota=1` 与排序后的 `key_quota`；
- `segment_id`、尝试次数、是否从 `unknown` 恢复。

时钟基准是每个进程调度段的本地单调计数毫秒。重启会创建新调度段；验证工具按段扫描区间，不把不同进程的单调计数直接混用。已验证全局与每键实际并发均不超过配置上限。

## SQLite 账本与恢复

账本包含整数格式版本 `ledger_format_version=1`，主要表：

- `runs`：规范/策略/清单/选项摘要、版本、当前/陈旧标记、进度、运行内容摘要、持久化报告。
- `scheduler_segments`：调度段与单调时钟基准。
- `window_tasks`：窗口任务状态、时钟、排队、配额、结果摘要。
- `event_decisions`：逐事件采样判定。
- `window_results`：窗口统计、采样摘要、排名摘要、完整结果 JSON。
- `key_aggregates`：窗口内按键中间量、估计、误差界与排名。

每个窗口完成时在**同一事务**中写入逐事件判定、按键中间量、窗口结果、排名/误差摘要、任务完成状态、尝试信息和运行进度游标，并重算运行内容摘要。任务状态机：

```text
queued -> running -> completed
             |
             +-- 进程终止 --> 重启时转为 unknown --> 重跑 --> completed
```

若进程在任务开始后终止，账本最多保留完整提交的前缀结果；重启先把遗留 `running` 标为 `unknown`，只按完整记录恢复，然后重跑未知窗口。未完成运行不能 `verify` 或 `replay`，也不会产生成功报告。最终任务记录保留 `recovered_from_unknown=true`，验证报告输出 `recovered_unknown_tasks`。

## 版本演进与陈旧防护

- 规范内容变化时 `spec_version` 必须严格增大；策略内容变化时 `policy_version` 必须严格增大。
- 版本号不得下降，即使对应内容未变。
- 新运行创建时，旧当前运行在同一事务中变为 `is_current=0, stale=1`；当前报告只读取新运行，不合并旧结果。
- 变化清单稳定列出新增/删除/修改的源、逐源策略与比例变化、受影响窗口集合，并声明 `overall_replacement=true`。
- 策略版本或盐变化会改变哈希域，因此受影响窗口集合包含所有源窗口。
- 账本中的历史报告 JSON 是不可变审计快照；其新鲜度以 `runs.is_current/stale` 为权威。`history` 显示旧运行已陈旧。

## 验证与重放

`verify` 独立读取当前账本并检查：

- 账本整数格式、当前运行唯一性、历史运行陈旧标记；
- 清单、规范、策略、报告和运行内容摘要；
- 任务状态、单调时钟、排队时长、进度游标；
- 按调度段扫描的开始/结束区间，全局和每键并发不越限；
- 每个事件的稳定哈希桶、阈值、采样标记、事件顺序、事件 ID 和判定摘要；
- 窗口计数、样本摘要、中间量、排名、并列决胜、误差界与 top-K 保证；
- 去规范化表与结果 JSON 一致。

不一致返回 `E_VERIFY`。

`replay` 从账本内保存的规范、种子、策略版本和参数重新生成事件流，重算全部已完成窗口，然后按窗口与规范化事件顺序比较。首个差异通过 `E_REPLAY` 返回：

```json
{
  "code": "E_REPLAY",
  "message": "replay mismatch",
  "details": {
    "source_id": "src-east",
    "window_start_ms": 1699999980000,
    "field": "event_decisions[0]/weight",
    "expected": 5,
    "actual": 6
  }
}
```

自动化测试会篡改一条已提交事件的权重，并确认上述源、窗口与字段定位。

## 已执行验证

使用 `samples/traffic-v1.json`、`samples/policy-v1.json`、`samples/traffic-v2.json`、`samples/policy-v2.json` 执行完整演示：

- v1：4 个窗口、165 个事件；`replay` 与 `verify` 均成功。
- v2：5 个窗口、240 个事件；新增 `src-north`，修改 `src-east`/`src-west`，比例和策略版本变化。
- 注入终止：退出码 `75`，未生成 `v2-report.json`；账本保留已提交窗口和一个 `running` 任务。
- 未完成时：`verify` 返回 `E_VERIFY`（退出 66），`replay` 返回 `E_REPLAY`（退出 65）。
- 重启：恢复 1 个 `unknown` 任务并完成全部 5 个窗口。
- 最终 `verify`：当前恢复运行 `current_runs=1`、`stale_runs=1`、`recovered_unknown_tasks=1`；全局实测并发 2（上限 3），每键实测并发 2（上限 2）。
- 最终 `replay`：5 个窗口、240 个事件逐项一致。
- Windows 交叉编译：`GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build` 成功生成纯 Go 可执行文件。
- v1 稳定摘要：`4d383a1e2063340079d646c2bb64810e27e462cd45b7f2593883091595815437`
- v2 稳定摘要：`5e5e8632ccd27bb7e0b2a89bf5214f2115047f5fd4c90aa63e4b50cf1b300c99`
- v2 样本集合摘要：`715cf773cbb605e8c49e78c7bacf8f5452df3f1dd6ea13e3a9d507b85ca3eac7`
- v2 运行内容摘要：`fc1b6dfbbbc5f2ee0caf56617ad7807b19cd0fa63474b999ff47502e5722d339`

运行：

```bash
GOTOOLCHAIN=local GOPROXY=off go test ./... -count=1
./scripts/demo.sh
```

测试输出与演示账本位于带时间戳的 `out/demo-*` 目录；`out/` 与构建产物不会进入版本控制。

## 设计边界

- 窗口任务为窗口级全键任务；它一次占用窗口内全部键配额，而不是在窗口内部再拆分键任务。这保证不绕过每键预算，但可能比细粒度调度更保守。
- 只有含事件的窗口被创建；分钟空洞不会生成空窗口记录。
- 全量估计使用聚类采样的 HT 缩放；样本内观测计数与缩放估计分开呈现，top-K 保证使用观测上下界而非把缩放估计当真实计数。
- 调度时间戳用于本地可核验记录，不用于业务排序或结果摘要；稳定结果摘要不包含调度时序。
