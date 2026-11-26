# HopLedger 设计说明

HopLedger 是一个仅依赖本地回环的定点整数诊断包。链路顺序固定为：

`client -> fwd1(forward) -> rev2(reverse) -> origin`

父进程按声明端口启动每个代理跳/源站子进程；调度器在线程池中通过真实
`127.0.0.1` TCP JSON Lines 套接字逐跳调用。不存在用 `sleep` 冒充套接字超时
的路径：`E_TIMEOUT` 只从 `socket.create_connection` 或 `socket.settimeout`
后的真实读写超时转换而来。

## 模块地图

- `src/hopledger/specs.py`：`chain.json`、`origin.json`、`diagnostic.json`
  的加载、交叉校验和 `E_SPEC`。
- `src/hopledger/protocol.py`：JSONL 编码、真实握手、回执必填字段校验，
  以及 `E_TIMEOUT`/`E_FIELD`。
- `src/hopledger/nodes.py`：代理跳和源站子进程入口
  （`python -m hopledger.nodes ...`）。
- `src/hopledger/proc.py`：子进程启动、端口占用识别、握手重试、`SIGKILL`
  和重启。
- `src/hopledger/fixedpoint.py`：整数缩放、截断和 int64 饱和规则。
- `src/hopledger/metrics.py`：链序归并、中间量、误差上界和主导跳归因。
- `src/hopledger/invalidation.py`：版本/缩放因子失效依赖图。
- `src/hopledger/scheduler.py`：全局在途上限和每跳配额调度。
- `src/hopledger/ledger.py`：SQLite 原子批次、游标、格式版本和摘要。
- `src/hopledger/snapshot.py`：会话快照版本识别、整体替换和只读备份。
- `src/hopledger/replay.py`：全量运行与恢复结果的逐项比较及 `E_REPLAY`。
- `src/hopledger/runner.py`：把上述能力串成可集成的 `Runner` API。

## 声明校验

任一 schema、唯一性、取值、角色、缩放幂、拓扑或首尾衔接错误均抛出
`SpecError`，稳定错误码为 `E_SPEC`。错误对象包含首个非法位置的
`hop_id` 与 `field`。重复 ID/端口按声明顺序中第一个参与重复的跳上报。

缩放因子必须是正的 10 的整数幂：`1, 10, 100, ...`。诊断会话缩放因子还必须
出现在链声明的 `allowed_scale_factors` 中。

## JSONL 回执

每行一个 JSON 对象。成功回执至少包含：

- `hop_id` 与通用 `node_id`（二者对同一节点相同）
- `request_id`
- `config_version`
- `scale_factor`
- `monotonic_ms`
- `queue_ms`、`process_ms`、`forward_ms`

代理跳使用 `time.monotonic_ns()` 测量自身处理耗时；调度器另外记录排队等待的
单调时钟毫秒。单调时间是运行遥测，不进入确定性端到端数值结论。

## 定点口径

设跳缩放因子为 `f_h`，会话缩放因子为 `S`，毫秒量为 `m`：

1. 跳本地定点值：`L = m * f_h`
2. 统一到会话口径：`U = floor(L * S / f_h)`

所有乘除均为 Python 整数运算。非负量使用 floor division；对非负输入等价于
向零截断。每次除法的 `remainder_numerator` 与截断方向均写入结论。

溢出规则为饱和到有符号 int64：

- 上限：`2^63 - 1`
- 下限：`-2^63`

正常样本远低于上限；若发生饱和，节点输出 `"saturated": true`。

每跳节点值为 `queue + process + forward`，端到端值严格按
`client -> hops... -> origin` 的链序累加。结论包含每个组件、每跳累计值、
节点累加顺序、操作序列和所有 `config_version`。对象使用排序键的紧凑 JSON
规范化编码，因此回执到达/完成顺序不影响字节结果。

## 误差上界与归因

每个节点有三段整数上界（会话定点单位）：

- `quantization_truncation_bound_units = 3 * ceil(S / f_h)`：三个毫秒分量
  的单刻度量化上界。
- `chain_accumulation_bound_units = 3`：三次整数 rescale floor 的累计 ULP
  上界；整数加法本身结合律成立，不引入额外舍入。
- `timeout_estimation_bound_units = S * max(1, forward_ms)`：转发/超时估计
  偏差的保守上界。真实套接字超时不会被伪造成度量，而是令当前未提交批次保持
  `planned/unknown` 并抛出 `E_TIMEOUT`。

`E_h` 为三段之和，端到端上界为链序求和。贡献占比以整数 basis points 按最大
余数法精确分配，总和恒为 `10000`。

默认主导判定：

- `dominance_threshold_bp = 6000`
- `indistinguishable_margin_bp = 500`

最高占比达到阈值且领先第二名至少 margin 时，`primary_cause` 为
`dominant_hop_identified`；否则为 `error_indistinguishable`。若当前声明版本
与回执/结论引用冲突，最高优先级成因为 `config_version_conflict`；若任一跳
配置或观测的失败注入 basis points 达到 `high_failure_ratio_bp`（默认 5000），
成因为 `failure_injection_ratio_high`。所有适用标记仍保留在 `cause_codes`。

## 失效传播

每个请求的依赖节点为：

`receipt -> metric -> accumulation -> later accumulations -> e2e -> attribution`

配置版本或跳缩放因子变化会直接失效对应 `receipt`，因此需要真实重新下发；
会话 `scale_caliber_version` 变化只直接失效 `metric`，无需重新调用原始套接字。
某个累计值失效后，所有链序下游累计、e2e 和归因均为间接受影响节点。

`plan_invalidation()` 稳定输出：

- `invalid_nodes`
- `redispatch_nodes`
- `recompute_only_nodes`
- `affected_request_ids`
- 每一步的 `reason` 与 `propagated_from`

`Runner.apply_invalidation()` 只对 `redispatch_nodes` 发起套接字调用，只写
变化的回执行，并重算受影响结论/归因。提交前后的摘要索引逐项比较；非允许
集合中的回执、结论或归因摘要变化会列入 `digest_violations`。

## 并发调度

`DispatchScheduler` 使用 `ThreadPoolExecutor`：

- 全局 `BoundedSemaphore(max_in_flight)` 在整个请求链期间持有。
- 每个链节点一个 `BoundedSemaphore(hop_quota)`，仅在该跳真实调用期间持有。
- 获取信号量前后读取 `time.monotonic_ns()`，生成全局/每跳排队事件。
- 锁内维护实时和观测峰值；测试用慢假调用证明峰值不会超配额，端到端测试证明
  同样的信号量路径会发起真实子进程套接字调用。

## SQLite 账本与快照

每个批次先以独立事务写入 `batches(status='planned')`。调度和计算完成后，
`commit_batch()` 在同一个 SQLite 事务中写入：

- 逐跳回执
- 端到端结论
- 误差归因
- 调度事件
- 批次 `committed` 状态和摘要
- 进度游标
- 全账本内容摘要

账本 meta 含整数 `ledger_format_version=1` 和 SHA-256 `content_digest`。

快照是整体 JSON 状态。重新打开时比较：

- `hop_set_version`
- `scale_caliber_version`
- `content_digest`
- `ledger_format_version`

任一不一致都不会做字段级合并：旧快照先复制为模式 `0444` 的
`*.readonly.bak`，再通过临时文件 + `os.replace()` 整体替换。

## 强杀恢复与重放

代理跳可通过 `ProcessManager.kill_node()` 发送 `SIGKILL`。只有已经提交的
批次能恢复度量和归因；遗留的非 `committed` 批次由
`mark_unknown_batches()` 标记为 `unknown`，其请求必须重跑。系统不会生成
强杀时刻的退出报告。

重启节点后，`rerun_unknown_batches()` 按原批次 ID 重跑并在同一事务提交。
`compare_result_maps()` 对全量一次运行和恢复后运行逐项比较：

- 规范化回执（排除天然变化的 `monotonic_ms` 遥测）
- 结论 JSON
- 归因 JSON

发现差异时抛出 `ReplayError`（`E_REPLAY`），详情包含首个不一致的
`request_id`、`hop_id`、`field`、JSON 路径、期望值和实际值。

## 常用命令

```bash
python3 -m venv --without-pip .venv
. .venv/bin/activate
python /tmp/get-pip.py  # Debian/Ubuntu 缺少 ensurepip 时；正常 WSL 可直接 apt install python3.11-venv
pip install 'numpy>=1.26' 'pytest>=8'
pip install -e .
pytest -q

hopledger init-sample --dir artifacts/local-demo
hopledger run \
  --config-dir artifacts/local-demo/config \
  --ledger artifacts/local-demo/ledger.sqlite3 \
  --snapshot artifacts/local-demo/session-snapshot.json \
  --report artifacts/local-demo/run-report.json

python scripts/acceptance_demo.py
```
