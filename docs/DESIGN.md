# batchdiag 技术设计说明

## 1. 目标与边界

`batchdiag` 是一个纯本地、确定性的 Python 3.11 包，回答三个问题：

1. r1/r2/r3 三种历史回执能否归一为同一可复核账本；
2. 工作器参数版本变化时，能否沿
   `工作器参数版本 → 批次 → 请求归属` 依赖链精确判定失效边界；
3. 增量重算与从空白全量重建能否逐字节一致。

不使用任何容器、线上服务或真实账号；样本由固定整数种子生成，工作器是
127.0.0.1 高位端口上的真实 TCP 子进程。

## 2. 模块结构

| 模块 | 职责 |
| --- | --- |
| `errors.py` | `E_SPEC/E_FIELD/E_CHAIN/E_TIMEOUT/E_REPLAY/E_SNAPSHOT` 稳定错误码，携带字段路径、行号、来源文件 |
| `canonical.py` | 显式稳定序列化：键排序、无空白 JSON、float 固定 6 位小数、SHA-256 摘要 |
| `jsonlocate.py` | 保留源文件行号的递归下降 JSON 解析器，支撑“首个非法字段+行号” |
| `spec.py` | 请求流/工作器声明/归并规则加载与强校验 |
| `receipts.py` | r1/r2/r3 编解码、归一、补齐字段与补齐依据、字段/链路校验 |
| `planner.py` | 确定性切批与批次标识推导 |
| `model.py` | 固定种子 NumPy 两层 MLP 前向、参数摘要、顺序无关输出摘要 |
| `invalidation.py` | 依赖链失效传播、稳定差异报告、仅失效子集重算 |
| `worker.py` / `__main_worker__.py` | 回环 TCP 工作器子进程与带真实超时的客户端 |
| `scheduler.py` | 全局+每工作器双信号量的线程池调度与单调时钟排队记录 |
| `ledger.py` | SQLite 账本、单事务提交、格式版本/内容摘要、整体替换恢复、重放校验 |
| `engine.py` | 全量/增量/崩溃模拟/恢复编排 |
| `cli.py` / `gui.py` | 命令行入口与可选 Tk 仪表盘（无 Tk 时 CLI 降级） |

## 3. 稳定序列化与标识规则

* 规范字节：`json.dumps(sort_keys=True, separators=(",",":"), ensure_ascii=False)`，
  浮点统一为四舍五入 6 位小数字符串，拒绝 NaN/Inf，NumPy 标量/数组先转原生类型。
* 所有摘要都是规范字节的 SHA-256 十六进制截断（批次标识 12 位，其余 24/32 位）。
* 批次标识：`B:<worker_id>:<batch_seq>:<sha256(canonical({worker_id,batch_seq,arrivals}))[:12]>`。
  标识只依赖工作器标识、批序号与参与请求的到达序号，因此权重变化不会改变批标识，
  只有切批规则（如 batch_size）或请求归属变化才会改变。

## 4. 切批确定性

1. 请求按 `(arrival, request_id)` 全序排序；
2. 工作器按 `worker_id` 排序后 round-robin 分配请求；
3. 每个工作器按自身 `batch_size` 切批，批序号从 0 开始；
4. 批内按 `(arrival, request_id)` 排序得到稳定名次。

字典遍历顺序、派发顺序均不参与任何决策；测试以“逆序请求+逆序工作器”验证
逐字节一致的批划分与批标识。

## 5. 三格式归一

归一记录统一字段：`batch_id, request_id, output_digest, worker_id,
param_version, duration_ms, rank, duplicate_ref, param_digest`，并附带
`source_format`、`filled_fields`（字段名+补齐依据+补齐值）与源行号。

* r1 补齐：`worker_id` 依据 `batch_id` 前缀推导；`param_version=1`、
  `duration_ms=0` 取显式默认；r3 专属字段以 `absent:r1:<field>` 标记为空。
* r2 补齐 r3 的三个字段，依据 `absent:r2:<field>`。
* r3 无补齐字段。

校验分工：结构/取值错误（耗时为负、名次越界/重复、缺字段）报 `E_FIELD`；
引用关系错误（duplicate_ref 指向不存在请求、自引用、request_id 不在请求流）
报 `E_CHAIN`。错误均带数组下标字段路径与源行号。

## 6. 前向推理与摘要绑定

* 权重由 `np.random.default_rng(seed*1000003 + param_version)` 派生，
  网络为 `tanh(x W1 + b1) W2 + b2`，同规格任意时刻重建得到相同张量。
* 参数摘要绑定工作器种子、维度、batch_size、param_version 与全部权重。
* 批输出摘要绑定：批标识、排序后的请求标识集合、按请求标识排序的
  单请求输出摘要映射、参数版本与参数摘要。请求处理顺序变化不改变摘要
  （测试对批内逆序输入做对照）。
* 每条结果同时输出参数摘要、参与请求排序（rank_order 与 sorted_requests）
  与摘要依据字符串 `OUTPUT_DIGEST_BASIS`。

## 7. 失效传播

依赖链是显式三级：工作器声明 → 该工作器的批次 → 批内请求归属。

* 工作器种子、维度、batch_size、param_version、port、quota 任一变化，
  该工作器全部批次记为直接失效（`direct=true`），批内请求记为间接失效。
* 报告按 `(worker_id, batch_seq)` 稳定排序，并给出字段级 old/new 差异、
  直接/间接失效批、失效请求与传播轨迹（`worker->batch`、`batch->request`）。
* 仅失效子集重算：未失效批直接保留，重算结果与保留结果合并后与
  全量重建的规范字节比较，必须完全相等。

## 8. 并发调度与真实回环

* 全局 `Semaphore(global_concurrency)` + 每工作器
  `Semaphore(worker.quota)` 双约束，真实投射到 `ThreadPoolExecutor`；
  测试断言观测峰值不超过两类配额。
* 配额耗尽时批次在池内排队，记录单调时钟
  `time.monotonic_ns()//1_000_000` 的等待开始/结束、执行起止与等待毫秒。
* 工作器为 `python -m batchdiag.__main_worker__` 子进程，监听 127.0.0.1
  高位端口，使用 8 字节大端长度前缀的 JSON 分帧；启动前
  `ensure_port_free` 探测端口占用。
* 客户端 `socket.settimeout(timeout_ms/1000)` 是真实套接字超时；
  stall 工作器不回复时按 `E_TIMEOUT` 报告，附 host/port/timeout/elapsed。

## 9. SQLite 账本与快照

* 表：`schema_meta, batches, receipts, invalidation_traces, progress`。
* 每完成一批，在单个事务内写批次归属、归一回执、失效轨迹与进度游标，
  并更新整数格式版本 `LEDGER_FORMAT_VERSION=1` 与内容摘要。
* 内容摘要覆盖批次全列（含 output_digest/param_digest 等）、回执、
  失效轨迹与进度游标，任何列被篡改都会导致摘要不一致。
* 重开快照时版本或摘要不一致即**整体替换**（删除快照文件后按当前版本
  新建），绝不增量合并，并稳定列出差异来源
  （`schema_meta.format_version` / `schema_meta.content_digest`）。

## 10. 强制终止与重放

* `crash` 命令提交 N 批后对自身发 `SIGKILL`，当前批永不提交；
  未提交批在恢复时标记 `state=unknown, action=rerun_required`，
  不会生成退出报告。
* 恢复只信任已完整提交的批次；先对已提交批做重放校验，再补跑未知批。
* 重放按账本逐批重新前向计算并逐字段比较，首个不一致即抛 `E_REPLAY`，
  定位到批标识、请求标识、字段名，并给出 stored/recomputed 与判定依据。

## 11. Tkinter 说明

任务环境的 Python 3.11 虚拟环境无 `tkinter` 模块（宿主也缺
`libtk8.6.so`）。`batchdiag.gui` 保持标准 Tk 实现，`cli gui` 在导入失败时
捕获异常并输出结构化 CLI 降级信息；全部验收能力均不依赖 GUI。Windows
标准 Python 安装自带 Tk，可直接启用仪表盘。

## 12. 错误码对照表

| 错误码 | 触发条件 | 定位信息 |
| --- | --- | --- |
| `E_SPEC` | 规格非法：重复 id/arrival、维度不符、隐藏层非正、种子非整数、端口非高位/冲突、host 非回环 | 首个非法字段路径+行号+文件 |
| `E_FIELD` | 回执结构/取值错误：缺字段、耗时负、名次越界或批内重复、batch_id 形态错 | 字段路径（含数组下标）+行号 |
| `E_CHAIN` | 引用错误：duplicate_ref 不存在或自引用、request_id 不在请求流 | 字段路径+行号 |
| `E_TIMEOUT` | 真实 socket 超时 | host/port/timeout_ms/elapsed_ms |
| `E_REPLAY` | 恢复重放与账本不一致 | 首个分歧的 batch_id/request_id/field + stored/recomputed |
| `E_SNAPSHOT` | 快照保留异常（整体替换优先，正常路径不抛出） | 差异来源 |
