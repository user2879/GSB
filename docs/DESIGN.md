# 设计与技术说明：GPU 配额与分片调度领域组件

## 1. 模块划分

| 模块 | 职责 |
| --- | --- |
| `gpu_scheduler/errors.py` | 业务码与带字段路径的强类型错误（`E_SPEC/E_MEMORY/E_QUOTA/E_CYCLE/E_DEPEND/E_LEDGER/E_REPLAY`）。 |
| `gpu_scheduler/model.py` | 资源池、任务、分片、会话配置的数据模型与确定性 JSON/摘要工具。 |
| `gpu_scheduler/loader.py` | 三份 JSON 声明的加载与强校验，失败整体拒绝（顶层 `E_SPEC`，问题项保留细分子码）。 |
| `gpu_scheduler/graph.py` | 任务内+任务间合并的全局分片图、Kahn 稳定拓扑、传递失效集合推导。 |
| `gpu_scheduler/engine.py` | 纯内存确定性调度引擎：三重约束、优先级排队、显式抢占、失败级联、恢复重臂。 |
| `gpu_scheduler/fixedpoint.py` | 整数定点比例/均值、截断方向、单值/比较/聚合误差界、稳定决胜。 |
| `gpu_scheduler/stats.py` | 占用率、完成率、平均等待、窗口比较与多池聚合的定点统计。 |
| `gpu_scheduler/events.py` | 轨迹事件、抢占记录、资源占用与轮次记录的不可变结构。 |
| `gpu_scheduler/ledger.py` | SQLite 账本：单事务轮次写入、格式版本、链式摘要与篡改校验。 |
| `gpu_scheduler/snapshot.py` | 会话快照、版本识别、整体替换与只读备份。 |
| `gpu_scheduler/replay.py` | 账本重建、UNKNOWN 标注、恢复续跑与逐分片重放比对（`E_REPLAY`）。 |
| `gpu_scheduler/injector.py` | 四类异常注入，仅改写工作区内的输入副本或账本字节。 |
| `gpu_scheduler/logging_utils.py` | 仅写入工作区的 JSONL 结构化日志。 |
| `gpu_scheduler/diagnostics.py` | Tkinter 诊断界面，无 Tkinter 时降级为同口径控制台视图。 |
| `gpu_scheduler/cli.py` / `__main__.py` | 命令行入口。 |

## 2. 确定性保证

- 所有遍历都基于显式整数/字符串键排序（池 id、任务 id、分片号、时钟、优先级），不依赖 dict/hash 顺序。
- 就绪队列键为 `(-优先级, -累计等待, 任务id, 分片号)`；输入池/任务/分片乱序时，`tests/test_scheduler.py` 断言轮次与抢占序列逐字节一致。
- 时钟为从 0 起的整数毫秒单调时钟；任务可声明 `arrival_ms` 以制造“低优先级先占资源、高优先级后到达”的真实抢占场景。
- 抢占受害者规则固定为：优先级升序 → 最早派发时刻 → 任务 id → 分片号，且仅允许严格更低优先级任务成为受害者，因此每条抢占记录都满足触发优先级严格高于受害者。

## 3. 三重约束

每个派发决策同时校验，任一不满足即进入等待队列并记录 `wait_started_ms`：

1. 全局：运行分片 GPU 总数 ≤ `global_gpu_cap`，配额份额总和 ≤ `total_quota_shares`。
2. 资源池：池内 GPU、显存、份额、并发任务数均不超过池上限。
3. 任务：该任务所有在跑分片的显存之和 ≤ 任务 `memory_budget_mib`。

份额按 `quota_shares / gpu_count`（加载期强制整除）得到每卡整数份额，运行期份额占用全程整数运算。

## 4. 抢占与可恢复性

- 高优先级就绪分片在配额/显存不足、且存在严格低优先级运行分片可释放约束时，每轮最多执行一次抢占，保证抢占序列稳定。
- 抢占只在分片边界安全挂起在途分片：已 `SUCCESS` 分片的 `version` 与 `output_digest` 保持不变，被挂起分片标记 `PREEMPTED`，重新获得配额后仅重算该分片。
- 失败分片为终态 `FAILED`；其任务内下游与任务间下游通过全局分片图按稳定拓扑标记 `INVALIDATED`，起源节点本身不会被重复标记。

## 5. 定点口径与误差界

- 比例采用向零截断：`floor(|num|*scale/den)`，单值绝对误差严格小于 1 个缩放单位，误差上界记为 `1`（缩放单位）。
- 平均等待采用向下取整：`total*scale//count`，误差上界同样为 `1`。
- 两个定点值之差的合成误差界为 `1+1=2`；`|差值| <= 2` 判定为受误差影响，严格非零差值仍按整数方向排序，完全相等时按更小 id 决胜，并输出 `tie_break_applied` 与 `error_affected_against` 标注。
- 多池聚合在缩放前对精确整数总量求和后只做一次截断，因此聚合单值误差上界保持 `1`；两个聚合值比较的误差上界为 `2`。文档与 `stats.aggregate_pools` 均显式输出该口径。

## 6. 账本、快照与恢复

- 每个调度轮次在单个 SQLite 事务内写入：轮次摘要、轨迹事件、抢占、池占用、全分片状态与进度游标。
- 轮次摘要为 `sha256(canonical(prev_digest + canonical(round)))`，事件与池占用在摘要前按持久化相同顺序规范化，保证回读后链式校验一致；任意字节篡改在 `Ledger.verify` 处报 `E_LEDGER`。
- 快照保存池版本、任务版本、配置版本、缩放因子、配额状态、各分片结局与账本摘要。重开时版本不一致即整体替换：旧快照移动到只读备份目录（目录 `0555`、文件 `0444`），绝不增量合并。
- 强制终止后只信任已提交轮次。最后提交快照中仍为 `RUNNING` 的分片没有完成事件，统一标记 `UNKNOWN` 并要求重跑，不生成退出报告。恢复从账本的全分片状态重建，再续跑剩余轮次。
- 重放比对（`replay.compare_round_intervals`）针对恢复后缀：终态结局必须一致、每次派发的池/GPU/显存使用必须是参考运行同一分片的合法选择（多重集比较，兼容 UNKNOWN 重跑带来的额外轮次）、抢占记录须保持参考序列中的相对顺序；首个不一致以 `E_REPLAY` 报出轮次、任务 id 与分片号。

## 7. 异常注入与日志

| 注入 | 手段（仅改副本/参数） | 业务码 |
| --- | --- | --- |
| `memory` | 任务显存预算改为小于单分片显存 | `E_MEMORY` |
| `quota` | 资源池份额总和改为超过会话常量 | `E_QUOTA` |
| `cycle` | 引入分片 `0<->1` 依赖环 | `E_CYCLE` |
| `ledger` | 正常落账后篡改某轮 `round_digest` 字节 | `E_LEDGER` |

正常加载的顶层拒绝码为 `E_SPEC`，注入层依据每个 `Problem.code` 细分到上述精确业务码。所有日志写入工作区 `logs/events.jsonl`，业务码与样本不污染原始声明。

## 8. NumPy 的使用边界

`stats._busy_totals_via_numpy` 使用 dtype 为 `int64` 的毫秒时间轴数组对 GPU/显存占用做整数累加（`np.int64` 切片求和后显式转回 Python `int`）。定点量化、误差界与调度决策全部为 Python 整数运算，NumPy 不参与任何浮点舍入，因此不改变确定性与误差口径。
