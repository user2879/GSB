# 边缘缓存分片并行调度与租约账本验证平台 — 技术说明

## 1. 范围与运行约束

平台完全在本地运行，不依赖容器或外部服务：

* 边缘节点是独立 Python 子进程（`python -m edge_lease.node`），仅绑定
  `127.0.0.1` 高位端口；
* 进程间通信为回环 TCP 上的 JSON 行协议（一帧一行 JSON）；
* 唯一持久化后端是工作目录内的单个 SQLite 文件 `lease.sqlite`（WAL 模式）；
* 全部样本由固定种子（splitmix64 + FNV-1a）本地生成，结果可重复。

代码为跨平台 Python 3.11/3.12，子进程通过 `sys.executable -m edge_lease.node`
启动，Windows 与 WSL/Linux 行为一致；未使用任何 POSIX 专有 API。

## 2. 模块结构

| 模块 | 职责 |
| --- | --- |
| `edge_lease/errors.py` | `E_MANIFEST/E_CYCLE/E_GRID/E_PORT/E_HANDSHAKE/E_REPLY/E_CONVERGE/E_REPLAY/E_SNAPSHOT` 错误码与首个违规位置 |
| `edge_lease/content.py` | splitmix64 固定种子内容、32 位 FNV-1a 内容/分片摘要、规范化 JSON、分片连续性校验 |
| `edge_lease/fixedpoint.py` | 10 的整数幂网格量化、向零截断、int32 饱和 MAC、单值/派生传递误差上界（含 numpy 向量化） |
| `edge_lease/manifest.py` | 注册表/分片声明/版本清单强校验、单调版本号、派生图无环检测与稳定拓扑序（Kahn + 排序就绪集） |
| `edge_lease/compiler.py` | 版本→编译产物→计划项；逐字节稳定 `artifact_digest`；增量编译与失效传播（含每步依据） |
| `edge_lease/node.py` | 模拟边缘节点子进程：握手、定点重算、分片摘要、故障模式（timeout/exit/bad_reply/quant_mismatch） |
| `edge_lease/protocol.py` | 子进程生命周期、真实 socket 超时、E_PORT/E_HANDSHAKE/E_REPLY 分类、节点死亡识别与排除 |
| `edge_lease/scheduler.py` | 全局信号量 + 每节点配额信号量、单调时钟排队/服务计时、超时重试、attempt 区分、节点结局汇总 |
| `edge_lease/ledger.py` | SQLite 账本：格式版本、内容摘要、计划、批次原子事务、回执、配额、进度游标、状态机事件 |
| `edge_lease/converge.py` | 期望摘要重算与逐字节比对，稳定偏差列表（缺失/摘要/量化口径字段） |
| `edge_lease/replay.py` | 重开恢复、未提交批次标记 UNKNOWN/RERUN_REQUIRED、重编译重放并定位首个 E_REPLAY |
| `edge_lease/session.py` | 状态机迁移持久化、快照身份比对与整体替换（含替换前后差异条目） |
| `edge_lease/pipeline.py` | 端到端编排与三类故障注入（`corrupt_manifest/force_cycle/bad_grid`）及硬杀模拟 |
| `edge_lease/cli.py` | `generate-samples` / `run` / `replay` 命令 |

## 3. 定点精度口径（显式声明）

* 网格 `scale = 10**k`（`k >= 0`），非法值抛 `E_GRID`；
* 权重与派生系数量化采用 **向零截断（ROUND_DOWN）**，单值量化误差为整数上界
  `0`（恰好整除）或 `1`（截断了非零量子）；
* 所有乘加均为 Python 整数运算；派生合成唯一的除法同样向零截断，额外贡献
  至多 1 个量子的误差；
* 摘要域结果按有符号 32 位 `[-2^31, 2^31-1]` **饱和**并返回 `saturated` 标志，
  绝不静默溢出；
* 误差上界拆分为两个可见分量：
  `single`（本路径自身量化）与 `propagated`（上游摘要传递），
  并给出合计 `bound`；回执比对与汇总均携带这三个字段；
* 量化口径版本字符串 `Q32-S{scale}` 进入计划与每张回执，口径不一致在收敛阶段
  以 `quant_version` 差异字段报 `E_CONVERGE`。

## 4. 协议与故障码映射

握手与下发顺序：

```
platform -> {"type":"handshake","node_id":...,"quant_version":...}
node     -> {"type":"hello","node_id":...,"quant_version":...,"status":"ready"}
platform -> {"type":"dispatch", item_id, version, path, content_seed, path_table}
node     -> {"type":"receipt", node_id, version, path, derived_digest,
             shard_hit_digest, cost_ms, quant_version}
```

| 场景 | 错误码 | 判定依据 |
| --- | --- | --- |
| 端口占用/绑定失败/进程未就绪退出 | `E_PORT` | 子进程退出码 3 或就绪行超时 |
| 连接/握手超时、对端无响应 | `E_HANDSHAKE` | 真实 `socket.settimeout` 触发 |
| 回执缺字段/帧不可解析 | `E_REPLY` | 必需字段集合校验 |
| 节点异常退出 | 标记不可用并从后续调度排除 | `poll()` 存活检测 + `node_exited` 结构化日志 |
| 摘要不一致/无回执/口径不同 | `E_CONVERGE` | 收敛阶段稳定偏差列表 |
| 重放不可复现 | `E_REPLAY` | 定位首个 路径/节点/字段 |

所有失败同时写入 `events.jsonl`（JSON 行，含单调毫秒时间戳）。

## 5. 并发、重试与结局

* 全局 `BoundedSemaphore(global_limit)` 与每节点
  `BoundedSemaphore(per_node_quota)` 双重获取，先全局后节点，释放顺序相反；
* 信号量在任务线程启动后才获取，因此排队等待真实存在并被
  `task_waiting/task_running/task_finished` 事件记录；
* 测试通过运行窗口 `[ts_start, ts_end)` 的重叠计数证明任一时刻全局与单节点
  在跑任务数都不超限；
* 超时按节点 `retry_policy/max_retries` 重试，请求语义逐字节不变，回执以
  `attempt` 区分；最终每节点结局为 success/partial/timeout/failed/not_dispatched/
  unavailable。

## 6. 增量演进与失效

* 编译产物 `artifact_digest` 的规范化载荷 **不含版本号**，因此未变化路径跨
  版本逐字节保持不变，可逐项断言；
* 直接变更原因：`shards_changed/targets_changed/derived_coeff_changed/
  weight_or_strategy_changed/path_added/path_removed`；
* 沿派生边（源→依赖者）与节点引用关系传播，每跳输出
  `{path, parent, reason}`；失效集合按稳定拓扑序返回，算法仅使用排序后的
  就绪集合/队列，不依赖 dict 或哈希遍历顺序。

## 7. 账本、快照与崩溃恢复

* 元信息表带整数 `format_version`（当前为 1）与会话身份
  （清单版本、scale、seed、清单内容摘要、口径版本）；
* 每个批次在 **同一事务** 写批次行、全部回执、配额消耗与进度游标，
  状态从 `UNKNOWN` 翻转为 `COMMITTED`；硬杀后只可能留下未提交批次，重开时
  标记 `RERUN_REQUIRED` 并要求重跑，绝不生成退出报告；
* 快照重开比较同版本的清单内容摘要与 scale：一致则复用，漂移则
  **整体替换**（删除计划/批次/回执/状态事件），并稳定列出替换前后差异条目；
  正常的版本号演进只追加、保留全部历史版本；
* 重放按账本记录的版本与 scale 重新编译，校验 `plan_digest` 与每个
  `artifact_digest`，再逐项比对回执，首个不一致即 `E_REPLAY`。

## 8. 本地使用

```bash
pip install -r requirements.txt
python -m edge_lease.cli generate-samples --workdir workdir/demo
python -m edge_lease.cli run \
  --workdir workdir/demo \
  --registry workdir/demo/nodes.json \
  --shards workdir/demo/shards.json \
  --manifest workdir/demo/manifest.json \
  --scale 1000 --global-limit 3 --batch-size 3 \
  --report workdir/demo/report.json
python -m edge_lease.cli replay \
  --workdir workdir/demo \
  --registry workdir/demo/nodes.json \
  --shards workdir/demo/shards.json \
  --manifest workdir/demo/manifest.json \
  --scale 1000
```

故障注入：`--inject corrupt_manifest|force_cycle|bad_grid`
（分别对应 `E_MANIFEST/E_CYCLE/E_GRID`）。
