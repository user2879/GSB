# 设计说明：本地批推理调度与多版本回执系统

## 1. 总体结构

纯 Python 3.10+ 实现，运行期只依赖 NumPy；网络使用标准库 `socket`，
持久化使用标准库 `sqlite3`，监控面板可选 `tkinter`。

| 模块 | 职责 | 对应需求 |
| --- | --- | --- |
| `spec.py` | JSON 声明/请求流/配额事件强校验与归并 | 需求 1 |
| `receipts.py` | r1/r2/r3 编解码、归一、E_FIELD/E_REF 校验 | 需求 2 |
| `plan.py` | 确定性批切分、稳定序列化、批标识推导 | 需求 3 |
| `inference.py` | NumPy 整数定点前向、输出摘要 | 需求 4 |
| `protocol.py` / `worker.py` / `processes.py` | JSON 行回环 TCP、子进程编排 | 需求 4 |
| `leases.py` | 租约签发/续约/真实过期/唯一配额回收 | 需求 5 |
| `scheduler.py` | 全局+每工作进程双重上限线程池调度 | 需求 6 |
| `invalidation.py` | 失效传播、子集重算、全量重算逐项比较 | 需求 7 |
| `ledger.py` / `recovery.py` | SQLite 原子事务、快照替换、崩溃恢复 | 需求 8 |

## 2. 关键不变量

### 2.1 确定性

* 所有哈希输入都经 `plan.canonical_json`（`sort_keys=True`、
  固定分隔符、UTF-8）序列化，与字典遍历顺序无关。
* 请求先按 `arrival_seq` 排序再按 `batch_size` 切片；工作进程按
  `worker_id` 排序后以批序号取模轮转。
* 批标识 `b_<blake2b12>` 仅依赖 `(worker_id, batch_seq, arrival_seqs)`。
* 定点前向的权重由 `(seed, param_version)` 派生，输入向量由
  `(seed, request_id)` 派生；int64 累积先转 uint64 回绕，消除平台差异。

### 2.2 租约与配额

* 全局上限 = 信号量；每工作进程上限 = `LeaseManager` 条件变量等待。
* 在途批次数与活跃租约数一一对应，调度器运行期 `assert` 双重上限。
* 配额回收只有 `_recycle` 一个守卫路径，状态机
  `held -> released_done/released_expired/released_cancelled`，
  到期回收与撤销回收幂等互斥，同一份配额不会被回收两次。
* 续约先在工作进程侧做租约版本号乐观校验（`expected_version`），
  远端拒绝抛 `E_LEASE` 且本地版本号不变。
* 过期判定使用 `time.monotonic()` 毫秒，后台 reaper 与获取路径都会清扫。

### 2.3 三格式回执归一

归一结果携带 `source_format`、`filled_fields` 与 `fill_basis`：

| 补齐字段 | 依据 |
| --- | --- |
| `param_version` / `worker_id` | 批次归属上下文 |
| `elapsed_ms` | r1 无计时，默认 0 |
| `rank_in_batch` | 批内到达序稳定派生 |
| `dup_ref` | 请求归并（覆盖）映射派生 |

`elapsed_ms` 负数、名次越界/重复 -> `E_FIELD`；
`dup_ref` 指向不存在请求 -> `E_REF`。

### 2.4 账本与恢复

* 每张表都存 `content_digest`（canonical JSON 的 BLAKE2b）。
* 批回执（r1/r2/r3 三行）与租约状态在 `BEGIN IMMEDIATE` 单事务提交。
* 快照是单调递增的 `snapshot_version`，整体替换即新建版本，旧版本保留。
* 恢复只承认 `committed` 行；计划内缺失批写 `unknown` 并先回收配额。
* `replay_verify` 独立重算每行摘要，返回首个差异的表/键/双摘要。
* 子进程以 `close_fds=True` 启动，父进程被 SIGKILL 后不会继承
  SQLite WAL 文件描述符，恢复进程可立即读到全部已提交数据。

## 3. 错误码对照

| 错误码 | 触发场景 |
| --- | --- |
| `E_SPEC` | 声明/请求流/配额事件首个非法位置与字段 |
| `E_FIELD` | 回执耗时负数、名次越界或重复 |
| `E_REF` | 重复投递引用悬空 |
| `E_PORT` | 回环端口绑定失败/连接被拒绝 |
| `E_TIMEOUT` | 握手或读回执真实超时 |
| `E_LEASE` | 租约版本乐观校验失败、租约过期 |
| `E_LEDGER` | 快照缺失等账本错误 |
| `E_WORKER` | 工作进程意外退出 |

## 4. 运行方式

```bash
python3 samples/generate_samples.py     # 生成确定性样本
python3 demo.py all                     # 八个场景端到端演示
python3 -m unittest discover -s tests   # 48 个测试
```
