# MVCC 可见性诊断组件 — 架构与技术说明

## 1. 模块划分

| 模块 | 职责 | 对应业务要求 |
| --- | --- | --- |
| `mvcc_lab/loader.py` | 本地 JSON 加载与强校验，输出不可变 `Spec`；声明级摘要 | 1、7 |
| `mvcc_lab/model.py` | 事务、记录版本、读点、查询的冻结数据模型 | 1 |
| `mvcc_lab/visibility.py` | 快照可见性推导、理由链、稳定决胜、三类成因占比 | 2、6 |
| `mvcc_lab/executor.py` | 子进程托管的回环 TCP 重放执行器与 JSON 行客户端 | 3 |
| `mvcc_lab/scheduler.py` | 全局上限 + 每读点配额的线程池调度与占用轨迹 | 4 |
| `mvcc_lab/invalidation.py` | 结局/版本链变化的失效传播与仅失效子集重算 | 5 |
| `mvcc_lab/ledger.py` | SQLite 原子提交、版本识别整体替换、备份与恢复 | 7、8 |
| `mvcc_lab/runner.py` / `__main__.py` | 轮次编排、重放比对、强制终止模拟、CLI | 4、7、8 |

规则版本常量 `RULE_VERSION=mvcc-rules-1.0.0`、账本 `SCHEMA_VERSION=1`。

## 2. 可见性规则（显式声明）

1. 写入事务 `committed` 且 `commit_seq <= snapshot_seq` 才具备基础可见性；
2. 读事务对自身 **未提交（pending）** 版本可见；自身 **回滚（rolledback）**
   版本不可见；其它事务的未提交/回滚版本均不可见；
3. 沿版本链（版本号降序）取首个基础可见版本，其余基础可见候选记为
   `superseded`；
4. 断链候选追加 `chain_broken`；
5. 不可区分时按 `提交序号降序 -> 版本号降序 -> 事务标识序` 决胜，并置
   `tie_applied/error_influenced`。

排除理由稳定枚举：`chain_broken, commit_too_late, rolled_back,
superseded, uncommitted`，理由链始终按该固定字典序输出，与遍历顺序无关。

## 3. 回环 TCP JSON 行协议

* 子进程：`python -m mvcc_lab.executor <spec.json> --inject <mode>`；
* 启动后向 stdout 打印 `{"event":"ready","port":N,"rule_version":...}`；
* 父进程连接 `127.0.0.1:N`，发 `hello`，收 `hello_ack`；
* 批次请求含 `batch_id/snapshot_seq/reader_txn_id/query_ids/row_keys`，
  回执含 `batch_id/query_ids/summaries/rule_version`；
* 握手公告、连接、握手响应、批次读取均设置真实 socket 超时
  （`HANDSHAKE_TIMEOUT=2s`、`READ_TIMEOUT=2s`），不使用 `sleep` 伪装超时。

异常码：端口占用 `E_PORT`、真实超时 `E_TIMEOUT`、回执字段缺失 `E_FIELD`、
声明非法 `E_SPEC`、重放不一致 `E_REPLAY`。

## 4. 并发调度

* `ThreadPoolExecutor(max_workers=global_concurrency)` 真实投射工作线程；
* 每读点一个 `BoundedSemaphore(per_readpoint_quota)`，全局一个
  `BoundedSemaphore(global_concurrency)`；
* 每个任务先取读点配额、再取全局信号量，执行后立即释放，杜绝持有-等待
  循环；任何任务都必须经过这两道信号量，不能绕过配额直连执行器；
* 单套接字的请求/回执由 `_io_lock` 串行化，避免多线程写交错破坏 JSON 行；
* `TaskTrace` 以 `time.monotonic_ns()` 毫秒记录入队、两段等待、运行区间，
  并记录当时持有的读点/全局占用；`verify_within_limits()` 断言峰值不越界。

## 5. 失效传播

沿 `transaction -> version -> row -> query -> read_point` 边传播：

* 事务结局/提交序号变化，定位其写入版本与所属行键；
* 版本链追加（或结构变化）直接置该行键失效；
* 行键失效传播到引用它的全部查询，再汇聚到所属读点；
* 输出按读点标识序、行键序、查询标识序稳定排序，每步带完整边描述；
* 仅失效查询重算，未失效查询保留原 `summary_digest`，逐字节可校验。

## 6. 账本与恢复

* 每轮先在独立事务提交轮次意图（`rounds.state=open` + 进度游标 + 未知
  查询集合），再在 **同一事务** 写入逐查询结论、理由链、调度轨迹、游标并
  将轮次置 `committed`；
* 账本 `meta` 表携带 `schema_version` 与事务集/版本链/读点/查询/内容摘要；
* 重开时任一摘要或 schema 不一致即整体替换：旧库复制为 `0o444` 只读备份，
  差异写入 `replacement_log`，不做增量合并；
* 强制终止（SIGKILL）后只有 `committed` 轮次被承认，`open` 轮次查询全部
  标记为未知并要求重跑；终止时不产生退出报告；
* 恢复后按账本重放并与一次性完整判定逐字段比较，首个差异定位到
  `读点/行键/查询/字段` 并抛 `E_REPLAY`。

## 7. 误差决胜与三类成因

成因枚举：`error_indistinguishable`（误差不可区分）、
`rule_version_conflict`（规则版本冲突）、`version_chain_broken`
（版本链断链）。占比分母为受影响判定总数，使用 numpy 计算并保留 6 位
小数；`epsilon=0` 时不触发任何误差不可区分。
