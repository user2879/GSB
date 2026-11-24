# 回执归并诊断组件 — 技术说明

## 1. 目标与边界

组件把本地确定性推理工作器在多历史回执格式（r1/r2/r3）与多参数版本下的
输出，归并为一个可逐字节复核的 SQLite 权威账本，并保证 Tkinter 快速视图与
权威重算结论一致。全部通信只发生在 `127.0.0.1` 回环，不依赖容器、外部服务
或网络目标。

## 2. 模块结构

| 模块 | 职责 |
| --- | --- |
| `receipt_ledger/spec.py` | 三份 JSON 声明加载与强校验，E_SPEC 定位首个违规项/字段 |
| `receipt_ledger/digest.py` | 规范化 JSON、blake2b/sha256 内容摘要、批次链式摘要 |
| `receipt_ledger/deterministic.py` | 独立 RNG 的确定性参数、显式量化、NumPy 前向、输出摘要 |
| `receipt_ledger/codec.py` | r1/r2/r3 长度前缀信封编解码、类型/长度校验、归一视图 |
| `receipt_ledger/merge.py` | 批次规划、稳定采纳规则、否决理由、灰度结论合成 |
| `receipt_ledger/worker.py` | 回环子进程工作器、真实 TCP 客户端与超时重试 |
| `receipt_ledger/scheduler.py` | 子进程池、全局/每工作器双重信号量、排队单调时钟、硬崩溃注入 |
| `receipt_ledger/ledger.py` | SQLite 事务、链式证据、陈旧隔离、整体替换备份、恢复与重放 |
| `receipt_ledger/preview.py` | 单次读取快速视图、权威重算、E_PREVIEW、定点延迟统计 |
| `receipt_ledger/ui.py` | Tkinter 只读快速界面，无头环境显式降级 |
| `receipt_ledger/cli.py` | run/resume/preview/recompute/verify/ui 命令 |

## 3. 关键设计

### 3.1 确定性
* 每个 `(worker_id, param_version)` 用独立
  `numpy.random.default_rng`，种子由
  `blake2b(worker_id|model_dim|hidden_dim|seed|param_version)` 派生，
  不读取全局随机状态，因此与 JSON 加载顺序、线程调度无关。
* 量化口径显式：fp32/fp16 走 IEEE 舍入后回 fp64；int8 走对称整数量化。
* 所有跨进程比对基于 `canonical_json`（排序键、紧凑分隔），不依赖
  dict/hash 遍历顺序。

### 3.2 三格式回执与归一
信封为 `RLR1 | 版本字节 | 4 字节大端长度 | JSON 载荷`，解码校验魔数、
长度自洽、禁止尾随字节、字段类型（布尔不充当整数）。归一视图固定携带
`source_format`、`filled_fields`（含 field 与 basis）与原始信封长度，
原始信封与归一 JSON 都入 `receipts` 表，可回溯到补齐依据。

### 3.3 归并全序
同批多回执按 `参数版本降序 → 格式版本降序(r3>r2>r1) → merge_seq 升序`
选采纳回执；其余回执保留并记录 `superseded:*` 否决理由。

### 3.4 调度与套接字
* 全局 `BoundedSemaphore(global_concurrency)` 与每工作器
  `BoundedSemaphore(per_worker_concurrency)` 双重约束。
* 记录 `wait_start_ms/wait_end_ms/wait_ms`（单调时钟），耗时只来自真实
  前向与套接字，不注入人工延迟。
* 请求幂等，超时/断连按规则 `max_retries` 立即重试，归属批次与参数不变。
* 工作器以独立进程组启动，并在 Linux 上设置 `PR_SET_PDEATHSIG`，
  父进程被 `os._exit`/kill 后内核回收工作器，避免孤儿占端口。

### 3.5 账本事务与证据链
每个批次在单事务内写 `batches`、全部 `receipts` 与 `progress` 游标；
批次链式摘要 = `blake2b(上一批摘要 | 本批规范化载荷)`，创世摘要固定。
强制终止后只承认 `status='committed'` 批次，计划内其余批次标 `unknown`
并要求重跑，不生成退出报告（`exit_report=None`）。

### 3.6 快速视图 vs 权威重算
* 快速视图是一次 `SELECT`，只读缓存，不触发推理。
* 权威重算按账本声明重建批次计划，重放参数、前向、归并，逐批做
  规范化字节比对，首个偏差以 E_PREVIEW 报 `batch_id/field/request_seq`。
* 延迟以单调毫秒记录，均值用整数定点 `avg_q1000 = sum*1000//count`。

### 3.7 陈旧防护与整体替换
声明版本必须单调递增；摘要变化时旧快照与旧批次整体置 `stale`，不参与
当前展示，输出稳定排序的 added/removed/changed 差异，复制只读备份
（权限 `0440`）后再写新快照。

## 4. 错误码

| 错误码 | 触发场景 |
| --- | --- |
| E_SPEC | 声明校验首个违规，含 section/index/ident/field |
| E_CODEC | 回执魔数、长度或字段类型非法 |
| E_PREVIEW | 快速缓存与权威重算首个逐字节偏差 |
| E_REPLAY | 恢复重放与账本逐项不一致，定位批/请求/字段 |
| E_SNAPSHOT | 账本格式版本或磁盘摘要不一致、链式摘要断裂 |

## 5. 验证结果（本机实测）

* `python -m pytest tests/ -q`：35 passed，覆盖规格校验、三格式归一、
  确定性、归并全序、真实回环并发、E_PREVIEW、强制终止恢复、E_REPLAY
  定位、陈旧替换、重试幂等与定点延迟。
* `scripts/run_demo.sh`：7 批 r1/r2/r3 归并、权威重算 7/7 匹配、
  提交 3 批后 137 硬崩溃、恢复时 4 批 unknown 并重放 identical、
  v2 演进整体替换并生成只读备份。

## 6. 无头 Linux 的 Tk 处理

官方 Windows Python 自带 Tk；本机缺少 `libtk8.6` 时，可用
`apt-get download` 取 `libtk8.6/libtcl8.6/libXft2/libXss1` 等 deb，
`dpkg-deb -x` 解压到本地目录后以 `LD_LIBRARY_PATH` 指向，无需 root。
UI 在 Tk 不可用时抛清晰错误，CLI 的 preview/recompute 不依赖 Tk。
