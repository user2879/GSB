# 实际运行记录

记录日期：2026-11-25（Asia/Shanghai）

## 环境

```text
Python: 3.11.0rc1 (/home/zsl/.feedback/python-base/bin/python3)
NumPy: 1.26.4
OS: WSL Ubuntu 22.04
Docker: 未使用
业务网络依赖: 无；工作器仅监听 127.0.0.1
样本端口: 127.0.0.1:28431、127.0.0.1:28432 运行前 bind 检查均可用
```

宿主镜像默认缺少 `libtk8.6.so`。为完成环境确认，将 Ubuntu Jammy 的
`libtcl8.6/libtk8.6/tk8.6` 及相关 X 库下载并解包到 `/tmp/tk-verify`，通过
`LD_LIBRARY_PATH/TCL_LIBRARY/TK_LIBRARY` 验证：

```text
tkinter 8.6
root_created
```

该验证只用于确认 tkinter；推理、调度、账本和测试均不依赖 GUI 或这些临时库。

## 自动化测试

```bash
PYTHONPATH=src python3 -m pytest -q
```

实际结果：

```text
...........................                                              [100%]
27 passed in 7.12s
```

覆盖内容包括：严格规格错误、重复 JSON 键、稳定计划、三代回执编码/解码/归一并、负耗时、
重复/越界名次、未知重复投递引用、真实回环工作器、端口占用、协议缺字段、配额排队、
失效边界、未失效行逐字节保持、强杀恢复、快照整体替换、快照摘要注入和重放差异注入。

## 端到端演示

```bash
PYTHONPATH=src python3 scripts/run_demo.py
```

命令退出码为 `0`，完整机器可读结果在 `artifacts/demo/summary.json`。关键实际结果：

### 稳定计划

```text
plan content_digest = 235287da253c6e12932e9b0016f15320815be94dfeb9ff902d82371c8c357566
batches = 3
worker-alpha: batch_seq 0/1, 各 2 个请求
worker-beta:  batch_seq 0,   2 个请求
```

### 三版本回执归一

```text
normalized content_digest = 2376d29f9a14cb2ca672850c27c7b9a6b603ed47aab35c7e99320ffe7ed26f5d
归一批次数 = 3
与真实运行逐批归属摘要一致 = true
r1 补齐字段 = latency_ms, param_digest, param_version_seq, rank_in_batch,
                  superseded_request_ids, worker_id
```

### 失效边界

参数版本从 2 升到 3，仅 `worker-alpha`：

```text
direct_invalid_batches = 2
indirect_invalid_request_ids = req-001, req-002, req-004, req-006
unchanged_batch_ids = 1 个 worker-beta 批次
```

`worker-beta` 批大小从 2 改为 1：

```text
direct_invalid_batches = 1 个旧 beta 批 + 2 个新 beta 批
indirect_invalid_request_ids = req-003, req-005
unchanged_batch_ids = 2 个 worker-alpha 批次
```

另有一项集成测试在真实账本上只重算失效的 alpha 批，并逐字节确认 beta 行未变化。

### 全量运行

```text
progress_cursor = 3
ledger_digest = 27de8b30bd531386766459a57717bf4547032f694621e933678856a200559719
max_global_running = 2
max_worker_running = {worker-alpha: 1, worker-beta: 1}
```

实际调度记录中出现了同一工作器排队等待，例如 alpha 第二批 `wait_ms ≈ 12.946`，说明
配额由调度协调器执行，而非人为 sleep。

### 强制终止与恢复

```text
crash process return code = -9 (SIGKILL；shell 状态 137)
before resume:
  progress_cursor = 1
  completed_batch_ids = 1
  unknown_batch_ids = 2
  requires_rerun = true
  interrupted_process_exit_report = null
after restart:
  progress_cursor = 3
  unknown_batch_ids = []
  final ledger digest = 7ab8f80a3760ca2e48f2b7b8d60c0a7295ce262dabb3fae1f2e2d1eac265d526
  replay consistent with independent full run = true
```

全量账本与恢复账本的内容摘要因工作器观测耗时不同而不同；批次归属、回执摘要、参数绑定
和重放语义逐项一致。本记录不声称存在强杀时的可信退出报告；未提交批只作为未知批重跑。

### 快照恢复与异常注入

```text
worker set/version/digest 改变后 restore：
  whole_replacement = true
  reasons = worker_set_version, param_versions, missing_or_invalid_target_ledger
  restored rows equal full run = true

注入损坏声明：              E_SPEC   field=host
注入快照账本摘要不符：      E_SNAPSHOT field=ledger_digest
注入首批回执摘要重放差异：  E_REPLAY field=receipt_digest
```

## 复现命令摘要

```bash
PYTHONPATH=src python3 scripts/build_samples.py
PYTHONPATH=src python3 scripts/run_demo.py
PYTHONPATH=src python3 -m pytest -q
```
