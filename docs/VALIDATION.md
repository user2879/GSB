# 实际验证记录

验证时间：2026-11-26（Asia/Shanghai）。

## 环境

- Python：`3.11.0rc1`
- NumPy：`2.4.6`（满足 `numpy>=1.26`）
- pytest：`9.2.0`（满足 `pytest>=8`）
- 虚拟环境：工作区 `.venv`

当前 Ubuntu/Debian 环境的 `python3 -m venv` 缺少 `ensurepip`。本次使用已有的
`virtualenv --clear .venv` 引导出可用 `.venv`，随后执行：

```bash
. .venv/bin/activate
pip install 'numpy>=1.26' 'pytest>=8'
pip install -e .
```

在标准 WSL Ubuntu 中也可先执行
`sudo apt install python3.11-venv`，再按题目给出的
`python3 -m venv .venv && . .venv/bin/activate && pip install numpy pytest`
创建环境。

## 自动化测试

命令：

```bash
. .venv/bin/activate
pytest -v
```

实际结果：

```text
collected 31 items

tests/test_end_to_end.py .
tests/test_evolution.py ...
tests/test_failure_injection.py .
tests/test_invalidation.py .
tests/test_metrics.py ......
tests/test_protocol.py ....
tests/test_recovery.py ..
tests/test_scheduler.py .
tests/test_specs.py ............

============================== 31 passed in 7.14s ==============================
```

这些测试覆盖：

- `E_SPEC`：重复跳 ID/端口、非法角色、负/布尔毫秒、非法缩放因子、环和源站不匹配。
- `E_PORT`：真实占用声明端口后启动子进程。
- `E_TIMEOUT`：真实静默 TCP 服务导致握手读超时。
- `E_FIELD`：直接校验和真实 JSONL 套接字响应缺字段。
- 定点归并：样本结论为 `e2e_total_units=6000`，逆序回执产生相同规范化字节。
- 误差归因：`fwd1` 为主导跳，并覆盖不可区分、版本冲突、高失败比例。
- 失效传播：配置版本/缩放因子变化、请求范围、链下游传播和摘要不变量。
- 调度配额：全局 3、每跳 1 的压力测试不超发，并记录真实等待毫秒。
- 端到端回环：4 个请求经真实子进程和 TCP JSONL 完成，观测峰值为全局 2、每节点 1。
- 账本/快照：SQLite 摘要有效；失配快照生成 `0444` 只读备份并整体替换。
- 强杀恢复：`SIGKILL rev2` 后 `batch-0002` 标记 unknown，重启重跑并与全量结果一致。
- `E_REPLAY`：人为改写字段后定位到 `req-0001` / `fwd1` / `scale_factor`。
- 缩放口径版本演进：只重算派生指标和归因，不重新下发任何回执。

## 一键验收脚本

命令：

```bash
. .venv/bin/activate
python scripts/acceptance_demo.py --report artifacts/acceptance-report.json
```

实际报告摘要：

```json
{
  "ok": true,
  "interrupted_error_code": "E_DISPATCH",
  "marked_unknown_batches": ["batch-0002"],
  "replayed_batches": ["batch-0002"],
  "replay": {
    "first_mismatch": null,
    "ok": true,
    "receipt_count": 12,
    "request_count": 4
  },
  "evolution": {
    "affected_request_ids": [
      "req-0001",
      "req-0002",
      "req-0003",
      "req-0004"
    ],
    "digest_violations": [],
    "unchanged_byte_check_passed": true
  },
  "ledger_digest_valid": true,
  "max_observed_global": 2,
  "max_observed_hop": {
    "fwd1": 1,
    "origin": 1,
    "rev2": 1
  }
}
```

## CLI 样本

命令：

```bash
hopledger init-sample --dir artifacts/cli-sample
hopledger run \
  --config-dir artifacts/cli-sample/config \
  --ledger artifacts/cli-sample/ledger.sqlite3 \
  --snapshot artifacts/cli-sample/snapshot.json \
  --report artifacts/cli-sample/report.json
```

实际 4 条请求均得到：

- `e2e_total_units = 6000`
- `dominant_hop = "fwd1"`
- `primary_cause = "dominant_hop_identified"`
- 全局峰值 `2`
- `fwd1`、`rev2`、`origin` 的每节点峰值均为 `1`
- `ledger_digest_valid = true`

所有网络调用均绑定 `127.0.0.1` 高位端口；除 pip 依赖引导外，验证过程不需要外部网络。
