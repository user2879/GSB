# 实际验证记录

验证时间：2026-11-27（Asia/Shanghai）。

## 环境

- Python：`Python 3.11.0rc1`
- 虚拟环境：`.venv`
- 依赖：`numpy==1.26.4`、`pytest==8.2.2`
- SQLite：Python 标准库 `sqlite3`

基础镜像缺少 Debian `ensurepip` 数据包，直接执行 `python3.11 -m venv .venv` 报 `ensurepip is not available`，且当前账号无免密 sudo。已用以下等价用户态路径完成任务要求的环境：

```bash
python3.11 -m venv --without-pip .venv
python3 -m pip install --target .venv/lib/python3.11/site-packages pip==24.0
.venv/bin/python -m pip install numpy==1.26.4 pytest==8.2.2
```

标准 Ubuntu/WSL 若已安装 `python3.11-venv`，可直接使用 `bash scripts/setup_venv.sh` 或 `python3.11 -m venv .venv`。

## 自动化测试

命令：

```bash
.venv/bin/pytest -q
```

实际结果：

```text
.....................                                                    [100%]
21 passed in 9.29s
```

完整日志：`artifacts/verification/pytest-latest.txt`。

覆盖包括：

- `E_SPEC` 字段路径、重复 ID/端口、非法配额、缺失依赖和环；
- 样本确定性、乱序、批次切分、32 线程并发摘要一致；
- 定点 floor 截断与误差界；
- `E_PORT`、`E_TIMEOUT`；
- 真实节点租约 CAS、版本冲突不延期和真实 sleep 过期；
- 并行调度、依赖摘要、冻结分片零预留；
- 增量失效与未失效 payload 字节保持；
- `E_DIVERGE` 首差异路径；
- 真实子进程 `SIGKILL`、预算回收、unknown 标记和恢复重放。

## 端到端演示

命令：

```bash
.venv/bin/python -m waterlease.demo
```

实际退出码：`0`；总开关：`artifacts/demo-report.json -> ok = true`。

关键结果：

- 首版：6 个活动分片全部提交，`s7_frozen` 跳过；replay `ok=true`，current count 6，mismatch 0。
- 并行峰值：全局 `6/6` 资源单位；节点峰值 `node-a=1`、`node-b=1`、`node-c=2`。
- 真实排队：`s6_independent_c` 因 `global_resource_budget` 等待约 `193 ms` 后派发。
- v2 失效集合：`s2_raw_b`、`s4_dep_c`、`s5_join_b`、`s6_independent_c`、`s7_frozen(frozen)`。
- 稳定未失效集合：`s1_raw_a`、`s3_dep_a`。
- 字节保持：`s1_raw_a`（7040 bytes）和 `s3_dep_a`（7466 bytes）均 `identical=true`。
- 快照演进：`action=replaced`，差异数 `15`，旧快照备份在 `ledger/snapshot_backups/`（只读 `0444`）。
- 增量实际重算并提交：`s2_raw_b`、`s4_dep_c`、`s5_join_b`、`s6_independent_c`。
- 空白重建：6 个分片逐字段确定性比较通过，`first_difference=null`。
- v2 replay：按 seq 重放 `replayed_commit_count=10`，归约出 `replayed_current_count=6`；outcome 修订数 `verified_outcome_count=10`（6 个当前结论 + 4 个失效替换修订），`current_count=6`，mismatch 0。

## 强制终止恢复

演示以独立进程组启动：

```bash
python -m waterlease.cli ... run --recover --crash-after-inflight 2
```

在两个 active reservation 已提交且慢节点请求在途时，调度器对自身执行真实 `SIGKILL`。报告记录：

- `child_returncode = -9`；
- 回收 2 条 active reservation；
- 回收资源单位合计 `5`（`s1_raw_a=2`、`s2_raw_b=3`）；
- unknown 分片恰为 `s1_raw_a`、`s2_raw_b`；
- 恢复重放初始 `ok=true`，已提交 outcome 数为 0；
- 重新调度后 6 个活动分片全部完成；
- 最终 replay `current_count=6`、`replayed_commit_count=6`、`replayed_current_count=6`、`current_mismatches=0`、`ok=true`。

系统明确记录：SIGKILL 时的退出报告不可重建，只有完整提交事务可作为权威状态。

## CLI 抽查

```bash
.venv/bin/python -m waterlease.cli --spec-dir spec validate
```

实际摘要：

```text
spec_digest = ac53f26b7bd96caa9946c9b1d9eef3a76814654bc85ad8cd0f3974041688d913
topological_order = [s1_raw_a, s2_raw_b, s3_dep_a, s4_dep_c, s5_join_b, s6_independent_c, s7_frozen]
```

v2 digest：

```text
5b134c39d72d148097cd6ebb1be1653e902bcc3963e458605a5501489d23da04
```

最近一次命令输出保存在 `artifacts/verification/validate-latest.txt`、`artifacts/verification/validate-v2-latest.txt` 和 `artifacts/verification/replay-latest.txt`。

`bash scripts/verify.sh` 已实际执行通过（退出码 0），会依次重跑 pytest、完整 demo 和 v2 replay，并刷新上述日志。
