# 实际验证结果

完整原始输出保存在 `artifacts/acceptance-output.txt`。以下为本机 WSL Ubuntu 的实际结果摘要。

## 环境

```text
Python 3.11.0rc1
/home/zsl/feedback/uidfc5909d3-default-356/.venv/bin/python
numpy 1.26.4
tkinter: ModuleNotFoundError / libtk8.6.so unavailable -> gui-check skipped
```

## 自动化测试

```text
Ran 16 tests in 8.670s
OK
```

覆盖：E_SPEC、摘要元数据绑定、分批/乱序确定性、抽样、两类失效全量一致性、子进程 pass/corrupt/SIGKILL、主流程独立 verify、篡改 E_VERIFY、租约恢复、会话身份替换只读备份、真实 CLI SIGKILL 后恢复。

## 主样本回滚与校验

```text
replay fingerprint = 9503f0247aa2a66b7513bb479bc4ed93fa5c5008cc108f9d205d5e0f947bae09
tasks = 8
candidate blocks = 28
sampled/checked blocks = 17
passed before deletion propagation = 17
unchecked before deletion propagation = 11
```

样本事件 `seq=14` 删除 `snap-alpha-10`，其 8 个 `rb-alpha-10` 块结论被权威账本标记为 `invalidated`。独立校验结果：

```json
{
  "ok": true,
  "checked_blocks": 28,
  "status_counts": {"invalidated": 8, "pass": 12, "unchecked": 8}
}
```

这里的 `pass=12`、`unchecked=8` 是删除传播后的最新状态；删除前对应 17 通过、11 未校验，其中 5 个原 pass 和 3 个原 unchecked 被改为 invalidated。

配额等待实测（节选自 `artifacts/run-report.json`）：

```text
rb-alpha-10:a01-b000  wait_ms=0    global=1 vol=1
rb-beta-20:a01-b000   wait_ms=0    global=2 vol=1
rb-alpha-10:a01-b001  wait_ms=142  global=1 vol=1
rb-beta-20:a01-b001   wait_ms=142  global=2 vol=1
rb-alpha-11:a01-b000  wait_ms=364  global=1 vol=1
rb-gamma-30:a01-b000  wait_ms=364  global=2 vol=1
rb-gamma-30:a01-b001  wait_ms=525  global=2 vol=1
rb-alpha-11:a01-b001  wait_ms=565  global=1 vol=1
```

## 强制杀死单块校验子进程

注入点：`vol-beta:1`，该块按 600‰ 规则被抽中。

首次运行：

```text
unknown = 1
worker_returncode = -9
force_killed = true
```

重跑前独立校验如预期失败：

```json
{
  "code": "E_VERIFY",
  "volume_id": "vol-beta",
  "block_index": 1,
  "field": "restored_digest",
  "expected": "979fc299a53090f8fcc0927333e4deb4524421abbfc372d97f741d6810561c24"
}
```

执行 `recover --rerun` 后：

```json
{"ok": true, "checked_blocks": 28,
 "status_counts": {"invalidated": 8, "pass": 12, "unchecked": 8}}
```

完整 JSON：`artifacts/kill-run-report.json`、`artifacts/kill-recovery-report.json`、`artifacts/kill-verify-after-rerun.json`。

## 强制杀死整个进程后的账本恢复

注入点：`rb-alpha-11:a01-b000` 租约事务提交后、批次完成事务前，进程对自身发送 `SIGKILL`。

```text
crash_exit = 137
durable_running_batches = [('rb-alpha-11:a01-b000', 'running')]
```

此时失效分析从只读账本读到未完成块集合 4 个：块 `4,5,6,7`，见 `artifacts/invalidation-delete-unfinished.json` 的 `ledger_unfinished`。

恢复结果：

```text
recovered_interrupted = 1
unknown_before_rerun = 4
rerun_batch_count = 1
```

随后以 `--attempt 2` 补跑全量并独立校验：

```json
{"ok": true, "checked_blocks": 28,
 "status_counts": {"invalidated": 8, "pass": 12, "unchecked": 8}}
```

## 篡改检测

对 `vol-beta:1` 的校验 worker 注入内存翻字节：

```text
CLI exit = 3
code = E_CORRUPT
expected_digest = 979fc299a53090f8fcc0927333e4deb4524421abbfc372d97f741d6810561c24
actual_digest   = 0396bfbb86ea268cb27e8175c10d7e96d076db542019f83f11874652413e88c4
```

独立校验退出码 `4`，定位到 `vol-beta block 1 field=restored_digest`。

## 失效子集

```text
write change:
  affected_snapshots = [snap-alpha-10, snap-alpha-11]
  changed expected digests = snap-alpha-11 blocks 4,6
  full_rebuild_consistency.match = true

delete change:
  affected_rollbacks = [rb-alpha-10, rb-alpha-11]
  ledger_keys include rb-alpha-11 blocks 4..9
  full_rebuild_consistency.match = true
```

写入变更虽然也与 `snap-alpha-10` 区间相交，但该写入到达于 `snapshot_create seq=2` 之后，冻结摘要不变；传播报告保留了这条“检查但不变”的依据。这是事件到达顺序约束的直接体现。

## 会话整体替换

把抽样比例从 600‰ 改为 1000‰ 后，`content_digest` 不一致，运行时生成：

```text
ledger/backups/replacement-demo.replaced-001.sqlite   mode 0444
ledger/backups/replacement-demo.replaced-001.diff.json
```

新账本独立校验通过：28 块全部抽中，删除传播后为 `20 pass + 8 invalidated`。

## 复现实验

```bash
bash scripts/acceptance.sh | tee artifacts/acceptance-output.txt
```
