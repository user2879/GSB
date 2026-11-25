# 实际验证记录

验证时间：2026-11-25（Asia/Shanghai）。

实际执行环境：当前工作区中的 Python `3.11.0rc1`、SQLite `3.37.2`、NumPy `1.26.4`、Linux/WSL 用户态。代码只使用跨平台 Python API，并附带 Windows `run_validation.cmd` 与 `run_validation.ps1`；未把未实际执行的 Windows 运行结果记为已执行。

## 1. 执行命令

```bash
python -m pip install numpy==1.26.4
python -m qos_diag sample --workspace .
python -m qos_diag drill --workspace . --output artifacts/reports/acceptance-drill.json
python -m unittest discover -s tests -v
python -m qos_diag verify --spec-dir samples/edited_param --ledger artifacts/ledger/qos.db
python -m qos_diag verify --spec-dir samples/base --ledger artifacts/ledger/qos.db --revision 1
```

## 2. 自动化测试

结果文件：`artifacts/reports/unittest-output.txt`。

```text
Ran 12 tests in 6.178s

OK
```

覆盖：

- 样本强校验、重复端口 `E_SPEC`；
- 三级桶显式整数转移；
- event/window canonical bytes parity；
- 1/2/3/5/7/12/100 等任意分批边界逐字节一致；
- rate/burst/arrival 三类编辑的失效闭包与全量重放一致；
- 管道串联关系交换后的下游失效与全量重放一致；
- 误差三分类；
- 并发治理上限；
- 真实 9 个 TCP 子进程 + SQLite 账本 + 独立 verify；
- 篡改账本余量后，`E_VERIFY` 定位到 `event_seq=0`、`stage=ingress`、`field=balance_after_decision_q`；
- 摘要/版本不符时整体替换。

## 3. 一键验收演练

完整机器可读结果：`artifacts/reports/acceptance-drill.json`。顶层 `"ok": true`。

### 三级判定与路径

- 样本：3 条管道、9 个 shaper、36 个事件、108 个 event-stage 记录。
- event 与 window 路径 SHA/canonical bytes 完全一致。
- 当前样本参数：`P=3`、`E=36`、`W=18`。
- 成本估计：`C_event=111`、`C_window=93`、`deadband=8`、`advantage=18 > 8`，稳定选择 `window`。
- 未切换场景的边界字段仍始终输出：`required_advantage_at_least=deadband+1`、`current_advantage`、`margin_work`。

### 强杀恢复

- 首批 `0..5` 完整提交，cursor `sealed_seq=5`。
- 批次 `r1-b001`（事件 `6..11`）写入 planned 后，在调用前强杀 `sh-02-relay`，产生真实 `Connection refused`，批次未提交。
- 重开账本后该批次标记为 `unknown`。
- 重启 shaper 后重放 `6..35`，生成 5 个 recovery batches。
- 中断前封存前缀 `0..5` 恢复前后 canonical bytes/SHA-256 相同：`sealed_prefix_byte_identical=true`。
- `verify`：36 events、108 judgments 全部通过，`E_VERIFY_OK`。

### 编辑失效

主账本从 revision 1 编辑 p-02 relay 速率到 revision 2：

- 失效事件：12 个（p-02 全部事件）。
- 失效 cells：24 个（relay + egress）。
- 失效窗口：6 个。
- 未受影响 cells：84 个逐字节一致。
- 编辑后 `verify --spec-dir samples/edited_param` 通过。
- `verify --spec-dir samples/base --revision 1` 也通过，证明旧版本仍可复核。

到达毫秒编辑独立账本：

- 失效事件：11 个（从目标事件开始的 p-02 后缀）。
- 未受影响 cells：75 个逐字节一致。
- 未受影响事件：25 个完整 event JSON 逐字节一致。
- 独立 verify 通过。

出口突发容量编辑独立账本：

- 失效事件：12 个（p-03 全部事件）。
- 未受影响 cells：96 个逐字节一致。
- 未受影响事件：24 个完整 event JSON 逐字节一致。
- 独立 verify 通过。

### 并发与超时

- 并发探针使用 `global_limit=4`、`pipeline_limit=2` 和 20ms 服务延迟；实测最大全局在途 4，三个管道最大在途均为 2，未超限。
- 主演练中真实调度器实测最大全局在途 5，符合 `global_max_inflight=5`；每管道最大 2，符合配额。
- 超时探针让 shaper 延迟 100ms、客户端 timeout=30ms；最终报告实测 32ms 后得到 `real socket timeout after 30 ms`，`real_socket_timeout=true`（前一次运行为 40ms）。

### 版本/摘要替换

手工把 `meta.parameter_revision` 改为 `999` 但不更新 `content_digest` 后，重新打开账本检测到 digest mismatch，旧库被隔离为 `artifacts/ledger/replacement.db.invalid-*`，新库原子整体替换，`replaced=true`。

## 4. E_SPEC 示例

把第二个 shaper 的端口改成与第一个重复后：

```json
{
  "error": "E_SPEC",
  "field": "port",
  "message": "port is duplicated",
  "object_id": "sh-02-ingress",
  "ok": false
}
```

进程退出码为 `2`；`E_VERIFY` 不一致退出码为 `3`。

## 5. 已知边界

- 这是本地诊断/演练实现，回执用于确定性复核，不提供抗恶意进程的密码学签名。
- 滑窗批量物化采用“稳定顺序内连续同类 run”以保证与逐事件判定逐字节一致；窗口聚合统计另提供窗口级汇总。
- Windows 上可直接运行 `.cmd`/`.ps1`；强杀使用 `Popen.kill()`，清理脚本在 Windows 走 `taskkill /F /T`。
