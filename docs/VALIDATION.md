# 实际验证记录

记录时间：2026-11-21（Asia/Shanghai）。

## 环境

```text
Python 3.11.0rc1
pip 20.3.4
NumPy 1.26.4
pytest 8.3.3
```

当前自动化运行环境是 Linux 容器，`python -c "import tkinter"` 返回 `ModuleNotFoundError`。因此本报告只验证了 GUI 模块编译和缺 Tkinter 时的受控 `E_ENV` 错误，未在该容器中做窗口渲染验证。目标 Windows 环境应先执行：

```powershell
python -c "import tkinter; print(tkinter.TkVersion)"
```

## 自动化测试

命令：

```powershell
python -m pytest -q
```

结果：

```text
...........................................                              [100%]
45 passed
```

覆盖内容：

- JSON 重复键、未知字段、重复服务、种子/记录数、比例、阈值、缩放幂、桶数校验及字段路径。
- 相同种子/记录数/字段声明逐字节复现。
- 首个耗时越界和首个时间戳倒退记录的 `E_RECORD` 定位。
- 头/尾采样互斥、客户端键/版本/缩放稳定性、集合摘要可复现。
- 切点等值右归属、记录逆序后桶统计/归因/分位逐字节一致。
- 归因三段和精确等于估计差，所有绝对误差不超过总界。
- 误差界内判定、翻转集合稳定顺序、翻面前后状态和误差界数值。
- SQLite 批次原子提交、版本单调递增、旧 run 陈旧、当前视图整体替换。
- 模拟强制终止后 `committed` 复用、`unknown` 重跑。
- 快照摘要篡改、逐桶统计篡改后的首个服务/字段定位、深重放成功。
- CLI 三类注入分别返回 `E_SPEC`、`E_BUCKET`、`E_SNAPSHOT` 并写入 JSONL 日志。

## 本地演示

命令：

```powershell
python -m trace_diag demo
```

生成的 run：

```text
run_id           run-000001-35be0696ae9e
service_version  1
strategy_version 1
snapshot_digest  30de4436825296a62947d41c443e3c7a5df7a52755737ae9621498b09046b500
services         api-checkout, api-catalog, worker-ledger
policies/service head-balanced, tail-first
records/service  600
```

默认样本下选中 `head-balanced`，服务排序及估计 p95：

```text
api-checkout   172150 µs   (p95 合成误差界 7062 µs)
api-catalog    174540 µs   (p95 合成误差界 6492 µs)
worker-ledger  286690 µs   (p95 合成误差界 10486 µs)
```

误差界内判定 4 条。实际参数翻转 2 条：

```text
head-balanced scale_power 1 -> 0
  api-checkout: head-balanced -> tail-first

head-balanced scale_power 1 -> 2
  api-checkout: head-balanced -> tail-first
  worker-ledger: head-balanced -> tail-first
```

这两条翻转保留了策略版本，只改变缩放因子，证明诊断来自定点量化/采样敏感性，而不是文件版本变化。

## 深重放

命令：

```powershell
python -m trace_diag verify
```

结果：

```text
ok: true
run_id: run-000001-35be0696ae9e
services_checked: 3
service_payload_checks: 6
bucket_checks: 192
attribution_checks: 198
snapshot_digest: 30de4436825296a62947d41c443e3c7a5df7a52755737ae9621498b09046b500
```

即 3 服务 × 2 策略 × 32 桶的记录、命中、边界和桶摘要均被独立重算；逐桶归因和总计归因也逐项比较。

## 异常注入

实际命令及退出码：

```text
analyze --inject sample          exit 2  E_SPEC    $.services[0].id
analyze --inject policy-ratio    exit 2  E_BUCKET  $.policies[0].head_ratio_bp
analyze --inject snapshot        exit 4  E_SNAPSHOT $.runs[run-000001-35be0696ae9e].snapshot_digest
```

三类事件均写入 `data/logs/trace_diag.jsonl`。快照注入检测后已恢复原摘要，随后再次 `verify` 仍成功。

## 强制终止恢复

自动化测试使用工作流内部的测试停止点，在提交首个服务批次后立即抛出中断。重新打开账本后观察到：

```text
run status: recovering
svc-a: committed
svc-b: unknown
current view: none
```

随后调用恢复路径：

```text
newly committed: svc-b
skipped committed: svc-a
final run status: complete
active: true
```

恢复完成后快照重放一致。

## GUI 环境结论

GUI 模块通过编译。当前容器缺少 Tkinter，`python -m trace_diag gui` 返回受控错误：

```text
E_ENV: Tkinter is unavailable. On Windows, reinstall Python with the Tcl/Tk option
```

Windows 本地安装带 Tcl/Tk 的 Python 后，工作台入口不依赖其他外部服务；没有做本容器内的窗口渲染截图。
