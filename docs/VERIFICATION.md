# 实际验证记录

验证时间：2026-11-26（Asia/Shanghai）。

## 环境

```text
Python: Python 3.11.0rc1（满足 >= 3.10）
NumPy:  1.26.4（按任务要求固定）
OS:     Linux/WSL Ubuntu
Bind:   127.0.0.1 高位端口可由 OS 分配并监听
```

当前解释器不是 `/usr/bin/python3`，初始环境已有 `_tkinter` 扩展但缺少 `libtk8.6.so` / `libtcl8.6.so`，且 `sudo` 需要密码。验证时用 `apt-get download` 将官方 `libtcl8.6`、`libtk8.6` 及 X11 依赖解包到 `/tmp/reconflow-tkdebs/root`，通过 `LD_LIBRARY_PATH`、`TCL_LIBRARY`、`TK_LIBRARY` 启动真实 Tk 窗口。普通 Ubuntu 可直接执行：

```bash
sudo apt-get install -y python3-tk
```

## 一键结果

执行：

```bash
TKROOT=/tmp/reconflow-tkdebs/root \
LD_LIBRARY_PATH="$TKROOT/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" \
TCL_LIBRARY="$TKROOT/usr/share/tcltk/tcl8.6" \
TK_LIBRARY="$TKROOT/usr/share/tcltk/tk8.6" \
PYTHONPATH=src \
python3 -m reconflow demo --root artifacts/demo
```

`artifacts/demo/verification_report.json` 中 `success=true`。关键结果：

- 强制终止：子进程返回码 `-9`，确认 `forced_sigkill=true`。
- 恢复：识别出 `8` 个无完整提交结局的批次，首个为 `alpha#b0001`；重跑 `8` 个批次。
- 重放：`checked_batches=12`、`checked_items=15`、`identical=true`、`config_version=5`。
- 最终读取耗时：初始约 `0.294ms`，最终组合事务前读取约 `0.310ms`（单调时钟实测，机器负载下会有小幅波动）。
- 最终结论摘要：`82ac85cad6c7a0bd7717ba8d3a1393e24a0bd8fbb52dd467748c4c4f70ee7cea`。
- 快照数量：`2`（v1 与 v4）。
- E_SPEC 自检：负种子位于 NDJSON 第 `2` 行、字段 `seed`。
- TCP 错误自检：`E_PORT`、`E_TIMEOUT`、`E_FIELD` 均按预期触发。

## 增量与权威重算

三类连续编辑的计划失效集合与实际摘要变化集合完全相等，且快速/权威规范字节一致：

| 编辑 | 计划/实际失效分片 | 字节一致 |
| --- | --- | --- |
| `alpha` 权重改为 4000‱ | `S0003 S0004 S0005 S0006 S0007 S0008` | true |
| 阈值改为 8800‱ | `S0005 S0006` | true |
| 修改一篇选中文档的词项向量 | `S0001 S0002 S0003` | true |
| 分片大小从 6 改为 8 | 新分片 `S0001..S0006` 全部重算，旧 `S0007/S0008` 标记移除 | true |

最终字节文件校验：

```text
$ cmp artifacts/demo/cache/fast_conclusion.canonical.json \
      artifacts/demo/cache/authoritative_conclusion.canonical.json
CMP_IDENTICAL

sha256sum:
82ac85cad6c7a0bd7717ba8d3a1393e24a0bd8fbb52dd467748c4c4f70ee7cea  fast
82ac85cad6c7a0bd7717ba8d3a1393e24a0bd8fbb52dd467748c4c4f70ee7cea  authoritative
```

未受影响分片与配比项的逐项 `byte_identical` 检查包含在每次编辑返回的 `preservation_checks` 与 `verification_report.json` 中。

## TCP 双预算

全量 TCP 复核：

```json
{
  "task_count": 9,
  "peak_global_points": 18,
  "peak_domain_points": {"alpha": 8, "bravo": 7, "cyan": 6, "delta": 8},
  "wait_records": 6,
  "invariant_violations": [],
  "matches_conclusion": true
}
```

配置预算为 `global_sampling_point_budget=18`、`domain_sampling_point_quota=8`；全局峰值等于但不超过预算，领域峰值 `8 <= 8`。等待开始/结束 monotonic ms 记录在调度结果中。

## 自动化测试

执行：

```bash
TKROOT=/tmp/reconflow-tkdebs/root \
LD_LIBRARY_PATH="$TKROOT/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" \
TCL_LIBRARY="$TKROOT/usr/share/tcltk/tcl8.6" \
TK_LIBRARY="$TKROOT/usr/share/tcltk/tk8.6" \
PYTHONPATH=src \
python3 -m unittest discover -s tests -v
```

实际结果：

```text
Ran 14 tests in about 9s

OK
```

覆盖：

- 生成样本强校验与 `E_SPEC` 行号/字段。
- 领域顺序、NDJSON 批次逆序、传输重切分和结论字节确定性。
- TCP 远程分片摘要与本地一致、预算不越界。
- 权重/阈值失效子集精确且未受影响字节保持。
- 分片大小编辑与 `E_DIVERGE` 字段定位。
- 完整 SIGKILL、unknown 恢复、重跑、重放。
- 账本篡改触发带批次和字段的 `E_REPLAY`。
- `E_PORT/E_TIMEOUT/E_FIELD`。
- 真实 Tk 诊断窗口构建、刷新和 100ms 冒烟关闭。

## Tk 界面

实际冒烟：

```bash
PYTHONPATH=src python3 -m reconflow ui --root artifacts/demo --smoke-ms 500
# exit code 0
```

无显示环境时会返回 `E_UI_DISPLAY` 结构化错误；同一诊断数据可在 `artifacts/demo/diagnostic_view.txt` 查看稳定文本投影。

## 可重复入口

```bash
./scripts/verify.sh
```

该脚本会安装固定 NumPy、重建 `artifacts/demo`、运行端到端 demo 和 unittest。若系统尚未安装 Tk，请先执行 `sudo apt-get install -y python3-tk`。
