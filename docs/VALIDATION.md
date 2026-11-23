# 实际验证记录

验证日期：2026-11-24（Asia/Shanghai）

## 环境

| 项目 | 实际值 |
|---|---|
| 执行器 | Linux 6.18.33.2-microsoft-standard-WSL2 x86_64 |
| Python | 3.11.0rc1 |
| SQLite | 3.37.2 |
| Tkinter | 不可用：缺少 `libtk8.6.so` |
| 外部依赖 | 无 |
| Docker/容器 | 未使用 |

说明：当前执行器是 WSL/Linux，不是原生 Windows。核心业务、SQLite、文件原子替换、子进程强杀和 Tk 缺失降级已实际执行；Windows 专用的保留路径拒绝与 `.venv\Scripts\activate` 流程未在本执行器中直接运行。代码对 Windows 路径、`os.replace` 和 Tkinter 安装差异做了显式处理。

`python -m venv .venv` 已在本执行器尝试，但该精简 Python 缺少 `ensurepip`，故虚拟环境创建失败并已清理。Windows 官方 Python 安装器通常自带 `venv`/`ensurepip`；README 给出了完整命令。若仍失败，请在 Python 安装向导中勾选 `pip` 与 `tcl/tk` 组件。

## 命令与结果

```bash
python --version
# Python 3.11.0rc1

python -m unittest discover -s tests -v
# Ran 31 tests
# OK

python scripts/run_validation.py
# EXIT 0
# status: PASS
```

验证脚本实际执行了测试套件和完整 demo，并生成：

- `artifacts/validation-results.json`：demo 原始结构化报告。
- `artifacts/validation-summary.json`：环境、测试退出码、demo 阶段与 Tk 检查摘要。

## 容量证明

初始样本：

| 段 | 容量 | 已用 | 剩余 | 证明内容摘要 |
|---|---:|---:|---:|---|
| `seg-alpha` | 256 | 112 | 144 | `sha256:e21fa2557d617b680d04322ae341e86e4ec2574da6ce7d10a506b6831aa56f19` |
| `seg-beta` | 320 | 128 | 192 | `sha256:dab6c80ea6cce32dce14e8601143c1b95549da1edbcf2eb2bda150d3c367a3b8` |

`verify` 逐条重算内容文件摘要并确认：

- `used_bytes + remaining_bytes == capacity_bytes`
- 条目摘要清单与内容目录一致
- 存储证明与独立重建证明逐字节一致
- SQLite 账本内容摘要一致

## 容量拒绝

对 `seg-alpha` 写入 10000 字节：

```text
code=E_CAPACITY
required_bytes=10000
available_bytes=144
proof_unchanged=true
```

证明摘要仍为 `sha256:e21fa255...`。

## 写入、删除与视图重建

写入 `alpha/demo-new`（40 字节）后删除 `alpha/a`：

- 完成游标：写入 1，删除 2。
- `seg-alpha` 已用从 112 变为 104。
- 内容文件实际删除。
- 证明仍通过复核。

`view-audit` 从版本 2 整体重建为 3：

- `seg-alpha` 新增 `alpha/demo-new`
- `seg-alpha` 删除 `alpha/a`
- `seg-alpha` 摘要变化 `alpha/b`
- `seg-beta` 新增 `beta/x`
- `seg-beta` 删除 `beta/legacy`

`view-orders` 从版本 1 整体重建为 2：

- 新增 `alpha/demo-new`
- 删除 `alpha/a`

重复查询不再改变视图版本，差异 JSON 保持稳定。

## 快照重放

保存快照后再写入 `alpha/replay`：

```text
snapshot_replaced=true
replayed_operations=1
operation_codes=[{"status":"committed","recorded_code":"OK","computed_code":"OK"}]
```

恢复后快照整体替换，容量证明与账本复核仍为 `OK`。再次恢复返回 `replayed_operations=0` 且不替换快照。

另外通过篡改测试验证：

- 篡改快照条目摘要并重封证明/快照：`E_REPLAY`，定位 `seg-alpha / alpha/a / entries.digest`。
- 篡改操作业务码：`E_REPLAY`，定位 `seg-alpha / alpha/code / result_code`。
- 篡改快照整体摘要：`E_SNAPSHOT`，字段 `snapshot.content_digest`。

## 强制终止恢复

子进程在内容文件原子替换后、证明事务提交前调用 `os._exit(91)`。

重启结果：

```text
worker_exit_code=91
proofs_still_ok=true
exit_report_available=false
unknown:
  segment_id=seg-alpha
  entry_key=alpha/crash
  expected_digest=sha256:85aa2379a05e39a29815b094f328df2d8ebb68fc5aaf545b075c46d95ecf6c66
  observed_digest=sha256:85aa2379a05e39a29815b094f328df2d8ebb68fc5aaf545b075c46d95ecf6c66
  observed_size_bytes=24
  reason=interrupted_after_file_replace
```

该条目没有进入容量证明；未知文件被允许作为审计对象存在，但不会被冒充为已提交条目。恢复报告明确没有退出报告。

## 结构化异常注入

命令：

```bash
python -m cacheproof diagnose --workspace <workspace> --inject truncate --json
python -m cacheproof diagnose --workspace <workspace> --inject capacity --json
python -m cacheproof diagnose --workspace <workspace> --inject ledger-digest --json
```

实际结果：

| 注入 | 退出码 | 业务码 | 定位 |
|---|---:|---|---|
| 截断内容文件 | 4 | `E_INTEGRITY` | `seg-alpha / alpha/a / size_bytes`，48→47 |
| 容量参数小于已用 | 3 | `E_CAPACITY` | `seg-alpha / capacity_bytes`，required 112 / available 111 |
| 账本内容摘要不符 | 5 | `E_SNAPSHOT` | `ledger.content_digest` |

日志字段实际包含：

```json
{
  "event": "business_error",
  "business_code": "E_INTEGRITY",
  "event_seq": 16,
  "operation_seq": 16,
  "monotonic_ns": 15426548293707
}
```

`event_seq`、`operation_seq` 与 `monotonic_ns` 类型均为整数。

## Tkinter 降级

在本执行器运行：

```bash
python -m cacheproof ui --workspace demo-workspace --json
```

实际返回退出码 7 和：

```text
E_TK_UNAVAILABLE: Tkinter is unavailable; install the Python tcl/tk component
```

同时写入 `business_error` JSON 日志。无界面 `show` 与 `dashboard_snapshot` 仍正常输出段容量、已用/剩余、逐条目摘要结论、证明摘要、视图差异、未知条目、段集合版本和账本批次。

## 可重复性说明

- 31 项测试连续通过；demo 在同一操作序列下业务码、段证明和视图差异一致。
- 操作 UUID、真实单调时钟和日志序号天然反映本次运行，不作为“逐字节一致”的对象。
- 容量证明、内容文件、视图快照摘要由确定性规则生成。
