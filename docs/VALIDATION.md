# 实际验证记录

验证日期：2026-11-27（Asia/Shanghai）

## 环境

- WSL Ubuntu 工作目录：`/home/zsl/feedback/uidfc5909d3-default-405`
- Python：3.11.0rc1（满足 Python 3.10+）
- NumPy：1.26.4
- pytest：8.2.0
- Tkinter：可导入，且本环境 `tk.Tk()` 构造成功
- 网络：仅使用 `127.0.0.1` 随机/声明回环端口

## 自动化测试

命令：

```bash
python3 -m pytest
```

实际结果：

```text
....................................                                     [100%]
36 passed in 27.92s
```

完整终端输出保存于 `artifacts/pytest-validation.txt`。

覆盖：

- `E_SPEC` 首个字段/JSON Pointer 位置、重复 ID/端口、环、版本链和事件序号。
- 确定性样本、NumPy 聚合、批次切分/乱序/并发一致性。
- 真实套接字握手/读取超时、缺字段 `E_FIELD`、端口占用 `E_PORT`。
- 三节点真实子进程并行调度和资源释放。
- 全局预算排队、等待时间字段、每节点租约上限。
- 单调时钟真实 TTL 过期、`E_LEASE` 乐观续约、过期/容量换代回收互斥。
- 定点水位、三类误差归因、误差带决胜。
- 冻结分片不下发、不预占资源。
- 容量版本/配额演进的失效集合、传播依据和未失效记录字节不变。
- SQLite 原子提交、快照版本/摘要不符整体替换。
- 真实 `SIGKILL`/`wait()` 返回 `-9`、未提交分片 unknown、资源回收、重启和逐字段重建一致性。
- Tkinter 窗口构造冒烟（无 `DISPLAY` 时自动 skip）。

## 一键端到端验收

命令：

```bash
python3 -m workbench acceptance --output artifacts/acceptance
```

实际结果：

```json
{
  "ok": true,
  "checks": [
    "parallel_schedule",
    "baseline_replay",
    "water_error_model",
    "snapshot_version_replace",
    "sigkill_recovery_field_equality",
    "incremental_invalidation",
    "blank_rebuild_field_equality"
  ],
  "error": null
}
```

机器可读完整报告：`artifacts/acceptance/acceptance-report.json`。
终端输出：`artifacts/acceptance-cli.txt`。

其中 SIGKILL 验收包含：

- 对真实 `n1` 子进程调用 `kill()` 并观察 `returncode == -9`。
- `s3` 有已提交的在途租约/占用但无完整结局，恢复为 `unknown/E_REPLAY_RERUN_REQUIRED`。
- `s1` 已完整提交；杀进程前后其 shard、lease、分片水位和误差记录 canonical JSON 完全相同。
- 重启节点并重跑后，与 ScriptedClock 下从空白全量重建的六分片结论视图逐字段一致。

增量演进验收包含：

- 将 `n1.capacity_version` 从 2 升级到 3（版本 3 的父版本为 2）。
- 稳定失效集合为 `s1, s3, s5, s6`。
- 未失效集合为 `s2, s4`，其整行及租约/分片水位/误差记录字节不变。
- 演进后库与新规范空白重建通过 `compare_conclusion_views()` 逐字段比较。

## CLI 示例运行

```bash
python3 -m workbench validate examples/sample-spec.json
python3 -m workbench run examples/sample-spec.json --workdir artifacts/cli-run --serial
```

实际结果：`s1` 至 `s6` 全部 completed，并生成：

- `artifacts/cli-run/workbench.sqlite3`
- `artifacts/cli-run/session-snapshot.json`

## 复现说明

测试和验收会动态申请空闲回环端口；`examples/sample-spec.json` 使用固定端口 `39101-39103`。如固定端口被占用，可重新生成：

```bash
python3 -m workbench init examples/sample-spec.local.json
```
