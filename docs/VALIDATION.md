# 实际验证记录

验证时间：2026-11-25（Asia/Shanghai）。

## 环境

- Python：`3.11.0rc1`
- NumPy：会话最初检测为 `1.26.4`；交付前最终复跑时当前 shell 解析为 `2.4.6`。代码仅使用稳定的 `numpy.random`/`numpy.dot`/`floor_divide` 整数接口，两个环境下验收均通过。
- 操作系统：当前 WSL/Ubuntu 工作环境
- 网络：仅使用 `127.0.0.1` 高位端口
- Tkinter：模块加载失败，原因是系统缺少 `libtk8.6.so`；官方 Windows Python 安装包包含 Tk/Tcl。Tkinter 页面代码已通过 `py_compile`，其无界面视图模型和逐字节比对已包含在验收中。

## 已执行验证

### 1. 单元测试

命令：

```bash
python3 -m unittest discover -s tests -v
```

结果：

```text
Ran 7 tests in 0.145s

OK
```

覆盖：

- `E_SPEC` 首个非法实体/字段定位；
- `E_CYCLE` 与环路径；
- 定点向下截断和误差上界；
- 依赖传播稳定拓扑序；
- 快照首个不一致分片/字段定位；
- 执行器多分片批次的确定性摘要；
- 反转池遍历顺序后抢占牺牲集合仍为 `["alpha-old"]`。

### 2. 一键业务验收

命令：

```bash
python3 -m gpu_scheduler verify --workdir runtime/delivery-final
```

退出码：`0`。

摘要：

```json
{
  "ok": true,
  "failures": [],
  "sections": {
    "errors": {"ok": true, "check_count": 4},
    "kill": {"ok": true, "check_count": 5},
    "preemption": {"ok": true, "check_count": 11},
    "propagation": {"ok": true, "check_count": 7},
    "watermark": {"ok": true, "check_count": 6}
  }
}
```

完整报告：

```text
runtime/delivery-final/accept-1795548452243/acceptance-report.json
```

该次真实执行确认：

- 抢占样本中 `low-new` 被稳定选中，`low-new#0` 只有一次执行回执；
- 挂起行游标为 `next_shard_id=low-new#1`，恢复后只执行 `#1/#2`；
- 抢占传播集合为 `low-new#0/#1/#2` 的下游并集，账本含 `trigger_type=preempted`；
- 失效传播集合为 `etl#0, etl#1, etl#2, mart#0`；
- 重试后 `branch#0`、`branch#1` 的原始回执行、规范字节和摘要逐字节不变；
- `SIGKILL` 后恢复报告含 `unknown_shards=["recover-me#1"]`（不同调度时序下可能是其他 open attempt 分片，但必为非空 unknown 集合），重跑后 2/2 completed；
- `SIGKILL` 路径没有生成 exit report；
- `SIGTERM` 返回 `143`，并生成 `reports/latest-exit-report.json`，其中 `complete=true`、`signal=SIGTERM`；
- UI 缓存与 SQLite 权威重算规范字节一致；
- 快照逐轮重放和终局摘要比较通过。

### 3. 异常业务码抽查

```text
validate samples/cycle_injection.json                 -> exit 4 / E_CYCLE
run samples/preemption.json --inject E_PORT           -> exit 3 / E_PORT
run samples/propagation.json --inject snapshot        -> exit 5 / E_SNAPSHOT
```

快照外层 `content_sha256` 被改为 64 个 `0` 时，`read_snapshot` 同样定位到：

```text
E_SNAPSHOT locator.field=content_sha256
```

当前声明与快照的 `scale_power`/摘要不一致时，`restore_and_verify` 返回：

```text
merge_mode=whole_replace
declaration_matches=False
restored_spec_path=<workdir>/restored-spec.json
```

### 4. 编译检查

```bash
python3 -m compileall -q gpu_scheduler tests main.py
```

结果：退出码 `0`。

## 复现说明

优先运行：

```bash
python3 -m gpu_scheduler verify --workdir runtime/reproduce
```

该命令会创建新的时间戳验收目录，避免复用旧 SQLite 状态。手动分步命令见 `docs/USAGE.md`。
