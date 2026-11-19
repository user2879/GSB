# 实际验证记录

验证日期：2026-11-20  
验证环境：当前工作区 Linux 容器，Python 3.11.0rc1，SQLite 3.39.4，Tkinter 不可用。  
目标交付环境：Windows 10+ 的标准 CPython 3.11+（Tkinter 随安装器提供）。

## 自动化测试

命令：

```text
python3 -m unittest discover -v
```

结果：`Ran 18 tests in 1.150s`，`OK`。

覆盖点：

- `E_CHAIN`：重复标识、悬空上游、有环图、正整数/比例范围与字段路径。
- `E_TRACE`：时间戳倒退行号、未知服务、非法结局。
- 整数预算：窗口滚动、负余量截断、耗尽/恢复事件、逐字节确定性。
- 熔断：阈值打开、冷却、半开成功关闭、半开失败重开、迁移事件字段。
- 快速失败：触发源消费一次、上游不采样、传递深度与触发原因。
- 快照：内容摘要校验、整状态替换、结论作废计数、恢复后继续批次。
- 崩溃：未提交批次未知标注、从已提交游标重跑。
- 重放：7 条剩余调用一致；篡改状态时定位到调用 6 / 服务 `orders` / 字段 `circuit.open_count`；窗口参数变化报告 `kind=incompatible, basis=window_parameters_changed`。
- 注入与 CLI：三类业务码、NDJSON 日志、确定性批次运行。

## 完整演示

命令：

```text
python3 examples/run_demo.py
```

结果摘要：

```json
{
  "ok": true,
  "snapshot_replay": {
    "ok": true,
    "snapshot_id": "snap-000001",
    "start_cursor": 5,
    "compared_calls": 7,
    "final_cursor": 12
  },
  "unknown_batches": 1,
  "injections": [
    ["chain_corrupt", "E_CHAIN"],
    ["timestamp_regression", "E_TRACE"],
    ["snapshot_digest_mismatch", "E_SNAPSHOT"]
  ]
}
```

样本链版本：`chain-2026.11.20-r1`  
样本链摘要：`85d6a760d9533678238c7ecfdb9491a3eaf8876c73a8cfe787021cd7c7827f79`  
样本调用流摘要：`62f0ae48509fa759a03ba0861f8864b4e701c935a288e871e50a3ef77c5acc62`

状态结果：

- 12 条调用中 10 条发生快速失败。
- 预算耗尽触发 6 次，熔断打开触发 4 次。
- 快照在游标 4 做整状态恢复，作废 8 条结论，保留前 4 条有效结论；受影响服务稳定输出 `api, gateway, orders`。
- 强制终止演示中，子进程退出码为 86；批次 2 的调用 5–8 被标记为未知，`rerun_required=true`，从提交游标 4 继续处理 8 条调用至游标 12。

实际报告：`data/demo_report.json`。

## 环境说明

当前执行镜像的 `python3` 可以导入 `sqlite3`，但不能导入 `tkinter`。因此：

- `srebudget/*.py` 全部通过 `py_compile`；
- 界面数据装配逻辑由同一包、同一状态/快照 API 覆盖测试；
- 实际 Tk 窗口需在 Windows 标准 CPython 环境运行 `open_ui.bat` 或 `python -m srebudget.cli ui` 验证。

未安装 Docker、容器运行环境或第三方数值库，未访问外网。
