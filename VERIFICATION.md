# 实际验证记录

验证日期：2026-11-18  
项目目录：`/home/zsl/feedback/uid148`

## 环境

- Python：3.11.0rc1
- NumPy：2.4.6（用户目录安装，`import numpy` 成功）
- SQLite：3.37.2
- Tkinter：当前验证容器未安装 `python3-tk`，`import tkinter` 失败；因此未进行真实窗口渲染验证。`import exp_tracker.gui` 的无窗口安全导入已通过。
- 无 Docker、无端口监听、无外部服务。

## 自动化测试

命令：

```text
python3 -m unittest discover -v
```

结果：

```text
Ran 33 tests in 7.664s
OK
```

覆盖点：

- v1/v2/v3 归一后逐步指标、终值和指标摘要一致。
- 未知 v4 返回 `E_FORMAT`，生效路径为 `$.runs[0].format_version`；缺字段/类型错误同样带路径。
- 非法导入不创建记录库、检查点或快照数据库。
- 缺失父运行和父引用环返回 `E_RUN`，包含运行 ID 与引用链。
- 重复运行 ID、非单调记录版本返回 `E_RUN`。
- 指标序列首个偏移返回 `E_METRIC`，测试样本偏移步为 3、字段为 `loss`。
- `--stop-after=1` 中断后，第二轮只处理剩余一条；已完成记录不重复，只读备份只有一份。
- 模拟“记录已写、检查点完成标记未写”的活动条目，重启时识别目标摘要并恢复，不重复升级。
- 未知版本在备份或检查点写入前整体拒绝，`records.json` 字节不变。
- 提交 `run_version=2` 后当前记录整体替换，历史 v1 只读保留，差异报告字段稳定。
- 快照包含格式版本集合、来源版本集合、聚合口径版本、运行摘要与检查点位置。
- 快照恢复识别 `changed` 与 `removed_from_current`，恢复后重放全部通过。
- 缺失摘要、未知聚合版本、摘要不一致分别返回 `E_REPLAY`。
- 同一工作区重复执行三场景演示，JSON 输出逐字一致。
- CLI `init`、`list`、`commit` 与 `--workspace` 前后位置均通过。

## 演示验证

命令：

```text
python3 -m exp_tracker demo --workspace demo_workspace
```

结果摘要：

```text
scenarios:
  1_import_and_upgrade       — baseline v1、tuned v2 升级为 v3
  2_whole_record_replacement — baseline v1 -> v2
  3_snapshot_restore_and_replay — baseline v2 恢复，v3 变更被整体移除
replay:
  baseline pass
  ensemble pass
  tuned pass
```

`python3 -m exp_tracker curve baseline --workspace demo_workspace` 退出码为 0，`loss` 与 `accuracy` 均输出稳定 ASCII 曲线；`python3 scripts/demo.py` 退出码为 0，并以 `DEMO_OK` 结束。

演示完成后工作区包含 `records.json`、`upgrade-checkpoint.json`、`snapshots.sqlite3`、`backups/`、`reports/`。

同一工作区连续执行两次演示，输出 SHA-256 均为：

```text
bc0dd3bc18773b43e2800f4f1682e508288c5e3cf83dabb466ae723b2a33f67f
```

最终串行状态检查：

```text
formats [3]  sources [1, 2, 3]
baseline run_version=2 history=[1]
ensemble run_version=1
tuned run_version=1
snapshot_id=1 run_count=3 checkpoint_position=2
restore mode=whole_replace changed=[baseline]
replay: baseline pass, ensemble pass, tuned pass
backup modes: 444
```

CLI 实跑未知 v4 提交返回退出码 2，输出 `E_FORMAT`，生效路径为 `$.format_version`。

## 构建检查

```text
python3 -m compileall -q exp_tracker tests scripts tools
```

结果：退出码 0。
