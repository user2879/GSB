# 实际验证记录

验证日期：2026-11-20（Asia/Shanghai）

## 环境

- Python：`3.11.0rc1`
- 操作系统：当前容器为 Linux；交付目标为本地 Windows Python 3.11+。
- SQLite：标准库自带。
- 当前容器没有 `tkinter`，因此未能启动 GUI 窗口；`config_ledger/ui.py` 已通过 `py_compile`，目标 Windows 发行版自带 Tkinter。
- 未使用 Docker、第三方 Python 包、公网或真实集群。

## 语法与测试

```text
$ python3 -m py_compile config_ledger/*.py tools/*.py tests/*.py
exit=0

$ python3 -m unittest discover -v
Ran 21 tests in 13.8s
OK
```

覆盖点：

- 字符串/整数期望值合法，boolean/array 期望值整体 `E_SPEC` 拒绝。
- 重复节点、重复端口、非回环 host、非法万分比一次收集并输出节点/字段路径。
- 非法声明重新加载失败时不替换内存和账本状态。
- 全局并发峰值不超过 2，节点峰值不超过 1；排队任务记录单调时钟起止。
- 超时、非法 JSON、连接拒绝分别得到 `TIMEOUT`、`INVALID_JSON`、`CONNECTION_FAILED`。
- 非法 JSON 产生 `E_OBSERVE`；连接拒绝产生 `E_OBSERVE` 与 `E_PORT`。
- 同一输入两次拉取归并得到逐字节一致的结论、分歧集合和对照记录。
- 多数仲裁、时间最新并列仲裁、缺失/超时/分歧归因、法定观测数比例全部匹配。
- 参数变更后版本从 1 单调到 2，旧轮次陈旧化，当前视图整体清空，替换事件记录旧/新结论与翻转。
- 参数切换后、新轮次提交前跨进程重启，下一次提交仍能找回最后完整提交并生成旧/新对照。
- 界面预算为 0 时两次查询都记录 `E_LATBUDGET`，两个样本仍都进入聚合。
- 摘要篡改产生 `E_SNAPSHOT`，当前轮被陈旧化，游标变为 `RERUN_REQUIRED`。
- 结论与延迟聚合篡改都能被 replay 定位到首个键/节点/字段并记录 `E_REPLAY`。
- 真实子进程在提交 `IN_PROGRESS` 后被 `kill()`；重启只得到 `UNKNOWN` 尝试和重跑要求，没有退出报告。
- 端到端 demo 覆盖 `E_OBSERVE`、`E_PORT`、`E_SNAPSHOT` 三类注入。

## 端到端演示

执行：

```text
$ python3 -m config_ledger demo --runtime runtime/demo --all-anomalies
```

关键结果：

| 场景 | 结果 |
| --- | --- |
| `divergence / none` | `app.mode` 最新候选 `canary`，多数仲裁生效值 `stable`；`tie.rule` 最新并列仲裁生效值 `beta`；replay `MATCH` |
| `missing / none` | `cache.ttl` 因 node-b 缺失而不满足法定观测数；`db.timeout` 因 node-c 真实套接字超时而不满足；`partial.quorum` 在 2/3 节点有效时仍满足 6000bp 口径 |
| `divergence / invalid_json` | node-a 返回非法 JSON，日志出现 `E_OBSERVE`，节点本轮观测缺失 |
| `divergence / port_unreachable` | node-b 未启动，日志同时出现 `E_OBSERVE` 与 `E_PORT`，轮次正常提交但标注缺失 |
| `divergence / snapshot_mismatch` | 提交后篡改账本摘要，重开得到 `E_SNAPSHOT: current content digest mismatch`，当前视图被置为不可信 |

演示还确认：

- 三个节点均只绑定 `127.0.0.1` 高位端口。
- 账本、结构化日志和运行时声明均可写入工作区。
- replay 在基线、缺失、非法 JSON 和端口不可达场景均按账本样本/聚合口径完成比较。

## 界面检查

```text
$ python3 -m py_compile config_ledger/ui.py tools/run_gui.py
exit=0
```

因当前容器缺少 Tkinter，未在本环境执行 `mainloop()`。Windows 上运行 `py -3.11 tools\run_gui.py` 会自动启动分歧样本模拟器并打开诊断台。
