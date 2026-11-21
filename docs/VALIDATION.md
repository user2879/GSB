# 实际验证记录

环境：

```text
Linux 验证容器
Python 3.11.0rc1
NumPy 1.26.4
SQLite 可用
Tkinter 不可用（容器未安装；Windows 官方 Python 默认包含）
```

命令：

```text
python3 -m unittest discover -s tests -v
python3 scripts/demo.py
```

结果：

- 自动测试：29/29 通过。
- 演示检查：14/14 通过。
- 主报告摘要：`863e9a29bf31e8a9ff488ac8e0b63049d5b5faf53ec3407d5ad81b0713ab269d`。
- 回环正常重放：通过并持久化 ACK。
- 回环不一致注入：`E_ECHO`，首个位置为 A 流序号 1 的 `gap_presence`。
- SIGKILL 后恢复：`complete_incident_reconstruction=false`，未确认作业列出三项未确认字段。
- 样本损坏：`E_WIRE`，第 27 帧，偏移 1404。
- 清单版本冲突：`E_FORMAT`，字段 `version`。
- 快照摘要不符：`E_SNAPSHOT`。
- 账本输入漂移：`E_REPLAY`，首个位置为 A 流缺口 `3..3`，字段 `gap_presence`。

演示报告位于 `artifacts/demo/demo_report.json`。
