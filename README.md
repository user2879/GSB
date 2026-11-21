# 跨组件配置分发实验台（configbench）

面向 Windows 本地环境的配置分发实验台：平台以独立子进程管理多个模拟组件
节点，节点经 127.0.0.1 回环 TCP 的 JSON 行协议接收配置切片，用整数定点
口径对固定种子样本做分组聚合；支持增量编译/下发、界面缓存与账本权威重算
双路径一致性、交互延迟预算判定，以及强制终止后仅凭持久化账本恢复重放。

## 环境要求

- Windows 10/11（当前亦在无头 Linux 验证），Python 3.11
- 标准库：`sqlite3 socket subprocess threading tkinter json`
- NumPy（节点侧整数数组运算）：`pip install -r requirements.txt`

## 目录结构

```
configbench/   平台与节点实现（见 docs/DESIGN.md 的模块表）
samples/       registry.json / manifest.json / samples.json 本地声明
tests/         34 个 unittest 用例（纯算法 + 真实子进程 + 崩溃恢复）
scripts/       demo_crash_recovery.py 强制终止恢复演示
docs/          DESIGN.md 技术说明、VERIFICATION.md 实测结果、REQUIREMENTS.md 任务书
workspace/     运行产物（账本、报告、确定性摘要，运行后生成）
```

## 快速开始

```bash
pip install -r requirements.txt
# 1. 强校验清单/注册表/样本
python -m configbench.cli validate
# 2. 启动子进程节点，发布全部版本并做双路径一致性与延迟统计
python -m configbench.cli --ledger workspace/bench.db run
# 3. 强制终止 -> 账本恢复 -> 重放校验演示
python scripts/demo_crash_recovery.py
# 4. 针对已有账本做恢复/查询/权威重算/延迟报告
python -m configbench.cli --ledger workspace/bench.db recover
python -m configbench.cli --ledger workspace/bench.db query
python -m configbench.cli --ledger workspace/bench.db recompute 2
python -m configbench.cli --ledger workspace/bench.db latency
```

图形界面（Windows 自带 tkinter）：

```python
from configbench.ui import open_gui
open_gui(platform).run()
```

无 tkinter/无显示环境可用 `configbench.ui.render_terminal_view(platform)`
得到等价的终端视图。

## 运行测试

```bash
python -m unittest discover -s tests -v
```

## 关键设计约束（摘要）

- 定点：`raw/10**scale`，向零截断，int64 饱和；规范化 JSON 保证摘要逐字节一致。
- 增量：内容寻址产物，未变项字节不变；仅受影响节点下发，切片含派生依赖闭包。
- 调度：全局 + 每节点并发配额、排队单调时钟、真实超时、可重试错误才重试。
- 双路径：缓存快速展示 vs 账本逐条重放，摘要逐字节一致、耗时在误差界内。
- 账本：SQLite 事件溯源 + 链式 SHA256 + 整数格式版本；损坏整体替换并保留
  只读备份；未提交任务标记 UNKNOWN 重跑，不生成退出报告。

详细规则见 `docs/DESIGN.md`，实测数据见 `docs/VERIFICATION.md`。
