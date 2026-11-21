# 应急预案演练诊断工作台 — 技术说明与验证报告

## 1. 环境与约束
- 目标环境：本地 Windows，Python 3.11+；仅使用标准库 `tkinter / sqlite3 / json / hashlib / threading / logging`。
- 无外部服务、账号、网络下载或容器；样本由构建脚本本地确定性生成。
- 逻辑时钟完全由注入偏移与阶段耗时推导，代码中不存在 `time.sleep` 驱动的状态推进。
- 本仓库在 Linux 沙箱开发，该沙箱未安装 `tkinter` 运行库；Tk 在 `emergency_drill/ui.py` 中惰性导入，
  无显示环境自动降级为稳定文本视图。Windows 官方 Python 安装包默认包含 Tkinter，可直接开窗。

## 2. 目录结构
- `emergency_drill/models.py`：不可变数据模型、规范化 JSON（固定键序、紧凑分隔）、SHA-256 摘要、强校验。
- `emergency_drill/graph.py`：Kahn 稳定拓扑序、E_CYCLE 环检测、直接/间接下游、传播依据、跳过溯源链、引用者。
- `emergency_drill/loader.py`：本地 JSON 加载，损坏 JSON 与结构违规均定位到首个违规位置。
- `emergency_drill/engine.py`：显式状态迁移许可表、单调逻辑时钟、确定性模拟、失败级联、增量重算、E_INCR 比对。
- `emergency_drill/editing.py`：五类变更集、E_DEPEND 引用保护、失效集合与传播依据、逐字节冻结校验。
- `emergency_drill/repository.py`：SQLite 账本，迁移/溯源/版本/游标同事务，摘要链与重放校验。
- `emergency_drill/session.py`：首轮演练、增量重算、新一轮整体替换、快照存取、E_SNAPSHOT/E_REPLAY。
- `emergency_drill/ui.py` / `viewmodel.py`：Tk 诊断界面与只读文本视图（刷新不写账本）。
- `emergency_drill/cli.py`：统一命令行；`crash_worker.py`：强制终止子进程。
- `tests/`：29 个 unittest 用例；`scripts/`：Windows `.bat` 与跨平台 `.sh` 验收脚本。

## 3. 业务码映射
| 业务码 | 触发条件 | 首个定位信息 |
| --- | --- | --- |
| E_PLAN | JSON 损坏、标识重复、序号不连续、耗时/偏移非法、引用不存在、事件作用阶段缺失 | `$.stages[i].field` 或 `文件:line col` |
| E_CYCLE | 前置引用成环 | 环上阶段列表 |
| E_DEPEND | 删除仍被引用的阶段 | 引用者稳定列表 |
| E_INCR | 增量结论与整体重跑首个字段差异 | `stage=<id>.field=<field>` |
| E_SNAPSHOT | 快照内容摘要不符或版本/预案/事件摘要不一致 | 期望/实际摘要、差异列表 |
| E_REPLAY | 按记录版本与注入序列重放，首个阶段字段不一致 | `stage=<id>.field=<field>` |
| E_LEDGER | SQLite 摘要链或载荷摘要被篡改/写入不完整 | 表与版本 |
| E_STATE | 在存在 unknown 阶段时声称生成强制终止退出报告 | 阶段标识 |

## 4. 关键设计决策
- 稳定顺序：拓扑序同层按 `(order, stage_id)` 排序；所有列表输出可重复，界面渲染顺序固定。
- 逻辑时钟：`LogicalClock.advance` 拒绝回退；阶段起点取前置结束与当前时钟最大值，终点叠加整数毫秒耗时。
- 失败级联：失败阶段的全部下游按拓扑序迁入 `skipped`，`skip_trace` 保存失败根到该阶段的稳定最短路径。
- 增量重算：冻结阶段沿用历史业务载荷（除当前版本归属字段外逐字节一致），仅失效子集重新推进状态机；
  `verify-incr` 对全部阶段整体重跑并逐字段比较，首个差异即 E_INCR。
- 版本与陈旧：每次编辑版本号 +1；旧版本结局/事件结论 `stale=1`，当前统计只聚合非陈旧行。
- 原子持久化：每次迁移在单个 SQLite 事务内写 `transitions / skip_traces / progress`，并推进哈希链头摘要。
- 强制终止：只恢复已完整提交的迁移；存在 running 而无终态迁移的阶段写入 `unknown` 结局，
  `recover` 返回码 2 且拒绝退出报告，必须执行 `rerun-after-crash` 后才能产出报告。
- 快照：摘要计算排除 `content_digest` 自身；重放按快照版本的预案与注入序列完整模拟后逐阶段比对。

## 5. Windows 本地操作
```bat
py -3 -m emergency_drill.cli generate-samples --sample-dir samples
scripts\run_demo_windows.bat
scripts\crash_recovery_windows.bat
py -3 -m emergency_drill.cli view --ledger work\acceptance\drill.sqlite3 --text
py -3 -m emergency_drill.cli view --ledger work\acceptance\drill.sqlite3
py -3 -m unittest discover -s tests
```

## 6. 实际验证结果（Linux 沙箱 / Python 3.11.0rc1，逻辑与 Windows 一致）
- 单元测试：`Ran 29 tests ... OK`，覆盖强校验、状态机、失效传播、增量一致性、账本、快照、子进程崩溃恢复。
- 增量验收：修改 `recover` 耗时与判据后，失效集合仅 `["recover"]`，冻结 4 个上游阶段；
  `verify-incr` 返回 `matched=true`，冻结阶段剔除版本归属字段后逐字节一致。
- 整体重跑比对：5 个阶段全部字段匹配，未出现 E_INCR；人为篡改结论的单测能定位到 `stage=recover.field=end_clock_ms`。
- 强制终止：子进程退出码 137，恢复识别 `unknown_stages=["detect"]`、`can_build_exit_report=false`、返回码 2；
  重跑后 `unknown=0` 且可生成报告。
- 异常注入：损坏 JSON -> E_PLAN（含 line/col），成环 -> E_CYCLE（含环列表），篡改快照 -> E_SNAPSHOT（含期望/实际摘要）。
- 结构化日志：`work/acceptance/drill.log.jsonl` 为 JSON Lines，含时间、级别、业务码、位置与上下文。
- 只读界面：连续两次渲染前后 `ledger_digest` 不变，失效集合、时间轴、跳过链、陈旧版本均稳定展示。

## 7. 备注
- `threading` 属要求确认的可用标准库；本方案因确定性模拟不需要并发，未引入线程等待，避免非确定性。
- 若在 Windows 启动 Tk 窗口报缺少 Tk，请重新运行 python.org 安装包并勾选 “tcl/tk and IDLE”。
