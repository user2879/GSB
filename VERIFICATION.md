# 验收验证记录

验证日期：2026-11-23（Asia/Shanghai）

本记录对应项目根目录下的当前实现。所有持久化数据均写入项目内的 `artifacts/` 目录；演示和测试不使用网络端口、Docker 或第三方 Python 包。

## 验证环境

| 项目 | 实际值 |
| --- | --- |
| 运行环境 | WSL / Ubuntu 22.04，Linux |
| Python | 3.11.0rc1 |
| SQLite | 3.37.2 |
| Tk | 8.6 |
| 目标平台入口 | `run_gui.bat`、`run_demo.bat`，要求 Windows Python 3.10+ |

当前系统 Python 默认缺少 `libtk8.6.so`，GUI 冒烟时通过临时运行库补齐；这不改变项目依赖，Windows 标准 Python 自带 `tkinter`。桌面程序已使用真实 Tk 8.6 窗口完成冒烟。

## 复现命令

```bash
python3 -m imv.cli validate --json
python3 -m unittest discover -v
python3 -m imv.cli demo --data-dir artifacts/final-demo --json
python3 -m imv.cli demo --data-dir artifacts/verify-lease_conflict --inject lease_conflict --json
python3 -m imv.cli demo --data-dir artifacts/verify-base_corruption --inject base_corruption --json
python3 -m imv.cli demo --data-dir artifacts/verify-snapshot_tamper --inject snapshot_tamper --json
```

Windows 下可双击 `run_demo.bat` 或运行 `run_gui.bat`。交互界面复现步骤：

1. 点击“推进全部”，确认游标推进到 `10 / 10`。
2. 点击“权威重算”，确认增量缓存与权威结果逐字节一致。
3. 选择 `v_region`，点击“模拟租约过期”，确认视图先隐藏，随后自动全量重建并恢复。
4. 在“调度轨迹”“账本与重放”“诊断日志”页核对真实线程、批次事务与结构化日志。

## 输入校验

`python3 -m imv.cli validate --json` 退出码为 `0`，结果为 `status=valid`：

| 输入 | 结果 |
| --- | --- |
| 基表 | `sales_daily`，12 行，行标识 `sale_id`，摘要 `b2cbd98227dd1d77aa600b4f1f5911dc46b5fe51277fdc08903f9f2340d3fd9b` |
| 变更流 | 10 条，摘要 `188caae7bb0b49a7523d0c42f7c78eac5175da335865c69dbecf53b550cb50af` |
| 视图定义 | 3 个：`v_product`、`v_region`、`v_region_product` |
| 参数指纹 | `c5ee887d68abcfe8aa42b7f99ac4d1f92ce7fbf52c9814a3a26ca19071214dc4` |
| 缩放因子 | `100` |

## 单元测试

执行 `python3 -m unittest discover -v`，结果：

```text
Ran 18 tests in 3.214s
OK
```

| 测试文件 | 数量 | 覆盖内容 |
| --- | ---: | --- |
| `tests/test_fixedpoint.py` | 3 | 定点截断方向、`count/sum/min/max` 空值语义、int64 饱和 |
| `tests/test_scheduler.py` | 2 | 真实工作线程、全局/单视图配额、等待轨迹、续约、失败重试 |
| `tests/test_validation.py` | 5 | 列唯一、空值位图、序号连续、操作类型、行标识存在性及首个违规位置 |
| `tests/test_workbench.py` | 8 | 批次边界与视图顺序一致性、权威逐字节比较、租约重建、快照恢复、整体替换、`E_PREVI`/`E_REPLAY` 定位、异常注入 |

## 正常演示

`python3 -m imv.cli demo --data-dir artifacts/final-demo --json` 退出码为 `0`，报告 `status=pass`，9 个检查项全部为 `pass`：

| 检查项 | 实际结果 |
| --- | --- |
| 初始化 | 参数版本 `v1`，游标 `0`，快照版本 `1` |
| 增量批次 | 5 批完成，游标 `10`；批次耗时约 `18/14/18/15/17 ms` |
| 快速视图 | 从增量缓存渲染 3 个可见视图，渲染约 `0.38 ms`，未触发全量重算 |
| 权威一致性 | 3 个视图均逐字节一致；联合摘要 `a422cb36ac918b7ec26909733acf656045ac2033a80f41f5eb213d03a38081fd` |
| 分组规模 | `v_product=6`、`v_region=3`、`v_region_product=8` 个分组 |
| 租约过期 | `v_region` 在 TTL `100 ms` 后过期，结果被隐藏并进入不一致状态 |
| 全量重建 | 新租约刷新版本 `8`，重建后视图恢复，租约状态为 `completed` |
| 恢复后重放 | 游标 `10`，1 条重建轨迹，全部视图匹配 |
| 快照重开 | 快照版本 `9`，内容摘要 `78777448c78040aa9755ad3c6225c101c500b13cb2ce4cb95115fe9a4a5384d8`，匹配并重放成功 |

定义整体替换检查也为 `pass`：

- 参数版本从 `v1` 递增到 `v2`。
- `added=["v_active"]`
- `removed=["v_product"]`
- `changed=["v_region"]`
- `unchanged=["v_region_product"]`
- 缩放因子从 `100` 整体切换到 `10`。
- 替换前指纹 `c5ee887d68abcfe8aa42b7f99ac4d1f92ce7fbf52c9814a3a26ca19071214dc4`，替换后指纹 `d7a19bde77b20f9e0bdcafe5635ba8200465662d3b9acceb1ed6afcd502f1a79`。
- `v2` 快照重放与权威重算均为 `matched`。

计时数值受机器负载影响；权威摘要、分组规模和规范化 JSON 字节属于确定性结果。

## 故障注入

三组命令均退出码 `0`，报告 `status=pass`：

| 注入 | 业务码 | 结果 |
| --- | --- | --- |
| `lease_conflict` | `E_LEASE` | 以 `v_product`、`refresh_version`、实际 `1`/期望 `2` 定位乐观版本冲突 |
| `base_corruption` | `E_DATA` | 定位到 `row_id=1`、`column=region_id` 的基表行数据损坏 |
| `snapshot_tamper` | `E_SNAPSHOT` | 检测增量摘要被改为全零摘要，整体替换恢复后重放为 `matched` |

`snapshot_tamper` 的启动事件记录了快照内容摘要不符，并在同一流程中恢复结果、租约与游标；恢复后游标为 `10`，三个视图均逐字节一致。

## GUI 冒烟

在真实 Tk 8.6 窗口中自动执行“推进全部、权威重算、租约过期与自动重建”后，`artifacts/gui-smoke.txt` 记录：

```text
opened True
views 3
tabs 5
window 1440 900
cursor 10
scheduler_jobs 18
result_groups 6
advance_status 推进全部变更完成，用时 89.6 ms；界面受理 0.31 ms。
authority_status 权威逐行重算完成，用时 7.0 ms；界面受理 0.20 ms。
ledger_rows 6
lease_recovered True
log_chars 6487
tk_callback_errors 0
closed True
```

界面受理耗时低于 1 ms，权威重算在后台线程执行，租约重建后恢复正常展示，未出现 Tk 回调异常。

## 证据文件

- `artifacts/unittest-output.txt`：18 项单元测试完整输出。
- `artifacts/final-demo-report.json`：正常端到端演示报告。
- `artifacts/verify-lease_conflict.json`：`E_LEASE` 注入结果。
- `artifacts/verify-base_corruption.json`：`E_DATA` 注入结果。
- `artifacts/verify-snapshot_tamper.json`：`E_SNAPSHOT` 检测、恢复与重放结果。
- `artifacts/gui-smoke.txt`：真实 Tk 窗口冒烟读数。
- `artifacts/final-demo/ledger.sqlite3`：最终演示的 SQLite 账本与快照载荷。
- `artifacts/final-demo/logs/structured.ndjson`：最终演示的结构化业务日志。

## 结论

当前实现通过样本输入强校验、18 项自动化测试、正常端到端演示、三类异常注入和真实 Tk 界面冒烟。增量缓存与权威重算在分组集合、聚合值、空值处理、定点舍入与规范化 JSON 字节上保持一致；租约过期隐藏、全量重建、快照整体恢复、定义整体替换及恢复后重放均已验证。
