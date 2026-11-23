# 跨环境配置仲裁一致性验证工作台 — 技术说明

## 1. 模块结构

```
arbworkbench/
  errors.py           E_SPEC / E_GRID / E_VERSION / E_REPLAY 异常体系
  fixedpoint.py       10 的整数幂定点量化、饱和整数运算、误差界声明
  spec.py             环境/观测/视图 JSON 强校验（首个违规定位，整体拒绝）
  engine.py           三类仲裁口径、采纳/否决、不可区分决胜与三类归因
  canonical.py        排序键 + 固定分隔符的规范化 JSON 与 SHA-256 摘要
  versioning.py       视图版本注册表、单调递增、增删改键集合差异
  ledger.py           SQLite 账本、单视图事务、格式版本与内容摘要、崩溃恢复
  session.py          会话编排、整体替换、快照、E_VERSION/E_REPLAY 重放
  statistics_view.py  NumPy 只读观测分布统计（不参与仲裁决策）
  render.py           无显示环境文本渲染器（与 GUI 共用同一份会话数据）
  gui.py              Tkinter 操作界面（含无 Tk 时的 HeadlessUnavailable 守卫）
  cli.py / __main__.py  命令行入口与三类异常注入
sample_data/          确定性样本（5 环境、4 键、2 视图）
tests/                43 个 unittest 用例
output/               账本、快照、报告、结构化日志与验证结果
```

## 2. 强校验与错误码边界

* `E_SPEC`（结构/规范层，整体拒绝并定位首个违规）：环境标识唯一、同环境同键
  观测序号不重复、`timestamp_ms` 为非负整数、观测值为数值或二元区间、视图
  参与环境均存在且不重复、口径属于三类受支持值、JSON 可解析。
* `E_GRID`（结构合法但数值越界/规则冲突，仲裁期发现）：`scale_exponent` 超出
  `[-6, 6]`、量化后区间 `lo > hi`、标量口径遇到区间值、权重和非正、除零。
* `E_VERSION`：快照/账本格式版本、缩放口径版本、环境集合版本、视图指纹或内容
  摘要不一致，要求整体替换当前环境、观测与结论状态。
* `E_REPLAY`：恢复后按快照重算，定位到首个不一致的 `视图.键.字段`。

## 3. 三类仲裁口径（确定性）

所有集合先经全序排序（`seq` 降序 → `timestamp_ms` 降序 → 稳定标识升序），
环境按 id 升序遍历，因此与声明顺序、字典遍历顺序无关，规范化 JSON 逐字节一致。

* 最新观测优先 `latest_wins`：每个参与环境只取最新一条，再按全序取头名；其余
  否决理由 `outdated_lower_seq`，决胜时为 `stable_tie_break`。
* 加权多数 `weighted_majority`：每环境最新观测按量化值分组累计环境权重，最高
  累计权重组胜出；权威值取加权均值的整数地板；少数组否决理由
  `minority_candidate`。
* 区间交集 `interval_intersection`：每环境最新观测退化为 `[lo, hi]`（标量退化为
  零宽区间），交集为 `[max lo, min hi]`；空集标记 `policy_conflict` 并以
  `outside_intersection` 否决，非空时中点 `(lo+hi)//2`（奇数跨度向下取整）。

## 4. 定点表示与误差界（显式规则）

* 会话缩放 `S = 10 ** scale_exponent`，量化
  `q = round_half_away_from_zero(real * S)`，负指数用 Decimal 精确计算后仍得到
  整数 q 单位，决策路径不出现浮点。
* 截断方向：量化为半远离零；所有除法为地板除；区间中点奇数跨度取较低者。
* 溢出：`add/mul/div` 统一饱和到 `[-2^63, 2^63-1]` 并累计 `saturation_events`。
* 单观测量化误差 `|e_i| <= u/2`，`u = 10^-scale_exponent` 为一个量子。
* 合成中间项（加权）：`A = (u/2) * Σw`，随参与环境数与权重和**线性增长**；
  归一化后回到 `u/2`，叠加一次除法截断项得到判定界 `B_majority = 2u`。
* 判定界（q 单位）：最新 `B=1q`，加权多数 `B=2q`，区间交集 `B=2q`；权威值与
  相邻观测差值均以整数 q 表示并随结论持久化。

## 5. 不可区分决胜与三类归因

* 两个候选定点差 `<= B` 时按稳定标识（权重 → q 值 → 最小观测 id）决胜，并置
  `error_influenced=true`，归因 `error_indistinguishable`。
* `policy_conflict`：区间交集为空等口径无法合并的情形。
* `high_rejected_ratio`：被否决观测占比 `> 0.5`。
* 三类归因各自带 `attribution_contributions` 贡献说明，且仅依赖排序后的确定性
  数据，与口径执行顺序无关。

## 6. 视图版本整体替换

视图指纹 = 参与环境集合 + 每键口径的规范化摘要。新增视图版本从 1 开始，指纹
变化版本 +1，删除视图移出活动注册表；旧版本结论标记陈旧、不进入当前展示。
重算以整视图替换，并稳定输出 `added_keys / removed_keys / changed_keys`。

## 7. SQLite 账本与强制终止恢复

* 表：`ledger_meta`（整数 `format_version` 与滚动 `content_digest`）、
  `view_conclusions`（结论载荷）、`committed_views`（提交标记）。
* 每个视图在单个 `BEGIN IMMEDIATE` 事务内先写结论行、再写提交标记并更新摘要；
  进程被 kill 时只有结论行而无标记，重开账本清理孤儿行并把该视图列为
  `unknown`，要求重跑；已完整提交的视图可直接恢复。
* 重开账本先校验格式版本与内容摘要，不符即 `E_VERSION`，不混合状态。

## 8. 快照与恢复后重放

快照持久化快照格式版本、缩放口径版本、缩放指数、环境集合版本、视图版本指纹、
逐视图结论摘要与完整结论、已完成视图集合及内容摘要。重开先做版本/摘要识别，
不一致整体替换并报 `E_VERSION`；随后按快照重算，逐字段
（权威值、误差界、相邻差、误差标记、归因、采纳集合、否决集合、冲突细节）比对，
首个差异以 `E_REPLAY` 定位到 `视图.键.字段`。

## 9. Tkinter 界面与异常注入

* GUI 可编辑环境观测与视图口径、保存校验、触发仲裁、按稳定排序查看每键多环境
  分布、权威值、否决观测与理由、误差界、归因；刷新只读取会话状态，不写账本。
* 无 Tk 运行库时 `launch_gui` 抛 `HeadlessUnavailable` 并给出 Windows/ Debian
  安装指引，核心逻辑与文本渲染器仍可在无显示环境验证。
* CLI 注入仅改输入文件或参数：
  - `--inject spec_corruption`：复制样本并制造重复观测序号 → `E_SPEC`
  - `--inject grid_violation`：缩放指数改为 42（类型合法越界）→ `E_GRID`
  - `--inject version_mismatch`：篡改快照环境版本 → `E_VERSION`
  - `--crash-view <id>`：写入结论行后不提交即关闭连接，模拟 SIGKILL 恢复。
* 所有异常与生命周期事件写入 `output/events.jsonl` 结构化日志。

## 10. Windows 本地运行

```powershell
python --version                 # 需 3.11+，python.org 安装包自带 Tkinter
python -m pip install -r requirements.txt
python run_tests.py              # 单元测试
python run_demo.py               # 干净运行 + 三类注入 + 崩溃恢复
python -m arbworkbench --gui     # 图形界面（或双击 run_gui.bat）
```
