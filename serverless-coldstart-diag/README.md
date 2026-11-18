# Serverless 冷启动与并发控制诊断台

面向本地 Windows 环境的领域可视化与诊断工作台。以本地 JSON 清单声明版本化函数、
按分钟的确定性调用序列与并发控制策略，逐分钟推进时间轴，用**纯整数定点口径**估算
实例需求、热/冷调用拆分、冷启动次数、排队等待及其量化误差上界，并支持声明版本演进后的
陈旧结论整体替换、SQLite 账本持久化、强制终止恢复与一致性重放。

## 1. 环境准备

- Python 3.11 或以上（Windows 推荐 python.org 安装包，自带 `tkinter` 与 `sqlite3`）。
- 依赖：`numpy`（仅用于展示性派生分析，见 `coldstart_diag/analytics.py`）；
  `sqlite3`、`tkinter` 为标准库。
- 安装：`python -m pip install -r requirements.txt`
- 不使用 Docker 或任何容器；样本全部由确定性规则生成，不访问网络目标。

> 本机（Linux 容器）Debian Python 未发行 `tkinter` 且无 apt 权限，GUI 模块因此惰性
> 导入 Tkinter：无显示/缺 Tkinter 时打印友好提示，引擎、账本、测试、演示均可无头运行；
> 在标准 Windows Python 上执行 `run_gui.bat` 即可打开工作台。

## 2. 目录结构

```
serverless-coldstart-diag/
├─ coldstart_diag/
│  ├─ errors.py       # E_SPEC / E_REPLAY 错误码与带路径异常
│  ├─ fixedpoint.py   # 10^k 缩放因子、floor 定点比例、单值/比较误差界
│  ├─ spec.py         # 清单加载与强校验（整体拒绝，输出字段路径+违规值）
│  ├─ engine.py       # 逐分钟需求推导、实例裁决、热冷拆分、误差合成、聚合
│  ├─ ledger.py       # SQLite 账本（单分钟事务、格式版本、摘要、恢复）
│  ├─ replay.py       # 一致性重放与 E_REPLAY 不可复现定位
│  ├─ versioning.py   # 版本单调守卫、陈旧识别、整体替换、不可区分判定
│  ├─ workspace.py    # 快照、稳定视图模型、热力分档、等待曲线序列
│  ├─ analytics.py    # NumPy 展示性分析（分位数/热力直方图）
│  ├─ samples.py      # 确定性样本生成（无随机数）
│  ├─ demo.py         # 三场景可重复演示
│  └─ gui.py          # Tkinter 工作台（惰性导入，可无头保护）
├─ tests/             # 33 个 unittest 用例
├─ samples/           # manifest_v1/v2/v3.json（演示自动生成）
├─ workspace/         # ledger.sqlite、snapshots/、diagnostic_summary.json
├─ requirements.txt
├─ run_demo.bat / run_gui.bat / run_tests.bat
└─ README.md
```

## 3. 清单格式（强校验，违规整体拒绝 E_SPEC）

- `declaration_version` 正整数；`scale` 必须是 10 的非负整数次幂。
- `functions[]`：`function_id` 非空；`function_version` 同函数内唯一且单调递增；
  `memory_mib` ∈ [64,10240]、`init_ms` ∈ [1,600000]、`concurrency` ∈ [1,1000] 正整数；
  `min_instances`/`max_instances` 非负且 min ≤ max。
- `call_sequence[]`：`minute` 从 0 起连续不重复；`calls` 非负；`burst_permyriad` ∈ [0,10000]；
  且 `(function_id, function_version)` 必须引用已声明版本。
- `policy`：万分比参数 ∈ [0,10000]，并发/内存/预留为受界正整数。
- 校验失败时 `SpecError.details` 列出**全部**出错 `path`、违规 `value` 与原因。

## 4. 定点口径与误差界（全部整数运算）

- 缩放因子 `S=10^k`；比例 `q = floor(numerator*S/denominator)`，截断方向显式为
  floor（非负域即向下截断）。
- 单值量化误差：`0 ≤ 真值*S − q < 1`，上界 1 ulp；比例比较误差界为两侧之和 2 ulp，
  `|q_a−q_b| ≤ 2` 判定**不可区分**。
- 端到端冷启动比例误差界：对内存（±1 MiB）、并发度（±1）、预热比例（±1/10000）
  分别做 ±1 步长扰动重算，取最大偏差作为各输入贡献，再加本征 floor 的 1 ulp；
  同时输出各输入与本征量化的贡献占比（定点），见 `engine.synthesize_error_budget`。
- 均值/聚合比例同样 floor 输出并附 `agg_error_bound_ulp` 与舍入方向。

## 5. 实例裁决（确定性、可重裁、不依赖哈希顺序）

实例以有序整数集合表示（索引 0..N−1），余数按索引前缀稳定分配（`stable_allocate`）。
关键约束：全局并发换算实例上限 `floor(global_concurrency_cap/eff_conc)`、
内存预算 `floor(global_memory_mib/memory_mib)`；预热目标
`max(reserved_instances, ceil(desired*warmup/10000))` 再夹到容量上限。
预留与全局上限冲突时按“容量优先”稳定夹取，并在 `arbitration_reasons` 记录
`GLOBAL_CONCURRENCY_CAP_BINDING` / `RESERVED_VIOLATES_GLOBAL_CAP` 等依据。
同一分钟任意重裁得到同一实例集合与同一冷启动计数。

## 6. 版本演进、账本与重放

- 声明版本必须单调递增；旧版本分钟整体移入 `stale_minutes`，**不参与当前展示与汇总**；
  当前视图整体替换（`FULL_REPLACE`），稳定列出变化分钟、字段与旧新取值；
  界面每个数字都绑定单一声明版本。
- 每分钟在**同一 SQLite 事务**写入裁决/热冷拆分/定点统计/误差界，行级带 sha256 摘要，
  `runs` 表携带整数 `ledger_format_version` 与清单摘要。
- 强制终止后只承认已 COMMIT 的连续分钟；缺失分钟在恢复报告中标记 UNKNOWN 并要求重跑，
  不声称生成终止时刻退出报告。
- 重放按记录的声明版本、`scale`、并发参数重新推进并逐字段比对，首个不一致即
  `E_REPLAY` 定位到 `minute[m].field` 及期望/实算值；缩放因子或声明版本不一致时
  单独以 `path=config` 区分报告。

## 7. 运行方式

```bat
run_tests.bat    :: 或 python -m unittest discover -s tests -v
run_demo.bat     :: 三场景演示，运行前清空旧账本/快照
run_gui.bat      :: Tkinter 工作台（需桌面环境）
```

演示三场景顺序固定：
1. 首次全量推进 + 第 5 分钟强制终止的恢复与重放；
2. 预热比例 25%→40% 后推进，旧结论标陈旧并整体替换，输出分钟/字段级 diff 与版本统计对比；
3. 缩放因子 10000→100000 扰动后再推进，演示配置不一致的独立 E_REPLAY 报告与当前版本重放通过。

## 8. 实际验证结果（本机）

- `python -m unittest discover -s tests`：**33/33 通过**。
- `python -m coldstart_diag.demo`：三场景稳定输出；当前账本仅保留声明 v3 的 12 分钟，
  v1（24 行，含崩溃/全量两个运行）与 v2（12 行）全部在 `stale_minutes`；
  篡改分钟与缩放因子不匹配均按预期触发 `E_REPLAY`。
- 最新一次诊断总结见 `workspace/diagnostic_summary.json`，快照见 `workspace/snapshots/`。
