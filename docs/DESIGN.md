# 调度与亲和性仿真诊断工作台 — 技术说明

## 1. 范围与边界

- 输入完全来自本地 JSON（节点、Pod 队列），账本落在本地 SQLite；不连接任何真实集群、不使用容器。
- 所有调度判定（Filter/Score/误差界/决胜）只用 Python 整数完成；NumPy 仅用于已完成结论的只读汇总（`k8ssim/analytics.py`），不参与任何绑定决策。
- 目标运行平台为本地 Windows + Python 3.11（含 Tcl/Tk）+ NumPy。当前实现同样可在 Linux/macOS 运行；无显示环境下 GUI 仅做导入级校验。

## 2. 目录结构

```
k8ssim/
  constants.py        插件序、污点 effect、权重基准、账本格式版本等常量
  errors.py           E_SPEC / E_SNAPSHOT / E_REPLAY 错误类型与定位路径
  util.py             规范化 JSON 与 SHA-256 摘要
  models.py           Node/Pod/亲和/污点/结论等不可变数据模型
  loader.py           JSON 强校验（首个出错字段，确定性遍历顺序）
  analytics.py        NumPy 只读分析（均值、差距表、误差直方图）
  session.py          会话门面：引擎 + 账本 + 崩溃标记/恢复
  gui.py              Tkinter 诊断界面
  scheduler/
    match.py          标签选择器与污点容忍匹配原语
    filter.py         四闸门 Filter（固定顺序、节点按名字典序）
    plugins.py        四个整数定点打分插件与 quantize()
    scoring.py        加权合成、误差界传播、稳定决胜
    engine.py         参数版本、整体替换、逐 Pod 调度编排
  storage/
    ledger.py         SQLite 账本（格式版本、摘要、逐 Pod 事务）
    recovery.py       崩溃恢复、重放校验、快照整体替换
samples/              确定性节点与 Pod 样本
scripts/demo.py       三场景固定顺序演示
tests/                39 个 unittest 用例
```

## 3. JSON 规范与强校验

节点文件顶层 `{"nodes": [...]}`，Pod 文件 `{"pods": [...]}`。

- 节点：`name` 唯一；`labels` 为合法合格名/标签值；`cpu`、`memory` 为正整数（`bool` 显式拒绝）；`taints[].effect ∈ {NoSchedule, PreferNoSchedule, NoExecute}`。
- Pod：`pod_id` 唯一；资源请求为正整数；节点亲和支持 `In/NotIn/Exists/DoesNotExist`；反亲和带 `topology_key`；容忍支持 `Exists/Equal`，空 `effect` 表示容忍全部 effect。
- 引用完整性：正向操作符（`In/Exists`）引用的节点/Pod 标签键必须在集合中存在，`In` 的值也必须存在；负向操作符（`NotIn/DoesNotExist`）允许引用不存在的键（与 Kubernetes 语义一致）。
- 任一不合法整体拒绝，抛出 `E_SPEC`，`path` 定位到首个出错字段，如 `$.pods[3].node_affinity.required[0].key`。
- 字典按排序键、列表按声明顺序遍历，因此“首个错误”与文件书写顺序、哈希随机化无关。

## 4. Filter 阶段（硬约束）

固定闸门顺序，节点始终按名字典序评估，每个节点只记录第一个拒绝闸门：

1. `taint`：节点污点是否全部被容忍覆盖；
2. `node_affinity_hard`：required 节点亲和全部满足；
3. `pod_anti_affinity_hard`：同一 topology 域内不得存在匹配反亲和选择器的已放置 Pod；
4. `resource`：剩余 CPU、内存均满足请求（先 CPU 后内存）。

输出每个被过滤节点的 `node/gate/reason/detail`。同一输入多次运行、以及打乱节点加载顺序，候选集合与拒绝诊断逐字节一致（见 `tests/test_determinism.py`）。

## 5. Score 阶段：整数定点模型

### 5.1 单次 floor 除法

每个插件返回未缩放的整数 `numerator / divisor`（比值已在 `[0,1]`），再做**唯一一次** floor 除法得到 `0..100` 原始分，缩放因子 `S` 在除法前乘入：

```
raw = clamp(floor(100 * numerator * S / (divisor * S)), 0, 100)
```

截断方向统一声明为 `floor_div`（常量 `TRUNCATION`），并随每个插件结果持久化。

### 5.2 精确量化误差（而非保守上界）

利用整除余数计算精确绝对误差，再以整数 ceil 转成微点（micro-points，1 点 = 10^6 微点），保证误差界紧致且严格：

```
remainder = (100 * numerator * S) mod (divisor * S)
quant_error_mp = ceil(10^6 * remainder / (divisor * S))
```

余数为 0 时误差为 0。`tests/test_scoring.py` 用浮点参考值逐格验证该整数误差界始终包住真实误差。

### 5.3 四个插件

- `node_affinity_soft`：已匹配 preferred 权重 / 总 preferred 权重；无软约束时中性 50/100。
- `pod_anti_affinity_soft`：未与同域已放置 Pod 冲突的软权重占比（越分散越高）；无软约束中性 50。
- `resource_balance`：调度后 CPU/内存余量的整数“调和均值”型均衡度，按节点较小总量归一化到 `[0,1]`，全程整数乘除。
- `label_locality`：Pod 标签中与节点同键同值的比例；Pod 无标签时中性 50。

### 5.4 权重与合成

- 权重为万分比（basis points），四个插件权重之和必须恰好等于 10000，单项允许 0；不合法抛 `E_SPEC`。
- 插件贡献 `contribution_mp = raw * weight`；节点总分微点为各贡献之和，显示总分用一次最终 floor：`total_points = total_mp // 10000`，其余数即合成截断误差。
- 插件误差按绝对权重传播：`ceil(|weight| * quant_error_mp / 10000)`，与合成截断误差相加得到节点 `error_bound_mp`。

## 6. 不可区分判定与稳定决胜

节点排序键（全确定性）：

1. 加权总分微点降序；
2. 插件分向量按规范插件序（节点亲和软 → 反亲和软 → 资源均衡 → 标签本地性）字典序；
3. 节点名字典序。

取排序第一为胜者、第二为亚军，并区分：

- `unique`：无亚军或总分差严格大于两节点误差界之和；
- `exact_tie`：总分差恰为 0，不标注受误差影响，直接落到向量序/名字序；
- `error_bound_tie`：`0 < gap_mp <= winner.error_bound_mp + runner_up.error_bound_mp`，按同一稳定排序键绑定并置 `error_influenced=True`，界面与账本均显式标注。

误差标注只依赖最终微点总分与误差界，不依赖插件计算顺序；`test_plugin_computation_order_independence` 通过反转插件合成顺序证明绑定与标注不变。

## 7. 参数版本、陈旧防护与整体替换

- 缩放因子或任一项权重发生**语义变化**时，参数版本号单调 +1；内容未变则版本不增。
- 旧版本报告整体标记 `staleness=stale` 并保留在历史中，不参与当前展示；新一轮调度产出全新 `ScheduleReport`，`replaced_entries` 稳定列出被替换的 Pod 标识（队列顺序），不做逐项修补。
- 参数变化同时清空并重建账本中的 Pod 结果，保证界面视图与持久化状态一致。

## 8. SQLite 账本、快照与恢复

- 表：`meta`（格式版本，当前 `LEDGER_FORMAT_VERSION=1`）、`specs`（节点/Pod JSON + 摘要）、`parameter_versions`、`pod_results`、`snapshots`。
- 每个 Pod 的结论在独立事务中写入，携带决策 JSON 的 SHA-256 摘要；写入只发生在完整决策计算之后，因此不存在半条 Pod 记录。
- 快照保存格式版本、规格摘要、参数摘要与参数版本。重开快照逐项比对，任一不一致抛 `E_SNAPSHOT` 并列出不一致项，然后**整体替换**账本状态（`reset_pod_results` + 重写 specs/参数），不做增量合并。
- 强制终止：计算中及其后尚未提交的 Pod 写为 `UNKNOWN`（无决策 JSON）。重启只依据 `COMMITTED` 行恢复；`UNKNOWN` Pod 标记为未知并要求重跑，系统不生成也不声称存在“终止时退出报告”。
- 重放：按账本记录的参数版本重新执行 Filter+Score，逐 Pod、逐字段（候选集、拒绝三元组、各节点各插件原始分/贡献/误差、总分、误差界、绑定、决胜原因、误差标注）比较，首个不一致抛 `E_REPLAY` 并定位到 `pods[<id>].<field>`。

## 9. Tkinter 界面

- 左栏 Pod 列表（可行性、绑定、决胜原因、误差标注、参数版本）；右上候选节点总分/误差界与四插件原始分+贡献；右下 Filter 拒绝原因（闸门顺序）与快照列表。
- 所有 Treeview 渲染使用固定排序键；刷新与切换选中不触发任何账本写入，只有导入、重算、保存快照才落盘。
- 启动：`run_workbench.bat`（或 `python -m k8ssim.gui`）。

## 10. Windows 运行步骤

1. 安装 Python 3.11+，安装器勾选 *Tcl/Tk and IDLE*；`python -m pip install numpy`。
2. 生成样本（仓库已附带，也可重跑）：`python scripts/generate_samples.py`。
3. 运行演示：`run_demo.bat` 或 `python scripts/demo.py`。
4. 运行测试：`run_tests.bat` 或 `python -m unittest discover -s tests -v`。
5. 启动界面：`run_workbench.bat`。演示前关闭占用同一 SQLite 文件的其他进程与旧界面窗口。
