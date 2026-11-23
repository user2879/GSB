# 跨环境配置仲裁一致性诊断组件

这是一个面向 Windows 本地环境的配置领域工作台。它把多个部署环境中的基线源、覆盖源和运行时注入源构造成“环境内生效链”，再由 `latest`、`weighted_majority` 或 `interval_midpoint` 仲裁器合并为唯一权威视图。所有数值进入仲裁前都按会话缩放因子量化为有符号 64 位定点整数，结论、误差界、账本版本和内容摘要均可复核。

## 快速开始

目标环境为 Python 3.11+、标准库 `tkinter`/`sqlite3` 和 NumPy 1.26+。不需要网络服务、端口或容器。

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m tkinter
python -m unittest discover -v
```

直接使用仓库内固定样本运行：

```powershell
python -m config_arbiter run
python -m config_arbiter status
python -m config_arbiter replay
python -m config_arbiter gui
```

默认路径：

- 环境声明：`samples/environments.json`
- 视图口径：`samples/views.json`
- SQLite 账本：`runtime/arbiter-ledger.sqlite`
- 会话快照：`runtime/arbiter-snapshot.json`
- 结构化日志：`logs/arbiter.jsonl`

所有命令均支持 `--env`、`--views`、`--db`、`--snapshot`、`--log` 覆盖路径。

## 命令

| 命令 | 作用 |
| --- | --- |
| `run` | 校验输入；对变更视图重新裁决并逐视图事务提交；最后写快照 |
| `status` | 只读显示每个视图是 `complete`、`changed` 还是 `unknown` |
| `replay` | 按账本口径重新裁决已提交视图，逐项校验并报告首个 `E_REPLAY` |
| `restore` | 快照版本/状态一致时重放剩余视图；不一致时整体替换后重建 |
| `snapshot` | 从当前完整账本生成内容摘要快照 |
| `gui` | 打开 Tkinter 诊断工作台 |

异常注入（只改加载参数或校验参数，不改账本、不改样本文件）：

```powershell
python -m config_arbiter run --inject-spec-corrupt
python -m config_arbiter run --inject-priority-ref-missing-env
python -m config_arbiter status --inject-ledger-digest-mismatch
```

分别返回退出码 `2/3/4`，结构化报告 `E_SPEC`、`E_REF`、`E_SNAPSHOT`，并把 `command_failed` 事件写入 `logs/arbiter.jsonl`。

## 样本与语义

固定样本含有 `prod-cn`、`staging-eu`、`dr-ap` 三个环境和 `ops-authority`、`release-guard`、`cost-focus` 三个视图。样本 SHA-256：

```text
0f1f725e711096fe4afeb064119d84f74c0f398c53f12685fe33a9a689e6ff38  samples/environments.json
bd9e4fa012d6de3912658e280d5c41c1b9611bfaf4e71001b311447ebe01bacb  samples/views.json
```

环境内排序键为：

1. `priority_bias` 降序；
2. 全局 `source_type_order` 序号升序；
3. `source_id` 字典序升序。

链上首个满足 `bindings[key]` 的 `required_source_types` 与 `min_observed_seq` 的条目成为生效值。每个链步都记录来源标识、取值、定点值、观测序号、采集时刻、排序键、绑定失败原因和决策。样本中 `prod-cn/http.timeout_ms` 的高偏置 runtime 条目因序号不足被跳过，最终采用 override；`staging-eu` 则因绑定只允许 baseline 而采用 baseline，因此两环境链签名不同但可并排复核。

定点规则：

- 会话 `scale_power` 支持 `0..9`，缩放因子为 `10^scale_power`；
- 量化向零截断，即正数向下、负数向上；
- 所有定点运算最终饱和到有符号 64 位区间 `[-2^63, 2^63-1]`；
- 权重也量化为 `weight × scale_factor`，加权支持度使用饱和整数加减；
- 每个候选记录精确量化误差，单项上界为误差分数的向上取整；
- 逐键比较误差界 = `tolerance_fixed + 2 × 最大单项量化误差上界`；
- 视图级聚合上界取该视图中逐键聚合上界的最大值；
- 区间中点的除法明确向零截断，并计入 1 个定点单位的舍入余量。

误差带内出现多个候选时，统一按显式稳定决胜规则选择：环境优先级升序、环境标识字典序、来源标识字典序、定点值升序；加权分组先比较加权支持度降序。发生该情况的键会置 `error_influenced=true`，并在 `error_attribution` 中记录名义选择、最终选择、误差带配对和稳定规则。

## 权威性与恢复

账本每个视图在一个 SQLite 事务内写入视图版本、逐键权威值、链摘要、采纳/否决集合、误差界、归因和内容摘要。旧版本整体变为 `stale`，受影响键另写 `stale_findings` 传播轨迹。未被依赖变更影响的键直接从上一版本按规范化 JSON 原字节搬运，因此权威值和链摘要保持不变。

进程被强制终止后，只能看到已提交事务；没有完整提交记录的视图显示为 `unknown/requires_rerun`。组件不会生成或伪造进程终止时的退出报告。

快照包含环境集合版本、来源优先级版本、缩放口径版本、逐视图结论摘要、已完成视图集合、完整环境/视图文档和内容摘要。版本或状态摘要不一致时，`restore` 以快照为准整体替换环境、观测和结论状态，不增量合并。状态一致时先重算已提交视图，将权威值、采纳集合、比较误差界与聚合误差界逐项对照账本；首个差异定位到视图、键和字段并抛出 `E_REPLAY`，验证通过后才补齐未知视图。

## Tkinter 工作台

`gui` 提供五类稳定排序视图：环境/来源条目编辑、逐环境生效链、每键多环境分布与权威/否决、定点误差归因、版本与陈旧传播。编辑只写回本地环境 JSON；“裁决并提交”才写账本；`只读刷新` 只读取账本，内容摘要保持不变。

验收时已在容器显示服务上实际实例化完整窗口并检查五类视图，随后自动关闭；Windows 桌面可直接运行 `python -m config_arbiter gui`。

## 验收

```powershell
python -m unittest discover -v
```

实际结果与逐项运行证据见 [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md)，设计细节见 [docs/DESIGN.md](docs/DESIGN.md)。

## 目录

```text
config_arbiter/   规格、链、定点仲裁、账本、快照、CLI、Tkinter UI
samples/          固定确定性环境与视图 JSON
tests/            规格/顺序/定点/失效/SIGKILL/快照/CLI/只读刷新测试
docs/             技术设计与实际验收记录
```
