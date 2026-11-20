# Kubernetes 调度与亲和性仿真诊断工作台

面向本地 Windows 与 Python 3.11+ 的确定性仿真工具。它不连接 Kubernetes、Docker 或网络服务，而是从本地 JSON 读取 Node 与 Pod 队列，执行 Filter/Score、维护节点派生统计缓存、按依赖图做增量失效与重算，并用 SQLite 和只读快照保存、恢复、重放调度结论。

## 快速开始

目标 Windows 环境只需要 Python 官方安装包，无第三方依赖：

```bat
python --version
python -c "import tkinter"
run_tests.bat
run_demo.bat
run_gui.bat
```

等价的跨平台命令：

```bash
python -m unittest discover -v
python -m sched_diag demo
python -m sched_diag gui --nodes samples/nodes.json --pods samples/pods.json --plugins samples/plugins.json --workdir .sched-diag
```

运行产物位于工作目录，例如 `.sched-diag/state.sqlite3`、`.sched-diag/snapshots/` 和 `.sched-diag/logs/workbench.jsonl`。快照备份文件在写入后置为只读。

## 文件布局

```text
samples/nodes.json            确定性节点样本
samples/pods.json             确定性 Pod 队列样本
samples/plugins.json          四个打分插件的整数权重
sched_diag/manifest.py        清单加载、结构与引用强校验
sched_diag/model.py           不可变领域模型和序列化
sched_diag/scheduler.py       Filter、Score、会话、增量与全量校验
sched_diag/deps.py            事实/统计/Pod 结论依赖图
sched_diag/persistence.py     SQLite、只读快照、结构化日志
sched_diag/restore.py         整体替换恢复与快照重放
sched_diag/gui.py             Tkinter 诊断界面
sched_diag/cli.py             命令行入口
tests/test_workbench.py       32 项自动化测试
demo.py / run_*.bat           直接演示与 Windows 启动脚本
```

## 清单格式

根对象必须包含 `schemaVersion: 1`。Node 的核心字段：

```json
{
  "name": "node-east-a",
  "labels": {"zone": "east", "rack": "r1", "disk": "ssd"},
  "allocatable": {"cpu": 8, "memory": 16384},
  "taints": [{"key": "dedicated", "value": "batch", "effect": "NoSchedule"}],
  "images": ["registry.local/api:1"]
}
```

Pod 的核心字段：

```json
{
  "id": "api-front-a",
  "labels": {"app": "api", "tier": "web"},
  "resources": {"cpu": 2, "memory": 1024},
  "image": "registry.local/api:1",
  "nodeAffinity": {
    "hard": {"matchLabels": {"zone": "east"}},
    "soft": [{"selector": {"matchLabels": {"disk": "ssd"}}, "weight": 3}]
  },
  "antiAffinity": {
    "hard": [{"selector": {"matchLabels": {"app": "api"}}, "topologyKey": "rack"}],
    "soft": [{"selector": {"matchLabels": {"app": "api"}}, "topologyKey": "zone", "weight": 2}]
  },
  "tolerations": [
    {"key": "dedicated", "operator": "Equal", "value": "batch", "effect": "NoSchedule"}
  ]
}
```

选择器支持 `matchLabels` 与 `matchExpressions`，操作符为 `In`、`NotIn`、`Exists`、`DoesNotExist`。节点亲和选择器和反亲和 `topologyKey` 必须引用节点标签集合中的键；反亲和 selector 按 Kubernetes 语义引用 Pod 标签集合中的键。

强校验覆盖：节点名与 Pod ID 唯一、标签键值语法、CPU/内存正整数、污点 effect 属于 `NoSchedule/PreferNoSchedule/NoExecute`、选择器结构、软约束权重、Pod 请求不超过单节点最大可分配量。错误位置采用文件名加 JSON Pointer，例如 `nodes.json#/nodes/1/name`。

## 业务错误码

- `E_MANIFEST`：JSON、结构、类型、范围、唯一性等清单错误，整体拒绝。
- `E_REF`：语法有效但节点亲和/拓扑键或反亲和 Pod 标签键不存在。
- `E_SNAPSHOT`：快照缺失、格式错误、内容摘要或封套摘要不匹配。
- `E_REPLAY`：版本缺失、图签名不可复现、插件权重口径不同或重放首差异。
- `E_INCR`：增量结果与独立全量重算出现差异，并报告首个差异 Pod。
- `E_GRAPH`：依赖图引用不完整或存在环。

每次 CLI 或界面业务错误都会追加写入 `logs/workbench.jsonl`，包含时间、级别、事件、业务码、位置和细节。

## Filter 与拒绝顺序

每个 Pod 固定按以下顺序检查节点：

1. `node_affinity_hard`
2. `anti_affinity_hard`
3. `taint_toleration`
4. `allocatable_resources`

每个被过滤节点输出 `stage`、`stage_name` 与具体原因，包括实际标签值、命中同一 topology 的 Pod、未容忍污点或资源余量。通过全部检查的节点按字典序形成候选集合。`PreferNoSchedule` 不参与硬过滤。

## 整数定点 Score

四个插件均先输出 `0..100` 的整数原始分：

- 节点亲和软约束：已满足软规则权重之和 / 全部软规则权重之和，向下取整。
- 反亲和软约束：每条规则计分 `max(0, 100 - 20 * 同拓扑匹配 Pod 数)`，再按规则权重整数加权平均。
- 资源均衡：放置后 CPU 与内存剩余百分比之差的补数，`100 - abs(cpu_percent - memory_percent)`。
- 镜像本地性：Pod 镜像存在于节点镜像列表时为 `100`，否则为 `0`。

总权重为 `W`，插件权重为 `w_i`，贡献与总分的单位为千分之一分：

```text
contribution_i = floor(raw_i * w_i * 1000 / W)
total          = sum(contribution_i)
truncation     = floor_toward_zero_non_negative
error_bound    = 正权重插件数量（milli-point）
```

每个插件单独向下取整，因此相对精确加权和的误差严格小于“正权重插件数量”。若最高分与某候选的差值 `< error_bound`，进入稳定误差带；决胜键依次为同拓扑匹配 Pod 数升序、放置后最小剩余比例降序、节点名升序。结果中保存误差带、候选键和最终决胜依据。所有比较和求和均为整数运算。

## 依赖图与增量失效

图中的节点分为：

- 事实：节点逐标签键事实、污点、容量、镜像；Pod 标签、约束、请求、容忍、镜像；插件权重。
- 统计：`label_match_set`、`taint_gate`、`resource_remaining`、`same_label_pod_count`、`image_locality`。
- 结论：每个 Pod 的 Filter、Score 与绑定结果。

边方向为“依赖 → 派生数据”。例如节点标签事实指向标签匹配统计和反亲和拓扑计数；统计指向 Pod 结论；较早 Pod 结论指向较晚 Pod 的资源余量与反亲和统计。因此节点标签、污点、容量、Pod 约束、Pod 资源或插件权重变更时，可用依赖闭包精确得到失效统计与 Pod 集合，而不会反向形成环。

失效报告按稳定拓扑序输出每一步的 `id`、`kind` 和直接 `triggered_by`。增量重算只删除失效统计缓存，并仅对失效 Pod 重跑 Filter/Score；未受影响 Pod 的规范化 JSON 字节和结论版本号保持不变。`verify_full_consistency()` 会构建独立会话执行全量重算，按 Pod 队列比较结论，然后比较全部统计值；若有差异，抛出 `E_INCR` 并定位首个 Pod。

样本中把 `node-west-a` 的 `disk` 从 `ssd` 改为 `hdd` 后，失效 Pod 仅为 `cache-index`、`batch-report`；另外三个 Pod 的结论和版本不变，增量与全量校验一致。

## 快照、整体替换与重放

SQLite 表 `snapshots` 以 `(name, revision)` 为主键保留所有历史版本，`current_session` 保存最近一次调度状态。每个快照同时写入 `snapshots/<name>/rNNNN.json` 只读备份，包含：

- 节点、Pod、插件权重参数版本与依赖图版本；
- 事实修订号和每个 Pod 结论版本；
- 全部统计缓存、候选节点、插件原始分、加权贡献、误差界、决胜依据和绑定；
- 规范化内容摘要 `contentDigest` 与整个封套的 `envelopeDigest`。

恢复时先验证格式、版本块、内容摘要与封套摘要，再比较快照版本和内容摘要同当前会话是否一致。无论一致与否都会用快照内容整体重建会话，从不增量合并；报告稳定列出变化的节点和 Pod。差异不会破坏旧快照，历史 revision 与只读文件均保留。

重放流程在隔离副本上执行：

1. 从快照 `lastChange.roots` 在重建后的依赖图上重新推导失效集合；
2. 与快照记录的拓扑步骤、统计和 Pod 集合逐项比较；
3. 删除失效统计与结论，只重算失效 Pod；
4. 验证插件权重、定点口径和 Filter 顺序；
5. 逐 Pod 比较候选、得分明细和绑定，再执行独立全量重算交叉验证；
6. 任一差异以 `E_REPLAY` 报告首个 Pod 与字段；缺少版本或插件口径不一致也以 `E_REPLAY` 报告具体依据。

## 命令行

```bash
# 首次全量调度，可顺便保存快照
python -m sched_diag run --nodes samples/nodes.json --pods samples/pods.json \
  --plugins samples/plugins.json --workdir .sched-diag --snapshot baseline

# 增量修改节点标签并保留受影响范围
python -m sched_diag incremental --nodes samples/nodes.json --pods samples/pods.json \
  --plugins samples/plugins.json --workdir .sched-diag \
  --node node-west-a --key disk --value hdd

# 独立比较增量状态与全量重算
python -m sched_diag verify-incremental --nodes samples/nodes.json --pods samples/pods.json \
  --plugins samples/plugins.json --workdir .sched-diag \
  --node node-west-a --key disk --value hdd

# 快照列表、恢复、重放、SQLite 当前会话重新打开
python -m sched_diag list-snapshots --workdir .sched-diag
python -m sched_diag restore --workdir .sched-diag --name baseline
python -m sched_diag reopen --workdir .sched-diag

# 依赖图完整性/无环验证
python -m sched_diag graph --nodes samples/nodes.json --pods samples/pods.json \
  --plugins samples/plugins.json --workdir .sched-diag --json
```

## 故障注入

注入只创建或修改输入文件/参数，不修改调度逻辑：

```bash
# 复制节点清单并制造重复节点名
python -m sched_diag run ... --inject manifest       # E_MANIFEST

# 复制节点清单并删除 Pod 节点亲和引用的标签键
python -m sched_diag run ... --inject ref            # E_REF

# 先保存快照，再创建摘要被破坏的快照输入文件并恢复
python -m sched_diag restore --workdir .sched-diag --name baseline \
  --nodes samples/nodes.json --pods samples/pods.json \
  --plugins samples/plugins.json --inject snapshot-digest  # E_SNAPSHOT
```

## Tkinter 诊断界面

界面左侧按 Pod 队列稳定展示候选节点、最终选择和版本；右侧按节点名字典序展示 Filter 拒绝顺序、每节点四个插件原始分、milli 加权贡献、总误差界、决胜依据；下方展示本轮失效闭包、当前四个版本、内容摘要和只读快照列表。修改节点标签、值或删除键后立即执行增量重算并刷新全部视图。

界面还提供全量调度、增量/全量一致性校验、保存快照、恢复并重放按钮。CLI 的 `--inject manifest|ref|snapshot-digest` 在 GUI 启动时同样生效，错误码、位置和细节会写入结构化日志并显示在界面。

## 实测结果

当前执行环境为 Python 3.11.0rc1（Linux）。实际验证记录：

- `python3 -m unittest discover -v`：32 项测试全部通过。
- `python3 -m sched_diag demo`：输出 `DEMO_OK full_schedule=true incremental=true snapshot_restore=true replay=true`。
- 首次全量选择：`probe-west -> node-west-a`、`api-front-a -> node-east-a`、`api-front-b -> node-east-b`、`cache-index -> node-west-a`、`batch-report -> node-west-b`。
- `node-west-a disk: ssd -> hdd`：仅 `cache-index`、`batch-report` 失效并重算；`probe-west`、`api-front-a`、`api-front-b` 的 JSON 字节和版本保持不变；增量/全量校验通过。
- 快照后继续修改 `node-east-a` 标签与 `probe-west` 资源，再恢复旧快照：`replaceRequired=True`，变化项稳定报告为节点 `node-east-a`、Pod `probe-west`；重放失效 Pod 为 `cache-index`、`batch-report`，全量交叉验证通过。
- 三类注入分别实际返回 `E_MANIFEST`、`E_REF`、`E_SNAPSHOT`，退出码为 2，并写入 `logs/workbench.jsonl`。

当前 Linux 验证容器没有安装 Tkinter（`import tkinter` 失败），因此 GUI 启动在此环境按设计返回 `E_IO`；Windows 验收前应执行 README 顶部的 `python -c "import tkinter"` 确认。核心调度、增量、快照、重放和日志链路不依赖 Tkinter，已由上述测试覆盖。

## 设计边界

这是单机确定性调度仿真，不实现 Kubernetes 的抢占、优先级队列、拓扑分布、亲和性 namespace、生命周期回收或 API Server 交互。Pod 按输入数组顺序一次性调度，污点 `PreferNoSchedule` 不参与当前硬/软评分。所有派生数据、错误位置和输出顺序均围绕可重复诊断设计。
