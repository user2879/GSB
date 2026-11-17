# graphidx：图数据遍历与索引增量维护领域组件

面向本地 Windows / Linux 环境的纯 Python 领域组件（Python 3.11+，仅依赖 NumPy），
以本地 JSON 文件持久化有向属性图，在其上维护两类派生索引与可达性派生视图，
并对增量修改执行**最小化重算**：只重建受影响的索引条目与视图，未受影响条目的
版本号与摘要保持不变且可逐条核验。

## 目录结构

```
graphidx/            领域组件包
  errors.py          业务错误码：E_GRAPH / E_INDEX / E_DEPEND / E_BUDGET / E_STALE / E_REPLAY
  canon.py           规范 JSON 序列化与 sha256 内容摘要（确定性基石）
  graph.py           属性图加载、强校验、变更集应用（版本单调递增）
  indexes.py         标签倒排索引 + 边权重有序邻接索引、受影响条目推导
  views.py           可达性派生视图、依赖图、失效传播（NumPy 布尔矩阵闭包）
  scheduler.py       全局并发上限 + 每会话重算预算的调度器
  query.py           三类遍历查询（版本绑定 + 陈旧防护）
  session.py         维护会话编排、快照持久化、整体替换恢复
  replay.py          恢复后一致性重放校验
  structlog.py       JSON Lines 结构化日志
samples/             确定性样本：graph.json / indexes.json / views.json / changesets.json
demo.py              演示 CLI（build / update / query / restore / replay / inject / all）
tests/               unittest 测试套件（56 个用例）
snapshots/           演示生成的会话快照
logs/                结构化日志输出（*.jsonl）
```

## 快速开始

```bat
REM Windows 命令行（Linux 下用 python3）
python demo.py all                                  REM 一键验收流程
python demo.py build --out snapshots\session.json   REM 全量构建 + 快照
python demo.py update --budget 4                    REM 应用变更集（预算 4，演示排队）
python demo.py query --kind adjacency --start 1     REM 邻接遍历
python demo.py query --kind label --start 1 --label Person
python demo.py query --kind view --view reach_team
python demo.py restore --snapshot snapshots\session.json
python demo.py replay  --snapshot snapshots\session.json
python demo.py inject corrupt-graph                 REM E_GRAPH
python demo.py inject tamper-index                  REM E_INDEX
python demo.py inject zero-budget                   REM E_BUDGET
python -m unittest discover -s tests                REM 运行测试
```

## 设计要点

**1. 图与变更集（E_GRAPH）**
节点标识唯一、边两端存在、权重为整数（拒绝布尔）、删除节点须先删除其全部关联边；
任一校验失败即整体拒绝，图保持原状，错误明细携带 `element`（如 `nodes[1]`、`ops[0]`）
与 `field`。合法变更集应用后图版本号 +1；内容摘要基于显式排序（节点按 id、边按 id）
的规范 JSON，与输入顺序无关，同一变更集重复应用得到相同摘要。

**2. 索引构建（确定性）**
- `label_inverted`：每个标签一个条目，内容为升序节点 id 列表；
- `weight_adjacency`：每个有出边的源节点一个条目，边按 `(weight, dst, edge_id)` 升序。
每条目记录 `def_version`（索引定义版本）、`entry_version`（条目版本）、`status` 与
`digest`。索引摘要 = 按显式键序排列的 `[key, entry_digest]` 序列摘要；构建报告同时
写入图版本、图内容摘要、索引定义版本与各索引摘要。

**3. 受影响条目推导**
按变更操作精确推导：标签变化 → 新旧标签条目；增删边/改权重 → 源节点邻接条目；
增删节点 → 对应标签条目。推导结果按 `(索引名, 键)` 稳定排序输出，只重建这些条目并
递增其条目版本，未受影响条目版本与摘要逐条保持不变（`tests/test_incremental.py` 核验）。

**4. 视图依赖与失效传播（E_DEPEND）**
视图内容 = 源节点集合的可达闭包 ∪ 上游视图可达集合（NumPy 布尔矩阵闭包迭代至不动点）。
依赖成环在定义加载时拒绝并输出环上序列（如 `a->b->c->a`）；删除仍被引用的视图拒绝并
列出引用者。图结构或边权重变化时，变化节点集与视图「源 ∪ 可达区域」相交者直接失效，
再沿依赖图传播至全部下游，按稳定拓扑顺序输出。

**5. 预算约束调度（E_BUDGET）**
调度器按「重算条目数」消耗每会话预算；同一依赖波次内以有界线程池并发（全局并发上限），
视图依赖关系通过波次分层保证上游先于下游重算。预算耗尽时剩余任务进入排队并记录
`queued_at_ms`（等待开始时刻），绝不绕过预算。账本逐任务记录 `start_ms`/`end_ms`
（单调时钟毫秒）、目标条目/视图与所用预算；报告给出本次会话 `recomputed` 与 `skipped`。
预算为零且有待重算任务时以 E_BUDGET 拒绝。

**6. 遍历查询（E_STALE）**
邻接遍历（按权重序 BFS）、按标签过滤遍历、视图可达性查询的结果均绑定所用索引条目
版本与视图版本（`bindings`）。被引用条目/视图处于失效状态时返回 E_STALE 并列出全部
失效引用，不以旧缓存冒充当前结论；索引同步后同一查询重复执行得到逐字节一致的
规范序列（`canonical` 输出可比对）。

**7. 快照与整体替换恢复**
快照持久化：图版本与图内容摘要、索引定义版本、全部条目与视图版本/摘要/内容、
失效轨迹、重算账本、变更集序列与重算参数。重新打开时比较快照的图内容摘要与索引
定义版本：一致则无需动作；不一致则**整体替换**图、索引条目与视图状态（不增量合并），
并稳定列出发生变化的条目与视图及其新旧版本。快照装载时逐项校验内容摘要，
篡改条目摘要以 E_INDEX 拒绝。

**8. 重放校验与异常注入（E_REPLAY）**
`replay` 从快照记录的基础图出发，按快照中的变更集序列与重算参数重新推导受影响集合、
失效视图并重算，逐项比较：图摘要 → 失效轨迹 → 各条目版本/摘要/状态 → 各视图版本/摘要/
状态 → 归一化重算账本（单调时钟毫秒不参与比较）。首个不一致定位到条目标识与字段，
以 E_REPLAY 报告。三类异常注入仅改变输入文件或参数：图文件损坏（E_GRAPH）、
快照条目摘要篡改（E_INDEX）、重算预算为零（E_BUDGET），均写入 `logs/*.jsonl` 结构化日志。

## 实际验证结果

- `python3 -m unittest discover -s tests`：**56 个用例全部通过**（覆盖上述 8 项要求）。
- `python3 demo.py all`：全量构建 → 3 个变更集增量维护（预算 4 时每次重算 4 条、
  排队 1 条）→ 确定性查询 → 快照 → 分叉会话整体替换恢复（列出 6 个变化条目、
  2 个变化视图及新旧版本）→ 重放校验 `status: ok`（28 项比对）→ 三类异常注入
  分别输出 E_GRAPH / E_INDEX / E_BUDGET。
- 同一查询同步后重复执行 canonical 输出逐字节一致；同一图内容不同输入顺序下
  索引摘要逐字节一致（测试固化）。
