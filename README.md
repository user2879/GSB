# 战术决策与寻路系统（Deterministic Tactical Routing）

本项目在本地实现一个可复现的分层战术网格寻路实验环境。它覆盖 64×64、256×256、1024×1024 三档确定性地图，提供四邻域 A*、区域流场、区域门户分块跳点三种精确求解策略，并实现预算驱动策略选择、增量失效闭包、全量重建一致性核对、线程池调度、预算降级、原子快照与强制终止恢复。

运行依赖只有 NumPy 与 PyYAML；不访问远程服务，不使用 Docker，不依赖外部数据集。

## Windows 环境

```bat
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install numpy==2.2.6 PyYAML==6.0.2
.venv\Scripts\python -m pip list > artifacts\venv-pip-list.txt
```

本项目实际验证环境为 WSL2 Linux + Python 3.11.0rc1 + NumPy 2.2.6 + PyYAML 6.0.2；代码与路径处理保持 Windows 兼容，并提供 `run_demo.bat`。

## 一键演示

```bat
run_demo.bat
```

等价命令：

```bat
.venv\Scripts\python -m tactical_path demo --config config\demo.yaml --workspace workspace\demo
```

演示固定执行以下序列：

1. 三档地图各生成两次并逐文件比较；地图、单位、查询必须逐字节一致。
2. 按声明的时间/内存预算做策略选择，并验证相同输入零重算、查询分布变化只重算受影响组、预算变化只重算受影响档。
3. 并发求解全部查询，按查询键排序归并；记录展开数、逻辑工作量与估算峰值内存。
4. 小档对全部查询执行 A* 与区域流场双求解交叉核验。
5. 每档选择路线内部格做地形增量修改，计算依赖闭包并只重算闭包内节点。
6. 对编辑后地图做全量重建，按“代价 + 完整路线序列”双口径比较。
7. 小档用极紧内存预算演示固定优先级降级：串行、缓存淘汰、最小内存模式；所有查询仍返回。
8. 提交状态后启动子进程，在写完整未提交缓存后 `os._exit(91)`；重启只读取 `COMMITTED.json`，回收未提交快照并逐查询核验路线哈希。
9. 生成 `workspace/demo/report.json` 与报告 SHA；同一输入两次运行报告逐字节一致。

其他命令：

```bat
.venv\Scripts\python -m tactical_path generate --config config\demo.yaml --output workspace\samples
.venv\Scripts\python -m tactical_path recover --workspace workspace\demo\state\small
.venv\Scripts\python -m tactical_path error-codes
.venv\Scripts\python -m unittest discover -s tests -t . -v
```

## 三档配置与策略结果

| 档位 | 规模 | 区域数 | 单位/查询 | 自动选择 | 关键预算判据 |
|---|---:|---:|---:|---|---|
| small | 64×64 | 16 | 8 / 12 | 区域流场 | 流场均摊 0.073 ms，优于 A* 估计 0.388 ms |
| medium | 256×256 | 64 | 16 / 32 | 稠密 A* | 跨区域查询不适用流场；65,536 格低于 CJP 门槛；A* 估计 2.872 ms |
| large | 1024×1024 | 256 | 20 / 40 | 分块跳点 | CJP 查询估计 0.690 ms，优于 A* 3.596 ms，且预计算 65.536 ms 在预算内 |

每个策略决策都记录候选策略、是否可行、估算耗时、估算内存、误差上界与确定性理由。预算和查询分布签名决定复用或重算；不会因等价查询重排而重算无关查询。

## 目录

```text
tactical_path/
  mapgen.py        确定性地图、单位与查询生成
  strategy.py      预算驱动策略选择与受影响重规划
  engine.py        缓存、线程池、预算降级、闭包重算
  dependency.py    依赖图、版本戳、反向闭包
  consistency.py   增量 vs 全量重建双口径核对
  persistence.py   原子提交、提交指针、孤儿回收
  solvers/
    astar.py       四邻域加权 A*
    flowfield.py   区域反向 Dijkstra 流场
    chunkjump.py   精确区域门户分块跳点
config/demo.yaml   三档、预算、移动类别、估计模型参数
tests/             25 项 unittest
docs/DESIGN.md     算法、闭包、持久化与验证说明
```

运行产物在 `workspace/demo/`：`samples/` 为原始样本，`maps/` 为编辑后地图，`tiers/<档>/` 为策略、资源账本、闭包与一致性明细，`state/<档>/` 为提交快照与恢复记录。

## 实际验证结果

本次环境完整演示耗时约 69 秒；第二次同输入运行的整个 `workspace/demo/` 产物树均逐字节一致（不仅是主报告）。

- 地图生成：三档两次生成全部逐字节一致。
- 求解交叉核验：小档 12/12 个查询 A* 与流场代价偏差为 0；同一查询重复求解路线哈希一致。
- 增量闭包：small/medium/large 的闭包节点数分别为 8/7/15，实际重算集合与闭包完全相等；闭包外复用节点数分别为 8/25/531。
- 全量一致性：三档全部通过，最大代价偏差均为 0，路线序列完全相等。
- 预算降级：`serial → cache_eviction → minimal_memory` 三级触发，12/12 查询返回且哈希与降级前一致。
- 强制终止恢复：子进程退出码 91；只恢复最后提交快照，未提交的 `snap-000003` 与 pending 事务被回收，提交路线 12/12 哈希一致。
- 自动测试：`Ran 25 tests ... OK`。
- 当前报告：`workspace/demo/report.json`，SHA-256 `4253a3504354ebd59a7cf174b82e96bb59e0f31950845141200ed920d9bfd8af`，48,508 字节。

完整算法说明见 `docs/DESIGN.md`，环境与验证摘要见 `artifacts/`。
