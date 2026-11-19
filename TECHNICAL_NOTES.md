# 战术决策寻路诊断工作台 — 技术说明

面向确定性分层战术网格的 A* / 区域级流场双求解、精确增量失效、受控并发与崩溃恢复工作台。
纯 Python 3.11 + NumPy，无容器、无外部服务，样本由固定种子生成。

## 1. 目录结构

```
tactical_pathbench/
  errors.py        7 个业务错误码 + 结构化诊断异常
  encoding.py      固定小端确定性编码、SHA-256 内容哈希、Windows 路径与 UTF-8 落盘
  grid.py          三层栅格（代价/通行/覆盖）、矩形区域划分、边界门户、内容哈希
  generate.py      固定种子样本（地图、单位、查询），冻结 dataclass
  astar.py         四邻域 A*，可配置 manhattan/zero 启发，确定性并列裁决
  flowfield.py     区域方向场（区域内全对 Dijkstra）、跳点门户表、门户图路由
  dependency.py    tile/flow/jump/path 四类节点与带触发条件的有向边、失效闭包
  budget.py        线程预算、LRU 内存预算、固定优先级降级、按查询键归并
  consistency.py   代价 + 路线序列双口径一致性核对与首分歧定位
  persistence.py   原子提交、版本戳、孤儿回收、强制终止恢复
  workbench.py     门面：串联求解、增量更新、全量重建对照、持久化
scripts/           端到端/崩溃恢复/错误码/复现性 4 个演示脚本
tests/             19 个 unittest 用例
artifacts/reports/ 验收报告（JSON / TSV，UTF-8，Windows 风格路径）
artifacts/persist*/ 持久化工作区（manifest 原子文件、journal、orphan 回收目录）
```

## 2. 确定性约定

- 地形用 `numpy.random.default_rng(seed)` 的固定整数/阈值原语；单位与查询采样用
  `random.Random(seed)`，不依赖 dict 遍历顺序，所有集合落盘前排序。
- 内容哈希统一 SHA-256；整数小端 8 字节、浮点小端 double、文本 UTF-8 长度前缀；
  字典按键字节序排序后编码（`encoding.canonical_encoding`）。
- A* 堆条目 `(f, 单调序列号, cell)`；流场门户图同样使用序列号裁决，并列结果确定。
- 同种子两次 `manifest` 规范化字节完全相同（见 `reproducibility-check.py`）。

## 3. 双求解等价性

- A* 边代价 = 进入格地形代价；启发 `manhattan`（可切 `zero` 即 Dijkstra）。
- 流场对每个区域预计算 `dist[src][cell]` 与 `next_step[src][cell]`，
  跨区域走“边界门户/跳点”压缩图，再用方向场拼回完整格序列。
- 两者在全部样本查询上代价与路线序列完全一致（容差默认 0），结果记录
  `solver`、`expanded`、`cache_source`。

## 4. 依赖图与失效闭包（核心）

节点：`tile:*`、`flow:R###`、`jump:P####`、`path:<query>`。边方向为“源 -> 依赖者”，
带 `tile_change` / `unit_change` 触发标记。

- 区域内部格修改：仅该格所在区域的 flow 失效；闭包外 11 个区域 flow 全部复用。
- 边界格修改：所在区域 flow 失效 + 该格相邻的 jump 门户失效；
  jump 失效**不**反向传播到 flow（`assert_no_reverse_invalidation` 强制校验）。
- flow 失效正向传播到引用它的全部 path 缓存；jump 失效只使引用该 jump 的 path 失效。
- 单位移动只失效该单位名下 path，不触及任何 flow/jump。
- 闭包结果同时给出失效集合与复用集合、触发的依赖边、边界/内部命中格；
  与预期区域集合不符时抛 `E_CLOSURE_MISMATCH`，禁止整图重建。

## 5. 一致性核对

每次增量更新后，`consistency_check()` 新建独立工作台对当前网格做全量重建，
按查询键对比：① 代价差 ≤ 容差；② 完整格序列逐格相同。两者必须同时成立；
分歧报告首个查询、首个分歧格与相关依赖边，禁止只看代价。

## 6. 并发与资源预算

- `Budget(thread_budget, memory_budget, max_expansion_budget)`。
- 线程池任务按查询键排序提交、按键归并，结果与线程数 1 时逐路径哈希一致。
- 降级优先级固定：线程超订 -> 收缩线程池；不可并行 -> 串行回退；
  内存超预算 -> LRU 淘汰；禁止淘汰且超预算 -> `E_BUDGET_DENIED`。
- 每次降级记录原因、动作、前后累计展开节点数。

## 7. 持久化与强制终止恢复

- `manifest.json` 经 `.tmp` + `os.replace` 原子提交，附地图版本戳、
  规范化哈希与 JSON 序列化哈希（规避 int 键 JSON 往返差异），追加 `journal.log`。
- 退出报告 `exit-report.json` 只允许在存在已提交清单时写入；强制终止不会产生。
- 恢复只信任 manifest：`.tmp` 与无清单引用的松散缓存移入 `orphan/`，
  校验清单哈希，重算 flow/jump/tile 哈希与落盘值比对（`E_DEPENDENCY_STALE`）。
- `scripts/crash_worker.py` 用 `os._exit(137)` 在短生命周期子进程中模拟 SIGKILL，
  `crash_recovery_demo.py` 验证恢复后路径哈希与中断前已提交版本一致。

## 8. 业务错误码

| 错误码 | 触发链路 |
| --- | --- |
| `E_MAP_SHAPE` | 非二维层、层形状不一致、坐标越界、非邻接边 |
| `E_TERRAIN_RANGE` | 构造或运行时地形代价超出 1..10 |
| `E_DEPENDENCY_STALE` | 缺失节点、清单/节点哈希不匹配、无清单写退出报告 |
| `E_CLOSURE_MISMATCH` | 失效闭包区域集合不符、跳点反向污染 flow、增量/全量分歧 |
| `E_BUDGET_DENIED` | 非法预算、超展开预算、禁止淘汰却超内存 |
| `E_UNIT_INVALID` | 单位落在不可通行格、移动到障碍/越界 |
| `E_QUERY_UNREACHABLE` | 目标不可通行或无可达路径 |

所有异常均为携带 `code/context/dependency_edge` 的结构化诊断，写入路径全部走原子提交，
异常不会留下半写 manifest 损坏已提交状态。

## 9. Windows 运行

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
run_all.cmd
```

报告中的路径统一渲染为 `C:\...` 反斜杠形式；文件编码 UTF-8；数值固定小端。

## 10. 实际验证结果（本次执行环境）

- `python -m unittest discover -s tests`：19/19 通过。
- `run_demo.py`：三类增量变更后双口径一致性全部 `consistent=true`，
  产出 `artifacts/reports/acceptance-report.json`、`closure-report.tsv`。
- `crash_recovery_demo.py`：子进程退出码 137，回收 2 个孤儿文件，
  `path_hash_equal_after_recovery=true`，恢复版本戳 = 2。
- `error_chain_demo.py`：7 个错误码全部命中并结构化输出。
- `reproducibility_check.py`：清单字节、地图哈希、分区签名、路径哈希、代价五项全一致。
