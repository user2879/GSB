# ShardBench 分库分表路由可视化与诊断工作台

面向 Windows 本地环境、**仅依赖 Python 3.10+ 标准库**（`tkinter`/`sqlite3`/`threading` 等）
的分库分表路由领域工作台。用本地 JSON 描述逻辑表分片配置与路由单元依赖，把逻辑谓词
编译为确定性路由计划，在全局并发预算与每库配额双重约束下并行探查，并把路由理由、命中
库表、资源占用与结局写入本地 SQLite 账本，支持版本演进的缓存整体替换、依赖失效传播、
快照恢复与断言式账本重放。

## 目录结构

```
shardbench/
  errors.py            业务异常与业务码（E_CONFIG/E_CYCLE/E_RING/E_VERSION/E_REPLAY 等）
  determinism.py       FNV-1a 稳定哈希、SHA-256 摘要、规范 JSON、物理库表映射
  predicate.py         谓词 tokenizer/解析/规范化/分片键覆盖识别
  algorithms.py        取模、范围、一致性哈希环三种确定性算法
  config.py            配置与依赖图强校验、拓扑排序、版本摘要
  router.py            谓词 -> 路由计划、广播判定、范围区间、依赖派生合并
  cache.py             版本化缓存与整体替换视图、陈旧条目隔离
  versioning.py        四类版本（配置/环/依赖/预算）识别与单调递增
  invalidation.py      依赖图 BFS 失效传播、稳定拓扑、增量/全量比较
  scheduler.py         全局预算+每库配额双约束线程池、单调时钟轨迹、四类结局
  ledger.py            SQLite 账本，单事务整轮提交，崩溃只留未提交占位
  snapshot.py          会话快照（整数格式版本+内容摘要）、整体替换恢复
  replay.py            独立读账本重算并逐项比对，首个不一致即 E_REPLAY
  injection.py         三类结构化异常注入（只改输入/参数，不改源码）
  engine.py            编排完整路由轮次，串联上述全部模块
  gui.py               Tkinter 诊断界面（只读，稳定渲染；无 tkinter 时优雅降级）
  cli.py / __main__.py 命令行入口
  demo.py / *.bat      一键演示与 Windows 启动脚本
  samples/             确定性样本（三算法、组合键、广播、依赖链、异常变体）
  tests/               40 个 unittest 用例（标准库，无第三方依赖）
  data/                运行期产物：ledger.sqlite3 / snapshot.json / events.jsonl
```

## 物理映射与确定性约定

- 槽位总数 `P = db_count * tables_per_db`，统一映射：
  `db_index = slot % db_count`，`table_index = slot // db_count`，标识 `db{i}.t{j}`。
- 路由哈希一律使用 FNV-1a 64 位（`determinism.stable_hash`），**不使用** Python 内置
  `hash()`（其逐进程随机化），集合统一排序后序列化，保证跨进程、跨线程、跨字典遍历
  顺序结果逐字节一致。
- 取模：`slot = stable_hash(key) % base`；组合分片键按声明列顺序拼接后哈希。
- 范围：边界把整数域切为单调区间，`boundary_ownership` 显式声明 `lower`（边界归下界
  区间 `[b_i, b_{i+1})`）或 `upper`（归上界区间 `(b_{i-1}, b_i]`）。
- 一致性哈希：每节点按权重生成虚拟节点，位置 `FNV1a(node#vnode)` 升序成环，查询取
  顺时针首个虚拟点，越过末位回绕到首位。

## 快速开始（Windows）

```bat
:: 确认环境（官方安装包默认含 tkinter 与 sqlite3）
python --version

:: 一键端到端演示
python -m shardbench.demo

:: 图形诊断界面
shardbench\run_gui.bat
```

命令行（在仓库根目录执行）：

```bat
python -m shardbench --workspace "%CD%" run
python -m shardbench --workspace "%CD%" run --changed-table t_user
python -m shardbench --workspace "%CD%" run --fault u_user_eq=force_config_missing
python -m shardbench --workspace "%CD%" replay 1
python -m shardbench --workspace "%CD%" compare
python -m shardbench --workspace "%CD%" inject corrupt_config
python -m shardbench --workspace "%CD%" inject ring_conflict
python -m shardbench --workspace "%CD%" inject version_mismatch
python -m shardbench --workspace "%CD%" recover
python -m shardbench --workspace "%CD%" gui
```

运行测试：

```bat
python -m unittest discover -t . -s shardbench/tests -p "test_*.py" -v
```

## 需求与实现对照

1. **强校验加载**：`config.py` 校验表名唯一、分片键列存在、库/表数正整数、取模基数
   `>0`、范围边界严格递增、环节点权重正整数、依赖引用存在且无环；非法整体拒绝并带
   字段路径（E_CONFIG），成环以 E_CYCLE 列出环上单元。
2. **谓词规范化与覆盖识别**：`predicate.py` 支持等值/IN/范围/AND；空白、关键字大小写、
   IN 顺序、AND 顺序均产生相同规范文本与稳定路由标识；未覆盖完整分片键即显式广播并
   标注触发原因，禁止静默降级。
3. **三算法确定性路由**：`algorithms.py` 输出目标库表、算法名、参数、环版本与路由理由；
   `test_determinism.py` 验证任意单元处理顺序/多次重算结果逐字节一致。
4. **双约束并行调度**：`scheduler.py` 固定 `global_concurrency` 个工作线程，单一条件变量
   串行化准入，运行数与每库运行数永不超限；记录单调时钟毫秒的等待/运行区间；结局区分
   `T_SUCCESS / T_BUDGET_TIMEOUT / T_DEP_NOT_READY / T_CONFIG_MISSING`，
   `verify_never_exceeded()` 独立核验峰值。
5. **版本识别与缓存整体替换**：`versioning.py` 内容摘要判断变化、序号单调递增；
   `cache.replace_view()` 整体替换视图并稳定列出变化规则、受影响库表、清空条目数；
   旧条目进入 stale 区不参与汇总；命中校验版本，不一致视同未命中并记录原因。
6. **失效传播与子集重算**：`invalidation.propagate()` 沿依赖 BFS 推导直接/间接下游，
   按稳定拓扑输出失效集合与每步依据；未失效单元的路由标识、目标集合、摘要保持不变；
   `compare` 子命令做增量与全量重算逐项比较。
7. **快照与强制终止恢复**：`snapshot.py` 持久化四类版本、缓存/历史摘要、账本游标、整数
   格式版本与内容摘要；版本不一致整体替换而非合并；`ledger.begin_round()` 先写 unknown
   占位，崩溃后未提交轮次标记 `requires_rerun`，只依据已提交轮次恢复，不虚构退出报告。
8. **GUI/注入/重放**：`gui.py` 六个稳定排序的只读页签；`injection.py` 三类注入对应
   E_CONFIG/E_RING/E_VERSION 并写 JSONL；`replay.py` 独立读账本重算，差异定位到首个
   不一致的单元、库表与字段并以 E_REPLAY 报告。

## 实际验证结果

- `python -m unittest discover ...`：**40 个用例全部通过**（无 ResourceWarning）。
- 首轮 11 个路由单元全部 `T_SUCCESS`；调度核验峰值全局 3（预算 3）、每库 2（配额 2），
  `global_ok=True`、`per_db_ok=True`。
- 规则演进（`base 8 -> 16`）后 `config_version` 由 1 单调递增到 2，缓存整体替换并列出
  受影响库表；`replay 1` 与 `compare` 均报告完全一致。
- 三类注入分别返回 E_CONFIG/E_RING/E_VERSION 并写入 `events.jsonl`；篡改账本目标后
  重放抛出 E_REPLAY 并定位到具体单元、`targets` 字段与库表标识。
- 模拟强制终止后未提交轮次标记 `requires_rerun`，恢复只承认已提交游标。

> 当前开发容器为最小化 Linux，未安装 tkinter，`gui` 子命令会给出友好降级提示并返回
> 退出码 2；GUI 模块已通过语法编译校验，在 Windows 官方 Python（自带 tkinter）下可直接运行。
