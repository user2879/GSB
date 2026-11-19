# CSR SpMM 增量算子组件 — 技术说明

## 组件结构

- `spmm/csr.py` — CSR 构造/校验/行级编辑（替换、删除、追加）与输入指纹。
  校验失败抛 `E_CSR_INVALID`，违规项记录行号与字段
  （`indptr`/`indices`/`indices_order`/`nnz`）；非有限元素抛 `E_NONFINITE`；
  形状不一致抛 `E_SHAPE_MISMATCH`。
- `spmm/kernel.py` — 单精度参照内核（升序、单块、fp32 累加）与行分块内核。
  行内非零元按 `nnz_chunk_size` 切块，块内部分和二次累加；累加顺序支持
  `asc`/`desc`/`abs_desc`（按对应 B 行元素绝对值之和降序，平局按列索引升序，
  保证确定性）。每行记录 `nnz`、累加顺序标识、空行标识 `EMPTY_ROW` 与行结束
  指纹（blake2b，绑定行号/nnz/顺序/输出字节）。块参数越界抛 `E_BLOCK_INVALID`。
- `spmm/config.py` — 引擎配置与校验；`RESULT_PARAMS`（块大小、nnz 块、alpha、
  beta、精度档位、累加顺序）构成配置指纹，变更即触发全量失效。
- `spmm/engine.py` — 行块依赖图、精确失效判定、单次重算预算调度、线程池并发、
  快照提交与 JSON 报告。
- `spmm/snapshot.py` — 临时文件 + `os.replace` 原子替换；`snapshot.json` 最后
  落位作为提交标记，残留 `*.tmp` 一律忽略。
- `spmm/inject.py` — 异常注入隔离执行与业务码映射。
- `spmm/datasets.py` — 固定种子生成 uniform/powerlaw/extreme 三种分布样本。

## 失效与调度语义

- 每个输出行块记录依赖：读取的 A 行集合、B 行集合、参数指纹与产出指纹。
- 行替换：仅含被替换行的块失效；行删除：删除点所在块及之后所有块失效
  （行号平移导致块成员变化）；行追加：承载新行的块失效；`RESULT_PARAMS`
  任一变更：全部块失效。未失效块直接复用持久化结果，绝不重算。
- 重算集合按行块序号升序；每批最多消费 `recompute_budget` 个块，剩余进入
  待重算队列并标记 `E_RECOMPUTE_PENDING`，下次运行从队列头部继续；预算为零
  且有待算块时额外标记 `E_BUDGET_ZERO`。预算扣减序列写入报告可审计。
- 并发：线程池按块并行，块任务以行块序号为稳定排序键，汇总严格按序号序，
  与单线程执行逐位一致（`tests/test_concurrency.py`）。

## 快照与恢复

- 快照含全局版本号、配置指纹、输入指纹、已完成块列表、待重算队列、依赖图
  与产出指纹；按 `snapshot_commit_interval` 个块提交一次中间快照
  （`run_in_progress=true`）。
- 启动时校验基准输入/配置指纹，不一致抛 `E_SNAPSHOT_STALE` 并拒绝续跑；
  强杀后仅依据已完整替换的快照恢复：若快照指纹等于新输入指纹且
  `run_in_progress`，从待重算队列断点续算。
- SIGTERM/SIGINT：完成当前行块快照提交后以退出码 0 退出。

## 运行方式

```bash
python3 scripts/gen_samples.py      # 生成确定性样本到 data/
python3 scripts/run_demo.py         # 全量+编辑序列+增量+等价核对，报告到 reports/
python3 scripts/run_injections.py   # 异常注入，报告到 reports/injections_report.json
python3 -m pytest tests/ -q         # 43 项测试
```

## 实际验证结果（本机，Python 3.11 + NumPy 2.4.6 + pytest 9.2.0）

- `pytest`：43 passed。覆盖 CSR 校验违规、三种分布确定性、内核三种累加顺序
  与多种切块对 fp64 参照的一致性、空行标识、块参数越界、编辑序列（替换/删除/
  追加/块大小/alpha-beta/精度档位/累加顺序）在 3 分布 × 2 块大小下与全量重建
  的逐元素一致、精确失效范围、预算调度与待算队列、预算为零、快照失效拒绝、
  临时文件残留忽略、SIGTERM 退出码 0、SIGKILL 断点续算与全量重建一致、
  并发与串行逐位一致、7 项注入全部命中。
- `run_demo.py`：`equivalence_passed=true`（三种分布全部编辑步骤），
  快照自检 `stale_rejected=true`、`tmp_residue_ignored=true`。
- `run_injections.py`：7/7 命中（E_NONFINITE×2、E_SHAPE_MISMATCH、
  E_BLOCK_INVALID×2、E_BUDGET_ZERO、E_CSR_INVALID）。
