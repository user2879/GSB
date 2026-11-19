# 技术说明：本地确定性关系与事件抽取编辑应用

## 1. 环境与复现

- 解释器：Python 3.10.12（虚拟环境 `.venv`，系统 `python3.10 -m venv --without-pip` 后用 get-pip 引导）。
- 依赖：仅 `numpy==2.1.3`，固定在 `requirements.txt` 与 `pyproject.toml`；无模型/数据集下载。
- 随机源：`scripts/generate_samples.py` 仅使用 `numpy.random.default_rng(20261120)`，非法文档与边界文档显式构造，逐字节可复现。
- 一键运行：`./run.sh`（生成样本 → 全量 → 交互 → 验收 → 单测）。

## 2. 目录与模块

- `src/rexapp/config.py`：规则配置数据结构、加载与校验（词表非空/去重、阈值单调递增、未知实体类型）。
- `src/rexapp/documents.py`：JSONL 逐行解析与校验，输出稳定排序的失败清单。
- `src/rexapp/extraction.py`：最大匹配 NER（最长优先、同长按起始位置）、关系候选与窗口排除、事件论元与角色优先级。
- `src/rexapp/fixedpoint.py`：定点量化（`Decimal` + `ROUND_HALF_UP`）、阈值比较（等值归高档）、漂移。
- `src/rexapp/fingerprint.py`：BLAKE2b 内容指纹与分阶段缓存键。
- `src/rexapp/cache.py`：四阶段（entity/relation/event/confidence）缓存、下游失效、复用/重算/失效审计。
- `src/rexapp/engine.py`：线程池并发（排序键 `(doc_id, stage)`）、延迟预算语义、快照调度、诊断报告。
- `src/rexapp/snapshots.py`：临时文件 + `os.replace` 原子提交，内容指纹识别版本变化。
- `src/rexapp/commands.py` / `app.py`：六类交互命令与全量/交互两种编排，SIGTERM 完成快照后退出码 0。

## 3. 关键设计决策

### 3.1 确定性
- 所有浮点得分以固定运算顺序（触发衰减 → 兼容度 → 共现距离，权重固定）累加，量化前不做无序聚合；NumPy 仅用于 `float64` 标量运算。
- 实体、三元组、事件、失败与边界清单在输出前按稳定键排序，实体/三元组/事件 ID 在排序后重新编号，跨运行逐字段一致。

### 3.2 定点与边界
- 量化值 = `round_half_up(raw * 10^scale)`，阈值同样量化后比较；量化值等于阈值归入高档，并写入边界清单，保留 `raw_score` 与 `drift`。
- 样本阈值取真实量化值（关系 `0.5495`、事件 `0.5429`），保证每次运行必然产生边界样本。

### 3.3 增量失效与精确复用
- 缓存键由词序列指纹、触发词条目指纹、窗口/阈值/位宽等参数指纹按阶段派生；下游键包含上游键，级联失效天然正确。
- 引擎维护 `_dirty` 集合：触发词/阈值/窗口命令只标记受影响文档的相关阶段，文档替换/删除整体失效；未失效阶段直接复用且不重算，审计清单记录每次 reuse/recompute/invalidate。
- 置信度阶段即使整体命中缓存，也会以当前阈值重放分类（无缓存写入），确保阈值调整后分档字段与冷启动一致。

### 3.4 预算语义
- 每条命令在 `interaction_budget_ms` 的截止时间内处理；超时返回 `E_INTERACT_BUDGET_PENDING`、`latest=false` 的中间摘要，绝不阻塞、绝不把过期结果标记为最新；`drain_pending` 可在预算外收敛。
- 零预算立即进入 pending，测试通过注入 `work_hook` 延迟确定性触发，不依赖真实计时竞态。

### 3.5 并发顺序一致性
- 线程池大小取 `thread_pool_workers`，任务按 `(doc_id, stage)` 稳定键排序提交，结果按文档 ID 聚合导出；与 `workers=1` 的单线程结果逐字段一致（有专门测试）。

### 3.6 快照与退出
- 快照先写 `snapshot-*.tmp` 再 `os.replace` 整体替换；内容未变化时不提升版本。SIGTERM 在当前命令间隙被观察到，完成最后快照后以 0 退出。

## 4. 业务码映射

| 业务码 | 触发场景 |
| --- | --- |
| `E_DOC_INVALID` | 空词序列、重复文档 ID、配置结构非法 |
| `E_JSON_PARSE` | JSONL 行或配置非法 JSON |
| `E_ENTITY_TYPE_UNKNOWN` | 触发词/论元/文档提示引用未知实体类型 |
| `E_POSITION_OUT_OF_RANGE` | 词位置越界、非连续 |
| `E_ARGT_MISSING` | 事件必需角色缺失（保留已抽论元） |
| `E_INTERACT_BUDGET_PENDING` | 预算内无法给最新结果的中间状态 |

## 5. 验证结果（本机实测）

- `python -m unittest discover -s tests`：23 个用例全部通过，覆盖业务码、定点边界、确定性、增量/全量一致、并发顺序、预算 pending、快照版本、SIGTERM 退出码。
- `scripts/acceptance_check.py`：7 条命令逐条与冷启动全量比较，`all_commands_match_cold_full_run=true`，覆盖空结果、全部失败、单文档多次替换。
- 诊断报告 `output/diagnostic_report.json` 含分档分布、边界样本（含漂移）、复用/失效审计、各命令耗时；快照位于 `output/snapshots/snapshot.json`。
