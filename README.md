# 位置编码外推诊断工作台（PE-Diag）

面向长上下文 LLM 位置编码外推能力诊断的**可复用领域组件 + 可重复端到端示例**。
纯 Python 3.11 + NumPy，无模型权重、无网络、无 GPU、无容器；所有结果由固定种子与
内容指纹决定，保证「增量运行 == 空白全量重建 == 并发执行」逐字段一致。

## 1. 目录结构

```
pe_diag/                  可被外部反复调用的领域组件包
  common.py               规范 JSON/SHA-256 指纹/确定性 RNG/精度工具
  errors.py               业务错误码与 DiagError
  config.py               四类输入加载、按规模档生效的 EffectiveConfig、样本/注入模型
  encoding.py             RoPE / 线性内插 / NTK 感知缩放，统一 PositionEncoder 接口
  scoring.py              点积注意力打分、训练长度内外分层、single/stream 两条计算路径
  error_tracking.py       位置向量→打分→权重 三级偏差、首次超差、精度归因
  sensitivity.py          基频/缩放因子邻域扫描、边界事件、稳定敏感性排序
  strategy.py             三档规模算法取舍选择器（预置策略表 + 可审计评分）
  cache.py                内容指纹缓存键与精确失效
  injection.py            异常注入应用层（真实输入故障 → 业务错误码）
  snapshot.py             原子对象快照、清单、E_SNAPSHOT_STALE 恢复判定
  aggregation.py          训练长度内外分层统计（确定求和顺序）
  pipeline.py             确定性编排（串行/线程池、缓存复用、快照提交、报告组装）
scripts/
  generate_inputs.py      固定种子生成 samples/ 下四类输入
  run_demo.py             端到端运行入口（serial / parallel）
  verify_equivalence.py   编辑序列下增量/全量/并发等价性与精确失效验证
tests/                    28 个 pytest 用例
samples/                  生成的配置、JSONL 样本、规模描述、注入清单
snapshots/ reports/       快照输出目录与诊断报告输出目录（配置驱动）
```

## 2. 输入文件（`samples/`，全部固定种子生成）

- `position_encoding_config.json`：方案集合、基频、训练/目标长度、维度、缩放因子、
  精度档位、维度切分块大小、并发数、快照提交间隔、容差、敏感性扫描网格、
  单批点数预算与内存上限，以及按规模档覆盖的 `scale_profiles`。
- `attention_samples.jsonl`：每行含样本 ID、规模档、查询位置、键位置列表、
  固定种子生成的查询/键维度特征向量。
- `scale_descriptor.json`：三档规模（small/medium/large），样本条数、维度、
  外推倍率从「数百条/数十维/2x」到「上千条/近两百维/32x」。
- `injection_manifest.json`：非有限特征、键位置越界、基频为零、缩放因子非正四类注入。

三档规模与默认算法取舍（由预算/内存资格判定自然产生，而非硬编码）：

| 档位 | 样本数 | 维度 | 外推 | 选中策略 | 参与维度 | 分块方式 | 分层精度 |
|------|--------|------|------|----------|----------|----------|----------|
| small | 300 | 32 | 2x | S_FULL | 全维 | single | float64 |
| medium | 500 | 72(96×75%) | 8x | S_CHUNK | 截断 | single | float64 |
| large | 800 | 96(192×50%) | 32x | S_STREAM | 截断 | stream | float32 |

## 3. 快速开始

```bash
pip install -r requirements.txt          # 仅 numpy
python scripts/generate_inputs.py        # 生成确定性样本（可重复，字节级一致）

# 冷启动全量（串行）
python scripts/run_demo.py --samples-dir samples --mode serial \
    --report-name full_serial.json --snapshot-subdir demo

# 线程池并发（workers 取自配置）
python scripts/run_demo.py --samples-dir samples --mode parallel \
    --report-name full_parallel.json --snapshot-subdir demo_par

# 再次运行即增量续跑，未变化任务直接复用
python scripts/run_demo.py --samples-dir samples --mode serial

# 编辑序列等价性 + 精确失效审计（小规模快速演示）
python scripts/verify_equivalence.py

# 单元测试
python -m pytest tests/ -q
```

## 4. 三类位置编码的统一接口

`PositionEncoder(scheme, base, train_len, target_len, dim, scale, chunk_size,
precision, active_dim)` 对三方案返回相同的 `EncodingResult`（位置向量、cos/sin
旋转对、float64 角度、元数据）。元数据显式记录：方案、基频与有效基频、缩放因子、
有效位置缩放、生成维度范围 `dim_range`、`active_dim`、切分块大小、精度、角度归约方式。

- `rope`：角度 `pos · base^(-2j/d)`；
- `linear`：有效位置 `pos/scale`（基频不变）；
- `ntk`：固定 NTK，有效基频 `base · scale^(d/(d-2))`（位置不变）。

两处保证「改配置真实改数值」而非只改标注：
1. `active_dim` 真实截断旋转对，超出维度不参与旋转与打分；
2. 逆频率在每个 `chunk_size` 块内以**块起点重置的累乘**在工作精度中递推，
   不同块边界累积不同舍入历史：float64 差异约 1e-17、float32 约 1e-7~1e-5。

## 5. 打分、分层与误差追踪

- 打分：旋转查询/键后做缩放点积，`single` 批量 softmax 与 `stream` 在线 softmax
  是两条真实不同的计算/累加路径；权重归一，float64 两路径一致、float32 有可见舍入差。
- 分层：相对距离 `|key-query|`，`<= train_len` 判为训练长度内（**边界相等归 inside**），
  其余为 `extrap`；同距离在两区间分别汇总（均值/标准差/最小/最大/计数）。
- 三级误差：以全维 float64、无缩放 RoPE 为参考，依次计算位置向量相对 L2、
  点积打分偏差、归一化权重偏差，标注首个超差环节（`position_vector/score/weight`）。
- 精度归因：float32 与 float64 权重不一致时，分别量化「维度截断差」与
  「累加舍入差」，归因到 `position_encoding_dimension_truncation` 或
  `score_accumulation_rounding`。

## 6. 敏感性与规模策略

- 敏感性：对每方案扫描基频/缩放因子网格，输出外推打分平均偏差排序
  （偏差降序、取值升序的稳定决断）；`base=0` 记 `E_BASE_ZERO`、
  `scale<=0` 记 `E_SCALE_INVALID` 作为边界敏感事件与触发条件。
- 策略选择器：`POLICY_TABLE` 中三候选按点数预算与内存代理做资格判定，
  选择「最轻量可行」策略；输出各候选评分、资格、因子与最终决策，全程可审计，
  切换档位真实改变参与维度、分块路径与分层精度。

## 7. 指纹缓存、快照与等价性保证

- 缓存键由「阶段相关方案参数指纹 + 样本内容指纹 + 精度档 + 规模档 + 策略决策指纹」
  经 SHA-256 派生；改基频/缩放/训练长度/档位只失效对应阶段，替换单条样本只失效
  该样本任务。`verify_equivalence.py` 实测：单样本替换仅重算该样本相关任务。
- 快照：每个阶段对象先写临时目录再 `os.replace` 原子整体替换（不逐文件覆盖），
  清单记录格式/全局版本号、四类输入哈希、完成任务与失败记录；仅当磁盘格式不兼容
  才判定 `E_SNAPSHOT_STALE` 并新建 generation，旧快照保持只读不被改写。
- 确定性：任务按 `scheme|sample_id|stage` 稳定排序；线程池 `map` 保序回排，
  聚合按 `(sample_id, precision, key_index)` 固定顺序求和，报告以规范 JSON
  （键排序、无 NaN）落盘，因此并发/串行/续跑/冷重建业务字段逐字段一致。

## 8. 异常注入隔离

注入清单命中的任务在真实输入上施加故障，复用与坏数据相同的校验路径抛出业务码：
`E_NONFINITE_INPUT`、`E_KEY_POS_OOR`、`E_BASE_ZERO`、`E_SCALE_INVALID`。
单样本单阶段失败进入持久化失败清单（带命中注入项 ID），不进入统计汇总，其余任务继续。

## 9. 实测验证结果（本机 Python 3.11 + NumPy 1.26）

- `python -m pytest tests/ -q`：**28 passed**。
- 三档样本三方案外推打分、三级误差分级、档位取舍、敏感性排序、注入故障码均覆盖。
- 全量（1600 样本 × 3 方案 × 3 阶段 × 2 精度 + 敏感性）：
  串行冷跑约 3 分钟；`full_serial.json` 与 `full_parallel.json` 业务字段逐字段相等、
  失败清单顺序一致；四类故障码各有命中。
- `verify_equivalence.py`：方案切换、基频调整、训练长度变更、档位切换、单样本替换
  的编辑序列下，增量结果 == 空白全量 == 并发全量；并打印每步复用/重算/失效计数。

## 10. Windows 运行说明

全部路径使用 `pathlib`、无符号链接/无 fork/无容器，命令在 PowerShell 等价执行：
`python scripts\generate_inputs.py`、`python scripts\run_demo.py --samples-dir samples --mode serial`。
