# qcalib 设计与口径说明

## 1. 模块划分
- `qcalib/errors.py`：稳定业务码（`E_INPUT_SCHEMA`、`E_RECEIPT_MALFORMED`、
  `E_ZERO_SCALE`、`E_QUOTA_EXCEEDED`、`E_BUDGET_PENDING`）与确定性失败清单。
- `qcalib/yamlmini.py`：仅依赖标准库的确定性 YAML 子集解析/序列化，避免外部依赖。
- `qcalib/loaders.py`：权重包 JSONL、校准 JSON、配置 YAML、命令流的加载与校验，
  单条/单层失败只入清单不中断其余层。
- `qcalib/receipts.py`：q1/q2/q3 三版回执解析、缺失字段标记与重复条目裁决账本。
- `qcalib/calibration.py`：逐组稳定排序、固定分位插值、对称/非对称缩放因子与零点。
- `qcalib/quantize.py`：定点重量化、饱和截断、反量化与逐元素误差归因。
- `qcalib/distill.py`：校准前向、软/硬标签蒸馏校正、四环节逐级偏差与容差判定。
- `qcalib/cache.py`：层/组/参数/档位/校准集指纹、缓存键与失效/复用账本。
- `qcalib/scheduler.py`：全局预算、单层配额、单批点数三重约束与稳定排序键执行。
- `qcalib/snapshot.py`：阶段临时目录写入 + 整体替换提交 + 配置哈希/版本恢复判定。
- `qcalib/pipeline.py`、`runner.py`、`commands.py`、`cli.py`：编排、命令应用与 CLI。

## 2. 固定数值口径
- 分位：`h = p * (n - 1)`，对相邻次序统计量做线性插值；排序固定为
  NumPy `mergesort`（稳定排序），并列值保留首次出现次序。
- 对称方案：`scale = max(|q_lo|, |q_hi|) / (2^(frac_bits+bits-1) - 1)`，`zero=0`。
- 非对称方案：`scale = (q_hi - q_lo) / (2^(frac_bits+bits) - 1)`，
  `zero = clamp(round(low_int - q_lo/scale))`。
- 定点整数：`raw = round((x/scale + zero) * 2^frac_bits)`，随后按
  `int4/uint4/int8/uint8/int16/uint16` 档位区间裁剪或回绕。
- 反量化固定顺序：`(integer / 2^frac_bits - zero) * scale`。
- `scale=0`（全零组或非对称零跨度）输出 `E_ZERO_SCALE`，该组改用固定回退
  `scale=1.0` 并在产物中标注 `zero_scale_fallback`。

## 3. 误差归因优先级
逐元素按固定优先级判定：饱和截断 > 分位离散 > 反量化舍入 > 缩放量化。
真值为 0 时相对误差无定义，改用绝对误差并置 `zero_truth_branch=true`。

## 4. 回执裁决顺序
1. q3 且携带 `override_refs` 的条目优先；
2. 否则 `elapsed_ms` 降序，缺失视为 -1；
3. 耗时打平时 `param_version` 降序，缺失视为 -1；
4. 仍打平时回执文件名字典序升序、条目序号升序。
裁决规则与全部候选均写入 `receipt_ledger`。

## 5. 缓存键与失效
- 标定键：层内容指纹 + 分组内容指纹 + 标定参数指纹（不含档位/小数位宽）。
- 重量化/反量化误差键：再叠加量化档位指纹（level/bits/frac_bits）。
- 蒸馏键：再叠加校准样本子集指纹与蒸馏参数指纹。
- 回执键：回执目录与该层回执条目内容指纹。
单层权重替换只失效该层相关组；档位切换只失效重量化及下游；数据规模档位切换只
失效受影响层的蒸馏；未变化部分在增量运行中命中复用。

## 6. 全量/增量一致性
账本同时输出两份复用视图：
- `reuse_ledger`：本次运行的命中/未命中诊断视图（增量运行出现命中）。
- `canonical_invalidation_ledger`：剔除瞬态命中标记的规范视图，冷启动全量与
  任意命令序列后的增量运行逐字段一致。
除复用视图外，缩放因子、零点、定点整数、反量化值、误差归因、蒸馏校正量、
失败清单顺序在两种运行方式下完全相同。

## 7. 预算与快照
- 调度按 `(层名字典序, 组序号)` 稳定消耗预算；超单层配额中止该层并记
  `E_QUOTA_EXCEEDED`，全局预算耗尽保留待办并记 `E_BUDGET_PENDING`。
- 线程池最大并发数取配置 `workers`；任务带稳定排序键，并行结果按键重排，
  因此串行与并行结果一致。
- 五个阶段（解析校验、分位标定、定点重量化、蒸馏校正、账本汇总）均先写
  `tmp/` 再 `os.replace` 整体替换；提交清单记录配置哈希与层内容版本，
  恢复时哈希或版本不一致的阶段不会被复用。
