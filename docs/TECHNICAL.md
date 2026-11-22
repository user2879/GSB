# cqdiag 逐列量化标定诊断工作台 — 技术说明

## 概览

`cqdiag` 面向逐列分位标定、定点重缩放、饱和截断与反量化误差归因的诊断场景。
全程离线，仅依赖 Python 3.10+ 与 NumPy 1.24+，样本由固定种子确定性生成。

## 目录结构

- `cqdiag/config.py` — 配置加载与校验（E_CONFIG_INVALID / E_QUANT_WIDTH_INVALID / E_GROUP_MISMATCH）
- `cqdiag/samples.py` — 样本 JSONL 加载校验、固定种子生成、列替换与物化导出
- `cqdiag/calibration.py` — 逐列分位标定与缩放因子/零点推导
- `cqdiag/quantize.py` — 定点重缩放与饱和截断内核
- `cqdiag/dequant.py` — 反量化、容差判定与逐元素误差归因
- `cqdiag/stratify.py` — 列组分层敏感性分析
- `cqdiag/fingerprints.py` — 内容指纹与阶段缓存键派生
- `cqdiag/store.py` — 阶段快照存储（临时目录 + 整体替换提交 + 启动恢复）
- `cqdiag/pipeline.py` — 失效判定、预算约束并发调度、阶段提交
- `cqdiag/edits.py` — 编辑计划应用（样本替换/分位/位宽/饱和策略/列组重划）
- `cqdiag/cli.py` — `python -m cqdiag run` 入口，含全量重建一致性验证
- `scripts/gen_inputs.py` — 确定性生成 config.json / samples.jsonl / edits.json
- `tests/` — 57 个 pytest 用例

## 固定数值规则（写入产物元数据）

- 分位插值：`linear`，排序后位置 `pos = p*(n-1)` 线性插值；并列值不做特判，
  由稳定排序 + 线性插值天然确定（`tie_rule = stable_sort_linear_interp`）。
- 对称方案：`scale = max(|q_lo|, |q_hi|) / qmax`，`zp = 0`。
- 非对称方案：`scale = (q_hi - q_lo) / (qmax - qmin)`，
  `zp = round(qmin - q_lo/scale)` 并裁剪到 `[qmin, qmax]`；有符号区间
  `qmin = -2^(b-1)`, `qmax = 2^(b-1)-1`。
- 定点化：`fx = round_half_to_even(x * 2^frac_bits)`（NumPy 规则），
  `scale_fp = max(1, round(scale * 2^frac_bits))`，
  `q = round_half_away_from_zero(fx / scale_fp) + zp`（精确整数运算）。
  `frac_bits` 与量化位宽真实改变中间整数与截断集合（测试
  `test_frac_bits_change_intermediates` / `test_quant_bits_change_clip_set` 覆盖）。
- 截断方向：`+1` 上溢、`-1` 下溢、`0` 未截断；逐元素记录截断前定点值 `pre_clip`。
- 饱和策略：`clamp` 截断到区间；`mask` 将截断元素标记为无效，反量化直接判超差
  并归因 `saturation`。

## 误差归因

超差判定：`|x_hat - x| > abs_tol 且 rel_err > rel_tol`；真值为零时仅按绝对误差
判定并标注分支 `zero_truth_abs`。四类来源贡献（绝对值域）：

| 来源 | 贡献定义 |
|---|---|
| `saturation` | 元素超出可表示浮点区间的距离（截断元素） |
| `scale_quantization` | `|q - zp| * |scale - scale_fp/2^f|` |
| `quantile_discrete` | 元素超出分位标定区间 `[q_lo, q_hi]` 的距离 |
| `dequant_rounding` | `|x/scale_q - round(x/scale_q)| * scale_q`（未截断元素） |

决断取贡献最大者；并列按固定优先级
`saturation > scale_quantization > quantile_discrete > dequant_rounding`，
决断依据写入 `basis` 字段。

## 列级失效与缓存键

阶段缓存键链（sha256，JSON 规范化序列化）：

```
calib   = H(列内容指纹, 分位参数指纹)
scale   = H(calib, scheme, quant_bits)
quant   = H(scale, 列内容指纹, frac_bits, saturation)
dequant = H(quant, 列内容指纹, abs_tol, rel_tol)
stratify= H(张量, 组标号, 各列 dequant 键, group_oor_limit)
```

列内容指纹覆盖逐元素值与缺失标记；calib 产物记录 `element_refs`（引用的样本
元素下标集合）与 `valid_count`。替换某列元素仅使该列四阶段失效；分位参数调整
仅级联 calib 下游；位宽切换保留 calib 复用；列组重划仅失效 stratify
（`tests/test_incremental.py` 逐项覆盖）。

## 并发、预算与快照

- 线程池 `max_workers` 取自配置且不得超过；任务纯函数，携带
  `{stage}|{tensor}:{col}` 稳定排序键，按排序键顺序收集结果，因此列顺序、
  逐元素顺序、预算扣减序列、失败清单与单线程按列标识顺序执行完全一致
  （`test_parallel_matches_single_thread`）。
- 单批标定元素预算：按列标识顺序扣减列元素数，耗尽后列进入失败清单
  `E_BUDGET_EXCEEDED` 并跳过下游（`test_budget_exceeded_deterministic`）。
- 快照提交：产物先写 `staging/` 再整体 `os.replace` 到 `snapshots/`，清单
  `manifest.json` 原子写，记录全局版本号、配置哈希、已完成列与各产物版本。
  启动时：残留 `staging/` 判定为提交中断并丢弃；format 版本变化作废旧快照；
  清单引用产物缺失/损坏则剔除并触发该列重算（`tests/test_recovery.py`）。

## 使用

```bash
python3 scripts/gen_inputs.py --outdir runs/demo          # 生成确定性输入
python3 -m cqdiag run --config runs/demo/config.json \
    --samples runs/demo/samples.jsonl --workdir runs/demo/workdir \
    --report runs/demo/report.json --edits runs/demo/edits.json \
    --verify-rebuild                                       # 增量 + 全量重建一致性验证
python3 -m pytest tests/ -q                                # 测试
```

报告含：全局版本号、配置哈希、版本变化识别、恢复动作、逐列误差摘要
（scale/zp/valid_count/clipped/oor/max_err/主导来源）、分层结论与超限
（组, 位宽）组合、位宽切换前后截断元素数量对比、预算扣减日志、失败清单。

## 实际验证结果（2026-11-23，Python 3.11 + NumPy 1.26.4）

- `python3 -m pytest tests/ -q`：**57 passed**。
- 演示数据集（2 张量：48×12 与 6×8×8，20 列）+ 4 批编辑序列
  （样本替换、分位调整、8↔4 位宽切换、frac_bits 8→6、clamp↔mask、列组重划、
  容差调整）：增量运行 5 个全局版本，`--verify-rebuild` 输出
  `{"consistent": true, "diff_fields": []}`，即逐列缩放因子、零点、逐元素
  定点值、截断标记、误差归因与分层结论与空白全量重建逐字段一致。
- 位宽切换对比真实输出，例如 `weights_a` 由 (8bit, frac8) 38 个截断元素变为
  (4bit, frac6) 30 个，`weights_b` 25→25；切回 8bit 后 30/24。
- 复跑同配置：`version_change=unchanged`，四阶段 20/20 列全部复用、零重算。
- 异常输入：`quant_bits=6` → `E_QUANT_WIDTH_INVALID`；组标号不覆盖全部列 →
  `E_GROUP_MISMATCH`，均进入失败清单并以退出码 2 终止。
