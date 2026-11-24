# quantcalib 技术说明

## 1. 阶段与目录

管线分为六个持久化阶段：

1. `loading`：权重包、校准集、配置、命令、故障和回执加载校验。
2. `calibration`：逐层逐组分位、缩放因子与零点。
3. `quantization`：逐组定点整数、饱和截断和反量化。
4. `error_analysis`：校准投影输出、误差与来源归因。
5. `distillation`：软标签校正及校正前后偏差。
6. `receipt_ledger`：三版本回执归一、权威裁决和依赖绑定。

代码位置：

- `quantcalib/loading.py`：输入校验。
- `quantcalib/calibration.py`：分位与仿射参数。
- `quantcalib/quantization.py`：定点内核。
- `quantcalib/distillation.py`：响应、归因、蒸馏。
- `quantcalib/receipts.py`：账本与版本转换。
- `quantcalib/scheduler.py`：三重预算。
- `quantcalib/snapshot.py`：代际快照与原子指针。
- `quantcalib/pipeline.py`：缓存、并发、清单和报告。

## 2. 数值规则

### 2.1 分位

组内值按 `(value, original_index)` 升序排序。千分位 `p` 的位置固定为：

```text
position = (p / 1000) * (n - 1)
```

位置落在两个元素之间时使用线性插值；落在整数位置时取该元素。复制值在排序中保持原始下标顺序，等值边界可互换，因此不依赖排序实现的不稳定性。规则写入每组产物的 `percentile_metadata`。

### 2.2 缩放因子与零点

非对称方案：

```text
scale_base = (q_hi - q_lo) / (qmax - qmin)
zero = round((qmin + qmax)/2 - (min + max)/(2*scale))
```

对称方案：

```text
scale_base = max(abs(q_lo), abs(q_hi)) / qmax
zero = 0
```

其中 8 位非对称整数区间为 `[-128, 127]`，对称区间为 `[-127, 127]`。缩放因子先以 float64 计算，再按 `scale_effective_digits` 裁剪有效位；最终步长还会以 `2^-fractional_bits` 为量子，乘以 `quantization_levels` 后取整，因此切换小数位宽或量化档位会真实改变步长。

### 2.3 定点重量化

每个权重执行：

```text
q_real = w / scale + zero
q = round_half_away_from_zero(q_real)
q_sat = clamp(q, qmin, qmax)
w_pre = (q_sat - zero) * scale
w_hat = fixed_point_round(w_pre, fractional_bits)
```

`w_hat` 是实际反量化输出。只有 `q_sat != q` 的元素进入 `truncations`，记录方向 `low/high`、`pre_saturation_fixed`、`pre_saturation_integer`、饱和整数与反量化值。

`role` 和 `role_fingerprint` 包含位宽、小数位宽、量化档位、最终 scale/zero、qmin/qmax 与保留小数位，因此配置档位变化会出现在诊断报告中。

## 3. 校准响应、蒸馏与归因

### 3.1 确定性响应

每组权重经 SHA-256 派生的稳定基投影到校准输出维。组贡献为：

```text
contribution[j] = sum_i weight[i] * activation[i % input_dim] * basis(i, j, layer, group)
```

基系数除以组宽平方根。层输出为组贡献按组号顺序求和。参考输出单独记录，不与本地投影混淆。

### 3.2 蒸馏

对每个样本先计算一次 teacher 软目标 `softmax(raw/T)`。每一步：

```text
student = softmax(current/T)
agreement = dot(teacher, student)
alpha = soft_label_weight * T/(1+T) * agreement
current = current + alpha * (raw - current)
```

输出逐元素同步更新。温度、步数、软标签权重和 alpha/agreement 序列均写入蒸馏产物。

### 3.3 误差与分支

非零真值使用相对误差 `abs(actual-truth)/abs(truth)`；真值为零时改用绝对误差，分支固定为 `absolute_zero_truth`。超差判断使用配置 `tolerance`，并同时输出校正前、校正后绝对/相对误差。

四个来源探针分别为：

- `percentile_discretization`：分位范围与全范围标定造成的反量化差；
- `scale_quantization`：最终定点档位相对未档位化 base scale 的差；
- `saturation_truncation`：饱和后与未饱和定点值的差；
- `dequant_rounding`：反量化定点舍入前后的差。

每个输出元素取绝对贡献最大的来源。并列时固定优先级：

```text
saturation_truncation
percentile_discretization
scale_quantization
dequant_rounding
```

## 4. 回执归一与转换

回执按 `(layer_name, parameter_version)` 分组。q3 的 `supersedes` 先解析并移除被覆盖候选：

1. 存在 q3 覆盖引用时，在仍有效的 q3 中按 `elapsed_ms` 升序、参数版本降序、文件名字典序、行号决定权威项。
2. 没有 q3 覆盖引用时，按参数版本降序、文件名字典序、行号、耗时升序决定。
3. 裁决依据、被覆盖引用和候选集合全部写入账本。

q3 降级到 q1/q2 时，`group_scale_digest` 与 `supersedes` 写入旁路 sidecar；降级到 q2 时仅隐藏这两个 q3 专属字段，到 q1 时还会隐藏参数版本与耗时。sidecar 按输出行编号映射；重新转回 q3 时逐行恢复。转换报告记录 `convert_*`、`restore_from_sidecar`、保留字段和缺失字段。

## 5. 缓存键与失效边界

逐组各阶段键由以下内容派生：

- 层内容指纹：层名、shape、全部权重、层内容版本、层分组标记；
- 逐层参数指纹：scheme、位宽、分组、分位、小数位宽、量化档位、有效位；
- 位宽角色：最终 scale/zero、qmin/qmax、保留小数位；
- 校准集指纹：维度、样本 id、输入、参考输出。

阶段键链：

```text
content + calibration config + calibration set
  -> calibration key
  -> quantization key (adds bit profile and batch profile)
  -> error key (adds tolerance)
  -> distillation key (adds temperature/steps/soft weight)
  -> receipt ledger key (adds normalized receipts and all layer keys)
```

替换单层权重只改变该层内容指纹，因此该层的标定、量化、误差、蒸馏和全局账本失效；其他层键不变，直接从持久化产物复用。报告 `reuse.events` 和 `reuse.by_layer` 逐层记录 computed/reused，`invalidation.layers` 记录失效边界。

## 6. 预算与并发

调度先按层名字典序、组号升序生成确定性计划，再派发线程任务：

- 每组点数消耗全局预算和单层配额。
- 单层把下一组会超配额时中止该层，余项记 `E_QUOTA_EXCEEDED`。
- 全局预算不足时，当前及后续组保留待办，记 `E_BATCH_BUDGET_PENDING`。
- 量化点数按单批预算拆块；所有批块点数不超过批预算。
- 预算扣减、批块、待办全部在任务派发前确定，线程数与实际完成时序不改变序列或数值。

## 7. 快照事务与恢复

每次运行在 `.txn-*` 工作副本上读取旧缓存并写新产物。每次提交：

1. 将完整事务目录复制到 `.snap-*`；
2. 写入 `manifest.json`；
3. 目录改名为 `generations/gen-NNNNNNNN`；
4. 原子替换 `CURRENT` 指针文件。

因此 `CURRENT` 指向的永远是完整代际，不暴露半写入文件。清单记录全局内容版本、配置哈希、权重/校准/回执指纹、完成层、部分层、消耗预算、阶段产物版本、逐层缓存键和运行内快照序列。提交前比较旧全局版本和配置哈希；精确失效仍由逐层缓存键决定。

## 8. 本次实际验证

环境：Python 3.11.0rc1，NumPy 1.26.4，PyYAML 6.0.1，本地 Linux 工作区；代码不访问网络。

自测命令：

```text
python -m unittest discover -s tests -v
```

结果：`Ran 34 tests in 3.818s`，`OK`。

演示命令：

```text
python -m quantcalib demo --workdir artifacts/demo
```

结果摘要：

- 基线状态 `ok`，完成 3 层，`result_digest=2b697e125283974ecb1decb452e7e3959baa99fafe517b1e4626d656e56cfa0a`。
- 编辑 `retrieval.head` 后增量与空白全量均为 `ok`，共同摘要 `4c2080fd9d7c12c0d79080c2b81e608bf600ee9c298d0b7a737a290e4187964b`。
- 未变化的 `embedding.proj` 与 `encoder.block` 四个逐层阶段全部复用；仅 `retrieval.head` 及其下游账本失效。
- 二次增量同样逐层复用，核心结果与首次增量一致。
- 四类故障 `F01_NONFINITE`、`F02_GROUP_INDIVISIBLE`、`F03_BIT_WIDTH`、`F04_CALIBRATION_DIM` 全部 `hit=true`，报告为 `failed`，业务码计数为各 1：`E_WEIGHT_INVALID`、`E_GROUP_INVALID`、`E_QUANT_BOUNDS`、`E_CALIBRATION_INVALID`。
- q3→q1→q3 与 q3→q2→q3 均恢复 `group_scale_digest`/`supersedes`，逐行 JSON 语义一致，转换失败数 0。
- 示例预算总计消耗 36 点，按 8 点单批预算形成 5 个批；权威回执中两个 q3 通过覆盖引用和耗时裁决，一个 q3 通过参数版本降序裁决。

全部机器可读结果位于 `artifacts/demo/`。
