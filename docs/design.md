# 设计与确定性约束

## 位置编码

统一接口接收 `scheme, query_features, key_positions, key_features, base, scaling_factor, dimension, precision`，输出相同形状的 query/key 旋转向量、旋转对以及有效基频。实现位于 `rope_diag/encoding.py`。

对维度对 `j = 0..d/2-1`：

```text
theta_j = base_eff ^ (-2j/d)
angle   = effective_position * theta_j
[x'_2j, x'_2j+1] = [x_2j*cos(angle) - x_2j+1*sin(angle),
                    x_2j*sin(angle) + x_2j+1*cos(angle)]
```

方案差异：

- RoPE：`base_eff = base`，`effective_position = position`。
- 线性内插：`base_eff = base`，`effective_position = position / scaling_factor`。
- NTK 感知缩放：`base_eff = base * scaling_factor^(d/(d-2))`，`effective_position = position`。`d=2` 时使用成对极限 `base * scaling_factor`，并在方案参数指纹中记录公式版本。

三种方案共享同一特征形状、旋转对布局与 float64 偏差计算路径。缺失参数、奇数维度、非法精度等会生成 `E_CONFIG_INVALID`；基频为 0 和缩放因子非正分别映射 `E_BASE_ZERO`、`E_SCALE_INVALID`。

## 注意力与分层

键相对距离定义为 `abs(query_position - key_position)`。注意力打分：

```text
score_i = dot(rotate(q, query_position), rotate(k_i, key_position_i)) / sqrt(d)
weight  = softmax(score)
```

softmax 先减最大值再指数化，避免上溢。每个观察值按 `distance <= train_context_length` 分为 `in_train` 或 `out_of_train`；距离恰等于训练长度时归入 `in_train`，同时设置 `boundary_marked=true`，并写入 `boundary_samples`。

逐距离统计与层总计均包含独立样本数、评分数量、均值、总体方差（`ddof=0`）、最小/最大值及 P25/P50/P75/P90/P95/P99 分位数。

## 三阶段误差传播

参考扩展是“同基频、同维度的标准 RoPE 在真实位置上的解析扩展”。对每个外推键计算：

1. `position_vector`：方案编码与参考扩展在 query/key 向量上的逐元素最大绝对偏差，另附归一化 L2。
2. `dot_score`：方案与参考点积打分的绝对偏差。
3. `normalized_attention_weight`：对同一样本全部键分别做 softmax 后，目标键权重的绝对偏差。

阶段顺序固定为 `position_vector → dot_score → normalized_attention_weight`。首个超过对应容差的阶段写入 `first_exceeded_stage`；即使后续阶段出现相反判定，也保留全部偏差和 `exceeded` 标志，并设置 `contradictory_tolerance_judgement`。所有偏差使用 float64，序列化保留 12 位有效数字。

## 敏感性与数值边界

基频与缩放因子分别独立扫描，其他参数固定在配置值。每个扫描值对全部可计算外推键求“相对当前方案基线打分的平均绝对变化”，同时记录相对标准 RoPE 参考的平均偏差。排序键为：

```text
(-score_deviation, parameter_value)
```

仅 `status=scored` 的取值进入排序；边界值保留在 `scan_entries` 与 `boundary_sensitive_samples` 中。边界判据：

- `dimension_truncation`：有效位置非零、频率为正，但旋转角度绝对值低于当前精度档位的 `eps` 分辨率；或逆频率下溢为 0。
- `scaling_overflow`：有效基频、逆频率、角度、cos/sin 或最终旋转向量非有限。
- `base_zero`、`scale_nonpositive`：参数自身不满足数值前提。

每个边界记录附方案、样本、参数值、判据、原因和触发维度索引。

## 内容指纹缓存

阶段文件位于 `state-dir/cache/stages/<cache-key>.json`，缓存键为规范化 JSON 的 SHA-256：

```text
encoding: scheme parameters + sample fingerprint + precision
scoring:  encoding key + per-sample layer classification + score policy version
tracing:  scoring key + tolerances + trace policy version
```

影响范围：

- 改方案/基频/缩放因子：编码键变化，进而使打分和追踪失效。
- 改训练长度：编码全部复用；只有逐样本层分类（层、边界标记、相对距离）变化的样本使打分和追踪失效。
- 改精度：编码、打分、追踪全部失效。
- 改单条样本内容：只有该样本的三阶段失效。
- 改注入配置：受影响任务绕过缓存并进入失败清单，其他任务继续复用。

缓存索引和每个阶段结果均原子替换。报告分别列出 `reused`、`recomputed`、`invalidated` 与 `bypassed`。

## 并发排序

三个计算阶段分别向 `ThreadPoolExecutor` 提交任务，最大线程数严格取自配置 `concurrency`。每个任务携带：

```text
(方案规范序号, 样本 ID, 阶段序号)
```

线程完成顺序不参与报告构造；回收后先按稳定键排序，再写入缓存、失败清单与诊断报告。方案顺序固定为 RoPE、LinearInterpolation、NTK-aware；样本按 ID 排序；注入项按原始序号与 ID 排序；敏感性同分按参数值升序。

## 快照与恢复

运行清单 `manifest.json` 记录全局快照版本、配置/注入/样本哈希、已完成阶段、已完成方案、已完成样本和已完成任务键。

- 未完成快照：版本、配置、注入或样本指纹不一致时抛出 `E_CONFIG_MISMATCH`，拒绝续跑。
- 已完成快照：允许同目录开启新配置运行，历史缓存仍可按内容键复用；显式 `--resume` 时同样执行严格一致性检查。
- 信号：`SIGINT`/`SIGTERM` 设置线程事件；当前阶段全部任务回收并提交快照后返回 0。
- 强制终止：未原子替换的临时文件不会进入索引，恢复时只读取完整阶段文件。

`reuse_manifest` 反映真实执行路径，冷/热运行必然不同；`diagnostic_digest` 排除该执行态审计字段，用于验证诊断内容逐字段一致。
