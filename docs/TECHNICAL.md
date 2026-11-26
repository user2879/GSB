# 技术说明

## 1. 流水线与模块

处理顺序固定为：

1. 配置、模板、数据集、PNG、真值关键点和标注版本校验。
2. 滑窗粗定位：窗口切片、墨迹密度、竖直投影谷、纹理梯度、指纹派生投影。
3. 连通域细定位：连通碎片合并、超大区域投影分裂、合并/分裂依据逐条记录。
4. 关键点回归：块内加权墨迹重心与边界中心固定权融合，标注版本可覆盖坐标。
5. 多模板识别：3 个模板变体按权重融合，逐字符保留模板得分和置信度。
6. 阅读顺序：栏右到左、栏内上到下，冲突按固定优先级裁决。
7. 缓存失效、预算调度、阶段快照、误差追踪和异常注入。

对应模块：`config.py`、`samples.py`、`localize.py`、`keypoints.py`、`recognizer.py`、`reading.py`、`pipeline.py`、`commands.py`、`snapshot.py`、`trace.py`、`injections.py`。

## 2. 定点公式

量化档位定义为 `Q4/Q6/Q8/Q10`，定点小数位宽必须与档位一致。

```text
quantize_fixed(v, bits) = round_half_up(v / 2^bits) * 2^bits
ratio_fixed(n, d, bits) = round_half_up(n * 2^bits / d)
```

关键点回归先计算阈值内像素的加权重心：

```text
strength(x,y) = max(1, ink - gray(x,y))  if gray(x,y) <= ink
centroid = ratio_fixed(sum(strength * coordinate), sum(strength), bits)
bbox_center = (left + right, top + bottom) / 2
raw = round_half_up(3 * centroid + bbox_center, 4)
```

随后对 `raw` 量化，整数坐标由量化值四舍五入得到，并记录 `UP`、`DOWN` 或 `EXACT`。

模板匹配对每个变体尝试 `3 x 3` 的 `(-1,0,1)` 对齐，取最佳交并比：

```text
score_scale = 2^(fixed_bits + 10)
variant_score = round_half_up(intersection * score_scale / union)
glyph_score = round_half_up(sum(weight_i * variant_score_i) / sum(weight_i))
```

当前变体权重为 `(64, 2, 1)`。最高分字符胜出；同分时按 Unicode 码点升序裁决。

置信度把模板质量和关键点量化质量合成为 ppm 整数：

```text
score_ppm = round_half_up(best_score * 1_000_000 / score_scale)
quality_ppm = max(0, 1_000_000 - residual_ppm)
confidence = round_half_up(
    score_ppm * confidence_score_weight
    + quality_ppm * confidence_residual_weight,
    confidence_score_weight + confidence_residual_weight,
)
```

默认权重为 `4:1`。切换量化档位会改变 `score_scale`、量化残差、得分整数和置信度，不会只修改显示文本。

## 3. 定位与阅读顺序

滑窗任务排序键由文件名、阶段名、窗口列/行序号和附加坐标派生，因此线程完成顺序不会改变输出顺序。窗口指纹由图像内容指纹、阶段参数指纹、模板指纹和窗口坐标哈希得到；投影权重由该指纹的 SHA-256 字节派生。

连通域先按 `cc_merge_gap`、色调差和最大跨度决定是否合并。合并记录包含左右片段、边界框间隙、色调差、并集跨度和使用的规则。超大区域按竖直投影零间隙分裂，并记录父片段、部分序号和分裂规则。

阅读顺序先按观测栏号、`y`、`-x` 排序。相邻项发生异常时，同时收集可能冲突：

- `CROSS_COLUMN`：观测栏号回退，或与真值栏号不一致。
- `MISORDER`：真值阅读序号不递增，或同栏内 `y` 回退。
- `MISSING`：真值阅读序号不是上一项加一。

若同一相邻对命中多个类型，按配置中的优先级裁决。默认优先级为 `CROSS_COLUMN -> MISORDER -> MISSING`；原始类型集合和最终类型都会写入报告，不会静默丢弃。

## 4. 缓存键与精确失效

缓存键均为规范化 JSON 的 SHA-256：

```text
candidate_key = H(image, candidate_params, template_fingerprint,
                  template_version, quantization, window)
component_key = H(image, component_params, template_fingerprint,
                  template_version, quantization, candidate_key)
keypoint_key  = H(image, keypoint_params, template_fingerprint,
                  template_version, quantization, component_fingerprint,
                  annotation_coordinate_fingerprint, truth_id)
recognition_key = H(keypoint_key, template_fingerprint, template_version)
reading_key = H(image, reading_params, template_fingerprint,
                template_version, quantization, ordered_keypoint_keys,
                conflict_priority)
```

标注版本键由版本名和整版关键点内容指纹派生。替换标注版本时，只有坐标真正变化的 `keypoint_key` 改变；依赖它的识别、阅读条目和阅读顺序键随之失效，其余产物复用。失效记录包含产物类型、产物 ID、缓存键、`computed`/`invalidated`/`reused` 状态和原因。

## 5. 预算调度

每个识别任务的点数为：

```text
point_cost = character_block_pixel_count * template_variant_count
```

任务按文件名、阶段、栏序号、字符序号和块位置组成的稳定排序键处理。上一批待办任务在下一批具有最高优先级。当前批次无法容纳的任务写入 `pending`，带 `E_POINT_BUDGET_PENDING` 代码和点数；已识别结果在后续批次直接复用，不重复消耗预算。

默认全量预算消耗为 `37614`。以 `15000` 为上限时实测批次待办数为 `22 -> 8 -> 0`，合并后的逐字段核心结果与全量运行一致。

## 6. 阶段快照

`SnapshotStore.commit` 的协议为：

1. 在存储根目录创建临时目录。
2. 将 `report.json`、各阶段清单、待办清单和 `state.json` 写入临时目录。
3. 为每个文件计算 SHA-256 并写入 manifest。
4. manifest 记录全局版本、配置哈希、模板版本/指纹、命令游标、已消耗预算、待办数量和前序版本。
5. 完成后整体移动到 `versions/<global_version>`，再原子替换 `current.json` 指针。

`load_current` 先校验指针中的 manifest 哈希，再校验每个文件和报告。若当前版本损坏，则按版本目录倒序寻找第一个完整版本，返回 `recovered=True` 和原始问题列表。

## 7. 误差追踪与翻转归因

参考路径使用比当前配置高 1 个定点位的路径，并取消模板点数预算限制。对每个字符输出四个环节：

1. 关键点坐标量化：参考原始定点坐标与观测量化坐标之差。
2. 模板匹配得分舍入：把观测得分左移到参考定点尺度后的差值。
3. 置信度合成：ppm 差换算为参考定点小数后比较。
4. 阅读顺序裁决：阅读位次差乘以一个参考定点单位。

所有环节统一与参考尺度下的容差比较，记录绝对误差、相对误差 ppm、误差分支和首次超差环节。参考值为零时使用绝对误差分支，并显式标注 `truth_zero=true`。

两档位字符结论不一致时，归因顺序为：

1. 阅读位次变化：`reading_order_adjudication`。
2. 识别裁剪块边界变化：`keypoint_coordinate_quantization`。
3. 块边界相同但量化坐标/得分尺度变化：`template_match_score_rounding`。

翻转夹具的 `Q4` 与 `Q8` 识别裁剪块均为 `[148, 116, 172, 140]`，但量化坐标分别为 `[2560, 2032]` 与 `[40960, 32512]`，因此归因到匹配得分舍入，而不是块漂移或阅读重排。

## 8. 并发与确定性

- 滑窗和字符匹配使用有序线程池。
- 每个任务先有稳定排序键，线程池结果按提交时的有序键回填。
- 并发计数由锁保护，运行后断言从未超过配置的最大 worker 数。
- 缓存、待办、冲突和失败列表在输出前再次按稳定键排序。

验收中串行 1 worker 与并发 4 worker 的核心投影完全相同，观测最大并发为 4。

## 9. 异常注入

| 编号 | 注入 | 预期代码 |
| --- | --- | --- |
| `INJ-01` | 截断 PNG 数据块 | `E_SAMPLE_FORMAT` |
| `INJ-02` | 非整数关键点坐标 | `E_SAMPLE_FORMAT` |
| `INJ-03` | 模板变体维度与行宽不符 | `E_TEMPLATE_VARIANT_DIMENSION` |
| `INJ-04` | 量化档位和定点位宽越界 | `E_CONFIG_INVALID` |

每个注入在独立临时目录中执行。单个样本失败不会阻止其余样本载入；例如 `INJ-01` 和 `INJ-02` 命中 `page_a` 时，`page_b` 仍然有效。
