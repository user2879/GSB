# 技术设计

## 1. 模块边界

| 模块 | 职责 |
|---|---|
| `generator.py` | 固定种子合成 PNG、真值、v1/v2 关键点版本 |
| `dataset.py` / `config.py` / `validation.py` | 容器、真值、版本、配置和跨文件校验 |
| `sliding.py` | 滑窗特征、指纹投影、定点得分、NMS |
| `components.py` | 连通域标号、分裂、合并、过滤、候选关联 |
| `keypoints.py` | 扩展边界 + 墨迹重心回归、定点量化 |
| `reading.py` | 阅读串联、真值对照、冲突记录 |
| `adaptive.py` | 规模统计与策略表选择 |
| `cache.py` | 阶段缓存键、复用记录 |
| `scheduler.py` | 文件名/栏序号稳定排序与点数预算 |
| `snapshot.py` | 临时目录整体替换、`.previous` 恢复、续跑判定 |
| `engine.py` | 阶段编排、线程池、诊断产物 |
| `editing.py` | 可重放编辑命令流 |
| `cli.py` / `gui.py` | 无头 CLI 与可选 Tk 检查器 |

## 2. 确定性不变量

1. 所有列表顺序由稳定键决定，不依赖字典插入顺序、线程完成顺序或时间戳。
2. 所有浮点结果在输出前固定到 6 或 9 位小数；关键点坐标的唯一权威表示是定点整数 `x_q/y_q`。
3. 窗口 ID、连通域 ID、关键点 ID、缓存键都由内容和稳定顺序派生。
4. 线程池最多使用 `min(config.max_workers, task_count)`；结果按 `(文件名, 栏序号, 栏 ID)` 回收。
5. 快照中不含运行时间，因此同一最终数据集和配置生成相同 `global_version` 与相同诊断载荷。

## 3. 滑窗得分

每个窗口计算：

- `density_q`：墨迹像素占比，放大 `10^6` 后取整。
- `valley_q`：行投影平滑后局部谷值，转化为 `[0, 10^6]` 整数。
- `gradient_q`：水平/竖直一阶梯度均值的归一化整数。
- `projection_q`：`sha256(image_fp|x|y|window|step)` 派生的固定 `[-24, 24]` 偏置。

`score_q = round((4*density_q + 2*valley_q + 3*gradient_q)/10000 + projection_q)`。候选需同时满足密度下限和得分下限，再按得分降序做 IoU 0.5 的 NMS；输出阶段按窗口几何排序并重编 `cand*`。

## 4. 连通域规则

1. 对阈值以下像素做 8 邻域标号。
2. 若块高度超过中位高度 `split_ratio` 倍，在中间 50% 行内寻找最低水平投影；低于平均行墨迹 35% 时按该行分裂。
3. 任意两个碎片边界框的 Chebyshev 间隔 `<= merge_gap` 时合并。合并对按 `(间隔, 左 ID, 右 ID)` 排序后做并查集，保证传递合并稳定。
4. 每个成功 union 记录 `fragment_ids`、最终组 ID、水平/竖直间隔和阈值依据。
5. 面积小于 `min_area`、长宽比大于 `max_aspect`、或与任意粗候选无交集的组被过滤并写入 `filtered`。

## 5. 关键点定点域

设块边界向外扩展 `e=max(1, round(0.06*max(width,height)))`，令 `scale=2^fixed_bits`：

```text
regressed_x = 0.65 * ink_centroid_x + 0.35 * expanded_bbox_center_x
ideal_x_q   = round(regressed_x * scale)
x_q         = round(ideal_x_q / quant_step) * quant_step
residual_x_q = x_q - ideal_x_q
```

`y` 同理。切换 `fixed_bits` 会改变整数坐标尺度；切换 `quant_step` 会改变舍入格点与残差。

## 6. 阅读顺序与冲突

排序优先级固定为：

1. 栏从右到左（`column.order` 升序）。
2. 栏内从上到下。
3. 同位置时 `x` 从右到左。
4. 关键点稳定 ID。

检测类型：

- `cross_column`：栏不存在于真值，或相邻栏的真值阅读锚点不连续。
- `wrong_order`：后一关键点匹配到的真值阅读序号不大于前一关键点。
- `missing_number`：相邻真值阅读序号间隔大于 1，或栏内检测点数少于真值数。

冲突记录按 `keypoint_index` 升序，再按业务码稳定排序。任何检测点都保留在 sequence 中，不通过丢弃来消除冲突。

## 7. 缓存键与精确失效

阶段键只包含该阶段真实依赖：

- `sliding_key` = 图像内容指纹 + 窗口/步长 + 墨迹阈值 + 算法版本。
- `component_key` = 图像内容指纹 + 全部连通域参数 + 算法版本。
- `keypoint_column_key` = 图像指纹 + 滑窗键 + 连通域键 + 栏 + 引用组件 ID + 分配点数 + 定点/量化参数。
- `reading_key` = 图像指纹 + 各栏键 + 关键点全局序 + 已处理栏 + 定点/量化参数。

因此：

- 只改合并阈值：滑窗命中，连通域及其下游失效。
- 只改窗口/步长：连通域命中，滑窗、关键点、阅读失效。
- 只改量化：滑窗和连通域命中，关键点、阅读失效。
- 替换单图：其他图像的阶段键不变，继续命中。

## 8. 预算、快照与恢复

调度先按 `(文件名, 栏序号)` 规划预算。每栏可完整、部分或待办；部分栏只回归前 `allocated_points` 个稳定组件，待办部分记录 `E_POINT_BUDGET_PENDING`。

提交流程：

1. 在 `diagnostics/<run>/.staging-*` 写入全部 JSON，最后写 `manifest.json`。
2. 若已有 `current`，先改名为 `.previous`。
3. 将 staging 整体改名为 `current`，成功后删除 `.previous`。
4. 启动时若只有 `.previous`，恢复为 `current`；若两者都在，表明上次替换已成功但清理未完成，删除旧备份。

`decide_run_mode` 比较数据集哈希、配置哈希和有效参数哈希，返回 `rebuild`、`incremental_rebuild`、`incremental`、`resume`。

## 9. 为什么增量与全量的 hit/miss 状态不同

增量运行会真实读取上一次快照的 `cache.json`，因此产生 `hit`；从空白目录强制全量重建没有旧缓存，必然全部 `miss`。这不影响确定性诊断载荷：坐标、阅读序号、冲突、失败清单和缓存键序列一致。`RunResult.canonical_results(include_reuse=False)` 专门用于比较这一不变量，`include_reuse=True` 可用于检查实际复用状态。

## 10. 扩展点

- 真实检测器可替换 `run_sliding_stage` / `run_component_stage`，保留阶段键和输出 schema。
- 新档位只需扩展 `DEFAULT_STRATEGIES` 与容量字段。
- 新编辑命令应在 `editing.py` 中做参数校验，并只修改数据集/配置的规范来源。
- 更大图像可把像素列表从组件产物中移出，只保留缓存内的紧凑坐标编码；当前样本规模优先可审计性。
