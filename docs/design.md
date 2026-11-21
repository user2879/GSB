# 设计说明

## 1. 数据流

1. `loader.load_inputs` 扫描视图与相机目录，完成 PNG/相机校验、像素边界校验与同名实例隔离。
2. `solver.solve_point` 对单个关键点执行四阶段：
   - `load`
   - `triangulation`（含视差筛选）
   - `rejection`
   - `finalization`（定点重算与误差传播）
3. `pipeline` 用固定线程池执行，但以关键点字典序汇总、仲裁预算、生成快照与报告。
4. `snapshot` 写完整临时目录，比较内容版本后整体替换 `current/`。

## 2. 确定性来源

- 视图、观测、剔除候选、失败、预算账本、待算队列、报告点列表全部显式排序。
- float64 残差与三维点不做 UI 层截断；JSON 使用 Python 最短往返表示。
- 缓存键使用 canonical JSON SHA-256，输入浮点的完整二进制值参与指纹。
- 线程完成顺序不参与任何决策；预算在并发结果汇总后按标识字典序扣减。
- 快照版本由阶段载荷、缓存、输入版本、配置哈希、相机哈希、完成列表与消耗预算共同决定。

## 3. 三角化

对观测 `(u_i, v_i)` 和相机投影矩阵 `P_i`：

```text
A[2i]   = u_i P_i[2] - P_i[0]
A[2i+1] = v_i P_i[2] - P_i[1]
```

先逐行归一化，再做 `SVD(A)`。取最小奇异值对应右奇异向量为齐次解 `X_h`，`X = X_h[:3] / X_h[3]`。若数值秩小于 4、最小两个奇异值间隔小于阈值，或齐次尺度不可用，则输出 `E_TRIANG_DEGENERATE`。

重投影残差：

```text
x' = P [X, 1]
r = hypot(x'[0]/x'[2] - u, x'[1]/x'[2] - v)
```

全部使用 float64。

## 4. 视差支撑图

对初始三维点 `X` 与相机中心 `C_i`，定义单位视线 `d_i = normalize(X - C_i)`：

```text
theta_ij = acos(clamp(d_i · d_j, -1, 1))
```

- `theta_ij < min_disparity_deg` 的视图对进入 `excluded_view_pairs`。
- 节点只有至少一条合格边时才被保留。
- 保留视图数少于 `min_views` 时失败，并保留排除对、排除视图和角度。
- 保留集合与全部集合不同时，只在保留集合上重新三角化。

## 5. 剔除与预算

每轮只计算当前活动观测的残差：

```text
worst = argmax(r_view)  // 并列时 view_index 最小
if worst.r <= threshold: stop("threshold")
if removed_count >= max_rounds: stop("round_limit")
if remaining_count - 1 < min_inliers: fail(E_INLIER_SHORTAGE)
else: 剔除、重新三角化、求值下一轮残差
```

一次残差求值记一次预算。`ResidualCounter` 在求值前检查单点配额；`BudgetLedger` 按字典序检查全局余额。全局余额不足时，从当前点开始的所有剩余点进入待算队列。

## 6. 定点链路

- `quantize(v, bits) = rint(v * 2^bits)`，采用 round-half-even。
- 静态主档位把基础定稿点和残差序列转为整数。
- 低档、高档各自完整重跑：DLT -> 点量化 -> 重投影 -> 残差量化 -> 剔除判定 -> 重三角化。
- 比较基础链路与定点链路的三角化、每轮剔除、最终定稿状态。
- 末轮剔除不一致时：
  - 两序列长度相同且前缀一致，只有末项不同：`quantization_truncation`。
  - 一条链没有剔除而另一条有：`quantization_truncation`。
  - 更早轮次已不同或序列长度不同：`residual_accumulation_rounding`。

## 7. 缓存键

```text
load_key = H(stage=load, keypoint_id, raw_obs_fp, camera_fp, min_views, load_injections)
tri_key  = H(stage=triangulation, load_key, triangulation_cfg_fp, camera_fp, tri_injections)
rej_key  = H(stage=rejection, tri_key, rejection_cfg_fp, per_point_quota, tri_cost, rej_injections)
fin_key  = H(stage=finalization, rej_key, quantization_cfg_fp, per_point_quota, rej_cost)
```

`camera_fp` 包含相机参数和 PNG 内容哈希，但不含工作区路径。快照只持久化当前输入可达的缓存键，因此增量快照与全量快照的缓存内容可比较。

## 8. 快照替换

1. 构造阶段载荷、缓存与 base manifest。
2. 计算内容寻址 `global_version`。
3. 与 `current/manifest.json` 的版本比较；一致则 `unchanged`。
4. 不一致则写入 `.snapshot-tmp-*`。
5. `current` 先重命名为备份，临时目录重命名为 `current`，成功后删除备份。
6. 清理失败时恢复备份。

这避免逐文件覆盖导致的半提交状态。
