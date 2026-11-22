# 技术说明（DESIGN）

## 1. 定点量化模型

- 约定：定点整数 `q` 表示实值 `q * 2^-f`，`f` 为小数位宽（frac_bits）。
- 量化档位 `gear ∈ 正整数`：真实量化步长 `step = gear * 2^-f`。档位越大，
  步长越粗，分辨率越低。
- 量化：`q = round_half_up(x / step)`，采用**远离零方向的半值舍入**
  （负数对称，见 `fixedpoint.round_half_up`）。
- 残差：`residual = x - q*step`，满足 `|residual| <= step/2`；
  方向 `direction ∈ {up, down, exact}`，由残差符号决定。
- 全部运算使用 Python 任意精度 `int`，不依赖 NumPy 计算路径，保证跨平台、
  跨次运行逐位确定。

切换档位时步长改变，整数结果或残差必然改变（测试
`test_gear_change_alters_integer_or_residual` 守护）。

## 2. 处理阶段与运算顺序

对每个请求、每个 `(frac_bits, gear)` 组合：

1. **归一化 normalize**：每通道独立，minmax 或 zscore。
   - minmax 分母 `max-min=0`、zscore 标准差为 0 时，按固定规则全部置 0，
     并写 `zero_denominator` 事件；通道缺候选写 `missing_channel` 事件。
   - zscore 结果裁剪到 `[-1,1]`，保证定点输入有界。
2. **定点化**：归一化值、通道权重、温度、特征、特征权重均按当前档位量化。
3. **通道融合 channel_fusion**：通道名**字典序**固定顺序，
   `raw_product = norm_q * weight_q`，单通道贡献 `= raw_product >> f`
   （half-up），求和后再乘温度定点值并 `>> f`，得到融合分 `fusion_q`。
   保留每通道 `raw_product` 与 `contribution_q`，实现逐通道可分解。
4. **特征叠加 feature_overlay**：逐特征 `feat_q * weight_q >> f` 后按
   特征索引顺序求和，`overlay_q = fusion_q + feature_sum`。
5. **裁剪 clip**：按配置 `score_clip` 换算的定点上下界裁剪，
   记录 `clipped` 标记。
6. **排序决断**：裁剪分 → 融合分 → 特征贡献和 → 物品 ID 字典序，
   全部在确定的整数/字符串比较下完成。

## 3. 跨档位审计账本与公共单位

不同小数位宽的整数尺度不同（如 f=4 的 16 与 f=8 的 256 可能表示同一实值），
直接相减会把尺度差异误判为量化误差。因此：

- 取候选集合中最大位宽 `F`，公共单位 `U = 2^-F`；
- 任意阶段值统一换算：`q_F = q_f << (F-f)`，再逐环节比较；
- 账本 `per_item[item].settings[]` 记录每个档位下四阶段定点中间值、
  各通道/各特征定点贡献、舍入方向与名次；
- `setting_pair_audit` 对任意两组档位，给出每候选名次变化、
  **首个分歧环节** `first_diverged_stage` 与该环节在公共单位下的定点差值
  `stage_diff_common_unit`，可直接回答“变化发生在归一化/通道合成/特征叠加/
  裁剪的哪一环、差值是多少”。

名次比较也在公共单位下以同一排序键重算（`_common_unit_rank`），
使跨位宽比较只反映真实量化差异而非尺度本身。

## 4. 翻转归因与边界对

- 翻转 `flips`：记录翻转前后名次、对端候选（另一档位下占据原名次的物品）、
  融合分前后值、**各通道贡献差（公共单位）**、以及是否由并列决断触发
  （裁剪分与融合分在两档位公共单位下均相等时为 true）。
- 边界对 `boundary_pairs`：在每个 `(f, gear)` 下，融合分实值差
  `diff_real <= step/2`（量化半步）的候选对，附：
  - 当前档位融合差（定点与实值）、半步阈值；
  - **最高位宽最细档位**下的参考差值 `highest_frac_ref_diff_*`；
  - 相对最细档位是否发生名次变化及判定结论
    `flip / stable_within_half_step`。
- 档位收紧（gear 增大 → 半步变大 → 更多对落入阈值），边界对数量单调不减，
  样本中至少一个位宽严格增加（测试守护单调性与严格增长）。

## 5. 重算点预算与待办保留

- 工作单元为 `(request_id, item_id)`，按请求 ID、物品 ID 字典序稳定排序。
- 每单元消耗 1 个重算点；单批预算不足时，剩余单元原样保留在待办队列，
  并产出 `E_RECOMPUTE_BUDGET_PENDING` 失败项；下一批从同一队列队首继续，
  即待办优先。
- `BatchLedger` 记录每批预算、实际消耗点数、已处理键、保留项、来源
  （fresh/recomputed）与实际并发度。
- 最终账本是纯函数产物，与分批方式无关：小预算多批续跑与一次性足额预算
  的 ledger/flips/boundary 逐字段相同（集成测试守护）。

## 6. 并发确定性

- 线程池大小取自配置 `concurrency`，`concurrency_used = min(配置值, 任务数)`，
  绝不超过配置。
- 每个任务携带由请求 ID、物品 ID、阶段名派生的稳定排序键
  （`fixedpoint.stable_key`），future 结果按稳定键回收排序，
  消除线程完成顺序的不确定性。串行（concurrency=1）与并发产物逐字段相同。

## 7. 增量一致性

- 以 `(request_id, item_id)` 输入指纹（含原始得分、特征、名次、分片与配置哈希）
  命中缓存即复用，未命中才重算。
- 最终产物只由确定性输入决定，与到达顺序无关。编辑序列（分片追加、单请求候选
  替换、通道权重调整、位宽/档位增删）后的增量运行，与空白目录全量重建
  canonical JSON 逐字节相同（集成测试守护，同时断言编辑确实改变了产物，
  排除“假一致”）。

## 8. 阶段化快照与恢复判定

- 临时目录 `snapshot_tmp/v<N>/` 下按阶段（ledger/flips/boundary/report）
  分目录写入，每阶段产出 sha256 版本；全部写齐后**整体移动**到
  `snapshot_commit/versions/v<N>/`，再用 `os.replace` 原子切换 `current`
  符号链接——不逐文件覆盖已提交产物。
- 清单 manifest 记录：全局版本号、配置哈希、已完成请求列表、已消耗重算点、
  各阶段产物版本哈希。每累计 `snapshot_commit_interval_batches` 个批次
  （以及 drain 的最后一批）触发一次整体提交，因此单次运行可产生 v1、v2…
  多个阶段版本，`current` 始终原子指向最新版本；最终报告以
  `intermediate_manifests` 列出中间版本号。
- 启动校验：配置哈希不符或全局版本号非法 → 输出 `E_SNAPSHOT_STALE`，
  旧版本目录移动到 `stale/` 隔离、移除 current 链接，随后全量重建。
- 同配置连续运行版本号递增（v1、v2 ……），历史版本目录并存。

## 9. 错误处理

所有加载/运行错误落入失败清单而非中断：
`E_RECORD_INVALID`（候选记录）、`E_CONFIG_INVALID`（配置）、
`E_RECOMPUTE_BUDGET_PENDING`（预算待办）、`E_SNAPSHOT_STALE`（快照失效）。
