# 架构与确定性规则

## 阶段

1. `load_validation`：扫描 PNG、验证签名/尺寸/模式、解析和校验配置、材质、命令。
2. `layer_noise`：按层生成坐标值噪声，叠加基础纹理，逐层定点量化并记录残差。
3. `material_mix`：把层场映射成四通道，按材质层权重和混合模式逐通道定点聚合。
4. `channel_synthesis`：按通道上下界裁剪，产出最终定点通道、贡献分解和边界样本。

快照清单为四个阶段各记录内容哈希和状态，并把所有缓存数组写入同一暂存目录。

## 噪声与量化

每层噪声使用 SplitMix64 派生的二维值噪声。种子材料包括：

```text
config.seed, 层序号, 基础纹理内容键, frequency, amplitude, resolution
```

坐标网格由 `x * frequency`、`y * frequency` 构造，四角哈希后平滑插值，不使用 `random`、`numpy.random` 或任何进程全局随机状态。

逐层合成顺序固定为层序号升序。第 `i` 层：

```text
signal_i = clip(texture_i + amplitude_i * noise_i, 0, 1)
composed_i = signal_i                                      (i == 0)
composed_i = (prior_quantized * i + signal_i) / (i + 1)    (i > 0)
raw_i = round_to_fixed(composed_i, fraction_bits)
quant_i = round_to_multiple(raw_i, quantization_level)
residual_i = raw_i - quant_i
```

`quantization_level` 是按最低位整数计的量化步长，因此调整量化档位会改变中间整数与后续输入，而不是仅改变显示。

## 定点混合

权重先按 `weight_fraction_bits` 转为整数。通道映射固定为：

- `diffuse = field`
- `roughness = scale - field`
- `metallic = field^2 / scale`
- `normal_perturbation = 2 * field - scale`

`weighted` 模式计算整数加权和后除以总权重；`multiply`、`screen`、`overlay` 按材质层序号稳定顺序做定点非线性组合。每个通道使用配置中自己的舍入模式。等权最高层按最小层序号决断；恰好半值舍入进入边界样本清单。

## 缓存键与失效

```text
texture_key = H(content_hash, mode, width, height)
layer_key   = H(texture_key, 层参数, 定点参数, 量化档位, 上游 layer_key)
channel_key = H(material_id, channel, 正权重层集合, 对应 layer_key,
                mix_mode, 位宽/小数位/量化档位, 通道舍入)
final_key   = H(全部 channel_key, 输出通道, 分辨率, 定点参数)
```

替换纹理只改变引用它的层及其下游；调整层频率改变该层及下游；调整一个通道权重只改变该通道与最终合成。复用账本逐项输出 `key`、上游引用、引用层集合、内容哈希和 `reused/recomputed` 状态。

## 预算调度

每个工作项携带 `(stage, layer_index, row_start, row_end, col_start, col_end)` 稳定键。主线程按稳定键从待办列表扣除预算并提交线程池；工作项完成顺序不影响预算扣减序列。预算不足时未提交部分留在待办，并记录 `E_POINT_BUDGET_PENDING`；完整构建会继续下一批直到待办清空。

## 交互语义

命令先完整应用到内存状态，再以真实状态提交后台构建。主线程最多等待 `latency_budget_ms`：

- 新修订已完成：返回最新摘要、完成层数、进度、最终通道摘要、受影响层/通道。
- 未完成：返回最近已完成修订的摘要，标记 `E_RESPONSE_STALE`，命令仍留在队列中，最终排空队列。

预览只使用真实构建结果，不会跳过量化、降低分辨率或使用近似通道。

## 快照与恢复

提交过程：

1. 计算全量状态哈希；与当前清单相同则跳过。
2. 在 `.staging-*` 写入 `cache.npz`、`cache_entries.json`、`ledger.json`、`manifest.json` 并 `fsync`。
3. 如存在 `current`，先改名为 `.previous-*`。
4. 暂存目录整体改名为 `current`，再删除备份。
5. 启动时若 `current` 不完整，从最新完整 `.previous-*` 回滚；残留暂存目录被隔离清理。

恢复要求输入配置哈希、命令流哈希和可选全局版本一致；不一致返回 `E_SNAPSHOT_VERSION` 并从空白缓存重跑。
