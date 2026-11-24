# 设计说明

## 1. 阶段数据流

```text
activations.jsonl ──┐
weights_v1.txt ─────┼─> load_validation
weights_v2.jsonl ───┤        │
config/state ───────┘        v
                       version_conversion
                              │
             ┌────────────────┴────────────────┐
             v                                 v
          softmax                     layernorm_statistics
             │                                 │
             │                    parameter fingerprint
             │                                 v
             │                       layernorm_output
             └────────────────┬────────────────┘
                              v
                           summary
```

每阶段产物独立序列化。`summary` 引用各阶段行记录，不复制可变对象。

## 2. 统一中间表示

加载后的权重统一为 `WeightRow`：

```text
layer, row, scale, bias, scheme, precision,
quant_bits, source_version, raw, sidecar
```

v1 的 `raw` 保留 `scale_permille`、`bias_code` 与定宽行信息；v2 的 `raw` 保留精度标签。`sidecar` 只在发生损失转换时出现，并带自身哈希。

## 3. 指纹结构

所有键使用规范化 JSON 的 SHA-256：

```text
activation_fp = H(kind, layer, row, values)
parameter_fp  = H(weight source payload, conversion sidecar hash)
conversion_key = H(stage, source payload, direction, version, bits, level)
softmax_key    = H(activation_fp, version, bits, level, exp approximation)
stats_key      = H(activation_fp, version, bits, level, variance floor)
output_key     = H(stats key, stats content hash, parameter_fp, scheme,
                   bias enabled, version, bits, level)
```

记录内容哈希为：

```text
content_hash = H(result, dependencies)
```

读快照时会重新计算并比对，损坏或不完整的缓存不会被复用。

## 4. 精确失效矩阵

| 变化 | 转换 | Softmax | LN 统计 | LN 输出 |
|---|---|---|---|---|
| 单行激活 | 复用 | 目标行 | 目标行 | 目标行 |
| 单层参数 | 该层 | 复用 | 复用 | 该层 |
| 交换版本 | 全部 | 全部 | 全部 | 全部 |
| 定点小数位宽 | 全部 | 全部 | 全部 | 全部 |
| 量化档位 | 全部 | 全部 | 全部 | 全部 |
| 并发数 | 复用 | 复用 | 复用 | 复用 |

LayerNorm 统计量不使用可学习参数，因此单层参数替换不应触发统计量重算。

## 5. 调度模型

调度器先构造不可变计划，再交给线程池：

1. 按层名排序；
2. 层内按行号排序；
3. 先检查批次预算，再检查层配额；
4. 记录预算扣减前后的值；
5. 余额不足或配额越界时写入待办队列和失败记录；
6. 其余任务按相同顺序提交线程池。

线程完成顺序只影响内部完成队列，不影响结果映射。最终层、行、预算序列、待办队列和失败清单均从计划结构生成。

## 6. 快照原子性

快照版本目录在临时目录中完整构建。提交分为：

```text
tmp/<snapshot>  --atomic rename--> snapshots/rXXXX-sYYYY
tmp/manifest    --atomic replace--> manifest.json
```

目录名包含逻辑版本与快照序号，不复用或逐文件覆盖旧快照。清单最后写入，因此清单指向的快照必定完整。

提交前比较：

- 配置哈希；
- 完整状态哈希；
- 五阶段内容版本。

三者都相同才跳过写入。

## 7. 全量/增量清单一致性

增量运行自然可以从上一快照得到真实缓存命中集。空白全量没有输出缓存，因此实现会：

1. 仍计算全部产物；
2. 从当前状态的命令历史重建上一逻辑状态；
3. 计算上一状态各阶段的内容键集合；
4. 用最终键与上一键集合比较，生成同一份 `cache_plan`。

这样 `cache_plan` 描述的是“相对上一编辑状态的键稳定性”，而不是本次模式的物理命中。增量模式额外保证物理上未受影响行不调用算子。

## 8. 异常隔离

加载阶段按层/行捕获；转换阶段逐行捕获；执行阶段每个调度任务有局部失败列表。线程任务即使出现未预期异常，也会被调度器转为 `E_INTERNAL` 结果，因此其他行继续。

注入运行器为每项故障创建独立临时 runtime，避免前一项产生的错误状态影响后一项。

## 9. 确定性边界

以下行为被固定：

- 浮点输入到整数的舍入方向；
- 定点乘除的舍入方向；
- 指数范围归约与多项式阶数；
- 最大并列选择；
- 归一化余数放置；
- 方差下限替换；
- 统计量累加顺序；
- 层、行、预算、待办、失败的排序；
- JSON 规范化与 SHA-256。

运行时间、线程 ID 和实际并发峰值不进入业务产物与缓存键。

