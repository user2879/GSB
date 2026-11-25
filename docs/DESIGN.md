# 技术设计与复核边界

## 分层

```text
models/validation      输入模型与 E_INPUT_INVALID 隔离
fixedpoint             确定性整数定点算术
fingerprint            内容/参数/阶段缓存键
stages                 四阶段纯函数
engine                 命令事务、线程池、失效规划、预算
diagnostics            参考域对比与翻转归因
snapshot               临时目录、整体发布、CURRENT、恢复
cli/gui                可重复演示与只读诊断面板
```

业务计算均为纯函数；调度层只决定何时计算，不决定累加顺序。召回证据按 `evidence_id` 升序；置信度贡献按相同顺序加入；边界任务按 `(left_turn_id,right_turn_id)` 排序。

## 互相依赖的边界

去重结果提供每轮证据分布，话题边界使用该分布与查询相似度的组合分数。话题建立后，证据归属得到跨话题集合，再进入置信度的跨话题惩罚。因此：

- 改查询/通道声明/通道权重 -> 召回 -> 去重 -> 边界 -> 话题 -> 置信度；
- 改证据文本/权重 -> 查询轮次的内容指纹与归属 -> 话题/置信度；
- 改话题阈值 -> 边界 -> 话题归属 -> 置信度；
- 改定点域 -> 所有定点阶段及权威选择。

话题 ID 刻意不使用通道权重或定点值，只使用起始轮查询、通道声明和去重证据内容分布，因此相同逻辑话题在不同定点域保持同一 ID；误差报告中的话题差异只表示真实归属变化。

## 缓存键和失效清单

缓存键输入只包含实际依赖。示例：

```text
recall = H(turn_content_fp, query_fp, declared_channel_weights, domain)
dedupe = H(recall_payload, domain)
boundary = H(left_query_fp, right_query_fp,
             left_dedupe_distribution, right_dedupe_distribution,
             threshold_bp, domain)
confidence = H(turn_content_fp, dedupe_payload, topic_assignment,
               cited_evidence_content/weight, attribution,
               decay/penalty/answer_threshold, domain)
```

`full` 模式可以强制重算所有已调度任务，但逻辑失效/复用清单仍由前后缓存键比较生成，所以可与 `incremental` 的模式无关输出对照。真正的执行命中/未命中只写在 `execution` 元数据，不进入规范化业务结果。

## 预算和恢复边界

预算只限制置信度合成阶段的证据点数，不跳过召回/去重/话题归属。待办按轮次升序保留。若单轮成本大于整批预算，则该轮持续待办，直到预算提高；不会产生半合成置信度。

快照恢复只信任 `CURRENT` 指向的完整版本。临时目录、已发布但指针未切换的目录都不构成提交；`CURRENT` 缺失或损坏时直接全量重算，绝不在未提交目录中猜测“最新版本”。恢复时同时核对：

- 基准配置哈希；
- 基准证据库指纹；
- 已执行命令前缀哈希；
- 快照命令游标；
- 清单自身哈希和所有阶段文件哈希。

任何不一致都走全量重算，而不是拼接不同版本的阶段文件。

## 定点公式精度说明

定点域使用 Python 整数存储单位值，所有乘除通过 `Decimal` 高精度中间值后执行 `quantization` 步长量化。舍入模式：

- `half_up`：远离零的半值进位；
- `half_even`：半值取偶；
- `down`：向零截断。

参考域仅替换小数位宽/量化/舍入，不改变检索、话题或惩罚语义，因此误差归因比较的是同一逻辑计算在不同定点口径下的传播。

## 错误隔离

初始化输入错误直接输出 `E_INPUT_INVALID` 并落盘失败清单。命令错误在事务副本上校验，失败时恢复前一配置/样本状态，只追加失败记录并推进命令游标。下一条合法命令仍可继续处理，失败清单顺序按命令序号稳定保留。
