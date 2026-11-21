# 架构与确定性设计

## 1. 四阶段流水线

`loading -> quantization -> calibration -> deployment` 四个阶段各自：

1. 在 `stages/<phase>/.v*-*.tmp/` 写入 `payload.json`；
2. 写入包含 payload 哈希的 `COMMITTED` 标记并 `fsync`；
3. 将临时目录整体 `os.replace` 到唯一不可变目录；
4. 最后原子发布 `snapshots/vNNNNNNNN.json` 清单。

清单只引用已完整替换的阶段目录。恢复时从最高版本倒序验证 `COMMITTED`、payload 哈希和阶段路径，跳过任何损坏或未完成清单。清单版本未变化时直接返回已有版本，不逐文件覆盖。

## 2. 指纹与缓存

缓存键分为四层：

- 量化参数键：层内容指纹 + 有效量化设置；
- 反量化键：量化参数键 + 层内容指纹；
- 校准键：当前层校准样本内容指纹 + 所有上游层样本指纹 + 上游反量化键 + 蒸馏设置；
- 部署清单键：归一化配置哈希 + 校准集哈希 + 所有层量化/校准摘要。

因此：

- 修改某层位宽/分组/方案/权重：仅该层量化与反量化失效；
- 校准键包含上游依赖：该层及其字典序下游校准失效；
- 定向校准样本只影响其 `layers` 覆盖的层及下游；
- 任意校准样本或蒸馏设置变化会重建部署清单；
- 未失效层直接加载 `cache/quant/*.json` 与 `cache/calibration/*.json`。

## 3. 量化规则

### 对称

`qmax = 2^(bits-1)-1`，组缩放为 `max(abs(x)) / qmax`，零点为 0。

### 非对称

`qmax = 2^bits-1`，`scale=(max-min)/qmax`，`zero_point=round(-min/scale)` 并裁剪到 `[0,qmax]`。

缩放因子为 `float64`，然后按有效位向零截断。截断后为零的组使用整层回退缩放，并在 `degeneration_sources` 记录 `source=whole_layer`。整层全零时固定回退为 `1/qmax`。

量化取整固定为 nearest-ties-to-even。误差统计按原数组 C 顺序逐元素累计，输出最大绝对误差、MAE、SSE、MSE 和元素数。

## 4. 校准与蒸馏

对每个样本：

1. 以固定顺序点积计算原始输出和反量化输出；
2. 对原始输出按温度做数值稳定 softmax；
3. 每个元素校正为 `q + gain * softmax_i * (original_i - q_i)`；
4. 输出校正前、校正后和校正量统计。

`soft_label_weight=0` 时不跳过样本，而是记录 `correction_disabled=true` 和 `reason=soft_label_weight_zero`。样本集合为空时也返回零误差摘要和 `no_applicable_samples` 标记。

## 5. 预算调度

调度器先按 `(阶段序号, 层名)` 稳定排序，在启动任何线程前按串行顺序预留点数。单批预算不足时阻塞队首，并把剩余任务放入待算队列；下一批重置预算后继续。线程池大小严格等于配置并发数，结果按稳定键重新排序，因此并发度、完成顺序和批次切换不会改变：

- 层结果顺序；
- 误差表顺序；
- 复用/失效清单顺序；
- 预算扣减序列。

任务点数：量化阶段为权重元素数；校准阶段为权重元素数乘适用样本数。

## 6. 交互语义

`CompressionSession.submit_command` 从请求触达时刻开始计时。命令在私有会话副本上执行：

- 预算内有终态：返回 `frame_type=final`；
- 预算内无终态：返回 `frame_type=interim`、`E_INTERACT_BUDGET_PENDING`、最近一次完成摘要、受影响层和进度；
- 中间帧 `persisted=false`，不会更新 `session/current.json`；
- 完成后 `wait_for_final` 返回最终帧，只有终态才保存会话。

## 7. 信号与恢复

`SIGINT`/`SIGTERM` 仅设置协作退出标记。引擎在阶段边界检查标记，完成并提交当前阶段后以 `interrupted` 状态返回，CLI 退出码为 0。强制终止无法执行退出报告，但已发布阶段和清单仍是原子的；下一次 `status` 或运行只读取最后一个完整清单。

## 8. 一致性的关键约束

- 输入归一化后配置哈希稳定；
- 缓存中的浮点值通过 JSON 最短往返表示保存，重新加载与内存值相同；
- 完整重建忽略缓存，但仍沿同一确定性代码路径计算；
- 核心比较字段仅含层结果、校准结果、部署清单、错误报告和失败清单，不包含运行时长、复用计数或快照版本。
