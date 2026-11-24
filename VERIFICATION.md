# 实际验证记录

验证日期：2026-11-24  
环境：Linux 工作区 / Python 3.11.0rc1 / numpy 1.26.4  
样本种子：`20261124`  
组件版本：`mmwin 1.0.0`

## 样本与首轮结果

命令：

```bash
python3 tools/generate_samples.py --samples samples --seed 20261124
python3 -m mmwin run --samples samples \
  --config samples/window_config.json \
  --plan samples/injection_plan.json \
  --checkpoint .mmwin-checkpoint --output out/report.json --reset
```

观察结果：

- 图像事件 16、文本事件 34、传感器事件 40、计划到达 90。
- 窗口 14 个，全部关闭；无帧窗口 2 个：
  `w-0000004200-0000005400`、`w-0000004800-0000006000`。
- 最大已投递事件时间戳 `7860 ms`，允许迟到 `400 ms`，最终水位线 `7460 ms`。
- 迟到再结算记录 135 条，策略均为 `resettle`，均含旧/新快照与旧/新定点结论。
- 投递轨迹与 `injection_plan.json` 的 90 条记录逐条一致。
- 失败清单为空。

## 定点与空窗

- Q8 (`fractional_bits=8`) 与 Q12 (`fractional_bits=12`) 对样本全部 12 个非空视觉窗口产生不同的中间视觉整数；所有 14 个窗口的置信度也发生变化。
- 2 个视觉空窗在两种档位下都输出 `empty_visual=true` 和 12 维全零视觉向量。

## 检查点与预算

- 使用 `--max-batches 1` 只处理前 7 条到达，83 条进入待办并输出 83 个 `E_EVENT_BUDGET_PENDING`。
- `MANIFEST.json` 记录 `processed_arrival_count=7`、83 个待办到达序号、全局版本和全部 8 个阶段。
- 二次运行正确续跑至 90/90，业务字段与空白全量重放逐字段一致。
- 完成后再运行一次：`resumed=true`，视觉计算 0 次、融合计算 0 次，仍与全量结论一致。
- 修改配置中的允许迟到后，旧检查点输出 `E_CHECKPOINT_VERSION_MISMATCH` 并从到达 0 全量重放；结果等于空白全量重放。

## 增量编辑

自动化测试依次施加：

1. 反向替换乱序注入计划；
2. 替换一个文本断言的值与极性；
3. 从文本 JSONL 删除一个事件；
4. 将窗口由 1200/600 调整为 1600/400；
5. 将允许迟到由 400 调整为 900；
6. 将量化档位由 Q8 切换到 Q12。

每一步稳定后，增量运行与按相同最终事件集/配置的空白全量重放，在窗口结论、再结算、空窗标记、失败清单、投递轨迹、水位线、待办与内容失效清单上逐字段一致。

单文本事件替换后：

- 未引用该事件的窗口四个阶段全部为 `reused`。
- 引用该事件的两个最终窗口：视觉阶段仍复用，归属/融合/再结算按引用关系重算。
- 全运行 `visual_computations=0`，证明未命中窗口没有被重新解码或投影。

单文本事件删除后，未引用被删除事件的窗口也全部保持 `reused`；再结算键已去除全局快照序号依赖，不会因其他窗口快照编号移动而误失效。

## 并发一致性

对同一最终样本分别使用 `thread_pool_size=1` 和 `thread_pool_size=6`：

- `windows`、`resettlements`、`empty_windows`、`failures`、`delivery_order`、`watermark`、`pending`、`invalidation_manifest` 完全一致。
- 并发参数不进入语义配置指纹，因此投影、视觉整数和融合结论一致。
- 调度任务按键 `(window_start_ms, window_end_ms, phase, snapshot_id)` 单调提交；最终合并顺序稳定。

## 自动化测试

执行：

```bash
python3 -m unittest discover -s tests -v
```

结果：7 个测试场景全部通过。覆盖校验、边界水位线、乱序回放、迟到再结算、空窗、量化、溢出、预算待办、检查点续跑、检查点不匹配、重复到达、完整编辑序列、精确缓存失效及串并行一致性。
