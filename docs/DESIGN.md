# 设计与不变量

## 1. 处理阶段

1. **加载校验**：扫描 PNG 文件名字典序；解析 PNG signature、IHDR/IEND、chunk 长度与 CRC；Pillow 只接受 `L/RGB/RGBA`，尺寸必须等于配置的固定值；同步读取 truth 并验证整数半开边框、范围、栏/阅读序号连续唯一和字符非空。
2. **栏检测**：按阈值二值化，统计每列墨迹行数，同时要求至少 3 行连续竖笔；以投影窗口闭塞小空隙，从左到右形成 run，再按右边优先排序。输出整数 `x0/x1`。
3. **字符候选**：栏内按行统计墨迹，投影窗口闭塞笔画内部空隙；低于最小高度的 run 丢弃并写 `E_SEGMENT_TOO_SHORT`。候选段指纹由图像内容指纹、栏内容指纹、边界、阶段参数指纹派生。
4. **边框与粘连分割**：对候选段做 2x2 二值开运算，去除 1 像素桥线和尘点，再按行/列密度裁剪留白并记录墨迹重心。高度超过上限时，用 DP 在投影谷选择切点；切点代价为 `split_penalty + projection[cut] * split_valley_weight`，谷值越低越优；累计代价并列时取更靠上的切点。子块 ID 形如 `<parent>/p0001`，并保留父段 ID/指纹。
5. **定点匹配**：字符块压缩为 24x24 灰度覆盖，三个内置变体先裁紧再归一化；`overlap` 使用 1 像素容差的对称 F1，`l1` 使用严格平均绝对误差。分数按位宽量化为整数，融合权重转为 1e6 定点整数后加权累加再除权重和。
6. **置信度**：`margin = top - second`，覆盖率来自最佳模板变体。二者以定点整数合成，再按 `scale // quantization_steps` 向下量化。输出原始定点、量化定点、数值、档位与保留位数。

## 2. 缓存键

| 阶段 | 主要依赖 |
| --- | --- |
| columns | 图像内容指纹、阈值、投影窗口、引擎版本 |
| split | columns 键、栏区间、投影窗口、高度约束、段内容 |
| border | split 键、段指纹、分割罚项/谷权重、边框参数 |
| match | border 键、段指纹、边框、匹配度量、权重、位宽、量化档位、模板版本、可用模板版本 |

匹配缓存键包含模板版本；栏/切分/边框不依赖模板内容，因此模板缺失只影响匹配。所有键和任务键均可稳定排序。

## 3. 快照协议

每次提交先写入 `state/.staging-*`，包含本代全部阶段 JSON、`results.json` 和 `manifest.json`；完成后一次性重命名到 `state/snapshots/gen-...`，再用临时文件 + `os.replace` 原子替换 `state/current.json`。清单保存：

- `generation` 与 `global_version = vNNNNNN-<content hash>`；
- 配置哈希、模板版本、图像集哈希、环境哈希；
- 各阶段 artifact key/path 映射和内容级阶段版本；
- 产物逻辑 ID、预算待办、账本、运行时版本；
- `snapshot_interval` 与批末整体提交范围。

启动时校验 current 指针。若指针损坏则扫描最晚的完整代并修复指针；报告 `E_SNAPSHOT_RECOVERED`。修改配置后重新计算环境哈希，报告 `config_changed/version_changed`，但阶段键未变的产物仍命中。

## 4. 预算调度

每个字符块成本为 `bbox_width * bbox_height * template_variant_count`。待办队列优先于新任务；待办内部和普通任务内部都按 `文件:栏:字符` 稳定键排序。预算不足时不扣分，字符结果标 `E_MA_BUDGET_PENDING`。每笔成功预留写入累计账本：

```json
{
  "sequence": 7,
  "batch": 2,
  "stable_key": "page_01.png:c0001:h0001:match",
  "points": 8379,
  "remaining_before": 1952576,
  "remaining_after": 1944197
}
```

同一冷启动/编辑序列重放得到完全相同的扣减序列；缓存命中的字符不会重复扣点。

## 5. 并发

栏检测任务键为 `<file>:columns`，匹配任务键为 `<file>:cNNNN:hNNNN:match`。线程池大小严格取配置值，任务数更少时取任务数；引擎记录观察到的峰值任务数，但结果聚合按稳定键排序，不依赖完成顺序。

## 6. 注入隔离

注入清单按目标文件应用，不修改输入文件：

- `truncate_png`：读取字节后在内存截断，CRC/IEND 校验失败；
- `truth_bbox_out_of_bounds`：truth 深拷贝后替换越界边框；
- `missing_template`：仅从目标图的可用模板集合移除指定字符；
- `nonfinite_fusion_weight`：仅目标图的有效配置写入 NaN，再由配置校验拒绝。

每个注入项记录 `applied`、`hit` 与实际观测错误码；其它图片继续处理。

## 7. 确定性边界

- 样本、样本 PNG 字节、真值、配置、注入清单和模板由固定种子/排名生成。
- 哈希一律使用 UTF-8 canonical JSON；整数运算不依赖浮点排序。
- 失败、图像、栏、字符、缓存事件、待办、注入项均显式排序。
- 快照临时目录名包含 UUID，但不进入任何清单或结果。
- PNG 编码依赖 Pillow 版本；本项目在固定依赖版本下验证字节级样本复现。
