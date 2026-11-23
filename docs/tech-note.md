# 跨模态图文检索诊断工作台 · 技术说明

面向本地确定性样本的检索诊断工作台，解决数据规模连续变化时对齐算法取舍的
**可审计、可复现、可增量**问题。全部计算仅依赖 Python 标准库、NumPy 与 Pillow，
不访问网络或外部服务。

## 1. 环境与目录

- Python 3.10+（实测 3.11）；虚拟环境由 `scripts/bootstrap.sh` 离线创建。
- 依赖 wheel 固化在 `vendor/wheels/`：`numpy==1.26.4`、`Pillow==11.0.0`。
- 运行期目录（均在 YAML 中声明、可写）：`snapshot.temp_dir`、`snapshot.snapshot_dir`。

```
retbench/
  errors.py        错误码（E_SAMPLE_INVALID / E_CONFIG_INVALID /
                   E_INTERACT_BUDGET_EXCEEDED / E_SNAPSHOT_STALE）
  miniyaml.py      零依赖受限 YAML 解析器（环境含 PyYAML 时优先 PyYAML，输出一致）
  fixedpoint.py    定点量化（frac_bits 8/12/16 × tier coarse/standard/fine）
  fingerprint.py   BLAKE2b 规范化内容/参数指纹
  config.py        配置与命令流严格校验、预置策略表模型
  samples.py       PNG/区域真值/文本断言扫描与 E_SAMPLE_INVALID 失败清单
  generator.py     固定种子样本/配置/命令流/注入清单生成
  features.py      视觉分块特征 + 文本断言特征（确定性投影/补齐标注/量化记录）
  alignment.py     稳定归一化、定点对齐、三通道召回融合、同分裁决
  strategy.py      规模自适应策略选择器（评分、理由编码、真实计算路径切换）
  ledger.py        阶段 DAG 缓存键、精确失效传播、复用/重算与等价性校验
  concurrency.py   有界线程池 + 稳定排序键（样本名/分块行列号/阶段名）
  snapshot.py      临时目录写入 + os.replace 整体替换 + 清单版本判定
  engine.py        命令应用、阶段流水线、预算语义、快照与诊断报告
  runner.py        全量/增量执行与逐字段等价校验 CLI
  acceptance.py    端到端验收（异常、并发、预算、快照陈旧）
tests/             31 个 unittest 用例
```

## 2. 确定性设计

| 环节 | 确定性手段 |
| --- | --- |
| 样本生成 | 固定种子 `20261123` + 用途键派生独立 `Generator` 子流 |
| 投影矩阵 | 仅由 `(维度, 配置指纹, 用途键)` 派生，无全局随机态 |
| 累加顺序 | 分块按行列号升序、特征按维度升序，线程池按稳定键回排 |
| 归一化 | 减最大值 softmax，避免大数值溢出 |
| 量化 | `step = tier_scale * 2^-frac_bits`，`q=rint(x/step)`，记录步长/整数/残差 |
| 指纹 | 规范化 JSON（键排序、紧凑分隔）后 BLAKE2b，跨进程稳定 |

同一输入与配置下多次运行特征整数逐元素一致（见 `tests/test_features.py`）。

## 3. 业务要求到实现的映射

1. **加载校验**：`samples.scan_samples` 校验 PNG 容器头/尺寸/区域真值整数且落在图内/
   受控词表与断言字段/样本名唯一，非法项进入失败清单且不致命；`config.build_config`
   对分块整除、维度、权重有限性、候选上限、并发数、量化枚举与规模档单调性做致命校验。
2. **特征抽取**：`features.tile_raw_stats` 产出颜色均值/梯度/边缘密度，经固定投影到
   视觉维度；文本按实体/属性/谓词/对象 one-hot 投影，缺失断言用配置指纹派生默认向量
   补齐并在 `padded_positions` 标注；`fixedpoint.quantize_array` 记录步长/整数/残差。
3. **对齐与融合**：`alignment.align_pair` 按权重做定点乘加，词面/视觉/混合三通道由
   `fuse_candidate` 以通道权重与温度合成定点分；同分依次按对齐分、通道贡献和、样本名
   裁决；量化档位改变步长与整数，粗档位会制造同分从而改变排序。
4. **规模自适应**：`strategy.select_strategy` 依据样本数、图元总数、断言平均长度与
   单样本估计代价对预置策略评分，输出指标、逐策略评分与理由编码；`feature_precision`
   切换边缘统计、`tile_range` 切换中心半区、`channels` 切换融合通道、`cap_discount`
   切换候选上限，均真实改变后续计算路径。
5. **指纹账本**：`ledger.Ledger` 维护 visual/text/alignment/recall/diagnosis 五阶段
   DAG，缓存键由上游内容指纹、参数指纹与量化档位派生，并记录引用的分块与断言集合；
   失效沿 DAG 下游传播，未失效阶段直接复用。
6. **增量等价**：`runner.equivalence_check` 对比规范化后的全量报告与增量报告，
   忽略运行期元数据后逐字段比较 BLAKE2b；编辑序列覆盖规模收窄/扩张、单图/文本替换、
   权重调整、通道增删与量化切换。
7. **预算与并发**：`Engine.handle_command` 超预算时返回带
   `E_INTERACT_BUDGET_EXCEEDED` 的最近完成摘要；`BoundedPool` 硬限并发并记录峰值，
   `stable_task_key` 保证任意并发度下结果顺序一致（测试覆盖 workers=1/2/4/8）。
8. **快照与报告**：`SnapshotStore` 先写临时目录再 `os.replace` 整体替换，清单记录
   全局版本、配置哈希、命令游标与阶段版本；配置哈希漂移返回 `E_SNAPSHOT_STALE` 并
   清空转全量；报告含逐查询归因、边界样本、量化审计、失败项与账本清单。

## 4. 精确失效语义（实测）

| 编辑 | 重算阶段 | 复用阶段 |
| --- | --- | --- |
| 改对齐权重 | alignment → recall → diagnosis | visual、text |
| 增删召回通道 | recall → diagnosis | visual、text、alignment |
| 切量化档位 | visual、text 及其全部下游 | 无 |
| 切规模档位 | 受影响样本的 alignment 起；新样本补特征 | 未变化样本的 visual/text |
| 替换单图 | 该样本 visual 及其下游 | 该样本 text、其余样本全部阶段 |
| 替换单文本 | 该样本 text 及其下游 | 该样本 visual、其余样本全部阶段 |

## 5. 复现步骤

```bash
# 1) 离线环境
bash scripts/bootstrap.sh

# 2) 全量测试（31 用例）
.venv/bin/python -m unittest discover -s tests -v

# 3) 端到端验收（生成样本、全量/增量等价、异常与并发报告）
.venv/bin/python -m retbench.acceptance sample_accept --samples 12

# 4) 在任意工作区上运行 CLI（输出 reports/*.json）
.venv/bin/python -m retbench.runner sample_accept
```

## 6. 实际验证结果

- 单元测试：31/31 通过，开启 `-W error::ResourceWarning` 无资源泄漏。
- 增量等价：`equivalence.equivalent = true`，全量与增量规范化指纹相同。
- 并发一致：workers=1/2/4 报告指纹（剔除并发观测元数据）完全相同，峰值不超过配置。
- 预算语义：5ms 预算 + 每任务 20ms 模拟耗时稳定触发
  `E_INTERACT_BUDGET_EXCEEDED` 并返回过期摘要。
- 快照陈旧：配置哈希漂移稳定触发 `E_SNAPSHOT_STALE`，旧清单被清理并转入全量。
- 失败清单：固定注入 2 个故障样本（区域越界、实体不在词表），均以
  `E_SAMPLE_INVALID` 进入报告且不影响其余样本计算。

## 7. 关键设计取舍

- **全量与增量同构**：全量路径先重放命令流的最终参数/替换，再从空白账本跑同一
  流水线，避免两套实现漂移；等价性校验作为回归闸门。
- **隔离副本**：CLI 在 `_full_samples` 副本上跑全量，避免增量路径的文件物化污染
  原始样本，保证替换类命令的失效轨迹可审计。
- **定点优先**：所有对外得分均为 int64 定点整数，温度与权重同样量化，保证档位切换
  能真实改变整数与排序，而不是仅改变浮点显示。
