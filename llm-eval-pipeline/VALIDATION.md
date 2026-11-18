# 验证记录

环境：Python 3.11、NumPy 2.x（与 1.26 的 ndarray API 兼容）、PyYAML，全程离线、
无外部网络依赖；样本由固定种子 LCG 从受控词表确定性生成。

## 单元 / 端到端测试

命令：`python3 -m unittest discover -s tests -v`

结果：`Ran 24 tests ... OK`（24 个全部通过）。覆盖：

- 平台无关稳定哈希、三档子词切分、特征与奖励打分逐字段确定性；
- `E_BAD_JSON` / `E_EMPTY_TEXT` / `E_DUP_ID` / `E_CONFIG_INVALID` /
  `E_TIER_LIMIT_ZERO` / `E_RULE_WEIGHT` / `E_THRESHOLD_BOUNDARY` 业务码；
- 三类幻觉规则与引用缺失检出、固定优先级同分决断；
- 策略选择器随规模切换、候选评分与决策记录可审计、同规模确定性；
- 流式与批量两条指标算法路径结果完全相等；
- 内容指纹缓存复用、下游失效与缓存键稳定性；
- 编辑序列增量运行与空白工作区全量重建逐字段一致；
- 并发度 1 与 8 的产物逐字段一致、偏好对与桶顺序稳定；
- 快照崩溃恢复与不中断运行逐字段一致、严格续跑血缘变更返回 `E_CONFIG_MISMATCH`；
- NaN 注入隔离，命中注入编号并记录 `E_NON_FINITE`，其余样本正常处理。

## 实测数据（samples/demo：600 偏好对、300 评测样本）

基线运行：

- 规模档 `medium`，选中策略 `S1_medium`，幻觉检出率 `0.2733333333333333`；
- 首次运行 901 个阶段产物全部新算（600 打分 + 300 幻觉 + 1 指标）。

精确增量失效（同档替换单条偏好对，规模/参数不变）：

- 第二次运行 `900 reuse / 1 recompute / 0 invalidated`，
  即仅被替换的 1 条偏好对重算，其余 900 个未失效产物直接复用，未静默重算。

编辑序列增量 vs 全量重建（替换偏好对 → 改规则权重 → 追加 1000 对触发切档）：

- 规模档 `medium -> large`，策略 `S1_medium -> S2_large`；
- 最终 1600 偏好对，增量结果与从空白工作区全量重建：
  `INCREMENTAL == CLEAN REBUILD: True`（规范化后逐字段一致）；
- 切档改变特征精度/桶数参数，相关缓存键整体失效，符合“参数指纹变化即失效”的设计。

并发一致性（concurrency 1 vs 8）：

- `serial == parallel: True`；偏好对顺序稳定且按整数键排序，共 600 条。

快照恢复（snapshot_interval=20，处理 37 步后注入崩溃）：

- 崩溃后磁盘保留 `snapshot_features.json`（游标快照，临时文件 + 原子替换）；
- 续跑 `snapshot_resume=True, resumed_stages=['features']`；
- 恢复后结果与不中断运行逐字段一致：`recovered == uninterrupted: True`；
- 改写配置后严格续跑返回业务码 `E_CONFIG_MISMATCH`。

异常注入隔离（启用注入项 1 `non_finite_feature`，目标 pair_000003）：

- 报告 `injections` 记录 `item_no=1`；失败清单含 `E_NON_FINITE`；
- 命中样本 `reward_diff=null, reward_diff_ref="nan"`，其余 599 对正常产出；
- 报告以临时文件 + 原子替换写入 `report.json`。

## 复现实验命令

```bash
python3 -m eval_pipeline generate --sample-dir samples/demo --preferences 600 --evaluations 300
python3 -m eval_pipeline run --sample-dir samples/demo --workdir work/demo
python3 -m eval_pipeline run --sample-dir samples/demo --workdir work/demo-inject \
    --injection samples/demo/injection.json
python3 -m eval_pipeline verify --normalized work/a/report.json work/b/report.json
python3 -m unittest discover -s tests -v
```

## 关键取舍说明

- 规模增大时策略降低特征精度、放大哈希桶并下调幻觉阈值修正量，以控制计算与碰撞；
  策略选择输出候选评分与理由，使精度/阈值敏感性/指标稳定性的取舍可审计、可复现。
- 缓存复用计数、续跑标记属于运行态审计信息，不参与核心结果的逐字段一致性比对；
  奖励差值、幻觉标记、指标与失败清单排序在任意增量路径下与全量重建严格一致。
