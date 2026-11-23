# 实际验证记录

## 环境

```text
CPython 3.11.0rc1
numpy 1.26.4
Pillow 10.3.0
pytest 8.2.0
```

已执行：

```text
python -m pip install numpy==1.26.4 pillow==10.3.0 pytest==8.2.0
python -m ancient_vlayout validate-runtime --json
```

校验返回 `ok: true`、`diagnostics: []`。应用不使用容器，样本生成与运行过程不访问网络。

## 自动化测试

```text
python -m pytest
19 passed in 70.27s
```

覆盖内容：

- PNG/truth 有效加载、截断 PNG、truth 越界、重复阅读序号、失败清单字典序；
- 16/16 栏区间、80/80 字符边框、80/80 识别字符；
- 段指纹逐字段稳定，4 个粘连块全部 DP 分割；
- DP 并列取上侧、不可行切点诊断、父块继承；
- 8/12/16 位切换改变定点整数、数值与保留位数，且只重算匹配；
- 匹配度量/高度/罚项/阈值/投影窗口/单图替换的阶段失效矩阵；
- 编辑序列终点与空白全量重建逐字段一致；
- 低预算待办、下一批优先、账本重放一致、最终全量等价；
- 四类注入全部命中且互不阻断；
- 快照指针恢复、阶段 artifact 损坏重建、配置变更版本递增；
- CLI `generate` 与 `run` 产物生成。

## 正常样本结果

样本：4 页、每页 4 栏、每栏 5 字符，共 80 字符；每页一个桥接粘连对。

```text
images: 4
characters: 80
recognized: 80
failures: 0
pending: 0
columns matches truth: 16/16
bbox matches truth: 80/80
```

实际产物：

- `artifacts/recognition.json`
- `artifacts/report.json`
- `artifacts/diagnostics.json`
- `artifacts/run_summary.json`
- `artifacts/snapshot_manifest.json`

## 故障注入结果

| 编号 | 类型 | 目标 | 观测错误 | 命中 |
| --- | --- | --- | --- | --- |
| INJ-001 | 截断 PNG | page_04.png | E_IMAGE_INVALID | 是 |
| INJ-002 | truth 越界 | page_03.png | E_TRUTH_INVALID | 是 |
| INJ-003 | 点阵模板缺失 | page_02.png | E_TEMPLATE_MISSING | 是 |
| INJ-004 | 非有限融合权重 | page_01.png | E_CONFIG_INVALID | 是 |

`page_02.png` 在缺失一个模板后仍完成其余字符识别；另外三页各自进入失败/诊断，不阻断其它目标。故障报告位于 `artifacts/injections/`。

## 预算验证

将 `ma_point_budget` 设为 `26000`：

```text
第一批：扣减 3 笔，77 个字符待办
第二批：首笔稳定键 = 第一批待办首项，累计扣减 6 笔
持续 28 批后：待办 0，账本共 80 笔
最终识别与充足预算的全量运行逐字段一致
相同编辑序列在独立目录重放：前两批账本完全相同
```

## 增量验证

实际观测的失效阶段：

| 编辑 | 重算阶段 | 命中阶段 |
| --- | --- | --- |
| 仅改匹配度量 | match | columns/split/border |
| 仅改高度约束 | split/border/match | columns |
| 仅改分割罚项 | border/match | columns/split |
| 改阈值或投影窗口 | columns/split/border/match | 无 |
| 替换单图 | 仅该图的 columns/split/border/match | 其余图片全部命中 |

最终编辑状态与空白全量重建的 `recognition.json` 相同（包括栏区间、边框、字符、置信度、失败与待办顺序）。

## 定点位宽示例

同一字符（`page_01` 首栏首字符）：

| bits | confidence_fixed | confidence_value | 保留位数 | quantum |
| --- | ---: | ---: | ---: | ---: |
| 8 | 14 | 0.0546875 | 3 | 2 |
| 12 | 200 | 0.048828125 | 4 | 40 |
| 16 | 3275 | 0.0499725341796875 | 5 | 655 |

这说明位宽切换真实进入匹配和置信度整数运算，而非只改格式化显示。
