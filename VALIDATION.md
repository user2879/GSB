# 验证记录

验证日期：2026-11-21（Asia/Shanghai）。未使用 Docker、网络、模型或外部
图像数据。

## 环境检查

```text
Python 3.11.0rc1
numpy 1.26.4
PyYAML 6.0.2
pytest 8.3.3
```

## 自动化测试

```bash
python3 -m pytest -q
```

结果：`13 passed`（最终复跑耗时 8.25s）。覆盖：

1. 并行与串行完整报告 canonical JSON 完全相等；
2. 表格行列线、60 个单元格边界/跨度、25 个图表点像素/数值/归一化与真值一致；
3. 非法真值、坏 PNG、缺失真值分别进入稳定 `E_PAGE_INVALID` 失败清单；
4. 线检测重复运行一致，短线和超宽候选丢弃原因、合并来源可追溯；
5. 1500 点预算下第 3/4 页 pending，下一次调用按队列继续完成；
6. 图表阈值、网格面积参数、归一化方式的分阶段失效；
7. 单页替换只触发该页阶段，单元格与数据点裁切级复用；
8. 第一次快照后强制终止、无最终报告、从版本 1 恢复且业务结果一致；
9. 图表点配额和表格/图表重叠的表格优先裁决；
10. 配置并发度 3 时观察到实际并发峰值为 2–3，未超限；
11. CURRENT 指针哈希、整代清单和全部产物哈希校验，未发布临时代被忽略；
12. 配置与真值错误码单元验证。
13. 全缓存运行完全不枚举候选，已消耗检测点为 0。

## 默认样本运行

```bash
python3 -m parseapp run --samples samples \
  --config config/parse_config.json --state-dir .parse_state/demo
```

观察结果：

```json
{"counts":{"failed":0,"ok":4,"pages":4,"pending":0},
 "snapshot_version":3,"status":"completed"}
```

检测点成本/扣减序列：

```text
page_01.png 1289   remaining 3711
page_02.png   60   remaining 3651
page_03.png 1083   remaining 2568
page_04.png   87   remaining 2481
```

逐页真值对照：表格行线、列线、单元格数量、合并跨度、边界完全一致；
柱状图顶部和折线标记像素与真值完全一致，数值与 `minmax` 归一化坐标误差
小于 `1e-6`。

串行和并行在新状态目录上运行后，完整 `report.json` 的 canonical JSON
SHA-256 完全相同：

```text
canonical_equal True
sha256 970f28054d773c0eae6af3173b370b8142d101dcc8fa602fcddb071b6c773c33
```

## 低预算与续算

```bash
python3 -m parseapp run --samples samples \
  --config config/parse_config.budget_demo.json --state-dir .parse_state/budget
python3 -m parseapp run --samples samples \
  --config config/parse_config.budget_demo.json --state-dir .parse_state/budget
```

第一次：第 1/2 页完成；第 3/4 页 `E_POINT_BUDGET_PENDING`，队列顺序为
`page_03.png`、`page_04.png`。第二次：恢复该队列并完成全部页面，报告
`pending=0`。

## 强制终止与恢复

```bash
python3 -m parseapp run --samples samples \
  --config config/parse_config.json --state-dir .parse_state/crash \
  --fail-after-commits 1
python3 -m parseapp run --samples samples \
  --config config/parse_config.json --state-dir .parse_state/crash
```

第一次退出码 `75`，无 `report.json`。提交的代状态为 `active`，完成页面
`page_01.png`、`page_02.png`，清单仍保留完整四页预算计划。第二次恢复报告
`snapshot_version=1`，恢复页面为前两页，预算账本四条扣减序列与无中断运行
相同，最终全部页面完成。

## 缓存失效

- 仅把 `chart_decision_threshold` 从 `0.55` 改为 `0.60`：全部页面线、网格、
  单元格复用；图表点重算。
- 再把 `cell_min_area` 从 `100` 改为 `150`：线复用，网格重算，满足条件的
  单元格仍复用，图表点复用。
- 再把归一化改为 `pixel_relative`：线、网格复用，图表点重算。
- 在 `page_04.png` 的 `[400,540,407,547]` 增加无关暗点：前三页全部复用；
  `page_04` 三阶段重算，但 14 个单元格和 7 个数据点裁切级复用。

## Windows 原子替换检查

当前实际环境为 Linux/WSL2，运行：

```bash
python3 scripts/check_windows_atomic.py
```

结果：

```json
{"atomic_replace_supported": true, "error": null,
 "platform": "Linux", "temp_dir_created": true}
```

Windows 上应运行同一脚本确认。该结论不等同于已在 Windows 本机实际执行。
