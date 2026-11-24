# 实际验证记录

验证日期：2026-11-24（Asia/Shanghai）

## 环境

```text
Python 3.11.0rc1
numpy 1.26.4
Pillow 10.3.0
平台：本地 Linux/WSL 工作区（代码未使用容器）
```

依赖安装命令已由项目 `requirements.txt` 固定为本地 `numpy` + `pillow`。

## 自动化测试

命令：

```bash
python3 -m unittest discover -s tests -v
```

结果：

```text
Ran 9 tests in 63.931s
OK
```

覆盖内容：

- 固定种子样本两次生成的 PNG/JSON 字节一致。
- 合法样本加载与非法 PNG、非单调刻度真值的 `E_CHART_INVALID` 隔离和稳定排序。
- 线性、对数、断轴、遮挡标签、高密度刻度与柱/折线数据点恢复。
- 串行与 7 线程报告完整对象相等。
- 量化档位只失效数据阶段；对数阈值只失效映射及下游；最短刻度长度改变刻度检出集合。
- `ols`/`endpoints` 改变真实还原值；断轴阈值改变分段；线宽容差和遮挡容忍度改变业务结果。
- 替换图像只重算该页、新增页重算、删除页识别，未受影响页面复用。
- 低预算待办与单页数据点配额 `E_PAGE_QUOTA_EXCEEDED`。
- 编辑命令流的替换、新增、删除和配置修改均可落到本地文件，并能得到精确失效清单。
- 中途停止后的快照续跑与全量基线核心结果一致；无变化重跑 21 个阶段全部复用。

## 正式自检

命令：

```bash
python3 -m chart_workbench selfcheck --config config/calibration.json --output out/selfcheck
```

结果：

```text
exact_equal: true
result_digest_equal: true
pages_ok: 7 / 7
pages_pending: 0
data_points: 42
tick_comparison_points: 77
budget_consumed_total: 77
budget_batch_count: 4
verification_passed_pages: 7 / 7
result_digest: 37a016e457ff65c16fff107525382aa0c626c5894c1edef4499728c3a72c527f
```

三份报告完整文件 SHA-256 相同：

```text
5b98acd88b3be7f18e4e3760ebe7a2200f555db3ed8f4f102f493037d4d31b32  out/report.json
5b98acd88b3be7f18e4e3760ebe7a2200f555db3ed8f4f102f493037d4d31b32  out/selfcheck/serial/report.json
5b98acd88b3be7f18e4e3760ebe7a2200f555db3ed8f4f102f493037d4d31b32  out/selfcheck/parallel/report.json
```

严格比较结果见 `out/selfcheck/comparison.json`，`exact_equal`、`pages_equal`、`budget_equal`、`failures_equal` 均为 `true`。

## 逐页恢复指标

| 页面 | 映射 | 数据点 | 真值通过 | 最大绝对误差 | 最大相对误差 |
|---|---|---:|---|---:|---:|
| `page_01_linear_bars.png` | x=linear, y=linear | 6 | 是 | 0.1 | 0.008333 |
| `page_02_log_line.png` | x=linear, y=log10 | 6 | 是 | 4.4 | 0.006452 |
| `page_03_broken_line.png` | x=linear, y=broken_linear（2 段） | 4 | 是 | 0.3 | 0.010000 |
| `page_04_linear_line_occluded.png` | x=linear, y=linear，排除 1 个 y 标签 | 6 | 是 | 0.2 | 0.020000 |
| `page_05_dense_bars.png` | x=linear, y=linear，11 个 x 刻度 | 11 | 是 | 0.1 | 0.020000 |
| `page_06_linear_bar_occluded.png` | x=linear（排除 1 个标签）, y=linear | 4 | 是 | 0.1 | 0.005556 |
| `page_07_log_bars.png` | x=linear, y=log10 | 5 | 是 | 1.0 | 0.001429 |

## 强制终止恢复实验

构造 24 个独立文件名的本地页面副本，配置并发度 2、快照间隔 2 页。启动独立 CLI 进程；轮询到首个原子 `CURRENT` 快照出现后发送 `SIGKILL`。

观察结果：

```text
hard_kill: true
process returncode: -9
snapshot version: 1
snapshot completed_pages: kill_page_000.png, kill_page_001.png
snapshot pages_recorded: 24
```

随后在同一输出目录续跑：

```text
resumed pages_completed: 24
resumed pages_pending: 0
failures: 0
pages_equal_to_fresh: true
failures_equal_to_fresh: true
budget_equal_to_fresh: true
result_digest_equal_to_fresh: true
ticks_reused: 24
mapping_reused: 2
data_reused: 2
mapping_recomputed: 22
data_recomputed: 22
```

这验证了周期性原子快照能在进程被强杀后恢复，并复用已完成页和已提交刻度产物。

## 阈值/反解切换实验

对同一基线观察到：

- `quantization_step: 0.1 -> 0.5`：7 个页面的刻度与映射复用，7 个数据阶段重算，首柱值 `11.9 -> 12.0`。
- `log_axis_detection_threshold: 0.01 -> 1e-9`：7 个刻度复用、7 个映射重算；对数页映射从 `log10` 变为非对数类型。
- `tick_min_length: 6 -> 9`：7 个刻度阶段重算，页面输出 `E_TICKS_NOT_FOUND`。
- `line_width_tolerance: 2 -> 0`：刻度集合真实变化，7 页均无法通过宽度约束得到刻度。
- `broken_axis_detection_threshold: 0.35 -> 10`：断轴页映射从 `broken_linear` 变为 `linear`。
- `inverse_mode: ols -> endpoints`：数据点数组真实变化，而不仅是显示格式。
- 替换 `page_01`、新增 `page_08`、删除 `page_06`：仅替换页和新增页重算，其余 5 页全部复用；增删清单精确。
- 预算 10、并发 1/3/7：完整报告逐字段相同，待办页与预算扣减序列稳定。

## 常用验收命令

```bash
python3 -m chart_workbench validate-report --report out/report.json
python3 -m chart_workbench compare \
  --left out/selfcheck/serial/report.json \
  --right out/selfcheck/parallel/report.json \
  --strict
```
