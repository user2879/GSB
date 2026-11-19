# 响度与动态范围并行测量工作台（loudbench）

面向广播播控的 5.1 响度/真峰值并行测量工作台。本地确定性合成 48kHz/24bit/6 声道
素材，按时间窗切分测量任务，在受限线程数与内存预算下并行执行 K 加权、400ms
两级门控响度统计与 4x 过采样真峰值检测，并按时间轴归并出整段报告。

## 环境准备

- Python 3.10+（本机在 Python 3.10.12 验证；系统无 pip 时用 get-pip 引导）
- 安装依赖：`python -m pip install -r requirements.txt`（numpy、PyQt5）
- 工作区可写，报告/哈希清单/持久化目录默认写入 `workspace/`

## 快速开始

```bash
python run_demo.py --workspace workspace/demo --verify-twice
python run_tests.py
python run_gui.py workspace/demo/runs/S1_unrestricted_4w.json
```

Windows 双击或执行 `run_demo.bat`、`run_gui.bat`。无显示环境时用
`QT_QPA_PLATFORM=offscreen` 做界面离屏自检。

## 演示固定序列（run_demo）

- S1 无预算限制 4 线程；S2 单线程串行；S3 紧张内存（LRU 淘汰 + RECOMPUTE）；
  S4 极限内存（E_BUDGET_DENIED + UNCACHED 重算）；S5 单任务耗时预算触发串行降级。
- 异常注入覆盖 E_AUDIO_FORMAT、E_WINDOW_INVALID、E_BLOCK_BOUNDS、E_THREAD_RANGE、
  E_MERGE_ORDER、E_PEAK_OVER，每个错误带 location/evidence 可复算。
- 产物：合成 WAV、真值标签、任务清单、整段参考、5 份运行报告、demo 索引、
  SHA256 哈希清单。

## 关键设计

- 积分块锚定整段起点的全局 100ms 网格，块末端归属唯一，窗口任意合法切分下整段
  门控响度与一次算完一致。
- K 加权用 0.5s 冲激响应 + 分块 FFT 卷积，任务以固定 warm-up 重建滤波状态。
- 驻留决策先做时间轴顺序的规范 LRU 计划，运行时只执行计划；线程数变化不改变任何
  测量结果与账本，两次运行逐字节一致。
- 详细原理、错误码、容差与验证记录见 `loudbench/DESIGN.md`。

## 验收结论

本地合成 5.1 样本上：并行与串行结果逐字段一致；预算降级不缺失任何窗口；两次相同
输入与预算的运行输出与哈希逐字节一致。测试命令 `python run_tests.py`（10 项全过）。
