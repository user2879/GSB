# 本地确定性跨模态对齐与检索管线

面向图文检索场景的离线数据处理与结果审计工具：完成图像/文本特征抽取、
跨模态对齐打分、Top-K 检索、v1 二进制 / v2 JSON 交换格式互操作、基于
内容指纹的增量失效、阶段化快照恢复与异常注入验证。仅依赖 Python 3.10+
标准库与 NumPy，全部数据本地生成，无网络、无容器。

## 环境准备

```bash
# Python 3.10+；NumPy 安装（无 pip 时先用 get-pip.py 引导）
python3.10 -m pip install --user -r requirements.txt
```

## 目录结构

```
config/
  alignment.json        跨模态对齐配置（分块/量化/维度/权重/阈值/并发/提交间隔）
  format_spec.json      v1 二进制布局与 v2 JSON 字段规范
  injections.json       六类异常注入配置
  injections_signal.json 可处理退出信号演示用延迟注入
xmodal/                 管线源码（模块职责见 docs_design.md）
xmodal/tests/           34 个 unittest 用例
scripts/demo.sh         端到端演示脚本
scripts/signal_demo.py  SIGTERM 优雅退出与恢复演示
artifacts/verification/ 一次完整验证的报告、快照与 VERIFICATION.json
```

## 快速开始

```bash
# 1. 生成 8 个固定种子图文样本（PNG + regions.json + text.json）
python3.10 -m xmodal.cli generate --input demo_output/input --count 8

# 2. 全量运行（默认 v2 JSON 特征包）
python3.10 -m xmodal.cli run --input demo_output/input \
  --workdir demo_output/work --config config/alignment.json \
  --spec config/format_spec.json

# 3. 异常注入 / v1 二进制 / 调整并发度
python3.10 -m xmodal.cli run --input demo_output/input \
  --workdir demo_output/work_inj --config config/alignment.json \
  --spec config/format_spec.json --injections config/injections.json \
  --format-version 1 --workers 8

# 4. 查看恢复点；中断后续跑同一 workdir 即可
python3.10 -m xmodal.cli recovery --workdir demo_output/work

# 5. 测试与一键演示
python3.10 -m unittest discover -s xmodal/tests -p "test_*.py"
DEMO_ROOT=demo_output bash scripts/demo.sh
```

## 关键设计

- **确定性**：sha256 内容指纹替代 Python 随机哈希；得分固定 9 位小数；
  报告、缓存键、快照版本均由输入与配置派生，可重复复现。
- **精确增量**：样本替换只失效该样本链路；权重变更只失效对齐/检索；
  分块尺寸变更失效抽取及下游；格式切换只失效转换；未失效产物直接复用。
- **一致性保证**：增量与空白全量的 `content_hash` 相同；workers=1/4/8
  结果逐字段一致；v1/v2 读回的图像块与文本向量逐元素相等，权重残差由
  float32 精度标签解释并写入 sidecar。
- **快照恢复**：阶段临时目录整体 `os.replace` 提交；SIGTERM 在样本边界
  完成原子提交后以退出码 0 退出；SIGKILL 后清理 `.tmp`，仅依据完整提交
  产物与清单恢复。
- **异常隔离**：单样本故障记录样本名、阶段、业务码、异常类型后继续其余
  样本；注入编号 INJ-001~006 写入报告证明命中。

## 业务码

| 业务码 | 触发场景 |
| --- | --- |
| `E_SAMPLE_INVALID` | PNG 头/尺寸/截断、JSON 字段缺失、偏移表越界、非有限 float32、未知版本 |
| `E_RETRIEVAL_LOW_SCORE` | 查询目标得分严格低于检索阈值，候选清单保留供审计 |
| `E_FORMAT_LOSSY` | 特征超出 float32 表达范围或非有限，记录具体特征键 |

## 实际验证结果

最近一次完整验证（Python 3.10.12 / NumPy 2.2.6，8 样本）：

- 34 个 unittest 全部通过；
- v2 与 v1 跨格式图像/文本最大绝对偏差 `0.0`；
- 串行与并发（workers=1/8）`content_hash` 完全相同；
- 格式切换仅 convert 阶段 8 个样本重算，其余阶段零重算；
- 六类注入全部命中并映射到预期业务码，5 个失败样本隔离且清单有序；
- SIGTERM 退出码 0、SIGKILL 重启恢复后业务结果与全量运行一致。

证据见 `artifacts/verification/VERIFICATION.json` 与各 `report.json`，
模块级细节见 `docs_design.md`。
