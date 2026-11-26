# 古籍竖排版面定点流水线（GSB）

这是一个从空目录落地、可本地重复演示的 Python 包。它面向确定性合成的竖排古籍版面 PNG，完成滑窗定位、连通域细化、关键点回归、多模板识别、阅读顺序串联、增量缓存、预算调度、快照提交和定点误差归因。

整个项目只读取本地文件，不访问外部网络，也不依赖 Docker 或其他容器环境。

## 交付内容

- `gsb/`：可安装的 Python 包与 `gsb` 命令行入口。
- `samples/`：确定性 PNG、manifest、真值关键点、标注员关键点版本和翻转夹具。
- `configs/`：默认配置、量化翻转配置、模板路径注册表和 12 条编辑命令流。
- `fonts/`：内置 16 字点阵模板 `bitmap-v1` 与翻转夹具模板 `bitmap-flip-v1`。
- `injections/`：带编号的四类异常注入配置。
- `tests/`：按业务要求覆盖校验、定位、阅读、识别、量化、增量、并发、预算、快照和注入。
- `scripts/verify.py`：一次运行完整验收并写出 `artifacts/validation.json`。
- `docs/TECHNICAL.md`：定点公式、缓存依赖图、调度和快照协议说明。

## 当前环境

本机实际环境为 Python `3.11.0rc1`，系统已提供 NumPy `1.26.4`、Pillow `10.3.0`、PyYAML `6.0.1` 和 pytest `8.2.0`。

标准 Ubuntu/WSL 环境推荐：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

本机的 Python 缺少 `ensurepip`，因此实际创建虚拟环境时使用了系统站点包兼容方式；依赖安装与 editable 安装均通过本地 `--no-index` 完成，没有联网：

```bash
python3 -m venv --without-pip --system-site-packages .venv
. .venv/bin/activate
python -m pip install --no-index -r requirements.txt
python -m pip install --no-index --no-build-isolation -e .
```

Tkinter 状态：`python3 -c "import tkinter"` 当前失败，原因是系统缺少 `libtk8.6.so`。本项目运行时不依赖 Tk；若需要桌面调试，可在有权限的 WSL 中安装 `python3-tk` 与 Tk 运行库后再次验证。

## 快速开始

```bash
. .venv/bin/activate

# 校验配置、模板、数据集和所有 PNG
gsb validate

# 全量流水线
gsb run --out artifacts/report.json

# 12 条编辑命令的增量执行
gsb commands

# 高一位宽参考路径与四阶段误差追踪
gsb trace

# 四类编号异常注入
gsb inject

# 提交版本化阶段快照
gsb snapshot

# 完整验收并生成 artifacts/validation.json
gsb verify

# 单元/集成测试
pytest -q
```

不使用 console script 时，也可以执行 `python -m gsb ...` 或 `python scripts/verify.py`。

## 样本

| 数据集 | 样本 | 字符数 | 用途 |
| --- | --- | ---: | --- |
| `samples/dataset.yaml` | `page_a` | 20 | 主页面、标注版本替换、图像替换 |
| `samples/dataset.yaml` | `page_b` | 16 | 未受影响样本与故障隔离验证 |
| `samples/quantization_flip.dataset.yaml` | `quantization_flip` | 1 | 量化档位真实改变识别结论 |

生成器可随时重建样本与夹具：

```bash
python -m gsb.synth --root .
```

生成过程使用整数种子和固定流程；重复执行不会改变主样本 PNG 的内容哈希。

## 实测结果

完整验收的机器可读结果在 `artifacts/validation.json`，共 18 项检查，当前全部通过。

- 默认配置哈希：`747bb643b601228ad3972042271d98eff1083d30d38f4b986f758d0f04f14418`。
- 主模板指纹：`3e6802d79c561f163544f6ab1023a11e437ddb44c39917f6962d3dfff055f91b`。
- 关键点、识别字符、阅读位次：`36/36` 全部与真值一致。
- 连通域边界框：`page_a` `20/20`、`page_b` `16/16` 与真值块边界框完全一致。
- 串行 1 worker 与并发 4 worker 核心结果逐字段一致；观测并发不超过配置上限 4。
- 替换 `page_a` 标注版本后，仅 2 个关键点、2 条识别、2 个阅读位次失效；其余 34 个复用，`page_b` 全部复用。
- 12 条编辑命令全部成功应用，增量结果与从最终状态全量重建逐字段一致。
- 单批模板点数限制 `15000` 时，待办字符数按 `22 -> 8 -> 0` 逐批消化，合并结果与无预算限制全量运行一致。
- 快照提交两个版本后破坏最新版本 `stages/keypoints.json`，恢复判定会回退到前一有效版本并记录哈希不匹配。
- 四类注入全部命中：`INJ-01`、`INJ-02` 为 `E_SAMPLE_FORMAT`，`INJ-03` 为 `E_TEMPLATE_VARIANT_DIMENSION`，`INJ-04` 为 `E_CONFIG_INVALID`；截断 PNG 和非整数关键点均未中断 `page_b`。
- 量化翻转夹具：`Q4` 为“己”，`Q6/Q8/Q10` 为“曰”；`Q4` 与 `Q8` 的块边界相同但定点坐标、得分、置信度不同，归因结果为 `template_match_score_rounding`。

## 集成 API

主要入口位于：

- `gsb.pipeline.run_from_paths(...)`：从配置、数据集和模板路径运行。
- `gsb.pipeline.run_pipeline(...)`：使用已加载对象和缓存运行。
- `gsb.commands.run_command_stream(...)`：执行增量编辑命令。
- `gsb.trace.analyze_attribution(...)`：误差追踪与翻转归因。
- `gsb.snapshot.SnapshotStore`：临时目录提交、版本指针和恢复。
- `gsb.acceptance.run_acceptance(...)`：可重复验收与 JSON 报告。

## 目录

```text
gsb/                 Python 包与 CLI
samples/             PNG、manifest、真值与标注版本
fonts/               点阵模板清单
configs/             运行配置、模板注册表、编辑命令
injections/          编号异常注入配置
tests/               pytest 用例
scripts/verify.py    验收入口
docs/TECHNICAL.md    技术说明
artifacts/           实际运行产物
```
