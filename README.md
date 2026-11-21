# 跨模态对齐与检索诊断工作台

这是一个完全本地、确定性的图文检索工作台。它从固定种子生成的 PNG 图像、区域真值 JSON 和结构化文本 JSON 开始，执行以下链路：

1. 样本加载与索引校验；
2. 图像分块 + 定点视觉特征；
3. 文本断言 + 实体组合 + 定点文本特征；
4. 稳定跨模态对齐打分；
5. 阈值检索、并列决断和边界诊断；
6. 内容指纹缓存、阶段级增量复用；
7. 线程池并发和特征点预算调度；
8. 异常注入、阶段快照原子提交和 JSON 诊断报告。

项目不依赖容器、真实账号、线上服务或外部数据集。

## 快速开始

```bash
# 本工作区已使用 virtualenv 创建 .venv；新环境可执行：
python3 -m virtualenv .venv
.venv/bin/python -m pip install -r requirements.txt

# 重新生成确定性样本与配置
.venv/bin/python -m workbench generate \
  --samples samples --config config --seed 20261122 --sample-count 8

# 增量运行（无旧状态时会自发冷启动）
.venv/bin/python -m workbench run \
  --samples samples \
  --config config/alignment.json \
  --injections config/injections.json \
  --output output \
  --mode incremental

# 从空白状态做全量重建
.venv/bin/python -m workbench run \
  --samples samples \
  --config config/alignment.json \
  --injections config/injections.json \
  --output output_full \
  --mode full

# 异常注入演示
.venv/bin/python -m workbench run \
  --samples samples \
  --config config/alignment.json \
  --injections config/injections.demo.json \
  --output output/injection_demo \
  --mode full

# 测试与端到端验收
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_acceptance.py
```

## 目录

```text
workbench/                 核心包（加载、特征、对齐、缓存、快照、调度、报告）
samples/sample-NN/         每个样本目录
  image.png                固定 256x192 PNG
  regions.json             图像尺寸与图元整数矩形真值
  text.json                实体、属性、断言与原始文本
config/alignment.json      对齐与运行配置
config/injections.json     空注入清单（正常回归）
config/injections.demo.json 5 类故障注入演示
output/                    默认产物、快照与报告
  report.json              主报告
  acceptance.json          端到端验收结果
  artifacts/               展开后的四阶段产物
  injection_demo/report.json
  state/current/           当前原子快照
tests/                     11 项本地回归/一致性测试
scripts/run_acceptance.py  验收流程
```

## 输入格式

样本目录名即样本名，必须唯一。加载器按排序后的目录扫描，不支持远程路径。

`regions.json`：

```json
{
  "schema_version": 1,
  "sample_name": "sample-01",
  "image_size": [256, 192],
  "regions": [
    {
      "label": "circle",
      "rect": [20, 16, 86, 72],
      "color": "red",
      "entity_id": "e1"
    }
  ]
}
```

`rect` 必须是四个非布尔整数，宽高大于 0，且完全落在 `image_size` 内；`entity_id` 若出现必须能在文本实体中找到。

`text.json`：

```json
{
  "schema_version": 1,
  "sample_name": "sample-01",
  "text": "sample-01: a red circle at top-left.",
  "entities": [{"id": "e1", "type": "circle", "attributes": {"shape": "circle"}}],
  "attributes": [{"entity": "e1", "name": "color", "value": "red"}],
  "assertions": [{"id": "a1-color", "entity": "e1", "attribute": "color", "value": "red"}]
}
```

加载层校验 PNG 签名、IHDR/IEND、Pillow 解码、PNG 尺寸与真值尺寸、区域矩形、文本实体/属性/断言字段、引用关系和样本名一致性。单样本失败不会中断其他样本；失败项进入排序后的 `failures`，并带有 `E_SAMPLE_INVALID` 父业务码。

## 对齐配置

`config/alignment.json` 的主要字段：

| 字段 | 作用 |
| --- | --- |
| `chunk_size` | 图像分块边长；必须整除宽高 |
| `quantization_level` / `quantization_levels` | 量化档位及可用于相邻档位诊断的档位集合 |
| `visual_dim` / `text_dim` | 两路特征维度；相等时共享同一语义投影 |
| `visual_weight` / `text_weight` | 对齐贡献权重；内部按和归一化 |
| `similarity_threshold` / `top_k` | 定点阈值与最大返回条数 |
| `fixed_decimal_bits` | 定点小数位宽；有效位宽 = 配置位宽 - 量化档位 |
| `thread_pool_size` | 线程池上限，代码不会创建超过该值的 worker |
| `feature_point_budget` | 单批特征点预算；视觉块按 `visual_dim`，文本按断言数乘 `text_dim` 计费 |
| `snapshot_commit_interval` | 每完成多少个样本提交一次整体快照 |
| `required_attributes` | 缺失断言补齐的必需属性 |
| `edge_threshold` / `temperature` | 边缘密度阈值与稳定 softmax 温度 |
| `boundary_neighborhood` | 阈值邻域宽度 |
| `diagnostic_tolerance` | 判断阶段差值“首次超过容差”的门限 |

当前默认阈值 `0.7451171875` 恰好等于一个默认样本的定点得分，用于稳定演示“恰等阈值”和 `E_BOUNDARY_SENSITIVE`。换数据集或大幅改特征参数后，可重新把它校准到某条真实得分。

## 特征与打分

- 视觉块统计：RGB 均值、色度、梯度强度、边缘密度、块中心坐标和活动度，并按 `base_stats_schema` 记录。
- 投影：由 `projection_salt + 特征维度` 派生固定矩阵，对视觉和文本共享语义通道；输出记录 `projection_fingerprint`。
- 文本：每条断言按实体类型、属性和值映射到语义向量；同一实体的断言再组成实体组合向量。缺失必需断言使用配置指纹派生默认向量，并记录 `padding_positions`。
- 定点：归一化向量乘 `2^有效位宽` 后按银行家舍入变为整数，同时保存每个元素的 `rounding_residual`、L2 残差和最大绝对残差。
- 对齐：对文本实体组合与视觉块求余弦；每个组合执行 `shift = cosine - max(cosine)` 后做稳定 softmax，再构造融合视觉向量并计算融合余弦。多个实体通过全局唯一块分配避免重复证据，加权得到文本质量。
- 分项：`visual_component_fixed + text_component_fixed = score_fixed`，报告同时输出浮点贡献与定点贡献。
- 检索：按 `score_fixed` 降序、样本名字典序升序裁决并列；仅返回 `score_fixed >= threshold_fixed` 的前 `top_k`。等于阈值时包含并标为 `boundary: true`。

## 缓存与增量

每个产物使用 SHA-256 内容键：

```text
visual    = 图像内容 + 区域真值 + 分块/维度/定点参数 + 量化档位 + 阶段注入
text      = 文本内容 + 文本维度/补齐参数 + 量化档位 + 阶段注入
alignment = visual_key + text_key + 对齐权重/温度 + 量化档位
retrieval = 所有 alignment_key + 冲突样本签名 + 阈值/top_k + 注入上下文
```

因此：

- 替换单张图：该样本视觉、该样本对齐、全局检索失效；文本复用。
- 修改一条断言：该样本文本、该样本对齐、全局检索失效；视觉复用。
- 调整权重或温度：视觉/文本复用，对齐和检索失效。
- 调整分块尺寸：视觉失效；文本复用。
- 切换量化档位或位宽：两路特征及全部下游失效。

`reuse_ledger` 逐阶段列出 `reused` / `recomputed` 的样本、缓存键和块/断言引用，可直接审计“未失效阶段没有重算”。

## 并发与预算

调度任务键形如 `sample|stage|r000000|c000000|tail`，阶段顺序为视觉、文本、对齐。任务先按稳定键排序，再按特征点预算顺序准入；第一个放不下的任务及其后的任务全部保留到 `pending_tasks`，并输出 `E_POINT_BUDGET_PENDING`。线程完成顺序不影响最终组装顺序。

实际线程池上限等于 `thread_pool_size`。`output/artifacts/execution_trace.json` 记录 worker 上限、点数消耗、峰值并发等执行信息；核心语义摘要不包含会随线程调度波动的字段。

## 快照与续跑

- 阶段 JSON 先写入临时 staging 目录，再整体替换 `output/state/current`。
- Manifest 记录 schema/engine 版本、全局版本号、配置哈希、已完成样本、各阶段产物版本与键列表、注入清单哈希。
- 配置哈希或 manifest 完整性不一致时输出 `E_SNAPSHOT_STALE`；内容键仍安全的阶段可复用，必要时全量模式从空 staging 重建。
- 快照按 `snapshot_commit_interval` 做中间提交，最终提交包含全部阶段。

## 异常注入

`config/injections.demo.json` 包含 5 项真实故障：

| 注入 ID | 故障 | 映射业务码 |
| --- | --- | --- |
| `INJ-PNG-TRUNCATED` | 内存中截断 PNG 末尾字节，破坏 IEND | `E_PNG_TRUNCATED` + `E_SAMPLE_INVALID` |
| `INJ-ASSERTION-FIELD` | 删除断言的 `value` 字段 | `E_TEXT_SCHEMA` + `E_SAMPLE_INVALID` |
| `INJ-FEATURE-NONFINITE` | 首个视觉投影值改为 NaN | `E_FEATURE_NONFINITE` + `E_SAMPLE_INVALID` |
| `INJ-THRESHOLD-RANGE` | 以 1.5 作为阈值执行校验 | `E_CONFIG_THRESHOLD`，隔离后使用安全阈值继续 |
| `INJ-CHUNK-INDIVISIBLE` | 指定样本使用 30 像素分块 | `E_CHUNK_MISMATCH` + `E_SAMPLE_INVALID` |

每项命中写入 `injection_results.hits`，包含注入编号、阶段、证明字段和业务码；未命中项进入 `missed` 并输出 `E_INJECTION_MISSED`。

## 报告与业务码

主报告包含：

- `retrieval_quality`：自命中 top-k、top-1、MRR；
- `retrieval_index.queries[].results`：得分、贡献分解、排名、阈值通过、边界标记、相邻量化档位比较；
- `boundary_diagnostics`：两档得分、差值、翻转、阈值翻转、排名翻转、首次超过容差阶段；
- `failures`、`pending_tasks`、`reuse_ledger`、`snapshot`、`injection_results`；
- `reproducibility_sha256`：对样本缓存键/引用/自对齐、检索结果、质量、边界诊断和失败清单计算的稳定摘要。

常用业务码：`E_SAMPLE_INVALID`、`E_PNG_TRUNCATED`、`E_TEXT_SCHEMA`、`E_REGION_INVALID`、`E_FEATURE_NONFINITE`、`E_CHUNK_MISMATCH`、`E_CONFIG_THRESHOLD`、`E_POINT_BUDGET_PENDING`、`E_BOUNDARY_SENSITIVE`、`E_SNAPSHOT_STALE`、`E_CACHE_CORRUPT`、`E_INJECTION_MISSED`。

## 实际验证结果

环境：Python 3.11.0rc1、NumPy 2.4.6、Pillow 13.0.0、pytest 9.2.0。

```text
.venv/bin/python -m pytest -q
11 passed in 27.84s
```

```text
.venv/bin/python scripts/run_acceptance.py
status: passed
```

关键结果：

- 基线 8 查询：`self_hit_at_k_rate = 1.0`、`top1_self_hit_rate = 0.875`、`MRR = 0.9375`。
- 基线语义摘要：`5de0902e56f06ad790ff558e3caca63e134d8c738fedfa49d28d476abbc51eed`。
- 无改动增量与全量摘要一致，四阶段全部复用：视觉 8、文本 8、对齐 8、检索 1。
- 样本替换 + 断言修改 + 权重调整 + 分块 16 + 量化档位 2 的编辑序列，最终增量摘要与干净全量摘要一致。
- 5 项注入全部命中、0 项漏命中；4 个目标样本被隔离，其余 4 个样本继续检索。
- 预算降至 10 时 392 个任务稳定进入待算队列，输出 `E_POINT_BUDGET_PENDING` 且无越预算执行。
- 线程数 1 与 8 的语义摘要一致。
- 默认边界样本 `sample-01 -> sample-02` 得分恰等于阈值，相邻档位发生阈值翻转，首次超差阶段为视觉定点量化。

机器可读结果见 `output/acceptance.json`、`output/report.json` 和 `output/injection_demo/report.json`。
