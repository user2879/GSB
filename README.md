# Table Cluster Diagnostics Workbench

面向本地合成表格数据的确定性聚类、密度估计与资源预算诊断工作台。项目只依赖 NumPy 与 PyYAML，不访问外部数据源、不使用 Docker，提供 Python CLI、自动化验收测试和离线交互控制台。

## What It Does

- 按 YAML 数据描述和固定种子生成含数值列、类别列、缺失值、高基数类别列、目标结构与冗余列的表格。
- 完成类别编码、训练段缺失值填充、标准化或分位数变换，并保留统计量与内容指纹。
- 提供自实现、可复现的 KMeans、DBSCAN、GaussianMixture，输出硬归属、每轮目标函数和算法诊断量。
- 计算轮廓系数、Calinski-Harabasz、候选条件密度对数似然，并以主指标和固定键生成排名。
- 在全局迭代预算、单候选配额、线程池并发和块缓存驻留预算下执行确定性调度。
- 通过内容指纹链实现跨运行增量复用，逐阶段记录复用/重建账本。
- 以临时目录写入 + 原子指针替换方式发布预处理、聚类、评估、诊断四阶段快照。
- 支持非有限特征、超参数越界、阈值恰等于边界分数、数据规模超限四类注入，并隔离故障。

## Quick Start

Windows PowerShell / CMD（Python 3.10+）：

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py run --config configs/demo.yaml
python app.py serve --config configs/demo.yaml --open
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py run --config configs/demo.yaml
python app.py serve --config configs/demo.yaml --open
```

默认控制台地址为 `http://127.0.0.1:8765`。页面完全离线，无 CDN 或远程字体依赖。

## Commands

```bash
# 运行一次演示实验，打印摘要
python app.py run --config configs/demo.yaml

# 运行并打印完整 JSON
python app.py run --config configs/demo.yaml --json

# 在隔离临时工作区执行串行/并行 A/B 校验
python app.py verify --config configs/demo.yaml

# 启动本地交互控制台
python app.py serve --config configs/demo.yaml --host 127.0.0.1 --port 8765

# 自动化验收
python -m unittest discover -s tests -v
```

## Demo Data

`samples/demo_dataset.yaml` 由程序生成，不包含外部附件：

- 基础 240 行，可配置放大；注入清单将本次演示上限压到 360 行。
- 3 个潜在客群、4 个数值信号列、区域类别、48 档高基数渠道、二元会员列、重复冗余列和随机噪声列。
- `segment` 作为目标结构列保留，不参与无监督特征矩阵。
- 缺失率按列固定；缺失值仅使用训练段统计量填充。
- `value_copy` 与 `order_value` 冗余，演示运行会在训练段检测并剔除。

## Determinism Contract

- 每条候选具有由 `method + normalized hyperparameter fingerprint` 派生的稳定键；预算按稳定键排序预分配。
- 串行与并行只改变任务完成顺序，不改变排名、逐样本归属、指标顺序、预算扣减序列或失败清单顺序。
- 特征和分数分块大小来自 `scheduler.block_size`，与缓存驻留块数解耦；缓存淘汰只影响重算次数，不影响浮点累加顺序。
- 同一输入/配置重复运行产生相同内容指纹、产物指纹和 `determinism_digest`。
- 快照版本、配置哈希或注入清单哈希任一不一致时，不续跑旧产物，并记录 `SNAPSHOT_VERSION_INVALID`。

## Outputs

- `outputs/latest_report.json`：完整审计报告，含候选、标签、指标、预算、复用、快照、注入与失败清单。
- `outputs/latest_core.json`：便于下游比对的报告核心。
- `outputs/cache/`：按内容指纹寻址的跨运行产物缓存与复用账本。
- `outputs/snapshots/`：不可变阶段版本、原子发布指针与清单；失效快照归档到 `snapshots/invalidated/`。

## Project Layout

```text
app.py                              CLI / server 入口
clustering_workbench/
  config.py, data.py                YAML、确定性数据生成、结构校验
  preprocess.py                     编码、缺失填充、变换、指纹
  methods.py                        KMeans / DBSCAN / GMM
  metrics.py                        指标、边界判定、稳定排名
  cache.py                          块级 LRU 驻留缓存
  pipeline.py                       调度、预算、复用、快照、报告
  snapshots.py                      版本化阶段快照与原子发布
  injections.py                     注入命中与业务码映射
  server.py                         本地 JSON API
  static/                           离线交互控制台
configs/demo.yaml                   演示配置与注入清单
samples/demo_dataset.yaml           合成数据描述
tests/test_workbench.py             14 项自动化验收
docs/TECHNICAL.md                   技术设计与语义说明
docs/VERIFICATION.md                实际运行结果
```

更完整的算法、指纹、预算和快照时序说明见 [docs/TECHNICAL.md](docs/TECHNICAL.md)，本次实测结果见 [docs/VERIFICATION.md](docs/VERIFICATION.md)。
