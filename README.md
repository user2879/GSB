# 多模态特征融合本地实验台

面向图文样本的联合特征构建、定点融合打分与并行调度诊断的纯本地实验台。
不依赖容器与外部服务，Python 3.11 + NumPy/PyYAML/Pillow 即可运行。
原始任务说明见 `TASK.md`。

## 快速开始

```bash
python3 -m venv .venv --system-site-packages   # 或使用任意带依赖的环境
pip install -r requirements.txt
python main.py generate                        # 固定种子生成 32 个样本 -> data/samples
python main.py run --output output/run         # 全量运行（含异常注入）
python main.py verify --output output/verify   # 串行 vs 并行逐字段比对
python -m pytest tests/ -q                     # 13 项自动化测试
```

预算受限与续跑：

```bash
python main.py run --output output/budget --budget 100000  # 预算耗尽 -> E_BUDGET_EXHAUSTED + 待算队列
python main.py run --output output/budget                  # 自动续跑，余量继续消耗
```

## 目录结构

- `configs/fusion_config.yaml` — 融合配置（分块、维度、权重、温度、位宽、量化档位、并发、预算、缓存、快照间隔）
- `configs/injection.yaml` — 独立异常注入配置（非有限特征值 / 分块不整除 / 配额为零 / 分块截断 / 断言损坏）
- `scripts/generate_samples.py` — 固定种子样本生成（PNG 几何图元+噪声、JSONL 断言，≥30 样本）
- `fusion_bench/` — 核心包：加载校验、定点运算、图像/文本特征、融合、调度、缓存、快照、编排
- `main.py` — 运行入口（generate / run / verify）
- `tests/test_pipeline.py` — 端到端与单元测试
- `output/` — 结果、审计清单、诊断报告与快照目录

## 业务码

| 业务码 | 触发场景 |
| --- | --- |
| `E_IMAGE_FORMAT` | PNG 容器头非法、解码失败、尺寸不一致、分块尺寸不整除图像边长 |
| `E_SAMPLE_INVALID` | 样本 ID 缺失/重复、断言结构非法、图像无文本描述、特征值非有限 |
| `E_CONFIG_INVALID` | 配置维度/位宽/量化档位/并发数等不合法（运行级，退出码 2） |
| `E_BUDGET_EXHAUSTED` | 单样本成本超配额，或全局乘加预算耗尽（样本保留在待算队列） |
| `E_CONFIG_MISMATCH` | 快照版本号或配置哈希不一致，拒绝续跑（退出码 3） |

## 确定性设计（核心）

- **定点整数运算**：全程 int64，舍入固定 half-even；`exp` 用"折半 + Taylor + 平方还原"
  避免大负值抵消；产物元数据显式记录位宽、舍入方式与量化档位。
- **派生种子**：投影矩阵由 `(visual_dim, 配置指纹)` 派生；默认补齐向量由
  `default_fill_seed_tag + 配置指纹` 派生；禁止任何运行时随机状态。
- **并行纯计算 + 串行提交**：线程池只执行无副作用的样本级纯函数（任务携带
  `(样本ID, 阶段名)` 排序键）；配额判定、全局预算扣减、缓存驻留/淘汰、输出排序
  全部在提交阶段按样本 ID 稳定顺序完成。因此任意并发度、任意预算耗尽时机下，
  样本输出顺序、贡献分解顺序、失效复用清单顺序、预算扣减序列、失败清单顺序
  与单线程串行执行逐字段一致。
- **缓存审计**：块级 LRU，同访问序号按块键字典序决断；被淘汰块再次引用时从原始
  输入重算并逐元素核验，淘汰/重算清单落盘。提交阶段用缓存重建的特征重算融合分
  并与并行结果比对（`commit_verified`）。
- **阶段化快照**：加载校验 → 特征抽取 → 融合打分 → 产物汇总；各阶段产物先写临时
  目录再 `os.replace` 整体替换；清单记录全局版本号、配置哈希、输入指纹、已完成
  样本、已消耗预算与各阶段产物版本。缓存状态一并快照，保证续跑后淘汰/重算序列
  与一次性全量执行逐字段一致。收到 SIGTERM/SIGINT 时完成当前阶段提交后以退出码 0 退出。
- **预算记账**：分配阶段只做预演，实际扣减在提交时逐样本记账，因此信号中断后
  续跑不会出现"已扣未算"的卡死；预算台账忠实记录每一轮实际扣减历史（两轮不同
  预算上限的续跑场景下，台账与单次全量运行不同属预期行为）。

## 输出产物（output/run/）

- `results.jsonl` — 逐样本定点得分、判定、边界标注、贡献分解、特征与元数据
- `failures.jsonl` — 失败清单（样本 ID、阶段、业务码、注入编号）
- `budget_ledger.jsonl` — 预算扣减序列（样本 ID 稳定顺序）
- `cache_evictions.jsonl` / `cache_recomputes.jsonl` — 淘汰与重算审计
- `pending_queue.json` — 待算队列（预算耗尽样本保留位置）
- `summary.json` — 汇总与同分排名（`-score, 样本ID` 字典序）
- `diagnostics.json` — 注入命中报告、并发度、预算来源、指纹信息
- `snapshots/` — 阶段产物与清单（manifest.json）

## 实际验证结果（本机，Python 3.11.0rc1 + NumPy 1.26.4）

- `pytest`：13/13 通过（定点舍入、串并行一致、重复运行一致、缓存淘汰/重算、
  注入命中、预算耗尽与续跑、配置不匹配拒绝续跑、非法输入、边界判定、信号退出）。
- `python main.py verify`：串行（workers=1）与并行（workers=4）的 7 类产物
  全部 `identical`。
- 预算 100000 运行：4 个样本完成、25 个进入待算队列并输出 `E_BUDGET_EXHAUSTED`；
  续跑后 27 个样本全部完成，结果/失败清单/缓存审计/待算队列/汇总与全新全量
  运行逐字节一致。
- 注入配置 5 项（INJ-NONFINITE-001 … INJ-CORRUPT-005）全部命中并写入诊断报告。
- 信号注入（SIGTERM）：退出码 0，快照保留部分进度；续跑后产物与全量运行一致。
