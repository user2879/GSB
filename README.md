# 定点量化融合交互工作台（语音对话理解）

面向语音对话理解场景的本地确定性交互工作台：将**声学证据、转写文本证据、工具调用结果**三路证据沿定点量化融合链路生成轮次情感置信度，在配置档位与高一档位两套精度下诊断数值精度与误差传播，并支持交互式反复调参。全程本地运行，不依赖 Docker、真实账号或任何外部服务。

## 环境准备

- Python 3.11+（本仓库在 Python 3.11 上验证；Windows 命令行下直接 `python` / `pip` 可用即可）
- 创建虚拟环境并安装依赖（版本以本地镜像可获取的稳定版为准）：

```bash
python -m venv .venv
# Linux/macOS: .venv\Scripts\activate 对应 Windows 下 .venv\Scripts\activate
.venv/bin/pip install -r requirements.txt
```

本次实测安装版本：numpy 2.4.6、pytest 9.2.0、pyyaml 6.0.4（pip 26.3）。

## 快速开始

```bash
# 1. 生成固定种子本地样本（turns/acoustics/tools/fusion.yaml/vocab）
.venv/bin/python -m workbench.cli generate-samples --out samples

# 2. 基线运行：三路打分 + 定点融合 + 双档位误差传播
.venv/bin/python -m workbench.cli run --samples samples --run-dir run

# 3. 异常注入运行（四类故障，注入编号写入报告）
.venv/bin/python -m workbench.cli run --samples samples --run-dir run_inj --inject config/injection.yaml

# 4. 交互模式（stdin 每行一条命令，stdout 返回 JSON）
printf 'run\nset-weight text 0.65\nset-precision 16 1\nsummary\nquit\n' \
  | .venv/bin/python -m workbench.cli interactive --samples samples --run-dir run_interactive

# 5. 一键验收场景（A–G 全部检查）
.venv/bin/python -m workbench.cli verify --work-dir verify_out

# 6. 测试套件
.venv/bin/python -m pytest tests/ -q
```

## 样本目录（固定种子 20261122 生成）

| 文件 | 内容 |
| --- | --- |
| `samples/turns.jsonl` | 24 轮对话：轮次 ID、说话人、受控词表词项序列、毫秒起止、情感标注、工具调用 |
| `samples/acoustics.jsonl` | 每轮音高均值/方差、能量包络分位数（0.25/0.5/0.75），与轮次一一对应 |
| `samples/tools.json` | 工具清单：名称、参数模式、确定性公式、量纲（`sentiment_polarity` / `ratio`） |
| `samples/fusion.yaml` | 融合权重、温度、位宽/档位及允许集合、有效位数、容差、并发数、预算、快照间隔 |
| `samples/vocab.json` | 受控词表 + 情感触发词表 |
| `config/injection.yaml` | 独立异常注入配置，四类样例各带唯一注入编号 |

## 设计要点

**定点融合链路（`workbench/fusion.py`）**：归一化（批次 min-max，轮次升序累加，按配置保留有效位）→ 量化（`round(v·2^frac_bits / quantum)·quantum`，`Fraction` 精确有理数 + 确定性 half-up 舍入）→ 定点整数加权求和（权重同尺度定点化，除以融合温度，加偏置）→ 裁剪到置信区间后输出定点整数置信度。权重和为零（`E_FUSE_WEIGHT_SUM_ZERO`）、量纲不兼容（`E_FUSE_DIM_INCOMPAT`）、证据缺失（`E_FUSE_EVIDENCE_MISSING`）、裁剪到端点（`E_FUSE_CLIPPED`）均输出业务码与归一化前原始量，不静默替换默认值。

**逐级误差传播（`workbench/errorprop.py`）**：配置档位（12,2）与高一档位（16,2）完整重算四个环节，逐轮输出各环节差值并标注首次超容差环节；跨档位整数先换算回各自定点尺度再比较。判定翻转时：量化整数（按浮点等值）相同而加权和不同 → `weighted_sum_accumulation`；量化整数不同 → `quantization_truncation`，归因证据由实际计算路径派生。

**内容指纹缓存（`workbench/cache.py` + `workbench/pipeline.py`）**：归一化统计量、量化整数、加权求和、裁剪判定各自以「上游内容指纹 + 本阶段参数指纹」派生缓存键并记录引用集合；参数指纹按阶段精确限定作用域（如权重只进 `weighted_sum` 的参数指纹），因此改权重只重算加权求和/裁剪，归一化与量化直接复用。

**预算约束并发调度（`workbench/scheduler.py`）**：线程池最大并发取自配置；单批融合点数预算按轮次 ID 升序消耗（每轮 3 点），不足者保留待办并标记 `E_FUSE_BUDGET_PENDING`；任务携带（轮次 ID, 阶段名）稳定排序键，报告装配前统一排序，并发/串行结果逐字段一致。

**交互与快照（`workbench/interactive.py`、`workbench/snapshot.py`）**：五类命令（改权重、切温度、调位宽/档位、替换单轮、增删工具）触发增量重算；每条命令在交互延迟预算内响应，超时返回带 `E_PREVIEW_BUDGET` 与显式版本号的最近一次完成摘要。各阶段产物先写 `run/tmp` 再整体替换提交到 `run/snapshots`，清单记录全局版本号、配置哈希与各产物指纹；启动时配置哈希不一致输出 `E_SNAPSHOT_STALE`。

**异常注入（`workbench/injection.py`）**：四类故障（非有限声学统计量、工具量纲不符、位宽越界、权重全零）施加于真实输入副本后流经正常校验/融合路径，业务码由真实计算产生；注入编号写入报告与失败清单，单轮异常记录轮次 ID + 阶段名 + 业务码并隔离，其余轮次继续。

## 业务码

`E_LOAD_*`（加载校验）、`E_FUSE_WEIGHT_SUM_ZERO` / `E_FUSE_DIM_INCOMPAT` / `E_FUSE_EVIDENCE_MISSING` / `E_FUSE_CLIPPED`（融合）、`E_FUSE_BUDGET_PENDING`（预算）、`E_PREVIEW_BUDGET`（延迟预算）、`E_SNAPSHOT_STALE`（快照）。完整定义见 `workbench/codes.py`。

## 实测验证结果（本仓库当前样本）

- `pytest tests/ -q`：**47 passed**
- `verify` 一键验收：**all_passed = true**，覆盖
  - A 三路定点融合完成（24/24 轮，置信度区间分布 low=9 / high=15）
  - B 每轮四环节误差传播输出（24 轮首次超容差环节均为 `quantize`，量化差值约 6e-5–9e-5，容差 1e-6）
  - C 并发 4 与串行 1 运行逐字段一致
  - D 预算=6 时 2 轮入批、22 轮 `E_FUSE_BUDGET_PENDING`，扣减序列按轮次 ID 升序
  - E 五类命令序列后增量运行与空白全量重建逐字段一致（含复用/重算清单：热运行 241 项全部复用、0 重算）
  - F 四类注入全部命中（`INJ-ACOUSTIC-NONFINITE-001` → `E_LOAD_ACOUSTIC_NONFINITE`，`INJ-TOOL-DIMENSION-002` → `E_FUSE_DIM_INCOMPAT`，`INJ-PRECISION-RANGE-003` → `E_LOAD_PRECISION`，`INJ-WEIGHTS-ZERO-004` → `E_FUSE_WEIGHT_SUM_ZERO`）
  - G 快照配置哈希变化检出 `E_SNAPSHOT_STALE`
- 翻转归因：样本数据在默认容差下无自然翻转；两种归因（`quantization_truncation` 与 `weighted_sum_accumulation`）由 `tests/test_errorprop.py` 经真实融合路径构造的边界案例验证（如归一化值 0.499755859375 在 12 位档上舍入越界、权重 0.3/0.2/0.5 在两位宽下有效权重差异导致加权和翻转）。

## 目录结构

```
workbench/            # 核心包
  codes.py            # 业务码定义
  loader.py           # 加载校验层（E_LOAD_*）
  scoring.py          # 三路确定性打分（文本/声学/工具公式白名单求值）
  fusion.py           # 定点融合链路（Fraction 精确运算）
  errorprop.py        # 双档位误差传播与翻转归因
  cache.py            # 内容指纹缓存与精确失效
  scheduler.py        # 预算约束并发调度
  pipeline.py         # 流水线编排（证据→四环节→报告）
  interactive.py      # 交互命令与延迟预算响应
  snapshot.py         # 阶段快照整体替换与过期检测
  injection.py        # 异常注入
  verify.py           # 端到端验收场景
  cli.py              # 命令行入口
scripts/generate_samples.py  # 固定种子样本生成
config/injection.yaml        # 独立异常注入配置
tests/                       # pytest 套件（47 项）
samples/                     # 生成的本地样本
```
