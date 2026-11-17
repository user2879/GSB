# qdemo 交付说明：LLM 权重量化交互式策略配置组件与技术演示环境

## 1. 环境依赖

- Python 3.10 及以上（本环境验证：Python 3.10，无 GPU、无网络下载、无真实模型权重）
- numpy **1.26.4**（固定版本；`python3.10 -m pip install numpy==1.26.4`）
- 仅使用 Python 标准库 + numpy；Windows 本地命令行与线程池（`ThreadPoolExecutor`）可运行

## 2. 目录结构

```
qdemo/
  __main__.py     # CLI 入口：gen-weights / gen-calib / validate / full / interactive / replay
  common.py       # 业务码、规范化指纹、有效位舍入、原子写（临时文件 + os.replace）
  weights_io.py   # 权重样本包与校准集生成/加载/校验（固定种子，逐层张量）
  config.py       # 量化与蒸馏配置加载、默认值合并、校验、规范化指纹
  quantize.py     # 逐层对称/非对称量化与反量化核心（float64 缩放因子 + 有效位裁剪）
  evaluate.py     # 校准集前向、误差统计、温度软标签蒸馏校正
  engine.py       # 缓存键派生、依赖失效与增量重算、线程池并发、异常隔离
  interactive.py  # 交互命令会话、延迟预算语义、快照原子提交、退出信号处理
  report.py       # 实验记录与 JSON 报告构建（内容完全确定，无时间戳）
config.sample.json    # 配置样例（位宽/分组粒度/蒸馏温度/预算/并发/快照间隔）
commands.sample.txt   # 交互命令脚本样例（五类修改命令 + status/report/quit）
tests/run_acceptance.py  # 验收套件（16 项检查）
data/                 # 生成的本地样本（weights.npz[.meta.json]、calib.npz[.meta.json]）
out/                  # 运行产物（报告、实验记录、交互日志）
```

## 3. 快速复现

```bash
cd <工作区根目录>
# 1) 生成确定性权重样本包与校准集（种子可控，无需真实模型文件）
python3.10 -m qdemo gen-weights --out data/weights.npz --seed 12345 \
    --layers "fc1:16:32:normal,fc2:8:16:uniform,head:4:8:normal"
python3.10 -m qdemo gen-calib --weights data/weights.npz --out data/calib.npz --seed 777 --samples 32

# 2) 校验配置
python3.10 -m qdemo validate --config config.sample.json

# 3) 一次性全量执行（生成 JSON 报告 + 实验记录，原子提交）
python3.10 -m qdemo full --config config.sample.json \
    --report out/report_full.json --record out/record_full.json

# 4) 交互式会话（脚本模式；去掉 --script 进入 stdin 交互，提示符 qdemo>）
python3.10 -m qdemo interactive --config config.sample.json \
    --workdir out/interactive1 --script commands.sample.txt

# 5) 从实验记录重放并逐字段比对
python3.10 -m qdemo replay --record out/record_full.json

# 6) 验收套件
python3.10 tests/run_acceptance.py
```

## 4. 交互命令

| 命令 | 说明 |
|---|---|
| `set-bits <layer> <2..8>` | 修改某层量化位宽 |
| `set-scheme <layer> symmetric\|asymmetric` | 切换量化方案 |
| `set-group <layer> <N>` | 调整分组粒度（须整除对应维度） |
| `set-per-channel <layer> on\|off` | 开关逐通道缩放 |
| `set-distill <temperature> <alpha>` | 修改蒸馏温度与软标签权重 |
| `status` | 查询最近一次已完成摘要 |
| `report` | 立即提交实验记录并生成 JSON 报告 |
| `quit` | 完成当前命令的记录原子提交后以退出码 0 退出 |

每条命令的响应同时打印并追加到 `<workdir>/interactive_log.jsonl`，包含业务码、
已完成层数、当前误差统计、失效/复用清单与进度。

## 5. 业务码与退出码

| 业务码 | 含义 |
|---|---|
| `OK` | 成功 |
| `E_CONFIG_INVALID` | 配置/权重有限性/分组粒度等校验失败（退出码 2） |
| `E_WEIGHT_SHAPE_MISMATCH` | 张量形状与描述不一致或层间不可链接（退出码 3） |
| `E_INTERACT_BUDGET_PENDING` | 延迟预算内未得到最新参数结果，返回最近一次已完成摘要，后台继续计算 |
| `E_LAYER_FAILED` | 单层单阶段异常（已隔离，记录层名/阶段/异常类型） |
| `E_DEPENDENCY_FAILED` | 上游阶段失败导致的下游阶段跳过 |
| `E_REPLAY_MISMATCH` | 重放结果与记录不一致（退出码 4） |

正常结束（含收到 SIGINT/SIGTERM 并完成当前命令记录提交）退出码为 0；
强制终止后仅依据已完整替换（`os.replace`）的记录文件恢复。

## 6. 设计要点

- **量化核心**：逐层按位宽与分组粒度分别计算对称（`scale=maxabs/(2^(b-1)-1)`，零点 0）
  与非对称（`scale=(max-min)/(2^b-1)`，零点取整裁剪）方案；缩放因子用 float64 计算并
  按 `rounding.scale_sig_digits`（默认 12 位有效数字）裁剪，舍入参数写入记录；反量化后
  逐元素统计最大绝对误差与均方误差，两种方案摘要并列输出供取舍。
- **蒸馏校正**：软标签 `p=softmax(t/T)`、`q=softmax(s/T)`，校正
  `s_corr = s - alpha * T^2 * (q - p)`；逐层比较校正前后对教师输出的 MSE，
  校正后误差增大则标记为校正失稳层并保持原量化输出，失稳清单进入报告与记录。
- **增量重算**：缓存键由层内容指纹 + 层参数 + 舍入参数派生；评估键依赖模型链上
  前缀各层的量化键，因此修改第 i 层仅使 i 及其下游的评估/蒸馏失效；蒸馏键叠加温度
  与软标签权重。每条命令输出失效/复用清单供审计。
- **延迟预算**：命令触达后在 `interaction.budget_ms` 内等待最新结果；超时返回
  `E_INTERACT_BUDGET_PENDING` 与最近一次已完成摘要，计算转入后台继续（绝不描述为
  已终止），完成后按 `snapshot_interval` 原子提交记录。
- **并发一致性**：量化/评估/蒸馏任务在最大并发数取自配置的线程池中执行，任务携带
  层名+阶段名排序键，结果严格按层名字典序组装，失败清单按 (层名, 阶段) 排序，
  与单线程执行逐字段一致（验收项 `concurrency-order-consistency`）。
- **可复现性**：报告与记录不含时间戳等不确定字段；`replay` 从记录中的规范化配置
  重算并逐字段比对。测试钩子 `QDEMO_FAIL_LAYER`/`QDEMO_FAIL_STAGE` 用于注入单层
  单阶段异常以验证异常隔离。

## 7. 实际验证结果（本机，2026-11-18）

- `python3.10 tests/run_acceptance.py`：**16/16 通过**，覆盖：
  确定性样本生成、全量执行、记录重放逐字段一致、并发 1 vs 4 结果一致、
  交互增量结果与最终配置全量执行逐字段一致（layers/failures/distill/summary）、
  失效/复用审计（仅改蒸馏参数时 quant 全部复用、distill 全部失效）、
  延迟预算业务码 `E_INTERACT_BUDGET_PENDING`（pending 响应携带最近完成摘要、
  后台完成后记录已提交）、单层异常隔离（fc2 quant 注入失败后 eval/distill 标记
  依赖失败、其余 2 层正常完成、退出码 0）、非法配置 `E_CONFIG_INVALID`（退出码 2）、
  权重形状不符 `E_WEIGHT_SHAPE_MISMATCH`（退出码 3）、蒸馏失稳层标记
  （T=50, alpha=1.0 时 fc1/fc2/head 全部标记失稳并保持原量化输出）。
- 信号测试：慢速配置（每层 200ms、预算 5ms）的 stdin 会话中，命令计算期间发送
  SIGTERM，进程完成当前命令记录原子提交后以退出码 0 退出，记录文件完整可读。
