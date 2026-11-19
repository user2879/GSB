# RegBench：表格线性模型正则化训练调度与资源预算工作台

面向合成表格数据的**确定性**正则化训练工作台：按方案 × 系数网格展开训练任务，在
全局轮次预算与单方案配额约束下并发执行确定性小批量梯度下降，支持交互命令精确失效、
阶段化快照整体替换、任意中断恢复，并保证不同并发度/恢复点下结果**逐字段一致**。

- 纯本地运行，无网络、容器、GPU、外部账号依赖。
- 运行期仅依赖 **NumPy**；YAML 使用内置受控子集解析器 `regbench/yaml_lite.py`，
  因此 PyYAML 不是必需依赖（环境里存在 PyYAML 5.4.1，但本项目不 import 它）。

## 1. 环境与依赖记录

任务要求 Windows + Python 3.10/NumPy；本实现是跨平台纯 Python，实际验证环境为 Linux：

| 组件 | 要求 | 实际验证版本 |
| --- | --- | --- |
| Python | >= 3.10 | 3.11.0rc1（另有 3.10.12 解释器可用） |
| NumPy | 必需 | 2.4.6（通过 `get-pip.py` 引导 pip 后 `pip install --user numpy`） |
| PyYAML | 可选，本项目未使用 | 系统自带 5.4.1 |

Windows 等价检查命令：`python --version`、`python -c "import numpy"`。
若 Windows 缺少 pip，可从 https://bootstrap.pypa.io/get-pip.py 引导，再
`python -m pip install numpy`。

## 2. 目录结构

```
regbench/                 工作台包
  errors.py               业务码、失败清单、受控异常
  yaml_lite.py            零依赖受控 YAML 子集解析器
  config.py               配置 dataclass 与“收集而非中断”的全量校验
  data.py                 确定性数据生成/缺失填充/编码/划分/数据指纹
  models.py               L2/L1/弹性网络显式损失与梯度、确定性 SGD、检查点
  grid.py                 网格展开、稳定键、预算/配额纯函数规划器
  snapshot.py             临时目录整体替换 + 原子清单的三阶段快照
  runner.py               线程池执行、异常隔离、优雅退出、收敛汇总
  session.py              交互命令、精确失效、缓存复用、命令流回放
  report.py               JSON 诊断报告与并发一致性比对
  cli.py                  命令行入口
samples/                  确定性样本（YAML 配置、命令流、非法配置、长任务/小预算配置）
scripts/                  demo.sh、可中断驱动、恢复比对脚本
tests/                    20 个 unittest 用例
artifacts/                运行产物（报告 JSON、快照目录）
```

## 3. 快速开始

```bash
# 全部测试
python3 -m unittest discover -s tests -v

# 一键演示（校验/训练/预算/回放/异常/信号/强杀/三路一致性）
bash scripts/demo.sh
```

### 3.1 命令行参数（演示入口所需的四个核心参数）

| 参数 | 含义 |
| --- | --- |
| `--config` | YAML 实验配置路径 |
| `--snapshot-dir` | 快照目录（data/train/summary 三阶段 + manifest.json） |
| `--report` | JSON 诊断报告输出路径 |
| `--commands` | 交互命令流文件（replay 子命令） |

```bash
# 一次性训练
python3 -m regbench.cli run \
  --config samples/experiment.yaml \
  --snapshot-dir artifacts/run/snap \
  --report artifacts/run/report.json --reset

# 仅校验配置（非法项全部进入失败清单，退出码 4）
python3 -m regbench.cli check-config \
  --config samples/invalid_config.yaml --report artifacts/invalid.json

# 交互命令流回放
python3 -m regbench.cli replay \
  --config samples/experiment.yaml \
  --commands samples/commands.txt \
  --snapshot-dir artifacts/replay/snap \
  --report artifacts/replay/report.json --reset

# 串行(workers=1) / 并发(workers>1) / 中断恢复 三路逐字段比对
python3 -m regbench.cli consistency-check \
  --config samples/experiment.yaml \
  --snapshot-dir artifacts/cc/snap \
  --report artifacts/cc/report.json --interrupt-epoch 3

# 真实交互会话（stdin）
python3 -m regbench.cli interactive \
  --config samples/experiment.yaml --snapshot-dir artifacts/ix/snap
```

`run` 额外开关：`--workers N` 临时覆盖并发数（不改配置文件、不改配置哈希）、
`--force-full` 忽略快照全量重算、`--failpoint stage:epoch:任务键子串` 故障注入。

## 4. 关键设计与业务码映射

### 4.1 配置与数据（业务要求 1）

- 校验对每个非法条目产出一条 `E_CONFIG_INVALID` 并继续，互不中断；样本
  `samples/invalid_config.yaml` 一次收集 **18 条**。
- 覆盖：未知列（冗余列引用不存在源列）、类型不符、目标列缺失、学习率为负、
  惩罚系数为负、批量大小非正、并发数非正，以及 split_ratio/基数/种子等边界。
- 数据由 `base_seed` 的 `np.random.RandomState` 确定性生成：数值列、类别列、
  固定线性目标、冗余列，再用独立随机流注入缺失；固定种子洗牌切分训练/验证折。
- 数值列以**训练折中位数**填充并标准化，类别列以**训练折众数**填充并独热编码，
  首列恒为 1 的截距项；对预处理后的矩阵与填充参数计算数据指纹。

### 4.2 三种正则化的确定性训练（业务要求 2）

截距 `w[0]` 不正则，损失显式包含：

- ridge：`0.5*MSE + λ Σ_{j≥1} w_j²`，梯度项 `2λw`
- lasso：`0.5*MSE + λ Σ|w_j|`，次梯度 `λ·sign(w)`
- elasticnet：`0.5*MSE + λ(ρ||w||_1 + 0.5(1-ρ)||w||_2²)`，按混合比例线性组合

L1 次梯度在 `w_j=0` 处采用**固定取值规则 0**（`models.L1_ZERO_SUBGRADIENT`）。
每轮的批量划分与顺序由任务键派生的独立 `RandomState` 决定，与线程调度无关；
逐轮记录损失、参数摘要（L1/L2 范数、零系数个数等）与收敛状态。测试证明同任务
多次运行的逐轮损失轨迹、最终系数、验证指标完全相同，且“分段训练 + 检查点恢复”
与“一次性连续训练”逐轮一致。

### 4.3 网格、预算与配额（业务要求 3）

- 网格按 `(scheme, penalty, l1_ratio, lr, seed)` 稳定键排序，就绪队列位置即稳定顺序。
- 预算/配额裁决集中在纯函数 `grid.build_schedule_plan`，**与线程池执行顺序解耦**，
  这是并发一致性的根本保证：
  - 单任务轮次上限 = `min(max_epochs, 单方案配额)`，触及配额记 `E_TASK_QUOTA_EXCEEDED`；
  - 按稳定顺序预占全局预算，剩余预算 ≤ 0 的任务**不启动**，保留就绪位置并记
    `E_BUDGET_EXHAUSTED`；部分剩余预算下启动的任务标 `budget_truncated`。
- 执行层（`ThreadPoolExecutor`，并发数取配置）只按计划的 `epoch_limit` 跑确定性训练，
  真实消耗轮次回写清单。

### 4.4 交互命令与精确失效（业务要求 4）

| 命令 | 作用 | 失效范围 |
| --- | --- | --- |
| `switch <scheme>` | 切换当前方案视图 | 不失效任何结果 |
| `set-penalty <scheme> <v>` | 改惩罚系数 | 仅该方案的缓存键 |
| `set-ratio <scheme> <v>` | 改弹性网络混合比例 | 仅该方案的缓存键 |
| `set-lr <v>` | 改学习率 | 全部方案训练缓存 |
| `set-workers <n>` | 改并发数 | 不失效数值结果（并发不进配置哈希/缓存键） |

缓存键 = 任务参数指纹 + 数据指纹。未失效任务直接按 `cache_key` 复用已持久化结果，
**绝不重新训练**；二次回放演示中 6 个任务全部 `reused`、`recomputed=[]`。

### 4.5 阶段快照与恢复（业务要求 5、6）

- 三阶段 `data → train → summary`，每阶段先写同文件系统临时目录，再 `os.replace`
  整体替换；`manifest.json` 永远最后原子替换。强杀后只会看到上一版或新版完整快照，
  不会出现半成品（演示中 SIGKILL 后无临时目录残留）。
- 清单记录全局版本号、配置哈希、数据指纹、已完成任务键、已消耗轮次、各阶段产物版本、
  失败清单与交互覆盖层。
- 启动按配置哈希判定续跑，不一致返回 `E_CONFIG_MISMATCH`（CLI 退出码 3）。
- 训练按 `snapshot_interval` 逐轮提交检查点（含权重与 RNG 状态），恢复后从检查点
  继续，逐轮损失/系数/验证指标与一次性全量逐字段一致。

### 4.6 异常隔离与退出语义（业务要求 7）

- 数据折划分、目标计算、梯度更新、快照写入阶段的异常都被捕获，记录任务键、轮次、
  阶段名、业务码（`E_TASK_FAILED` / `E_STAGE_WRITE_FAILED`）、异常类型到失败清单，
  其余任务继续；失败任务不进收敛汇总。
- SIGTERM/SIGINT 触发优雅停止：完成当前轮次检查点原子替换与进度记录后退出，**退出码 0**，
  并记 `E_GRACEFUL_STOP`。
- SIGKILL（退出码 137）后重启只依据已完整替换的快照恢复并打印恢复点；按要求不承诺
  强杀一定生成退出报告。

### 4.7 报告字段（业务要求 8）

每次运行的 JSON 报告包含：各方案惩罚系数/混合比例、逐轮损失轨迹摘要（首尾各 5 点 +
长度）、收敛轮次、验证指标（MSE/RMSE/MAE/R²）、网格完成度、每任务开始/结束单调时钟
读数、是否触及配额、全局剩余预算、`E_BUDGET_EXHAUSTED` 清单、快照版本号、配置哈希、
失败清单、规范完成顺序、调度决策记录顺序与并发一致性比对结论。单调时钟与版本号不参与
逐字段一致性比较（天然随运行变化），比较载荷见 `report.comparable_payload`。

## 5. 实际验证结果（`bash scripts/demo.sh`）

| 验证项 | 结果 |
| --- | --- |
| 单元测试 | 20/20 通过 |
| 非法配置校验 | 收集 18 条 `E_CONFIG_INVALID`，未中断 |
| 全量训练（experiment.yaml） | 10/10 任务完成，10 个 `E_TASK_QUOTA_EXCEEDED`，剩余预算 130 |
| 小预算（tight_budget_experiment.yaml） | 3 任务启动、7 个 `E_BUDGET_EXHAUSTED`，剩余预算 0 |
| 串行 vs 并发 vs 中断恢复 | `status=identical`，mismatches=[] |
| 命令回放 | 首次 6 任务重算、13 个缓存键失效；二次回放全部复用、零重训 |
| 异常隔离 | failpoint 任务 `E_TASK_FAILED`，网格完成度 9/10，失败任务不进汇总 |
| SIGTERM | 退出码 0，恢复后与长任务全量结果逐字段一致 |
| SIGKILL | 退出码 137，无临时目录残留，恢复后与全量结果 `identical` |
| 配置哈希不一致 | `E_CONFIG_MISMATCH`，CLI 退出码 3 |

报告产物位于 `artifacts/demo/`，其中 `consistency_report.json` 含三路报告与比对结论，
`full_report.json` / `tight_budget_report.json` 分别覆盖配额与预算两类受控中止。

## 6. Windows 运行说明

`scripts/demo.sh` 面向 bash；Windows PowerShell 可按相同顺序直接调用模块，例如：

```powershell
python -m regbench.cli run --config samples/experiment.yaml `
  --snapshot-dir artifacts/run/snap --report artifacts/run/report.json --reset
python -m regbench.cli consistency-check --config samples/experiment.yaml `
  --snapshot-dir artifacts/cc/snap --report artifacts/cc/report.json
python -m unittest discover -s tests -v
```

原子替换使用 `os.replace`（Windows 上也是原子语义），路径均为相对/标准库拼接，无符号链接。
