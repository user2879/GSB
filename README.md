# 视频预测与运动时序领域组件

这是一个不依赖容器和网络服务的本地确定性示例。它把视频预测管线拆成
`encode -> motion -> predict -> decode` 四个阶段，在数据规模变化时选择不同的
帧编码精度、运动估计分块、预测路径和量化档位，并以内容寻址缓存、原子阶段快照和
线程池执行保证增量结果与空白全量重建一致。

## 快速开始

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 可选：重新生成本地固定种子样本
python samples/generate_samples.py

# 校验配置和 JSONL
python -m motion_pipeline validate \
  --config samples/config.yaml \
  --sequences samples/sequences.jsonl

# 从空白状态运行小规模档
python -m motion_pipeline run \
  --config samples/config.yaml \
  --sequences samples/sequences.jsonl \
  --tier 1 \
  --state-dir /tmp/motion-state \
  --output-dir /tmp/motion-output \
  --fresh

# 同一状态目录切换规模档，执行增量运行
python -m motion_pipeline run \
  --config samples/config.yaml \
  --sequences samples/sequences.jsonl \
  --tier 2 \
  --state-dir /tmp/motion-state \
  --output-dir /tmp/motion-output-tier2

# 跑完整验收场景（规模档扩张、单序列替换、参数/步长/量化编辑、增量 vs 全量）
python scripts/verify_demo.py
```

运行环境实测为 Python 3.11.0rc1、NumPy 1.26.4、PyYAML 6.0.2。项目不需要 Docker、
数据库、网络接口或外部渲染服务。

## 输入格式

配置 YAML 的示例字段如下：

```yaml
frame_width: 32
frame_height: 24
prediction_steps: 2
history_frames: 4
motion_model:
  linear_gain: 1.0
  damping: 0.82
  search_radius: 2
noise_layer:
  amplitude: 2.0
  seed_salt: 97
fixed_point_bits: 12
quantization_levels: [1, 2, 4]
precision_levels: [6, 8, 10, 12]
scale_tiers: [1, 2, 4]
latency_budget_ms: 250
thread_pool_workers: 4
snapshot_commit_interval_batches: 1
```

`history_frames` 是从序列开头取用的观测帧数；未来帧若存在，会被用作预测误差诊断的
真值。运行规模档时使用以下确定性缩放规则：

- 序列数：`round(有效序列总数 * tier / max(scale_tiers))`。
- 每条序列的轨迹数：`round(序列最大轨迹数 * tier / max(scale_tiers))`。
- 预测步长：`round(prediction_steps * tier / min(scale_tiers))`。

量化档位允许集合为 `[1, 2, 4, 8, 16, 32]`，精度档位允许集合为
`[4, 6, 8, 10, 12, 16]`，且至少一个精度档位不大于 `fixed_point_bits`。

JSONL 每行是一条序列：

```json
{
  "sequence_id": "seq-0001",
  "noise_seed": 20261124,
  "frames": [
    {
      "trajectories": [
        {"trajectory_id": "t-01", "shape": "circle", "x": 4.5, "y": 8.0, "radius": 2.1, "vx": 0.7, "vy": -0.2, "color": [220, 72, 48]}
      ],
      "color_blocks": [
        {"x": 1.0, "y": 1.0, "width": 7.0, "height": 4.0, "color": [30, 46, 64]}
      ],
      "noise_seed": 101
    }
  ]
}
```

支持 `circle` 与 `rect` 两种图元。所有坐标、速度、尺寸和颜色块参数必须为有限数，
颜色通道为 `0..255` 整数。样本生成器使用固定种子，生成的
`samples/sequences.jsonl` 包含 8 条序列、每条 16 帧、最多 8 条轨迹。

## 阶段实现

### 1. 帧编码

`encode` 把每帧的图元参数按所选精度转换为定点整数。坐标、半径、矩形宽高先执行
`round-half-away-from-zero`，再以整数子像素坐标光栅化。每帧记录：

- 量化值的数量；
- 有符号/绝对残差和及平均绝对残差；
- `up`、`down`、`exact` 舍入方向计数；
- 使用的精度位宽和缩放因子。

帧像素完全由输入轨迹、颜色块和定点参数决定，不读取进程全局随机状态。

### 2. 运动场估计

`motion` 对相邻帧做固定分块估计。每个分块先进行确定顺序的 SAD 整数搜索，再用
块内空间梯度和时域差做梯度修正，输出块级 `(dx, dy)` 定点运动矢量。结果保留每个
块的梯度能量、SAD、行优先矢量数组和摘要。同一序列、同一档位、同一参数下重复运行
得到逐元素一致的矢量。

### 3. 时序预测

`predict` 实现两条真实不同的路径：

- `linear`：用最近两个运动场做线性外推，外推系数为预测步序号。
- `weighted`：对最近最多四个运动场做固定权重加权平均，外推系数为 0。

两条路径都按固定整数顺序应用 `linear_gain`、逐步阻尼和量化。每一未来步输出运动
矢量、运动摘要、路径、外推系数、衰减系数和量化档位。

### 4. 逐帧解码与合成

`decode` 以最后一个观测帧为基准，按每个分块预测矢量平移采样，再叠加由序列 ID、
帧步序号、噪声种子和坐标派生的整数哈希噪声，最后裁剪到 `0..255`。每条序列输出
逐帧预测像素、运动摘要、预测系数和解码元数据。未来真值帧存在时，同时计算每步和
整体的 MAE、RMSE 与最大绝对误差。

### 定点、量化与随机性

- 所有像素/运动计算使用整数数组；浮点只用于输入解析、指纹和决策指标。
- 预测路径的量化单位是像素的整数倍，实际写回为定点单位。
- 噪声使用局部确定性整数哈希，不使用 `random` 或任何全局随机源。
- 所有序列、帧、运动对和预测步都携带由 `序列 ID + 阶段 + 序号` 派生的稳定排序键。

## 自适应策略选择

预置候选策略为：

| 候选 | 编码精度 | 分块 | 预测路径 | 量化 |
| --- | ---: | ---: | --- | ---: |
| `quality-linear` | 最高可用精度 | 4 | linear | 最低量化档 |
| `balanced-weighted` | 中间精度 | 8 | weighted | 中间量化档 |
| `throughput-weighted` | 最低精度 | 16 | weighted | 最高量化档 |

选择器读取实际序列数、平均轨迹数、预测步长、当前规模档、单序列估计代价和延迟预算，
计算 `target_load` 与候选分。候选分由质量权重、吞吐权重和超预算惩罚组成。决策记录
会写出所有规模指标、每个候选的评分/估计延迟比和最终选择理由。

本地样本的实测选择为：

- `tier=1`：`quality-linear`，精度 12、分块 4、线性外推、量化 1。
- `tier=2`：`balanced-weighted`，精度 8、分块 8、加权平均、量化 2。
- `tier=4`：`throughput-weighted`，精度 6、分块 16、加权平均、量化 4。

不同档位会实际改变帧像素、运动矢量、摘要和预测结果，而不是只改变决策文本。

## 精确失效与复用账本

每个序列的每个阶段使用内容缓存键：

```text
H(stage, 上游阶段内容键, 参数指纹, 适用量化档位, 算法版本)
```

缓存键不包含线程数、运行时间或输出目录。各阶段的依赖关系是：

```text
sequence content -> encode -> motion -> predict -> decode
```

因此：

- 替换单条序列：只有该序列的四个阶段失效，其他序列直接复用。
- 调整阻尼或线性增益：`encode`/`motion` 命中，`predict`/`decode` 重算。
- 调整搜索半径或分块：`encode` 命中，`motion` 及其下游重算。
- 调整预测步长：`encode`/`motion` 命中，`predict`/`decode` 重算。
- 切换量化档位：`encode`/`motion` 命中，`predict`/`decode` 重算。
- 切换规模档：按缩放后的序列内容、轨迹数、策略和步长逐序列计算键，只重算键发生变化
  的阶段及其下游。

输出包含两份互补审计文件：

- `cache_ledger.json`：当前最终状态的**规范内容依赖账本**。逐条记录阶段、序列集合、
  缓存键、上游键和最终可用状态；它只由最终输入、参数和量化档位决定，因此增量运行与
  空白全量运行逐字节一致。验收要求中的“失效复用账本一致性”以该文件为准。
- `cache_execution.json`：本次执行实际发生的 `HIT`/`RECOMPUTE` 事件及原因。例如
  `CONTENT_KEY_MATCH`、`NEW_SEQUENCE_OR_STAGE`、`UPSTREAM_STAGE_RECOMPUTED`、
  `PREDICTION_PARAMETERS_OR_QUANTIZATION_CHANGED`。该文件天然依赖本次运行的历史，
  与诊断报告中的阶段命中/重算计数一起用于观察具体增量行为。

把规范账本和实际执行日志分开，是为了同时满足两个不同目标：最终状态可审计且跨运行
完全一致，以及能还原某次增量运行究竟命中了什么、重算了什么。

## 快照与恢复

状态目录结构：

```text
state/
  manifest.json
  stages/
    encode/00000003/index.json
    motion/00000003/index.json
    predict/00000003/index.json
    decode/00000003/index.json
  tmp/
```

提交时先把新阶段快照写入 `tmp/` 并 `fsync`，然后通过目录重命名整体替换为新的版本目录，
最后单独原子替换 `manifest.json`。清单记录全局版本、配置哈希、已完成序列、规模档、
各阶段版本和内容哈希；已存在且哈希相同的阶段不会逐文件覆盖。

启动时清理未提交临时目录并检查清单、阶段版本、阶段文件和内容哈希。以下情况输出
`E_SNAPSHOT_STALE` 到诊断报告的 `snapshot.stale_events`：

- 配置哈希不一致；
- `--expected-version` 与清单全局版本不一致；
- 阶段快照缺失、损坏或内容哈希不一致；
- 有阶段目录但缺少清单。

配置哈希变化不会粗暴丢弃所有缓存，而是结合各阶段内容键做选择性失效；版本强制不一致
会清空可复用项并全量重算。强制终止时，只有已提交的阶段目录和清单可用于恢复；程序不承诺
在收到强杀后生成退出报告。

## 并发与确定性

`ThreadPoolExecutor` 的最大线程数严格来自 `thread_pool_workers`。运动估计按相邻帧对
提交任务，解码按未来步提交任务；每个任务都带稳定排序键。主流程按序列 ID、运动对序号
和步序号收集结果，因此线程完成顺序不会影响输出顺序。阶段缓存只由主线程更新，避免并发
写入状态。

测试将 `thread_pool_workers=1` 与 `=4` 的逐帧输出、决策、规范账本和失败清单逐字段比较，
两者完全一致。

## 输出文件

每次完整运行的输出目录包含：

- `predictions.json`：逐序列帧编码像素/摘要/量化元数据、全部运动矢量与摘要、逐帧预测
  像素、预测系数、解码元数据和误差统计。
- `decision_record.json`：规模指标、候选评分和最终策略。
- `cache_ledger.json`：最终状态规范内容依赖账本。
- `cache_execution.json`：实际命中/重算事件与原因。
- `diagnostic_report.json`：策略、量化档位、各阶段缓存计数、逐序列误差、单调时钟读数、
  线程并发观测值、延迟预算判定、失败业务码和快照状态。
- `failures.json`：按序列 ID、行号稳定排序的失败清单。

配置错误输出 `E_CONFIG_INVALID`，序列错误输出 `E_SEQ_INVALID`。非法序列不会阻断其他合法
序列；校验通过的行才进入管线。快照问题使用 `E_SNAPSHOT_STALE`。

## 验收测试

```bash
pytest -q
python scripts/verify_demo.py
```

当前实际结果：

- `pytest -q`：`11 passed`。
- `scripts/verify_demo.py`：`status=ok`，预测、决策、规范账本、失败清单和逐序列误差
  五项增量/全量比较全部为 `true`。
- 验收场景最终规模档选择 `balanced-weighted`，量化档位 4，预测步长 6。
- 最终增量运行命中 6 个阶段结果、重算 10 个；另一空白状态全量运行重算 16 个；两者逐帧
  输出完全一致。
- 编辑过程中产生 `E_SNAPSHOT_STALE`，用于标明配置哈希已变化；合法样本失败业务码为空。

`scripts/verify_demo.py` 的输入、快照、最终输出和 `verification.json` 默认保存在
`artifacts/demo/`（已加入 `.gitignore`，可随时重新生成）。
