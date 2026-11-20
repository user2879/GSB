# RoPE 位置编码外推诊断组件

为长上下文注意力计算提供三种位置编码方案（标准 RoPE、线性内插、NTK 感知缩放）的
可复现、可审计外推诊断。全部输入为本地确定性样本（固定种子 20261121 程序生成），
不依赖模型权重下载或线上服务，不使用容器环境。核心交付物为诊断报告
（`output/reports/report.json` / `report.md`）。

## 目录结构

```
rope_diag/            # 组件包
  schemes.py          # 三种 RoPE 方案 + 统一算子接口（返回方案名与参数指纹）
  scoring.py          # 点积打分 + 训练长度内/外推分层统计（均值/极值/计数）
  error_trace.py      # 位置向量→打分→注意力权重 三级误差追踪（float64）
  sensitivity.py      # 基频/缩放因子邻域扫描、敏感性排序、不可用档位标记
  pruning.py          # 维度剪枝策略选择 + 可审计决策记录
  scheduler.py        # 单批点数预算调度 + 异常注入与业务码映射
  pipeline.py         # 四阶段流水线、快照原子提交、并发/串行一致
  fingerprints.py     # 内容指纹与缓存键派生
  samples.py          # 确定性样本生成 + 编辑命令流应用
  report.py           # 诊断报告装配（JSON + Markdown）
scripts/
  generate_samples.py # 生成 config/samples/edits/injections/policy_table
  run_diagnosis.py    # 全量→编辑→增量→全新全量→一致性校验→报告
samples/              # 确定性输入样本
output/
  run_*/stage_tmp/    # 阶段临时目录（与 snapshots 同盘，支持原子替换）
  run_*/snapshots/    # 快照产物目录 + manifest.json
  reports/            # 诊断报告输出
tests/test_rope_diag.py
```

## 运行

```bash
pip install -r requirements.txt        # Python 3.11, numpy
python scripts/generate_samples.py     # 生成确定性样本
python scripts/run_diagnosis.py        # 端到端诊断 + 报告
python -m pytest tests/ -q             # 验收测试
```

## 关键设计

**统一算子接口**：`schemes.apply_scheme(params, query, keys, q_pos, k_pos)` 对三种方案
返回同一 `SchemeOutput`（方案名、参数指纹、旋转后 query/key、位置嵌入、边界标记）。
standard 用原始位置索引；linear 将位置压缩为 `pos/scale`；ntk 将基频调整为
`base * scale^(d/(d-2))`。同配置同输入逐元素可复现。

**误差三级追踪**：以方案自身在训练长度内位置（`pos % train_len`）的编码为参考，
依次计算位置向量相对 L2 偏差、点积打分相对偏差、减最大值 softmax 权重 L1 偏差，
全部 float64，标注首个超过 `error_tolerance` 的环节。

**边界敏感性**：按配置的 base/scale 邻域因子逐方案扫描，输出外推打分偏差的敏感性
排序；角度增量低于 1e-9 判定为维度级截断（`E_DIM_TRUNCATION`），非有限判定为缩放
溢出，基频为零/缩放非正直接映射业务码，命中即标记该档位不可用并记录边界敏感样本
与触发条件。

**剪枝决策**：按 `dim_block_size` 与 `prune_tolerance` 对预置策略表打分选档；选中策略
的 `dim_range` 实际裁剪参与打分的维度、`score_chunk` 实际改变点积分块方式；决策记录
含样本数、外推倍率、候选评分与选择依据。

**精确失效**：缓存键由方案参数指纹 + 样本内容指纹 + 精度档位 + 剪枝口径派生
（load_validate 仅依赖样本内容、校验上界与本阶段注入）。切换方案、改基频/缩放、
改训练长度/剪枝口径、替换单样本时，仅受影响样本与阶段进入重算集合；复用按内容
寻址直接读取持久化产物。

**预算与注入**：调度器按（方案名， 样本 ID）稳定顺序消耗单批点数预算，预算不足项
保留待算并标记 `E_POINT_BUDGET_PENDING`。注入清单对指定样本/阶段施加真实输入故障
（非有限维度特征、键位置越界、基频为零、缩放非正），由真实校验/算子逻辑检出，
每项命中唯一注入编号并映射业务码 `E_NONFINITE_FEATURE` / `E_KEY_POS_OOB` /
`E_BASE_ZERO` / `E_SCALE_NONPOSITIVE`，故障样本隔离不影响其他单元。

**快照与一致性**：四阶段（load_validate → position_encoding → scoring →
error_trace）产物先写 `stage_tmp/` 再 `os.replace` 原子提交到 `snapshots/`；
清单记录全局版本号、配置哈希、注入清单哈希、已完成方案与样本列表。线程池任务
携带（方案， 样本， 阶段）稳定排序键，结果装配顺序与串行完全一致。

## 实际验证结果（2026-11-21，Python 3.11.0rc1 + NumPy 1.26.4）

- `python -c "import numpy"` 可执行（1.26.4）。
- `python scripts/run_diagnosis.py` 六项校验全 PASS：
  serial_vs_concurrent / reproducible_rerun / incremental_vs_full /
  incremental_reused_artifacts / pending_marked / injection_codes_present。
- `python -m pytest tests/ -q`：9 passed。
- 增量运行复用 13 项、重算 33 项、失效 31 键；单样本替换仅使该样本下游阶段重算。
- 预算 430 点消耗 415 点，2 项待算标记 `E_POINT_BUDGET_PENDING`。
- 四类注入各命中唯一编号并映射对应业务码；base ×1e9 档位三方案均判
  `E_DIM_TRUNCATION` 不可用。
