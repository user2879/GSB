# 实际验证结果

验证时间：2026-11-27（Asia/Shanghai）。

环境：

- Python `3.11.0rc1`（满足 `>=3.10`）
- numpy `1.26.4`
- `Linux ... microsoft-standard-WSL2 x86_64`
- 未使用 Docker、外部服务或非回环网络

机器可读完整报告：`artifacts/acceptance_report.json`；控制台记录：`artifacts/acceptance_console.txt`。

## 执行命令

```bash
python3 -m compileall -q gsb scripts
python3 -m pytest -q tests/test_pipeline.py
python3 scripts/verify_acceptance.py
```

实际结果：

- `compileall`：通过。
- 单元测试：`9 passed`（含零计数边界与线程池会合）。
- 验收脚本：`9/9 acceptance groups passed`，覆盖 12 组确定性随机编辑序列和全部 107 个预算耗尽点。

## 验收矩阵

| 验收组 | 实际结果 |
| --- | --- |
| `input_validation_business_codes` | 13 个配置/样本违规用例均返回 `E_INPUT_INVALID`，并定位字段/批次 |
| `incremental_matches_full_rebuild` | 5 个编辑前缀全部与空白全量重建逐字段一致 |
| `random_edit_sequences` | 12 组、每组 2–5 个随机权重/阈值/分片/清洗编辑序列均与一次性全量重建一致 |
| `precise_fingerprint_invalidation` | 权重编辑仅重算 2 个 alpha 分片；4 个 beta/gamma 分片复用；替换样本仅重算所在 alpha 分片，其余 5 个分片复用 |
| `budget_cutpoints_and_concurrency` | 遍历全部 107 个工作单位预算耗尽点，续跑结果与串行参考一致；worker `1..4` 结果一致；另有 Barrier 单测证明 2 线程会合 |
| `warm_budget_reuse_plan_stability` | 9 个热增量预算切片后的规范结果和复用/重算清单一致 |
| `business_codes_quota_pending_limit` | 核对 `E_DOMAIN_QUOTA_EXCEEDED`、`E_BUDGET_PENDING`、`E_SHARD_COMPARISON_LIMIT` |
| `sigkill_recovery_matches_full` | 注入真实 SIGKILL，子进程返回码 `-9`；已提交版本仍为 1，恢复后版本 2，与独立空白全量一致 |
| `optional_tkinter_diagnostic` | 当前环境不可导入 Tk：`ImportError: libtk8.6.so ...`；核心流水线不受影响 |

## 关键业务码定位

- 领域配额用例：`E_DOMAIN_QUOTA_EXCEEDED`，字段 `domains.alpha.quota`，批次 `alpha-b0`。
- 零预算用例：`E_BUDGET_PENDING`，首个待办 `alpha-s01`。
- 分片比较上限用例：`E_SHARD_COMPARISON_LIMIT`。
- 输入违规示例字段：
  - `domains.alpha.sampling_weight`
  - `dedup.similarity_threshold_permille`
  - `dedup.shard_size`
  - `fixed_point.decimal_width`
  - `fixed_point.quantization_tier`
  - `samples.sample_id`
  - `samples.<id>.token_counts.<term>`
  - `samples.<id>.length_distribution`
  - `samples.<id>.missing_bitmap`
  - `samples.<id>.domain`

## 一致性口径

以下内容使用递归对象比较，不只比较哈希：

- 逐样本清洗状态、原因码、命中字段、填补来源和归一长度；
- 分片成员、两两相似度、边界标记、簇、代表样本和被剔除样本引用；
- 采样全局序位；
- 每领域占比、有效词项总量、定点偏差和方向；
- 账本条目顺序、缓存键、上游指纹、配置哈希和制品指纹。

冷启动全量构建没有旧制品，因此执行态全部为 `recomputed`；热增量中的未受影响条目为 `reused`。验收另外比较相同热基线下不同预算/并发路径的 `disposition + invalidation_reason` 执行清单，确保预算耗尽时机不改变最终复用/重算判定。

## 手动复现重点

```bash
# 全量本地样本运行
python3 -m gsb.cli run --config configs/pipeline.json \
  --samples data/batches --state /tmp/gsb-state --workers 2

# 零预算：应输出 E_BUDGET_PENDING 和首个待办
python3 -m gsb.cli run --config configs/pipeline.json \
  --samples data/batches --state /tmp/gsb-pending --budget 0; echo $?

# SIGKILL 注入后恢复
python3 -m gsb.cli run --config configs/pipeline.json \
  --samples data/batches --state /tmp/gsb-crash --crash-before-commit
python3 -m gsb.cli run --config configs/pipeline.json \
  --samples data/batches --state /tmp/gsb-crash
```

零预算 CLI 按设计返回退出码 `4`；输入违规返回 `2`，领域配额返回 `3`。

## Tkinter 说明

当前 WSL Python 能找到 `tkinter` Python 包，但系统缺少 `libtk8.6.so`。诊断界面代码和 `gsb ui --state <state>` 入口已交付；在 Ubuntu/WSL 安装系统包后可用：

```bash
sudo apt install python3-tk tk8.6
```
