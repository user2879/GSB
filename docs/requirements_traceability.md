# 需求逐条对照

| # | 需求要点 | 实现位置 | 验证 |
| --- | --- | --- | --- |
| 1 | JSONL/配置/属性加载与七项校验，确定性分桶，非法条目 E_LOG_INVALID/E_HASH_CONFIG_INVALID | `loader.py`, `hashing.py`, `config.py` | `tests/test_loader_validation.py`, `tests/test_hash_config.py` |
| 2 | 主/护栏指标 CTR/CVR，定点整数，请求 ID→物品 ID 字典序累加，精度标签与丢弃原因，逐字段可复现 | `stats.py`, `fixedpoint.py` | `tests/test_pipeline_behaviors.py::test_deterministic_across_runs` |
| 3 | 绝对/相对提升、固定方差式与固定 z 常量、定点区间端点；样本不足抑制；阈值恰好相等显式标注 | `estimation.py` | `EstimationBoundaryTests` 两个用例 |
| 4 | win/tie/lose 判定、按属性键分层、逐层区间、样本占比加权、漂移超容差 E_ATTRIBUTION_DRIFT 与逐层修正 | `attribution.py` | `test_attribution_drift_under_tight_tolerance` |
| 5 | 四阶段内容指纹缓存键、事件集合引用、按受影响分组精确失效、失效/复用清单 | `cache.py`, `fingerprint.py`, `pipeline.py` | 探针实验 + `equivalence_check.py` |
| 6 | 新增批次/单条修正/配置变更/位宽调整编辑序列，增量与全量逐字段等价，覆盖两种位宽×两种桶数 | `edits.py`, `scripts/equivalence_check.py` | 矩阵 `(6,18)`、`(4,24)` 全通过 |
| 7 | 线程池并发，稳定排序键，结果/顺序与串行完全一致，单独串行对照摘要 | `stats.py`, `attribution.py`, `scripts/serial_compare.py` | `test_concurrent_matches_serial`，摘要 `identical=true` |
| 8 | 四阶段临时目录写入+整体替换，manifest 版本/指纹/批次/阶段版本，续跑 E_CONFIG_MISMATCH，四类注入命中，信号退出码 0，强制终止按完整快照恢复 | `snapshot.py`, `pipeline.py`, `injection.py` | `tests/test_snapshot_resilience.py`、注入测试、SIGTERM 子进程实验 |

## 关键设计决策

- 定点优先：比率与份额全程整数缩放，浮点仅出现在方差/平方根，且立即按方向取整回整数，
  保证可复现与可审计。
- 区间保守取整：下界 floor、上界 ceil，量化对齐后仍保持名义覆盖率不收缩。
- 权重最大余数配平：层权重之和严格等于定点单位 1，消除权重侧系统性漂移；剩余份额截断
  偏差由确定性余数回填修正，修正明细可追溯。
- 缓存以内容指纹为唯一身份：相同逻辑内容必然同键，配置/位宽/桶数变化必然改键，避免
  TTL 或时间戳带来的不确定性。
- 配置变更与陈旧快照分离：正常配置演进走失效重建，只有显式注入的旧快照复用才报
  E_CONFIG_MISMATCH，兼顾增量能力与故障复现。
