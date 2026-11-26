# 验证记录

环境：WSL Ubuntu，Python 3.11.0rc1，NumPy 1.26.4（主流水线不依赖 NumPy 计算）。
运行前清空 `snapshots/`，命令：`python3 -m unittest`。

## 单元/集成测试（41 项全部通过，约 27s）
- tests/test_loading.py（10）：WAV 采样率/声道/位深/data 长度、受控字表、偏移越界、
  重复区间、版本回退、配置缺键，校验 E_AUDIO_FORMAT / E_INPUT_INVALID 与行号。
- tests/test_features.py（4）：窗特征逐元素可重复、跨边界区间引用与逐区间帧计数、
  特征维数与位宽、切换 Tier 整数变化。
- tests/test_evidence_calibration.py（7）：字到窗映射、共现整数确定性、Tier 残差、
  定点重定标、整数衰减、空轮继承、校准可重复。
- tests/test_aggregation.py（4）：阈值等值取高档、半步边界、重叠归属优先级与抑制依据、
  区间引用集合。
- tests/test_incremental.py（6）：首跑全算后二次全复用、单区间重标注仅失效相关语义
  路径且声学窗全复用、全局重标注经共现影响邻接区间、清空快照全量重建逐字段一致、
  单区间音频变更范围失效、账本审计原因。
- tests/test_scheduler.py（3）：配额 E_EVIDENCE_QUOTA_EXCEEDED、预算待办
  E_EVIDENCE_BUDGET_PENDING 且下批优先、稳定排序。
- tests/test_error_tracking.py（2）：四阶段逐区间差值与首个超容差环节、翻转归因取值域。
- tests/test_flip_attribution.py（3）：阈值比较/量化舍入归因与无翻转空结果。
- tests/test_recovery.py（2）：SIGKILL 后仅依据持久化快照续跑到 complete；
  SIGTERM 写 exit_report.json。

## 端到端与增量验收
- `bash run_demo.sh`：coarse 与 fine 两档均成功，error_tracking 报告首个超容差环节
  （word 阶段），区间标签含 positive/negative 与重叠抑制。
- 全量重建一致性：test_full_rebuild_field_identical 比对窗特征、字强度、校准值、
  区间汇总逐字段相等。
- 强制终止恢复（scripts/control_vdu.sh，仅 kill + 整数信号）：
  - `kill`（SIGKILL）后 state.complete=false 且 todo 非空，`resume` 后
    complete=true、todo=[]；无 exit_report（强制终止不可处理，符合预期）。
  - `term`（SIGTERM）后 current/exit_report.json 记录 reason=signal_15 与 todo。

## 确定性
- 相同输入与配置重复运行窗特征、字证据、校准、区间汇总逐元素一致（见 features/
  evidence/calibration 测试）；合成 WAV 字节级可重复。
