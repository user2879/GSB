# 验证结果记录（本地确定性环境）

环境：WSL Ubuntu，Python 3.11，numpy 1.26.4，pytest 8.2.0。无网络、无容器、无真实录音。
样本：固定种子 `20261126`，16 kHz / 单声道 / 16-bit PCM，20 个片段，
每片段 3200 采样（200ms），信号按 静音/语音/静音/语音 四块交替合成。

复现命令：

```bash
pip install -r requirements.txt
python3 -m pytest -v          # 46 passed
bash scripts/run_demo.sh      # 生成样本 + 常规运行 + 注入运行 + 全量重跑 + 恢复
```

## 1. 单元/集成测试（pytest）

`python3 -m pytest -v` 结果：**46 passed**，按需求映射：

| 测试文件 | 覆盖需求 | 用例数 |
| --- | --- | --- |
| `test_01_validation.py` | 需求 1 加载校验、错误码、失败清单不静默跳过 | 9 |
| `test_02_vad_windows_posterior.py` | 需求 2/3/4 迟滞 VAD、定点窗特征、后验与端点、阈值/半步边界 | 11 |
| `test_05_cache_invalidation.py` | 需求 5 内容指纹失效、精确复用、账本 | 5 |
| `test_06_latency_budget.py` | 需求 6 延迟预算、陈旧/最新区分、游标不推进 | 4 |
| `test_07_determinism.py` | 需求 7 增量==全量、乱序、编辑序列、并发上限 | 6 |
| `test_08_snapshot_injection.py` | 需求 8 原子快照、配置哈希恢复、强制中断、五类注入隔离 | 10 |
| `test_09_signals.py` | 需求 8 可处理信号退出报告 | 1 |

关键断言：
- 切换 `frac_bits/quant_mode` 后窗特征整数、反量化值与后验整数均真实改变。
- 单片段替换仅使引用该片段的窗重算，其余窗全部命中持久化复用并写入账本。
- 乱序投递、追加/替换/删除、门限与最短段调整、档位与位宽切换后，最终签名
  （后验整数、端点区间、边界窗/帧、缓存键）与一次性全量重跑完全一致。
- 线程池实际 worker 数被钳制在 `max_workers` 以内（测试以 cap=3、请求 16 验证）。

## 2. 端到端演示（artifacts/*.json）

### 常规命令流（`artifacts/run_report.json`）
- 34 条命令（追加、迟到片段、重排、替换、删除、门限调整、档位/位宽切换、查询、快照、退出）。
- 响应区分：**29 fresh，3 stale**；3 个陈旧响应均来自全局参数/量化变更，业务码
  统一为 `E_STALE_RESULT`，且都携带 `stale_watermark` 与 `current_watermark`。
- 最终端点区间 2 个，与合成信号的两个语音块对应。
- 账本（`ledger/ledger.jsonl`）事件统计：
  - `reuse_hit` 9109、`invalidation` 578、`recompute` 1253、
    `region_move` 12、`authoritative_summary` 44。
  - 陈旧摘要不写入 `authoritative_summary`，不推进已完成游标。

### 故障注入（`artifacts/injection_report.json`）
- 注入命中编号：`INJ-001`（load 阶段截断 S0005 数据块，`E_AUDIO_FORMAT`）、
  `INJ-002`（window 阶段 W0002 特征维度不符，`E_FIELD_INVALID`）。
- S0005 被隔离后接纳 9 个片段，其余窗仍产出 87 个窗特征与端点结论；
  `test_isolated_failure_keeps_others_consistent` 证明其与同存活集合的干净全量
  运行时间线指纹与逐窗后验完全一致。

### 全量重跑（`artifacts/rebuild_report.json`）
- 198 个窗、2 个端点区间 `[(98,201),(298,397)]`，无失败。
- 后验定点整数范围 10867..21969（Q15），全部为整数；缓存键覆盖
  timeline/params/quant/projection/region/endpoint 六类。

## 3. 快照与恢复
- 快照目录 `snapshots/store/current/` 含 `manifest.json` 与四阶段产品
  `load/vad/window/posterior.json`，经临时目录 + 整体替换原子提交。
- 清单字段：全局版本号、配置哈希、量化档位（`Q15_ROUND_HALF_EVEN`）、
  各阶段版本、已完成片段游标（20）、待办窗队列（空）、重算窗点预算（0）。
- `recover` 命令以版本+配置哈希识别：同配置恢复成功；改配置返回
  “no matching snapshot”，要求从空白重建而非逐文件覆盖。
- 强制中断测试：`before_swap` 崩溃点留下临时目录，恢复时丢弃临时目录并加载上一份
  完好快照；SIGKILL 不生成退出报告，恢复仅依赖持久化清单。
- 可处理信号（SIGTERM）测试：生成退出报告并落盘快照清单。

## 4. 关键设计约束的落实位置
- 定点饱和与三档量化：`vad_diag/fixedpoint.py`
- 内容/参数/量化缓存键派生：`vad_diag/fingerprint.py`、`vad_diag/engine.py`
- 窗不读取窗口外数据（断言 + 区间交集记录）：`vad_diag/windows.py`、`vad_diag/timeline.py`
- 阈值恰等/半步固定决断与边界标记：`vad_diag/posterior.py`
- 延迟预算与陈旧水印、游标控制：`vad_diag/engine.py`
- 原子快照与配置哈希恢复：`vad_diag/snapshot.py`
