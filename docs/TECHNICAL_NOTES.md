# CCObs 技术说明

面向 Windows 本地环境的 TCP 拥塞控制观测组件。仅使用 Python 3.11+ 标准库
(含 Tkinter / SQLite), 不使用 Docker, 不访问 127.0.0.1 以外的任何目标。

## 模块结构

| 模块 | 职责 |
| --- | --- |
| `ccobs/errors.py` | 稳定业务码 E_CONFIG / E_SCRIPT / E_CWND / E_SNAPSHOT / E_REPLAY 与结构化渲染 |
| `ccobs/config.py` | 会话配置与对端脚本 JSON 加载、强校验、内容摘要 |
| `ccobs/fixedpoint.py` | cwnd / ssthresh / flight / SRTT / RTO 整数定点累积与量化误差上界 |
| `ccobs/ledger.py` | 观测账本: 批次切分、窗口封存、失效子集重算、E_SCRIPT / E_CWND 判定 |
| `ccobs/peer.py` | 127.0.0.1 高位端口确定性模拟对端(长度前缀 JSON 线协议) |
| `ccobs/session.py` | 客户端会话: 发报、收应答、逻辑时钟、生成确定性事件流 |
| `ccobs/snapshot.py` | SQLite 快照: 版本识别、内容摘要、强制终止恢复、分窗口整体替换差异 |
| `ccobs/replay.py` | 恢复后重放校验, 首个不一致定位到事件序号与观测项(E_REPLAY) |
| `ccobs/gui.py` | Tkinter 诊断界面; 视图模型为纯函数, 渲染顺序稳定 |
| `run_session.py` | 命令行入口: run / recover / replay / gui |

## 关键设计

### 确定性口径
- 逻辑单调时钟: ack/delay/reorder/dup_ack 应答推进脚本 `delay_ms`;
  loss 条目推进固定逻辑超时 200ms。时钟只依赖脚本内容, 与墙上时间无关。
- 批次边界只由 `(initial_seq, batch_size)` 决定; 同一脚本任意次执行产出
  逐字节一致的批次边界与应答归属(集成测试 `test_loopback_session_deterministic`
  实际跑两次完整回环会话逐字节比对)。

### 定点累积与量化误差上界
- 定点单位 `U = 10 ** scale_factor`, scale_factor ∈ [0, 6]。
- 慢启动每确认 cwnd += 1 MSS; 拥塞避免每确认 cwnd += MSS²/cwnd (截断,
  误差上界 +1 单位); 三次重复确认触发快速重传: ssthresh = max(cwnd/2, 2MSS),
  cwnd = ssthresh + 3MSS; 超时: ssthresh 减半, cwnd = 1MSS 回慢启动。
- SRTT/RTTVAR 按 RFC 6298 风格整数定点维护, RTO = SRTT + 4·RTTVAR;
  每次截断除法向对应观测项的误差上界累加 1 个定点单位,
  每个观测项登记其累积所依赖的批次区间与误差上界。

### 窗口封存与失效子集重算
- 窗口 = 连续 `window_batches` 个批; 批 = 连续 `batch_size` 个事件。
- 新批到达时, 按事件类别与观测项依赖关系(ITEM_EVENT_KINDS)计算失效子集,
  仅重算当前未封存窗口内受影响的观测项; 不受影响项的记录逐字节不变
  (`test_only_affected_subset_recomputed`: 全 dup_ack 批次不触发 SRTT 重算)。
- 窗口封满即封存: 结论值与窗口版本号参与 digest, `verify_sealed()` 可逐项
  校验; 任何篡改返回首个被篡改窗口序号。

### 异常判定
- E_SCRIPT: 应答事件序号相对期望序号出现缺口, 报告 `gap_start` 与
  `missing`, 全部观测项自缺口点停止推进; 缺口前已封存窗口结论不变。
- E_CWND: 同一窗口内 dup_ack 计数超过声明上限 `dup_ack_limit`,
  报告 `over` 与 `limit`, CWND/SSTHRESH 在该窗口内停更, SRTT/RTO/FLIGHT
  等其他观测项的既有结论不受影响。

### 版本演进与整体替换
- `(config_digest, script_digest, scale_factor)` 任一变化 → 配置版本号单调
  递增; 旧版本已封存窗口标记为陈旧(stale_archive)不参与展示。
- 新一轮推进按窗口整体替换结论集, `diff_windows` 稳定输出
  added / removed / version_changed 三类窗口及其观测项清单。

### 快照与恢复
- SQLite 持久化: meta(配置版本、脚本摘要、缩放因子口径、完成标志)、
  events(仅完整批次覆盖的事件)、batches、windows、items、trace、anomalies,
  整体内容摘要存入 meta.content_digest。
- 强制终止(`--crash-after-batches N`)后重启: 仅依据完整批次记录恢复,
  未完成批次标记为未知并要求重跑, `exit_report_available=False`,
  不得声称能生成退出报告。
- 摘要或版本不一致 → 整体清空替换并以 E_SNAPSHOT 报告 stored/actual 摘要。

### 重放校验
- 按快照记录的事件序列重新推进账本, 与持久化的逐项轨迹
  `(event_seq, item, value)` 按序比对; 首个不一致以 E_REPLAY 报告
  event_seq / item / expected / actual。

## 实际验证结果(本机, Python 3.11)

- `python3 -m unittest discover -s tests` → 31 个测试全部通过。
- 正常样本: 4 批 2 窗口全部封存, 两次运行批次 digest 逐字节一致。
- `peer_script_gap.json`(gap skip=2) →
  `E_SCRIPT|script event sequence gap|gap_start=1010|missing=2`,
  缺口前窗口 0 的封存 digest 与正常运行逐字节相同。
- `peer_script_dupack.json`(窗口内 4 次 dup_ack, 上限 3) →
  `E_CWND|duplicate ack cluster over limit|limit=3|over=1|window=1`。
- `--crash-after-batches 2` → 恢复输出 `unknown_batches=2,3`,
  `exit_report=False`。
- 直接 UPDATE SQLite 篡改观测值 → 恢复报
  `E_SNAPSHOT|snapshot content digest mismatch|...`, 退出码 3, 数据整体替换。
- 脚本演进后重跑 → `VERSION|2|...` 且 `REPLACE|{"added":[],...,
  "version_changed":[window 0, window 1 ...]}`; `replay` → `REPLAY_OK|compared=80`。

## 复现命令

```bat
python run_session.py --config samples\session_config.json --script samples\peer_script.json --db artifacts\ccobs.db run
python run_session.py --config samples\session_config.json --script samples\peer_script.json --db artifacts\ccobs.db recover
python run_session.py --config samples\session_config.json --script samples\peer_script.json --db artifacts\ccobs.db replay
python run_session.py --config samples\session_config.json --script samples\peer_script.json --db artifacts\ccobs.db gui
```
