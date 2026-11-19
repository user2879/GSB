# 本地流媒体自适应缓冲仿真与恢复一致性引擎

本项目实现了视频点播客户端的确定性缓冲演进、自适应码率、事件语义、阶段化快照、中断恢复、篡改检测和并发预取验证。运行时只依赖 Python 3.11、NumPy 和标准库，不访问网络或外部服务。

## 快速运行

```bash
python3 -m streamsim generate-samples samples --clean
python3 -m streamsim run samples/session.json
python3 -m streamsim run samples/session_interrupt.json   # 在终止事件处退出码 75
python3 -m streamsim run samples/session_interrupt.json   # 从有效快照续跑
python3 -m streamsim verify-snapshots samples/session_interrupt.json
python3 -m unittest discover -s tests -v
python3 scripts/run_verification.py
```

`run` 将完整 JSON 报告写到 stdout，并原子写入 `state_dir/report.json`；逐事件日志写入 `state_dir/simulation.log.jsonl`。`generate-samples` 使用固定种子 `20261119` 生成本地网络轨迹、媒体清单、事件脚本和三个会话。会话中的 `state_dir` / `snapshot_dir` 是相对会话 JSON 所在目录解析的。

## 目录

- `streamsim/model.py`：会话、媒体、轨迹、事件解析和严格校验。
- `streamsim/engine.py`：整数毫秒事件时钟、播放缓冲状态机、ABR 和事件语义。
- `streamsim/snapshot.py`：版本化快照、原子写入、摘要校验、归档和回退。
- `streamsim/prefetch.py`：有界线程池、内存预算和首错取消。
- `streamsim/samples.py`：固定种子样本生成器。
- `streamsim/__main__.py`：CLI。
- `tests/`：16 个自动化测试。
- `scripts/run_verification.py`：端到端验收并生成 `artifacts/verification/`。

## 会话格式

```json
{
  "network_trace": "network.json",
  "media_manifest": "media.json",
  "event_script": "events.json",
  "client": {
    "startup_buffer_ms": 4000,
    "max_buffer_ms": 10000,
    "safety_margin_ms": 1200,
    "strategy": "hybrid",
    "switch_penalty": 0.25,
    "concurrency": 2,
    "memory_budget_bytes": 12000000,
    "throughput_safety_ratio": 0.88
  },
  "state_dir": "sample_runs/full/state",
  "snapshot_dir": "sample_runs/full/snapshots"
}
```

媒体清单中的每个媒体片段包含整数毫秒时长和同媒体时间的多个候选表示。每个表示给出 `bitrate_bps`、`size_bytes`。也兼容单表示分片：在片段对象上直接写 `bitrate_bps` 和 `size_bytes`。

```json
{
  "fragments": [
    {
      "duration_ms": 2000,
      "representations": [
        {"bitrate_bps": 400000, "size_bytes": 98000},
        {"bitrate_bps": 1100000, "size_bytes": 269500}
      ]
    }
  ]
}
```

网络轨迹按游标选择，每个轨迹段给出正数 `bandwidth_bps` 和非负整数 `rtt_ms`。事件脚本按 `(at_ms, order, 原始序号)` 排序。

```json
{"events": [
  {"at_ms": 8000, "action": "pause"},
  {"at_ms": 10000, "action": "resume"},
  {"at_ms": 14000, "action": "trace_switch", "segment_index": 3},
  {"at_ms": 22000, "action": "jump", "target_ms": 6000},
  {"at_ms": 15000, "action": "terminate"}
]}
```

支持 `pause`、`resume`、`jump`/`seek`、`trace_switch`/`network_switch`、`terminate`/`signal`/`interrupt`。同一毫秒的事件按声明顺序处理。

## 时间与确定性

- 网络带宽单位为 bit/s，分片大小单位为 byte。下载时长唯一采用向上取整整数规则：`ceil(size_bytes * 8000 / bandwidth_bps)`，完成时刻为 `start_ms + duration_ms`。
- 输入数字经 `Decimal`/`Fraction` 转为精确有理数；决策、完成时刻、缓冲水位和媒体进度全部用整数或 `Fraction`，不以浮点累加决定结果。浮点只出现在 JSON 输出。
- 事件时钟严格为整数毫秒。同一毫秒内部优先级固定为：下载完成、播放到达结尾、缓冲排空、脚本事件；全部同毫秒脚本事件处理完后才自动恢复播放或启动下一次下载。该规则保证暂停/跳转/轨迹切换与同刻完成事件的顺序可复现。
- 播放时每经过 1 ms 消耗 1 ms 媒体缓冲；暂停时预算不消耗，但在途下载继续并可在结束后增加缓冲。恢复播放保持暂停时刻的 `playhead_ms`。
- 跳转时，若新位置仍位于已下载连续缓冲区间内，则保留 `[target, buffer_end]` 并按缓冲尾部选择下一次分片；若跳到区间外或向后跳，清空旧区间并从包含新位置的分片重新下载。所有在途下载取消。
- `trace_switch` 只更新后续下载游标；在途下载继续使用启动时冻结的轨迹、带宽和完成时刻。

## 自适应码率

每次分片下载起点执行一次纯函数决策，只读取当前缓冲、当前轨迹段、历史已完成下载、当前媒体片段的候选表示和客户端参数：

1. 吞吐估计取最近 3 个已完成分片按精确 `Σbytes*8000 / Σduration_ms` 计算的平均吞吐，并与当前轨迹带宽取较小值；无历史时使用当前轨迹带宽。
2. `throughput_safety_ratio` 给吞吐估计留出余量。`hybrid` 还按安全余量与启动目标对低缓冲实施码率上限，`buffer` 使用更保守的分层上限。
3. 若当前缓冲加本片时长接近最大缓冲上限，则相对上次码率下降一级，落实“缓冲触顶降码率”规则。
4. 切换罚分通过 `rate * (1 - switch_penalty)` 调整候选得分，降低无谓切换。
5. 安全校验估计当前片完成后的水位，再以候选码率、当前轨迹带宽和本片时长估算下一片下载耗时；选择不会在下一片完成前跌至安全余量以下的候选。若物理带宽使所有候选均不可能安全，则仍选择最低码率并记录 `safety_ok=false` 和原因，后续排空会被显式计数。
6. 决策不读取下一片段大小、未来轨迹段或未来事件。最终平局按码率、大小和序号确定。

## 快照与恢复

每个下载完成、暂停、恢复、跳转、轨迹切换、排空、结束和中断边界都会生成阶段快照。快照包含：

- `now_ms`、`playhead_ms`、`buffer_start_ms`、`buffer_end_ms`、`buffer_ms`；
- 当前下载序号、网络轨迹游标、上次码率；
- 已下载分片摘要、完整决策计划、在途下载的绝对完成时刻和已下载进度；
- 事件游标、事件序号、切换次数、排空次数和逐事件日志。

版本标识是以下内容的规范化 SHA-256：排序后的媒体片段/表示、网络轨迹段、客户端参数。会话或目录路径不参与版本标识。

同一版本写入时，快照先写入不可变存档，再通过临时文件原子替换 `state_dir/snapshot.json`。版本变化、事件脚本变化时，旧状态先复制到 `snapshot_dir/stale/` 或 `snapshot_dir/quarantine/`，不会直接覆盖。恢复按事件序号和快照序号选择最新有效快照；当前快照无效时回退到存档，全部无效时从头重跑。

每份快照保存规范化内容摘要，并同时记录 `declared_event_seq`、实际状态事件序号、`declared_journal_length` 与日志实际长度。摘要缺失/不匹配、声明序号不符、日志不连续、版本或事件脚本摘要不符均判为 `E_DIGEST`/`E_SNAPSHOT` 失效。`verify-snapshots` 只读校验并返回 `E_DIGEST`。

SIGTERM/SIGINT 处理函数只设置标志；仿真在下一个完整事件边界写中断快照和 `state_dir/interrupted.json`，然后以退出码 `75` 退出，不承诺报告。再次运行同一版本时从该快照继续；版本不一致则归档旧状态并从头开始。

## 并发预取

确定性事件时间轴先生成分片决策计划；并发执行层只验证计划中的传输任务，不回写 ABR 或缓冲状态，因此并发数不会改变决策序列。执行层使用 `ThreadPoolExecutor` 和不超过并发数的批处理，单批任务字节总量不得超过 `memory_budget_bytes`；下一分片若放不下则按分片序号等待下一批。任一任务异常时设置取消标志、取消未完成任务，并按最小 `job_seq` 返回第一个 `E_PREFETCH`。

## 错误码与退出码

- `E_JSON`：JSON 语法错误。
- `E_INPUT`：通用字段或数值校验失败。
- `E_MEDIA` / `E_NETWORK` / `E_EVENT`：对应输入文件校验失败。
- `E_DIGEST` / `E_SNAPSHOT`：快照摘要、事件序号或恢复状态无效。
- `E_PREFETCH`：预取任务失败或资源限制不可满足。
- `E_IO`：本地文件读写失败。

退出码：`0` 成功，`2` 输入/业务错误，`3` I/O，`4` 快照校验失败，`5` 预取失败，`75` 可处理中断，`130` 键盘中断。非法输入在状态目录创建前完成校验，因此不会写出任何状态文件。

## 实际验证结果

运行 `python3 -m unittest discover -s tests -v`：16 个测试全部通过。运行 `python3 scripts/run_verification.py` 的实测摘要如下（原始报告在 `artifacts/verification/`）：

```json
{
  "environment": {"python": "3.11.0rc1", "numpy": "1.26.4"},
  "continuous": {
    "finished": true,
    "decisions": 31,
    "switches": 7,
    "drains": 0,
    "log_matches_report": true
  },
  "concurrency": {"invariant": true, "c1_peak_tasks": 1, "c2_peak_tasks": 2},
  "scripted_interrupt": {
    "exit": 75,
    "resume_mode": "resume",
    "resume_exact": true
  },
  "tamper": {
    "verify_exit": 4,
    "verify_code": "E_DIGEST",
    "resume_mode": "archive_fallback",
    "fallback_exact": true
  },
  "sigterm": {"exit": 75, "resume_exact": true},
  "version_change": {"mode": "version_reset", "stale_files": 1}
}
```

连续基线、脚本终止后续跑、SIGTERM 后续跑和篡改回退后的最终 `events`、码率决策、切换次数、排空次数及最终缓冲状态逐条一致。慢轨迹测试另外验证了缓冲排空计数和 `buffer_empty` 日志。
