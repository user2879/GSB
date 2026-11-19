# 技术说明：数字乐谱演奏表达数据处理管线

## 1. 模块结构

| 模块 | 职责 |
| --- | --- |
| `score_pipeline/errors.py` | 六个稳定业务错误码与 `PipelineError`，错误码语义集中维护 |
| `score_pipeline/model.py` | 数据模型、确定性 JSON 编码（`sort_keys`+紧凑分隔）、双层 SHA-256 哈希、原子文件写 |
| `score_pipeline/generator.py` | 固定种子 `numpy.random.Generator(PCG64)` 生成多声部音符、速度关键点、力度区间、踏板区间与逐声部真值清单 |
| `score_pipeline/tempo.py` | 速度曲线校验与拍→毫秒分段线性插值（bpm 线性变化的解析积分） |
| `score_pipeline/engine.py` | 参数校验、音符/踏板渲染、力度包络、拍区间失效闭包、版本化增量演进、历史片段重算 |
| `score_pipeline/persistence.py` | 内容寻址分片、原子提交协议、版本戳/哈希校验、孤儿回收、快照新旧判定、受控退出标记 |
| `score_pipeline/injection.py` | 可序列化异常注入配置；`HardExit` 表达强制终止 |
| `score_pipeline/lifecycle.py` | 固定增量序列、工作进程、崩溃恢复续写、受控退出报告 |
| `score_pipeline/scenarios.py` | 六个错误码的端到端异常链路覆盖 |
| `score_pipeline/validator.py` | 纯标准库独立验证器，不导入引擎/存储的任何函数 |
| `score_pipeline/cli.py` | 工作进程 CLI；业务错误写 stderr JSON，强制终止走 `os._exit(137)` |
| `run_demo.py` | 一条命令的固定序列演示与双跑逐字节比对 |

## 2. 确定性设计

- 随机源唯一：`np.random.Generator(np.random.PCG64(seed))`，抽样顺序固定；不使用 `set` 迭代、墙钟、PID、主机名。
- 所有落盘 JSON 走 `canonical_bytes`：键排序、`(",", ":")` 分隔、`allow_nan=False`，因此文件字节与哈希不依赖字典遍历顺序，Windows/Linux 一致。
- 浮点毫秒统一 `round(x, 6)` 后再入哈希；验证器与引擎使用同一数学规格但独立实现。
- 双跑验收：`run_demo.py --repeat-check` 在两个独立工作区跑完整序列并比对报告 SHA-256。

## 3. 拍→毫秒映射

速度关键点 `(beat, bpm)` 按拍严格递增。段内 bpm 线性变化，毫秒增量取解析积分：

```
anchor[i+1] = anchor[i] + (b1-b0) * 60000 / ((bpm0+bpm1)/2)
beat_to_ms(beat) = anchor[i] + (beat-b0) * 60000 / ((bpm0 + bpm(beat))/2)
```

关键点之外按端点 bpm 外推。音符实际时长再乘演奏法因子
（legato=1.0、normal=0.95、staccato=0.5）；力度包络为
`velocity * 力度区间增益 * 演奏法力度因子`，力度区间内对 `gain→end_gain` 线性插值。

## 4. 失效闭包与增量演进

每次修改生成带 `version/parent_version/kind/changed_range` 的 `Operation`（全量表达快照，便于无内存重放）：

- **速度变更**：以新旧 `TempoMap` 的首个分歧拍为界，分歧拍起所有音符进入 `recompute`；之前的音符仅按新曲线平移绝对时刻，`duration_ms` 用平移后的起止时刻按同一演奏法因子重算，力度字段不动。
- **力度/踏板变更**：只把与变更拍区间**闭区间相交**的音符放入 `recompute`（边界拍语义与力度增益的 `beat == end_beat` 规则一致），闭包外音符逐音符完全不变。
- 引擎断言 `recompute ∪ shift == 全部音符` 且二者不相交，并断言实际重算 ID 集合与闭包清单完全相等，否则抛 `E_TIMELINE_HASH`。
- 增量路径物理上不调用 `full_rebuild`；`full_rebuild` 仅用于基线、恢复装载和验收核对。每个版本同时产出整段重建哈希并与增量哈希比对。
- 历史重算：`Engine.replay_fragment(history, version, shard_id)` 从操作日志重放得到任意历史分片，与当时内嵌哈希比对。

## 5. 持久化与提交协议

```
state/
  manifest.json       # 唯一版本戳：版本、分片 blob 映射、分片内嵌哈希、时间轴哈希
  operations.jsonl    # 已提交操作（manifest 替换后才追加）
  blobs/<sha256>.json # 内容寻址、写后不可变
  closures/v<N>.json  # 每次提交的失效闭包与实际重算集合
  orphans/            # 恢复时隔离的未提交文件（带原因前缀）
  tmp/                # 暂存清单
  CLOSED / exit_report.json
```

提交顺序：① 写分片 blob（`crash_after_blob` 中断点）→ ② 暂存 manifest 并 fsync
（`crash_after_journal` 中断点）→ ③ `os.replace` 原子替换 manifest（唯一版本推进点）
→ ④ 追加操作日志与闭包。任一步中断，manifest 版本戳都不前进；重启时：

- 只信任 manifest 引用的 blob，逐个校验「文件名整包哈希」与「分片内嵌哈希」；
- 重算时间轴哈希与 manifest 记录比对，不符抛 `E_TIMELINE_HASH`；
- `tmp/` 残留标 `tmp_uncommitted`、无引用 blob 标 `dangling_blob`，移入 `orphans/` 并记录；
- 版本戳必须严格 +1，否则 `E_VERSION_REGRESS`；快照整体替换要求候选版本更新且哈希自洽，否则 `E_SNAPSHOT_STALE`。

## 6. 受控退出 vs 强制终止

- 受控退出：写完整退出报告 + `CLOSED` 标记。
- 强制终止：CLI 捕获 `HardExit` 后调用 `os._exit(137)`，跳过所有缓冲与 `finally`，
  因此不可能产生退出报告；恢复报告显式写 `exit_report_status = "missing_forced_termination"`，
  绝不伪造成功。恢复后从最近稳定版本续写固定操作序列，最终时间轴与无中断参考线逐音符、逐哈希一致。
- `os._exit` 与 `os.replace`/`tempfile` 均为 Windows 与 POSIX 共有语义，未使用信号或容器。

## 7. 错误码映射

| 错误码 | 触发链路 |
| --- | --- |
| `E_BEAT_RANGE` | 音符时值 ≤ 0、拍区间越界、力度/踏板区间非法、速度首点不在 beat=0 |
| `E_VELOCITY_MONO` | 速度关键点拍号非严格递增（序列单调性冲突） |
| `E_PEDAL_OVERLAP` | 同声部踏板区间交叉 |
| `E_VERSION_REGRESS` | 提交版本戳不等于当前版本 +1 / 操作父版本不匹配 |
| `E_SNAPSHOT_STALE` | 候选快照版本落后或内容哈希不自洽 |
| `E_TIMELINE_HASH` | blob 损坏、闭包分区与实际音符集合不一致、时间轴哈希重算不符 |

业务码同时出现在 JSON 报告字段与 CLI stderr（退出码 2）；注入配置带自身 SHA-256 并随报告落盘。

## 8. 独立验证器

`validator.py` 只导入 `model` 的 IO/哈希工具，速度积分、力度增益、演奏法因子全部本地重算：

1. 逐声部逐音符复核 `start_ms/end_ms/duration_ms/velocity_envelope/dynamics_gain`，记录首个不一致音符；
2. 复核踏板毫秒区间、分片双层哈希、时间轴整体哈希、操作版本链与 manifest 版本戳；
3. 输出 `validation-report/1` JSON，可被外部系统直接消费。

## 9. Windows 落地说明

- 要求 Python ≥ 3.10，依赖仅 `numpy==2.2.6`（见 `requirements.txt`）。
- 路径全部使用 `pathlib`；原子写依赖同目录 `os.replace`（Windows 上等价于原子替换）。
- 强制终止模拟用 `os._exit(137)`，不依赖 SIGKILL；在真实 Windows 上用任务管理器结束
  `python.exe` 进程等价，恢复路径不变。
- 无网络、无音频设备、无外部素材、无 Docker。
