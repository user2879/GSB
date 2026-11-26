# VDU 定点语音对话情感流水线 — 技术说明

## 1. 范围与约束
- 仅依赖 Python 3.11 标准库（`wave`、`struct`、`json`、`hashlib`、`threading`、
  `queue`、`signal`、`argparse` 等）与 NumPy（合成阶段可选，主流水线不使用）。
- 全程整数定点运算，禁止浮点累乘与全局随机状态；重复运行逐元素一致。
- 样本、转写、配置均由确定性脚本生成，不访问网络或外部数据。

## 2. 目录与模块
- `vdu/fixedpoint.py`：定点 Tier（coarse/fine/reference）、定点乘加、重定标与残差。
- `vdu/synth.py`：256 项整数正弦表合成 16kHz/单声道/16bit PCM 与逐字转写。
- `vdu/pcmio.py`、`vdu/loading.py`：PCM 读取与 RIFF/转写/配置强校验。
- `vdu/features.py`：帧/窗切分，跨区间引用与逐区间有效帧计数，六维窗特征。
- `vdu/evidence.py`：字到窗映射、覆盖帧加权声学证据、整数共现语义证据与残差。
- `vdu/calibration.py`：逐轮定点校准，整数衰减累乘，空轮继承标记。
- `vdu/aggregation.py`：区间置信度、阈值分档、边界标记与重叠归属裁决。
- `vdu/fingerprint.py`、`vdu/ledger.py`：内容指纹、缓存键与可复核账本。
- `vdu/concurrency.py`、`vdu/scheduler.py`：有界线程池与预算/配额/待办调度。
- `vdu/error_track.py`：高一位宽参考路径四阶段误差追踪与标签翻转归因。
- `vdu/snapshot.py`、`vdu/pipeline.py`：暂存目录 + 原子整体替换 + 续跑/重建。
- `scripts/control_vdu.sh`：仅用 `kill` 与整数信号的进程控制。

## 3. 需求逐条落点
1. 校验层：`validate_wav` 校验 RIFF、采样率、声道、位深与 data 长度，违规抛
   `E_AUDIO_FORMAT`；`validate_transcript` 校验受控字表、偏移有序不越界、区间
   （start,end,speaker）唯一、标签版本单调不减，违规抛 `E_INPUT_INVALID` 并带
   文件与行号（`find_line` 定位，JSON 语法错误使用 JSONDecodeError.lineno）。
2. 窗特征：`build_windows` 外层按帧序号升序、内层按特征维升序累加；跨边界窗记录
   `segment_refs`（逐区间有效帧数），窗 ID 与帧索引稳定，无随机/默认归约顺序。
3. 字证据：`windows_for_word` 以采样偏移映射覆盖窗与覆盖帧数；声学证据按覆盖帧
   加权，语义证据由共现窗口内极性桶整数计数合成；两源在定点域按维升序融合，并在
   `quantize_word_evidence` 记录逐维量化步长与残差；切换 Tier 改变中间整数与残差。
4. 逐轮校准：`calibrate` 按轮次前向推进，固定运算顺序
   `clamp(decay(prev) + gain*evidence)`；衰减 `decay_once` 仅整数乘除并按轮次间隔
   整数累乘，无浮点幂；空证据轮继承上轮并置 `inherited=true`、`gap`，不跳轮。
5. 区间汇总：`aggregate_segments` 定点求均值得置信度，`classify` 在等于阈值时取
   高档、半步内置边界标记；`arbitrate` 对重叠区间按置信度、起点、说话人固定优先级
   裁决，记录 `covered_by` 与 `suppression_reason`。
6. 指纹与账本：四级缓存键由上游内容指纹、配置指纹、量化档位、标签版本指纹派生；
   每条区间结论记录字项集合与窗集合。整体替换标签版本时通过 `label_fingerprint`
   识别差异：声学窗仅依赖音频指纹因此全部复用，语义邻域变化的字证据及其下游失效，
   其余命中复用，账本 `audit` 逐条记录 reused/valid 与原因。
7. 并发调度：`BoundedThreadPool` 真实并发不超过 `max_workers`；`BatchScheduler`
   按起点/说话人稳定顺序消耗预算，超配额置 `E_EVIDENCE_QUOTA_EXCEEDED`，预算不足
   保留待办并置 `E_EVIDENCE_BUDGET_PENDING`，下一批优先消费待办。
8. 误差追踪与快照：`reference` Tier 高一位宽且不做中间量化作为参考，四阶段输出逐
   区间差值与首个超容差环节；标签翻转按阈值比较/衰减累乘/量化舍入归因。每批先写
   `tmp` 暂存再符号链接整体替换；启动以版本号与配置哈希决定 resume/rebuild；
   SIGTERM 协同退出并写 `exit_report.json`；SIGKILL 后仅依据持久化快照续跑。

## 4. 定点与确定性
- 定点原始整数 `raw = real * 2**frac_bits`；乘法后按 frac_bits 确定性四舍五入。
- 能量在特征层整数归一化到有界区间，保证不超过 Tier 位宽。
- 所有累加使用显式 Python int 循环，不依赖 NumPy 归约顺序或随机种子。

## 5. 运行与测试
```
python3 -m vdu generate           # 生成 samples/ 与 configs/
python3 -m vdu run --tier coarse  # 全量/增量运行
python3 -m unittest               # 运行前清空 snapshots/
bash scripts/control_vdu.sh start|term|kill|wait|resume
```
