# 丢包与网络故障诊断工作台（netdiag）

面向本地 Windows 环境的确定性丢包/网络故障诊断工作台，聚焦**多版本观测格式互操作**
与**排查结论的版本兼容**。仅依赖 Python 3.10+ 标准库与 NumPy，无外部网络服务、
无数据集、无容器；回环复现只绑定 `127.0.0.1`。

## 环境要求

- Windows 上安装 Python 3.10+，命令行可调用 `python`；Tkinter 随官方安装包提供。
- 安装 NumPy：`python -m pip install -r requirements.txt`。
- 运行前确认回环端口未被占用；演示/重放端口可用 `--port` 调整。

## 目录结构

```
netdiag/
├── netdiag/
│   ├── errors.py          # E_WIRE/E_FORMAT/E_MERGE/E_REPLAY 结构化错误
│   ├── constants.py       # 版本、标志位、定点比率与成因标签
│   ├── wire.py            # 自定义紧凑二进制线格式编解码
│   ├── eventlog.py        # 事件清单 v1/v2/v3 读取与归一
│   ├── model.py           # 统一内部观测模型 + 仲裁留存
│   ├── topology.py        # 链路拓扑与流-链路映射
│   ├── merge.py           # 两类来源合并、显式仲裁、E_MERGE 冲突
│   ├── gaps.py            # 序号缺口分析与重传/乱序/待定归因
│   ├── statistics.py      # NumPy 时间窗聚合、定点误差界、稳定决胜
│   ├── replay.py          # 回环 TCP 对端重放与 E_REPLAY 校验
│   ├── snapshot.py        # 会话快照版本识别与整体替换恢复
│   ├── ledger.py          # 追加式账本与历史结论重放校验
│   ├── pipeline.py        # 端到端流水线与会话状态
│   ├── samples.py         # 确定性样本生成器
│   ├── logging_utils.py   # 结构化 JSON 行日志
│   ├── cli.py             # 命令行入口
│   └── gui.py             # 可选 Tkinter 图形外壳（惰性导入）
├── tests/                 # unittest 测试（24 个）
├── scripts/demo.py        # 端到端演示
├── requirements.txt
└── output/                # 样本、快照、账本与日志输出目录
```

## 线格式（自定义紧凑二进制帧）

单帧 = 36 字节头 + 负载 + 4 字节 CRC32，全部大端字段：

| 偏移 | 长度 | 字段 |
| --- | --- | --- |
| 0 | 4 | 魔数 `NDF1` |
| 4 | 8 | 时间戳（微秒，uint64） |
| 12 | 4 / 4 | 源 / 目的 IPv4 |
| 20 | 2 / 2 | 源 / 目的端口 |
| 24 | 1 | 协议（6 TCP / 17 UDP） |
| 25 | 1 | 标志位 SYN/ACK/FIN/RST/RETX（保留位须为 0） |
| 26 | 4 | 32 位序号 |
| 30 | 2 | 负载长度（上限 1400） |
| 32 | 4 | `flow_tag`：规范化五元组 CRC32，校验序号空间归属 |

- 帧截断、负载长度越界、CRC 失败、标志位非法组合、`flow_tag` 与五元组不符，
  一律以 `E_WIRE` 报告，并在 `context.frame_offset` 给出**首个出错帧的文件偏移**。
- 解码结果按流（方向不敏感键）重组，方向键区分双向序号空间。

## 事件清单 v1/v2/v3 与归一

- v1：仅 `event_id` + `loss_count`，流归属在信封顶层 `flow`。
- v2：事件增加 `link_id` 与毫秒时间窗 `window_ms`、事件级 `flow/direction`。
- v3：再增加布尔 `retransmitted` 与整数 `ooo_depth`。
- 未知版本、必需字段缺失、负计数、非法时间窗等整体拒绝为 `E_FORMAT`，
  上下文 `field_path` 定位到如 `events[3].loss_count`；归一事件保留
  `source_version` 供追溯。

## 合并与仲裁

- 仲裁规则版本化（`ARBITRATION_RULE_VERSION=1.0.0`），默认 `wire_preferred`，
  可选 `event_preferred`。
- 同一流两侧都出现时逐事件记录 `OverrideRecord`：获胜方、被覆盖方的原始
  时间窗与计数、采纳值；逐流统计来源构成与仲裁次数。
- 两侧对同一方向的序号覆盖区间不一致（不相交或边界不同）以 `E_MERGE`
  报告冲突区间；v1 不携带覆盖信息，不参与该检查。

## 缺口分析与归因

- **顺序无关**：入口按 `(direction, timestamp_us, file_offset, flags, seq)`
  确定性全序排序，输入帧任意打乱都得到相同结论（有专门的 shuffle 测试）。
- 三类成因：`true_loss`（闭窗未填充）、`retransmit_fill`（RETX 回填）、
  `ooo_late`（无 RETX 晚到帧，乱序深度与 v3 声明相容）。
- 填充语义冲突（同时 RETX 与非 RETX、深度超声明）标记 `undetermined`，
  相邻同因序号合并为缺口区间。

## 丢包率定点统计

- 整数毫秒网格窗口，窗口起点对齐到 `window_ms` 整数倍，可重复。
- 比率 `rate_bps = floor(1_000_000 * lost / expected)`，单位 BPS（1e-6），
  舍入方向恒为 `floor`，单窗量化误差上界 `ceil(scale/expected)` 随结果输出。
- 跨流比较的合成误差界为两侧误差上界之和；差值落在界内时按
  “比率降序、flow 字典序升序”稳定决胜，并置 `error_bound_decisive=true`。
- 期望样本数低于 `min_samples` 时置 `insufficient_samples=true`。

## 回环重放、快照与账本

- `replay.py` 在同进程启动回环 TCP 对端线程，长度前缀 + 线格式帧传输，
  对端回传观测序号集合；逐流逐区间比较边界与成因，首个不一致以 `E_REPLAY`
  报告流、方向、区间、两侧标签。`--tamper` 可注入真实丢包序号演示该错误。
- 快照持久化模型版本、事件版本集合、仲裁规则版本、拓扑、缺口结论与统计；
  重开时版本不一致即**整体替换**（`replaced=true`，绝不增量合并），
  恢复后可 `append_capture` 继续追加新样本并重算。
- 账本为追加式 JSON Lines，记录输入 SHA256 摘要、版本组合、缺口与统计结论；
  `verify-ledger` 重新解码/合并/归因并逐项比对，首个差异定位到流、区间与字段，
  输入缺失或哈希不符以 `E_REPLAY` 报明依据。日志为结构化 JSON 行。

## 常用命令

```bash
python -m netdiag.cli generate --outdir output/sample
python -m netdiag.cli diagnose --wire output/sample/capture.bin \
  --events output/sample/events_v1.json output/sample/events_v2.json \
  output/sample/events_v3.json --window-ms 100 --min-samples 4 \
  --snapshot output/run/snapshot.json --ledger output/run/ledger.jsonl \
  --log output/run/run.jsonl
python -m netdiag.cli replay --wire output/sample/capture.bin \
  --events output/sample/events_v3.json --port 41793
python -m netdiag.cli snapshot output/run/snapshot.json \
  --append-capture output/sample/capture.bin
python -m netdiag.cli verify-ledger output/run/ledger.jsonl --workspace .
python scripts/demo.py --port 41793
python -m unittest discover -s tests -v
```

## 实际验证结果

在 Python 3.11 + NumPy 2.4.6（Linux 沙箱等价验证；代码仅用跨平台标准库，
Windows 同版本 Python 行为一致）下：

- `python -m unittest discover -s tests`：**24 个测试全部通过**，覆盖
  线格式各类 E_WIRE、三版本 E_FORMAT、仲裁留存与 E_MERGE、顺序无关归因、
  待定分类、定点误差与稳定决胜、回环匹配/篡改 E_REPLAY、快照整体替换、
  账本往返与摘要篡改。
- `scripts/demo.py`：确定性样本产出 4 个缺口——
  `5-6 true_loss`、`7-8 retransmit_fill`、`9-10 ooo_late`、
  Flow B `102-104 true_loss`；回环 14 帧发/收一致、4 缺口匹配；
  快照恢复、追加样本、账本重放校验全部通过。
- 错误码 CLI 实测：截断样本返回 `E_WIRE` 且带 `frame_offset=0`；
  负计数清单返回 `E_FORMAT` 且带 `field_path=events[0].loss_count`。
