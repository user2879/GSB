# 实际验证记录

验证日期：2026-11-21  
环境：WSL2 Ubuntu / Python 3.11.0rc1 / NumPy 1.26.4；目标运行环境为 Windows 10+ 本地回环，代码仅使用跨平台标准库。

## 自动化测试

命令：

```bash
python3 -m unittest discover -s tests -v
```

结果：`Ran 16 tests ... OK`，耗时约 14 秒。

覆盖：

1. JSON 池/规则强校验：重复 ID、未知引用、低端口、越界权重、镜像跨界借用、自环。
2. 稳定哈希与桶范围。
3. 连接复用序列 0/1、达到复用上限后真实关闭、空闲阈值后回收。
4. 6 路并发借用：借出池 `peak_watermark=2 <= max_connections=2`、无不变式违规、至少 2 笔借用归还、至少一个请求记录排队等待。
5. 镜像目标不可达：4 个主请求全部 `ok`，4 个镜像均为 `E_TARGET`，日志按请求 ID 稳定排序且 phase 全为 `mirror`。
6. 镜像真实套接字超时：主请求 `ok`，镜像 `E_TIMEOUT`，耗时超过 60 ms 的注入超时。
7. SQLite 重放：逐请求字段篡改被定位为 `E_REPLAY`、请求 `l0`、字段 `destination`；独立镜像记录、借用和生命周期表也参与重放；强制摘要差异为 `E_SNAPSHOT`。
8. 崩溃恢复：1 个完整批次 + 1 个 inflight 批次恢复后，inflight 行不进入已提交集合；1 个活跃连接和 1 笔借用被标记 `unknown`，退出报告不可用，续跑游标为 3。
9. 断点续跑：只重跑游标 3–4，最终两个完整批次共 4 请求重放通过。
10. 版本替换：版本 1 -> 2，旧 epoch 标记 stale；2 笔活跃借用按 `force_return` 逐笔裁决；规则/池变化与去向分布输出。
11. 定点误差模型：反转请求顺序报告逐字节一致；等水位差落入误差界时稳定池固定决胜；NumPy 辅助矩阵可用。
12. CLI 端到端：3 批 8 请求执行与重放；`E_BORROW`、`E_TARGET`、`E_SNAPSHOT` 注入均返回预期业务码并写结构化日志。
13. 版本替换后继续 `--resume`：旧版本批次保留为 stale，活动 epoch 只允许当前版本，跨 epoch 游标连续重放。

## 本地确定性样例

命令：

```bash
python3 tools/target_stable_gray.py --delay-ms 20
python3 tools/target_mirror.py --delay-ms 35
python3 -m gray_arbiter run \
  --config samples/config.json \
  --requests samples/requests.ndjson \
  --ledger state/sample-verified.sqlite \
  --snapshots snapshots \
  --log logs/sample.ndjson \
  --reset-ledger --batch-size 12 --workers 16
python3 -m gray_arbiter replay --ledger state/sample-verified.sqlite --log logs/sample.ndjson
```

最终复核摘要（2026-11-21，另建临时账本运行，避免覆盖既有样本产物）：

```text
processed_requests: 36
committed_batches: 3
main_status: {"ok": 36}
mirror_status: {"ok": 10}
mirror_records: 10
borrows: 3
last_batch_peak_watermarks: {"stable": 2, "gray": 3, "mirror": 1}
invariant_violations: 0
replay: {"status":"ok","batches_checked":3,"requests_checked":36,"next_cursor":37}
exit_report_available: true
sqlite_mirror_rows: 10
ledger_format_version: 1
```

说明：镜像采样结果由请求 ID 与规则 ID 的稳定哈希决定，因此固定为 `ok=10, not_sampled=26`；36 个主请求全部成功，`mirror_ledger` 与快照中均写入 10 条独立镜像记录。借用笔数和瞬时峰值取决于并发线程取得连接的实际重叠关系，并非固定值；本次为 3 笔，灰度峰值 3（自身容量 2 + 1 个借入），稳定池已承诺水位峰值 2，均未越界。验收时以 `invariant_violations=0`、借用/排队测试和重放结果为准，不要求某个固定借用笔数。正常样例没有业务错误，因此 NDJSON 日志为空；异常日志由下方注入场景实际产生。

## 异常注入实测

命令（稳定池端点被命令行覆盖为已关闭的高位回环端口）：

```bash
python3 -m gray_arbiter run \
  --config samples/config.json \
  --requests samples/requests.ndjson \
  --ledger state/sample-fault.sqlite \
  --snapshots snapshots \
  --log logs/sample-fault.ndjson \
  --reset-ledger --batch-size 12 --workers 16 \
  --inject-target-unreachable stable
```

结果：

```text
exit_code: 3
main_status: ok=30, E_TARGET=6
structured_log_codes: E_TARGET=7 (1 条注入声明 + 6 条请求错误)
first_event: {"business_code":"E_TARGET","request_id":"*","pool_id":"stable","phase":"fault_injection","sequence":1}
```

稳定池的 6 条主请求按真实 `ConnectionRefusedError` 记为 `E_TARGET`；灰度池请求和独立镜像执行未被错误地改写。日志中包含 `business_code`、`request_id`、`pool_id`、`phase` 与 `sequence`。

快照与账本均落于本地，未访问任何外部地址。Tkinter 代码已做导入惰性化；当前 WSL 环境未安装 Tkinter，因此界面窗口未在本环境实际弹出，Windows Python 自带 Tkinter 时通过 `python -m gray_arbiter ui ...` 启动。
