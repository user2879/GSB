# 实际验证记录

## 环境

- 执行日期：2026-11-25
- 工作区：`/home/zsl/feedback/uidfc5909d3-default-257`
- Python：CPython `3.11.0rc1`
- pytest：`8.2.0`
- NumPy：`1.26.4`
- 当前执行容器：Linux；实现仅使用跨平台标准库 API，UDP 接收端通过 `sys.executable -m tunnel_interop.udp_receiver` 子进程启动，Windows 下使用同样的 `Popen`/UDP/SQLite 路径。

## 自动测试

命令：

```bash
python -m compileall -q tunnel_interop tests
pytest -q
```

结果：`64 passed`。

覆盖范围：

- 声明首错字段、端点/端口唯一性、MTU/开销、路由引用、分片参数、版本单调性。
- f1/f2/f3 往返、等价的统一关键字段、补齐字段与依据。
- 未知版本、必需字段缺失、总片数为零、序号越界、偏移未对齐、载荷上限、负重传、校验码和头摘要错误。
- 端点/路由重排后样本、报文标识、片偏移和封包摘要逐字节稳定。
- 正常、反序、重复真实 UDP 到达后的逐字节重组，以及 f1/f2 真实回环转发。
- 真实 socket 超时、真实重组截止期、回执字段缺失。
- 全局/单隧道双配额、真实线程池、可复现调度摘要。
- 端点 MTU、路由优先级、分片档位、重传、重组超时变化的精确失效传播和未失效分片复用。
- SQLite intent 恢复、强制进程退出、账本摘要注入、快照整体替换、稳定 diff、重放一致和首处 `E_REPLAY` 定位。
- 三类异常注入分别产生 `E_SPEC`、`E_REF`、`E_SNAPSHOT`。

## 完整验收矩阵

命令：

```bash
python -m tunnel_interop verify \
  --spec examples/spec.json \
  --evolved-spec examples/spec_evolved.json \
  --output-dir artifacts/verification \
  --log-file artifacts/verification/verify.log \
  --out artifacts/verification/verify-summary.json
```

退出码：`0`。

关键结果：

- f1/f2/f3：8 个分片全部 `all_equal=true`。
- 调度探针连续两次输出逐字节一致的确定性统计。
- 正常场景：2 个消息批次、8 个真实回执，2 个重组摘要匹配，重放 8 片通过。
- 反序场景：2 个消息批次、8 个真实回执，到达序号为 `[3,2,1,0]`，重组匹配，重放通过。
- 重复场景：2 个消息批次、16 个真实回执，每片重复到达并去重，重组匹配，重放通过。
- 演进：4 个依赖 `tun-alpha` 的分片和 1 个重组结果失效；4 个 `tun-beta` 分片保持原封包摘要与版本号。
- 快照替换：generation `1 → 2`，稳定列出 `ep-edge-a.mtu`、`tun-alpha.priority` 变更，`replacement_mode=whole_snapshot`。

产物：

- `artifacts/verification/verify-summary.json`
- `artifacts/verification/verification.json`
- `artifacts/verification/format_interop.json`
- `artifacts/verification/scheduler-determinism.json`
- `artifacts/verification/udp-normal.json`
- `artifacts/verification/udp-reverse.json`
- `artifacts/verification/udp-duplicate.json`
- `artifacts/verification/evolution.json`
- `artifacts/verification/snapshot-replacement.json`
- `artifacts/verification/verify.log`
- `artifacts/verification/timeout.json`
- `artifacts/verification/missing-ack.json`
- `artifacts/verification/crash-resume.json`
- `artifacts/verification/crash-rerun.json`
- `artifacts/verification/inject-corrupt.json`
- `artifacts/verification/inject-dangling.json`
- `artifacts/verification/inject-snapshot.json`

## 业务错误码实测

| 注入/场景 | 命令 | 结果 |
|---|---|---|
| 声明损坏 | `inject --kind corrupt-spec` | `E_SPEC`，路径 `$`，退出码 2 |
| 下一跳悬空 | `inject --kind dangling-route` | `E_REF`，路径 `$.routes[0].next_hop`，退出码 2 |
| 账本摘要不符 | `inject --kind snapshot-digest` | `E_SNAPSHOT`，路径 `ledger.meta.content_digest`，退出码 4 |
| 真实丢回执 | `forward --drop-all` | `E_TIMEOUT`，路径 `reassembly.deadline`，退出码 3 |
| 回执缺字段 | `forward --ack-mode missing-field` | `E_FIELD`，路径 `ack.received_at_ms`，退出码 2 |
| 强制退出后恢复 | `forward --crash-after-batch 1`，再 `resume` | 第一批已提交并可重放；第二批转 `unknown`，未知分片 `[0,1,2,3]`，`can_claim_exit_report=false` |
| 未知批次重跑 | `resume --rerun` | 真实 UDP 重组成功，随后重放 4 片通过 |

## 复现注意事项

- `examples/spec.json` 使用固定端口 `46371` 和 `46372`。若本机占用，请复制声明并替换为未占用回环端口。
- 所有 UDP 目标均为 `127.0.0.1`，不访问外部网络。
- 时间相关字段（真实回执时刻、真实单调时钟审计轨迹）在不同运行间可变；判定与可复现摘要使用固定逻辑时钟、内容摘要和真实完成条件。
