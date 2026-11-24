# 实际验证记录

## 环境

| 项目 | 实际值 |
| --- | --- |
| 执行时间 | 2026-11-25（Asia/Shanghai） |
| 平台 | Linux WSL2，`6.18.33.2-microsoft-standard-WSL2` |
| Python | `3.11.0rc1` |
| numpy | `1.26.4` |
| 标准库 | `sqlite3`、`socket`、`subprocess`、`json`、`threading` 可用 |
| Tkinter | 当前容器缺少 `libtk8.6.so`，无法启动窗口；无界面诊断模型已测试，Tkinter 代码采用延迟导入，目标 Windows 环境需按 README 自检 |
| 网络 | 仅使用 `127.0.0.1` 高位端口，无外部访问 |

本工作区不是 Windows 主机，因此未声称在 Windows 图形会话中实际打开窗口。组件代码不依赖 POSIX 专属路径或容器；Windows 强制终止分支使用 `TerminateProcess`，优雅路径使用可处理 `SIGINT`/`SIGTERM` 处理器。

## 自动化测试

命令：

```text
python -m unittest discover -v
```

结果：

```text
Ran 40 tests in 4.570s
OK
```

覆盖范围：

- 证书 ID 唯一、签发者存在、自环/互环、序列号/模数正整数、时间窗、算法集合、JSON 重复键。
- 消息序号连续、客户端/服务端计数、到达时间单调、回执/超时/端口错误。
- 单证书误差、累计误差、链级总界、边界不可区分、候选链稳定决胜。
- 增量与全量逐字段一致，未失效证书/消息逐字节一致，固定点参数和 issuer 图传播。
- 账本整数版本、内容摘要、记录摘要、快照整体替换、未知游标。
- 完整回环运行、完整恢复、优雅信号、强制终止、E_REPLAY 首差异。
- 诊断模型稳定性和只读账本验证。

固定种子样本连续生成两次的 SHA-256 完全一致：

```text
cf895e4200e56e99bcaec9b325e3e52faee8607674758e6ef6d126672c17336c  samples/certificates.json
e590e2a5d8f8cbeb8558e59ea4b8e81ac124ba3df7a9507fb775f960d10b1772  samples/handshake.json
920fc21d4ae96dae6c977816bdabadaec6d586fe2c774a022905281b1e9040c7  samples/certificates_modified.json
```

## 一键演示

命令：

```text
python tools/demo.py
```

关键结果（来自 `artifacts/demo/demo-summary.json`）：

```json
{
  "ok": true,
  "full_run": {
    "status": "COMPLETE",
    "progress_cursor": 8,
    "decision_counts": {"PASS": 5, "FAIL": 1, "INDETERMINATE": 2},
    "batches": [2, 4, 6, 8]
  },
  "incremental": {
    "comparison_equal": true,
    "invalidated_certificate_ids": ["leaf-expired"],
    "invalidated_message_seqs": [4, 5, 6, 7, 8],
    "retained_evidence": {
      "certificate_bytes_unchanged": true,
      "message_bytes_unchanged": true,
      "retained_message_seqs": [1, 2, 3]
    }
  },
  "graceful": {
    "status": "GRACEFUL_EXIT",
    "signal": "SIGINT",
    "progress_cursor": 4,
    "remaining_message_seqs": [5, 6, 7, 8],
    "exit_report": true
  },
  "hard": {
    "returncode": -9,
    "exit_report": false,
    "run_report": false
  },
  "hard_recovery": {
    "status": "COMPLETE",
    "initial_unknown_seqs": [5, 6, 7, 8],
    "progress_cursor": 8
  },
  "diagnostics_refresh_byte_identical": true
}
```

## 样本判定

| 消息 | 场景 | 定点结论 | 受误差影响 |
| --- | --- | --- | --- |
| 1–3 | 正常链 | `PASS` | 否 |
| 4 | 过期叶子证书 | `FAIL` | 否 |
| 5 | 到达时刻恰好落在有效窗口起点 | `INDETERMINATE` | 是 |
| 6 | 两条同分候选链 | `INDETERMINATE`，稳定选 `leaf-b-1` 链 | 是 |
| 7–8 | 正常链 | `PASS` | 否 |

计数：`PASS=5`、`FAIL=1`、`INDETERMINATE=2`，脚本期望差异为空。

## 定点误差归因实测

样本每条链长度 3：

```text
modulus_quantization = 6
time_window          = 6
path_length          = 1
chain_total          = 13
```

消息 6 两条候选链的同长度合成不可区分阈值为：

```text
13 + 13 = 26
```

8 条消息的诊断模型三段汇总为：

```text
modulus_quantization = 48
time_window          = 48
path_length          = 8
total                = 104
```

## 通信与恢复实测

- 占用的回环端口：`E_PORT`，字段 `socket.port`。
- 丢弃客户端序号 3 的回执：真实套接字在 750 ms 后触发 `E_TIMEOUT`，消息序号 5。
- 删除回执 `seq` 字段：`E_REPLY`，消息序号 2，字段 `reply.seq`。
- 优雅信号于批次 2 后停止：游标 4，退出报告存在；恢复先重放 1–4，再重跑 5–8。
- 强制终止于批次 2 后：SIGKILL 返回码 `-9`，无退出报告/运行报告；恢复标记未知 5–8 并完成至游标 8。
- 人为篡改已提交消息 4 的判定并同步重算其记录摘要：恢复在服务启动前以 `E_REPLAY` 报告消息 4、`leaf-expired`、字段 `$.messages[4].decision`。
- 诊断模型连续两次刷新输出逐字节一致；账本内容摘要与批次计数均未改变。

## 已知边界

- toy 签名只用于确定性和定点误差演示，不具备真实 TLS/PKI 安全属性。
- 握手服务是单连接确定性模拟器，不是通用网络服务。
- 当前容器没有 Tk 运行库；GUI 的视图模型已自动化验证，窗口渲染需在 Windows Python 3.11 图形会话中按 `python -m tlschain.cli gui ...` 复核。
