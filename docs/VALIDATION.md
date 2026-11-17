# 实际验证记录

验证日期：2026-11-18（Asia/Shanghai）

## 1. 环境

| 项 | 实际值 | 结果 |
|---|---|---|
| 操作系统 | WSL2 Linux，内核 6.18.33.2-microsoft-standard-WSL2 | 可用 |
| Python | 3.11.0rc1 | 满足 `>=3.11` |
| tkinter | 当前 WSL 解释器未安装 | 界面模块可导入；Windows 官方 Python 需勾选 Tcl/Tk 后运行 |
| NumPy | 当前 WSL 解释器未安装，且无 pip | 运行核心不依赖；提供本地 wheel 离线安装脚本 |
| Docker/容器 | 未使用 | 满足约束 |
| 回环端口 | `18101..18104`、`19101/19102/19104` 预检空闲 | 可用 |

环境预检命令：

```bash
python3 scripts/check_env.py --config data/demo_routes.json
```

输出 `code: OK`；Tk/NumPy 项目如实标记为未安装且默认非致命。

## 2. 自动化测试

```bash
python3 -m unittest discover -s tests -v
```

实际结果：

```text
Ran 46 tests in 7.382s
OK
```

覆盖点：

- 配置强校验、字段定位、整体拒绝；
- 确定性封装、摘要损坏偏移、长度/序号错误；
- MTU 分片、乱序、精确重复、冲突重复、缺片超时；
- 固定点统计、账本单写、插入顺序无关摘要；
- 两级并发上限与两级令牌桶；
- 真实 TCP 对端上的正常、分片、`E_DIGEST`、`E_PEER`、`E_TIMEOUT`；
- 直接注入缺片后的 `E_FRAG` 与缺失偏移；
- 快照校验、整体替换、版本差异、未完成会话转 `UNKNOWN`；
- 严格重放一致与首个字段差异定位。

## 3. 总演示

```bash
python3 scripts/run_demo.py --state-dir artifacts/demo-verify
```

实际结果：

```text
overall code: OK
normal: OK, fragment_count=1
queue: 3 个并发会话全部 OK，均有排队（队列事件 22/43/49）；观测最大全局/隧道活动会话=2/2（限额 5/2）
queue bandwidth bounds: route=true, global=true
fragment: OK, 1111 bytes -> 7 fragments, 回送摘要与原负载一致
missing-fragment injection: E_FRAG, missing_offsets=[172]
business codes: corrupt=E_DIGEST, dead=E_PEER, slow=E_TIMEOUT
restore: whole_replaced=true
replay: OK, first_mismatch=null
replay ledger digest equal: true
restart: 8 sessions restored, 0 unknown (演示在所有会话结束后保存)
ledger duplicate write check: E_STATE
global stats: forwarded=25756 B, payload=25769 B, throughput=5.471 KB/s,
              average_queue=942.875 ms, queue_events=121
```

演示在 `artifacts/demo-verify/demo_result.json` 写入完整结果，并在同目录写引擎/对端 JSONL 日志与快照。

## 4. 业务码验收

| 场景 | 期望 | 实际 |
|---|---|---|
| 正常接收 | `OK` | `OK` |
| 延迟超过响应预算 | `E_TIMEOUT` | `E_TIMEOUT` |
| 目标端口不可达 | `E_PEER` | `E_PEER` |
| 对端摘要回送被翻转 | `E_DIGEST` | `E_DIGEST` |
| 缺片重组超时 | `E_FRAG` | `E_FRAG` + `missing_offsets` |
| 配置非法 | `E_CONFIG` | `E_CONFIG` + 字段路径 |
| 快照损坏 | `E_STATE` | `E_STATE` |
| 重放差异 | `E_REPLAY` | `E_REPLAY` + 首差异字段 |

## 5. 未在当前 WSL 直接执行的项目

- Tkinter 窗口未在当前环境弹出验证，因为发行版解释器缺少 `tkinter`；代码使用延迟导入，核心与测试不受影响。Windows Python 官方安装包勾选 Tcl/Tk 后运行：

  ```powershell
  py -3.11 -m tunnelvpn.ui --config data\demo_routes.json --state-dir artifacts\ui-state
  ```

- NumPy 未在当前解释器安装，因为环境无 pip 且要求离线 wheel。项目运行不导入 NumPy；将 wheel 放入 `wheels\` 后执行 `scripts\install_numpy_offline.ps1` 即可离线安装。
