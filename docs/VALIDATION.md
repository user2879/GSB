# 实际验证记录

验证时间：2026-11-26（Asia/Shanghai）。

## 环境

- OS：WSL Ubuntu 工作区，Linux 回环网络可用。
- Python：`Python 3.11.0rc1`。
- 虚拟环境：`.venv/`，由 `virtualenv` 创建（系统 `python3 -m venv` 最初缺少 `ensurepip`）。
- 已安装：`numpy 2.4.6`、`pytest 9.2.0`，并以可编辑模式安装本包。
- Tkinter：默认 `.venv` 初始不能导入，因为系统缺少 `python3-tk`/Tcl/Tk 运行库且当前会话无免密 sudo。使用 `apt-get download libtcl8.6 libtk8.6 libxft2 libxss1` 免 root 解包后，配合本机已有 `_tkinter` 扩展验证到 `TkVersion 8.6`。正式 WSL 推荐执行 `sudo apt-get install -y python3-tk`。

## 一键验证

```bash
bash scripts/setup_env.sh
./scripts/check_ports.sh
bash scripts/full_check.sh
```

最终结果：

- `pytest`：`29 passed in 5.69s`。
- `scripts/full_check.sh`：退出码 `0`，输出 `FULL_CHECK_OK`。
- 端口预检：`48101`、`48102`、`48103`、`48191` 均空闲。
- 验证后无残留 `lease_middleware.node_server` 子进程，上述端口无监听。

原始输出保存在：

- `artifacts/full-check.log`
- `artifacts/pytest-final.log`
- `artifacts/demo/comparison.json`
- `artifacts/crash/recovery_report.json`
- `artifacts/injections/*/injection_result.json`

## 主样本结果

命令：

```bash
.venv/bin/python -m lease_middleware demo --root .
.venv/bin/python -m lease_middleware verify \
  --sample-dir samples \
  --ledger artifacts/demo/order-forward/ledger.sqlite
```

实际结论：

- 正序与逆序节点启动的 `conclusion.json` 逐字节相同：`byte_identical=true`。
- 结论 SHA-256：`fc4ead29445c5f249a1ced77ab6175b61b824e3fd697f2f76d451f4f1237b2c7`。
- 账本重放：`verified_records=32`。
- 账本摘要：`sha256:647c584ee652e3f71e9fa953332a88cbda32b6ec1e32fef0f3724d54865958d0`。
- 租约总数 6，终态有效租约 2。
- 有效回收 3 次：TTL 回收 `lease-c1`，换代回收 `lease-b1`、`lease-b2`。
- 抑制重复回收 2 次：cache-b 场景抑制重复 TTL，cache-c 场景抑制重复换代作废。

调度器报告的稳定峰值：

- 全局在途峰值：3（等于 `global_in_flight`）。
- 节点峰值：`cache-a=1`、`cache-b=2`、`cache-c=1`，均未超过节点配额。
- 14 条真实网络请求均有单调时钟等待开始/结束记录；统计后端为 `numpy`。

## 强杀恢复

命令：

```bash
.venv/bin/python -m lease_middleware crash-demo --root .
```

实际结果：

- victim 子进程返回码：`-9`（`SIGKILL`）。
- 已完整提交的授予记录：2 条。
- 未提交的续约结果不可见。
- 待决意图 `intent-renew-lease-crash` 被恢复为未知租约 `lease-crash`。
- 故意把会话快照 `format_version` 改为 `999` 后，恢复器从 SQLite 整体替换快照：`snapshot_replaced=true`。

## 异常注入业务码

| 注入 | 命令 kind | 实际业务码 | 首个定位/附加信息 |
| --- | --- | --- | --- |
| 声明字段损坏 | `spec` | `E_SPEC` | `nodes[0].port` |
| 原始 JSON 截断 | `spec-json` | `E_SPEC` | 文件行列位置 |
| 续约携带旧标签 | `lease` | `E_LEASE` | `version_tag`，节点当前标签 `2` |
| 账本摘要不符 | `snapshot` | `E_SNAPSHOT` | `ledger_header.content_digest` |
| 账本业务内容与重放不符 | `verify` | `E_VERIFY` | `entries[0].at_ms` |
| 端口占用 | `port` | `E_PORT` | `port` |
| 握手不响应 | `timeout` | `E_TIMEOUT` | `socket.timeout` |
| 回执缺字段 | `reply` | `E_REPLY` | `reply.protocol_version` |

参数化测试 `tests/test_injection_codes.py` 固定断言前表中的 7 个主要注入码；`spec-json` 可通过 CLI 单独复现。

## 复核入口

```bash
# 只读取声明和账本进行独立重放
.venv/bin/python -m lease_middleware verify \
  --sample-dir samples \
  --ledger artifacts/demo/order-reverse/ledger.sqlite

# 查看机器可读比较结果
cat artifacts/demo/comparison.json
```
