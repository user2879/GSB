# 配置说明（本地 Windows / 跨平台）

## 运行环境
- Python 3.11 或更高版本，仅使用标准库（`tkinter`、`sqlite3`、`socket`、`json`、`argparse`、`unittest`）。
- 无第三方网络库、无容器、无真实账号或线上站点。
- Windows 官方 Python 安装包默认勾选 Tcl/Tk，自带 Tkinter；本仓库 GUI 采用懒加载，
  缺少 Tkinter 时仅 GUI 不可用，命令行、账本与测试不受影响。

## 路径清单（均为工作目录相对路径）
| 用途 | 路径 |
| --- | --- |
| 有效规则集（9 条规则，3 层依赖：BASE→CHILD→GRAND） | `samples/rules.json` |
| 增量变更集（修改 R1 根规则） | `samples/changeset_modify.json` |
| 删除叶子规则变更集 | `samples/changeset_delete_leaf.json` |
| 损坏规则文件（JSON 语法错误，注入 E_RULE） | `samples/rules_corrupt.json` |
| 字段非法样本（非回环地址/越界端口/负队列等） | `samples/rules_invalid_fields.json` |
| 依赖成环样本（注入 E_DEPEND） | `samples/rules_cycle.json` |
| 缺版本字段快照（注入 E_SNAP） | `samples/snapshot_missing_versions.json` |
| 回环流量发生器+整形器+接收端脚本 | `traffic_sim.py` |
| SQLite 账本（默认，运行后生成） | `data/qos_ledger.db`（WAL 模式，附 `-wal`/`-shm`） |
| 结构化 JSONL 日志（默认，运行后生成） | `logs/qos.jsonl` |
| 会话快照（运行后生成） | `data/snapshot.json` |
| 诊断报告（demo 生成） | `data/diagnostic_report.json` |

## 端口约定
- 仅绑定回环字面量 `127.0.0.1`，全部使用高位端口（示例 `49100–49599`）。
- 运行前由 `qos.shaper.is_port_free` / `traffic_sim.py` 探测端口占用，被占用时直接报错退出。
- `traffic_sim.py run` 使用就绪文件（`.sink.ready`/`.shaper.ready`）编排子进程，
  健康探测不会消耗服务端唯一的 `accept` 槽位。

## 规则 JSON 结构
```json
{
  "rule_id": "R1_BASE",
  "match": {"dst_addr": "127.0.0.1", "port_prefixes": [14]},
  "shaping": {"rate_kbps": 256, "burst_bytes": 8192, "queue_limit_packets": 8},
  "upstream": [],
  "actions": ["police"]
}
```
- `match.dst_addr` 仅允许 `127.0.0.1` 或 `::1`。
- `match.port_prefixes` 为 1..65535 的整数数组，匹配语义为目的端口十进制字符串的前缀匹配，
  编译与转发均采用“最长前缀优先，同长按规则标识决胜”。
- `rate_kbps`、`burst_bytes` 为正整数；`queue_limit_packets` 为非负整数（0 表示不按队列上限丢包）。
- `upstream` 为上游规则标识数组，构成有向无环图；引用不存在或成环分别以 `E_RULE` / `E_DEPEND` 拒绝。

## 常用命令
```bash
python -m qos load
python -m qos change --changeset samples/changeset_modify.json --verify
python -m qos verify-incr --changeset samples/changeset_modify.json
python -m qos simulate --count 30 --size 1024 --gap-us 100000
python -m qos crash-recover
python -m qos snapshot-create --changeset samples/changeset_modify.json --out data/snapshot.json
python -m qos snapshot-restore --snapshot data/snapshot.json
python -m qos replay --snapshot data/snapshot.json --changeset samples/changeset_modify.json
python -m qos inject --fault corrupt --rules samples/rules_corrupt.json
python -m qos inject --fault cycle   --rules samples/rules_cycle.json
python -m qos inject --fault snapshot --snapshot samples/snapshot_missing_versions.json
python traffic_sim.py run --plan data/current_plan.json --shaper-port 49401 --sink-port 49402
python demo.py
python -m unittest discover -s tests -v
```
Windows PowerShell 下将 `python3` 换为 `python` 即可，路径分隔符无需调整（代码统一使用 `pathlib`）。
