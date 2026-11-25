# 实际验证记录

验证日期：2026-11-26（Asia/Shanghai）。

## 环境

- Python：`3.11.0rc1`，满足 Python 3.10+。
- NumPy：`1.26.4`。
- SQLite：`3.37.2`（标准库 `sqlite3`）。
- 平台：WSL Ubuntu / Linux loopback。
- Tkinter：Python 模块存在，但当前基础镜像缺少 `libtk8.6.so` 与 `libtcl8.6.so`；`python3 -m leasemesh tk` 实际返回 `E_IO` 和 `sudo apt-get install -y python3-tk` 提示。无头验收不依赖 Tk。

## 自动化测试

命令：

```bash
python3 -m unittest discover -s tests -v
```

实际结果：

```text
Ran 16 tests in 3.791s
OK
```

完整输出保存在 `artifacts/test-results-latest.txt`。

`python3 -m pytest -q` 的实际结果同样为 `16 passed in 3.81s`。

覆盖点：

- `E_SPEC`：非法续约间隔、重复端口、重复/倒退事件序号，并定位字段与行号。
- `E_PORT`：回环端口已占用。
- `E_TIMEOUT`：握手无响应、握手后读取回执超时。
- `E_FIELD`：回执缺少必填字段。
- `E_LEASE`：携带过期代数续约被拒，节点代数停留为 2。
- 调度器：全局上限 2、单节点上限 1 的真实阻塞线程测试。
- 归属裁决：两类回收竞争、唯一决策、20 组随机乱序后 canonical bytes 相同。
- 演进：失效集合为 `lease-a2`、`lease-c2`，未受影响记录字节证明全部为真。
- SQLite 账本：真实 3 节点子进程、13 个请求、独立重放、快照备份、篡改后 `E_VERIFY`。
- 强制终止：子进程返回 `-9`（shell 表现为 137），恢复出 1 个 unknown 请求和 1 个 reclaimed 配额预留，随后只读重放通过。

## 干净端到端命令结果

下列输出保存在 `artifacts/final-*.json`：

- `validate`：`ok=true`，3 个节点、13 条申请、11 个事件。
- `execute --allocate-ports`：13/13 成功；调度峰值为全局 3、每节点 2。
- `verify`：`checked_commits=26`、`replayed_events=13`、`first_difference=null`。
- `adjudicate`：4 个最终属主、2 个回收裁决；到期回收 1、配额作废回收 1、唯一代数周期 2。
- `evolve`：配额版本 3→4 且移除 `node-c` 后，失效集合按节点序为 `lease-a2`、`lease-c2`；`all_unaffected_bytes_identical=true`。
- `inject-stale`：退出码 13，错误码 `E_LEASE`，当前代数为 2，`node_did_not_advance=true`。
- `inject-spec --kind duplicate_port`：退出码 2，错误码 `E_SPEC`，字段 `nodes[1].port`，行号 35。
- `inject-spec --kind bad_interval`：字段 `renew_interval_ms`，行号 5。
- SIGKILL 恢复：shell 退出码 137；`recovery.count=1`，`reservations_reclaimed=1`；恢复后 `verify` 为 `OK`。

## 推荐复核顺序

```bash
python3 -m unittest discover -s tests -v

rm -rf work/manual-check
mkdir -p work/manual-check
python3 -m leasemesh execute \
  --declarations samples/declarations.json \
  --requests samples/requests.json \
  --allocate-ports \
  --workdir work/manual-check \
  --ledger work/manual-check/ledger.sqlite3 \
  --snapshot work/manual-check/snapshot.json \
  --reset

python3 -m leasemesh verify \
  --declarations work/manual-check/allocated-declarations.json \
  --ledger work/manual-check/ledger.sqlite3

python3 -m leasemesh inject-stale --allocate-ports --workdir work/manual-stale
python3 -m leasemesh inject-spec --kind bad_interval
```

注意：上面的 `rm -rf work/manual-check` 只用于清理你自己的临时复核目录；本仓库交付过程中未使用 Docker。
