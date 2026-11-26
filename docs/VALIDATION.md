# 实际验证记录

验证时间：2026-11-27（Asia/Shanghai）。

## 环境

- WSL Ubuntu 22.04 工作区，根目录即本仓库。
- Python：`3.11.0rc1`。
- 虚拟环境：`.venv/`，NumPy：`1.26.4`。
- Python 标准库 SQLite：`3.37.2`。
- `apt-cache policy python3-tk sqlite3` 显示软件源可用；当前会话的 `sudo` 需要密码，因此未能执行 `sudo apt-get install`。
- 镜像缺少 `ensurepip`，实际使用 `python3 -m venv --without-pip .venv` 后通过 `get-pip.py` 引导 pip。
- `sqlite3` CLI 当前未安装；所有账本验证使用标准库 `sqlite3` 和数据库文件完成。
- `.venv` 中暂时没有 `tkinter`（缺少系统 `python3-tk`）。已用基础 Python 和本机已有 Tk 动态库完成真实窗口构建冒烟：

```bash
LD_LIBRARY_PATH=/tmp/tkdeps2/usr/lib/x86_64-linux-gnu \
EDGE_UI_SMOKE_MS=300 \
python3 -m edge_shard_diag ui --spec config/demo --db artifacts/runtime/ledger.sqlite
# exit=0
```

在可输入 sudo 密码的标准 Ubuntu 上执行 `sudo apt-get install -y python3.11-venv python3-tk sqlite3` 后，`.venv/bin/python -m edge_shard_diag ui` 可直接启动。

## 自动化测试

命令：

```bash
./scripts/acceptance.sh
```

实际结果：

```text
Ran 16 tests in 9.900s

OK
ACCEPTANCE_OK report=artifacts/validation-report.json
```

完整输出保存在 `artifacts/final-validation.log`。

覆盖包括：

- `E_SPEC`：非法 host、重复端口、依赖环、非法 scale、快照版本不匹配。
- `E_FORMAT` / `E_FIELD` / `E_DUP`：未知回执版本、缺字段、稳定名次越界、悬空重复引用。
- 真实 TCP 调度、线程池、预算排队、`E_PORT`、`E_TIMEOUT`。
- 租约乐观版本错误 `E_LEASE`、真实短 TTL 过期。
- 容量换代撤销与到期回收不双重释放。
- `E_DIVERGE` / `E_REPLAY` 的首个分片和字段定位。
- 完整本地验收（含真实 SIGKILL）。

## 三格式回执归一

命令：

```bash
.venv/bin/python -m edge_shard_diag receipts
```

实测计数：

```json
{"d1": 4, "d2": 3, "d3": 4}
```

合并归一后 4 个分片各有一条采纳结论；容量版本相同的情况下 d3 优先于 d2/d1。d1 记录显式补齐 `capacity_version=0`、`scale_factor=1`、`stable_rank=-1`、空重复引用等字段及依据。

结构化结果：`artifacts/receipts-normalized.json`。

## 租约与定点归因

验收报告 `artifacts/validation-report.json` 中：

- 正确续约后 `lease_version=2`、`renew_count=1`。
- 用旧版本 `1` 再向节点子进程续约返回 `E_LEASE`，且账本到期时间未被延长：`wrong_version_did_not_extend=true`。
- TTL=40ms 的租约在真实 `time.sleep(0.09)` 后由单调时钟判定为 `EXPIRED`。
- `3/6` 与 `4/6` 在 scale=1 时定点值为 `(0, 0)`，判定受误差影响；scale=100 时为 `(50, 66)`，差值 `16 > epsilon_q=4`，且给定 `slack_q=100` 时 `error_affected=false`，结论转为可判定。

## 增量失效传播

命令：

```bash
.venv/bin/python -m edge_shard_diag verify-incremental
```

替换 `shard-0002` 直接引用的一条样本后，实测失效集合：

```json
["shard-0002", "shard-0004"]
```

未失效集合：

```json
["shard-0001", "shard-0003"]
```

未失效分片的规范记录 SHA-256 均保持不变：

```json
{"shard-0001": true, "shard-0003": true}
```

增量账本视图与空白全量真实 TCP 重建视图逐字段比较通过。结构化结果：`artifacts/incremental-verification.json`。

## 容量换代

完整验收中的容量场景实测：

- 短 TTL 旧租约先由真实到期路径裁决为 `EXPIRED`：`["lease-old-short"]`。
- 未到期长租约由 v2 换代路径裁决为 `REVOKED`：`["lease-old-long"]`。
- 再次执行到期回收得到 0 条。
- 换代后 `resource_owners` 中 HELD 行数为 0，`lease_slots` 中 HELD 行数为 0。
- v2 的 4 个分片全部在当前视图中完成；v1 行保留但标记 `stale=1`，不进入当前视图。

## 强制终止与重放

命令：

```bash
.venv/bin/python -m edge_shard_diag verify-recovery
```

实测：

```json
{
  "child_was_sigkilled": true,
  "unknown_shards": ["shard-0002"],
  "replay_compare": "passed: recovered ledger replay equals blank full rebuild"
}
```

子进程退出码为 `-9`。SQLite 回滚未提交结局，已提交的 `lease-shard-0002-v1` 授予记录被恢复流程回收，`shard-0002` 标记为未知并重跑；恢复视图与空白全量重建按 `E_REPLAY` 比较器逐字段一致。结构化结果：`artifacts/recovery-verification.json`。

完整机器可读验收报告：`artifacts/validation-report.json`。
