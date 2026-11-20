# 配置分发与故障恢复系统 — 技术说明

## 1. 目标与范围

平台在 Windows 本地（同样可在 Linux/macOS 运行）管理若干模拟组件节点。节点是
独立子进程，通过 `127.0.0.1` 回环 TCP 上的 JSON 行（每行一个 JSON 对象）协议通信。
系统聚焦三件事：

1. v1/v2/v3 三种历史清单格式的强校验、互操作与逐条升级；
2. 把当前版本清单编译为分发计划，按节点并行下发并收敛回执；
3. 在节点故障或平台中断后，依据检查点、SQLite 会话库与快照完成恢复与一致性重放。

全程只使用 Python 标准库 + NumPy + pytest，不依赖网络、容器或外部服务。

## 2. 目录结构

```
cfgdist/
  errors.py           业务码与 BizError（E_FORMAT/E_PORT/E_VERSION/...）
  logging_setup.py    线程安全的 JSON-lines 结构化日志
  manifest.py         v1/v2/v3 编解码、重复键检测、强校验、原子写
  upgrade.py          逐版本(v1->v2->v3)、逐条检查点升级与中断恢复
  derivation.py       NumPy 确定性聚合与节点派生状态摘要(sha256)
  planner.py          清单 -> 确定性分发计划（切片/广播/期望摘要）
  protocol.py         JSON 行 over TCP 收发与端口占用探测
  node_simulator.py   可独立运行的模拟节点子进程（含故障开关）
  runtime.py          节点声明加载、端口检查、子进程节点池
  distributor.py      线程池并行分发、超时重试、收敛判定
  session.py          SQLite 会话库、状态机、回执、幂等提交
  snapshot.py         快照版本识别与整体替换恢复
  replay.py           独立重放校验工具
  cli.py              命令行入口与三类异常注入
samples/              v1/v2/v3 样本清单与节点声明（端口 19001-19003）
tests/                pytest 套件（38 个用例）
run/                  运行期 db/snapshots/checkpoints/backups/logs 目录
```

## 3. 清单格式

- v1：`{"format_version":1,"entries":[{"key","value"}]}`，值仅支持
  `int/float/str/bool` 标量。
- v2：条目增加 `node`（目标节点）与 `config_version`（配置版本号）。
- v3：条目增加 `strategy`（`mode/retries/timeout_ms`），顶层增加 `derivations`
  （`name/inputs/op/target_node`，op 支持 sum/mean/min/max/count）。

强校验规则（任一失败整体拒绝，绝不跳过问题条目）：

- 未知 `format_version`、必需字段缺失、值类型不受支持、JSON 内重复键、条目键重复、
  未知字段、派生输入引用不到条目、非回环地址等，统一抛 `E_FORMAT`，并带
  `field_path`（如 `$.entries[2].node`）与 `reason`。

## 4. 升级与中断恢复（需求 2）

升级按“一个格式版本 × 每条目”为最小工作单元，v1->v3 共 2 个版本步：

- 每处理完一个 `(版本步, 条目)`，立即写检查点 `checkpoints/<name>.checkpoint.json`
  并向 `checkpoints/<name>.upgrade-trail.jsonl` 追加一条 before/after 轨迹。
- 中断时持久化中间工作清单 `*.working.json`；源清单在全部步骤完成前保持原版本不变，
  因此重启后可从检查点精确续跑，已完成的 `(步,条目)` 不会重复处理。
- 全部完成后一次性原子重写源清单为 v3，原始清单保留为只读备份
  `backups/<name>.v.backup.json`（`0o444`）。
- `--limit N` 用于确定性地模拟“每处理 N 条即被终止”。

## 5. 计划、并行分发与收敛（需求 3、4）

- `planner.build_plan` 按策略切片：`targeted` 只发目标节点，`broadcast` 发全部节点；
  切片按节点 ID、条目按 key 排序，计划指纹与节点到达顺序无关。
- 每个节点用 `derivation.node_digest` 基于“排序后的条目 + 排序后的派生结果”计算
  sha256 期望摘要；NumPy 聚合统一走 float64 并以稳定科学计数法渲染，保证顺序无关、
  跨恢复路径一致。
- `distributor.distribute` 用 `ThreadPoolExecutor` 在全局并发上限内并行下发，单次
  超时按策略重试；首次与重试回执分别落库（`attempt`、`retry` 标记不同）。
- 收敛判定按计划切片的稳定顺序比较期望/实际摘要；摘要不符或无成功回执抛
  `E_CONVERGE`，稳定列出偏差节点、期望摘要、实际摘要与原因，不受回执到达顺序影响。

## 6. 状态机与幂等提交（需求 5）

会话状态：`prepared -> distributing -> committed`，任意阶段可转 `failed`
（committed/failed 为终态）。每次迁移记录 `from/to`、配置版本、已确认节点集合与
UTC 时刻到 `state_transitions`。

提交按 `config_version` 去重：重复提交直接返回已提交结果并标记 `E_IDEMPOTENT`，
不会再次下发，也不会产生第二份回执记录。

## 7. 重放校验（需求 6）

`replay.replay_latest` 独立打开 SQLite 库，取最近已提交版本：

1. 校验当前清单仍为 v3 且包含该配置版本，缺失则 `E_REPLAY`；
2. 重新加载节点声明，库中计划里存在但声明中缺失的节点判为节点降级 -> `E_REPLAY`；
3. 用当前清单重算计划并比较计划指纹；
4. 逐节点比较库中最新成功回执摘要与重算期望摘要，差异定位到首个不一致的
   `node` 与 `field`（如 `node-a` / `receipt.digest`）。

## 8. 快照与整体替换恢复（需求 7）

快照持久化清单版本、配置版本、完整清单、计划、回执、状态机迁移与已提交版本号。
恢复时比较快照清单版本与当前清单版本：

- 一致：兼容恢复（compatible_restore）。
- 不一致：抛 `E_VERSION` 并执行**整体替换**——清空库中 sessions/transitions/
  receipts/committed 后按快照全量重建，绝不增量合并；冲突快照与替换前状态分别
  归档到 `snapshots/archive/` 并置只读，文件名带 `vX-replaced-current-vY` 标签。

## 9. 异常注入与业务码（需求 8）

注入只改变输入文件/参数，不修改任何实现代码：

| CLI 命令 | 注入手段 | 业务码 | 退出码 |
| --- | --- | --- | --- |
| `inject-corrupt --kind duplicate/unknown-version/truncate` | 改写输入清单 | E_FORMAT | 10 |
| `inject-port` | 先占用声明端口再跑端口检查 | E_PORT | 11 |
| `inject-snapshot` | 改写快照的 manifest_version | E_VERSION | 12 |

其余：E_CONVERGE=13、E_IDEMPOTENT=14、E_REPLAY=15、E_STATE=16、E_PROTO=17、
E_NODE=18。所有异常都写入 `logs/events.jsonl` 结构化日志。

## 10. Windows / PowerShell 运行步骤

```powershell
# 1. 环境（需 Python 3.11+，确认 python/pip 可直接调用）
python --version
pip install -r requirements.txt

# 2. 端口检查：若 19001-19003 被占用，改 samples/nodes.json 的高位端口
python -m cfgdist.cli plan samples/manifests/manifest_v3.json --nodes samples/nodes.json

# 3. 三版本升级（--limit 模拟中断后再续跑）
python -m cfgdist.cli upgrade samples/manifests/manifest_v1.json --nodes samples/nodes.json --workdir run --limit 3
python -m cfgdist.cli upgrade samples/manifests/manifest_v1.json --nodes samples/nodes.json --workdir run

# 4. 并行分发 + 收敛 + 提交
python -m cfgdist.cli distribute samples/manifests/manifest_v3.json --nodes samples/nodes.json --workdir run --max-concurrency 2

# 5. 重放校验 / 快照 / 异常注入
python -m cfgdist.cli replay samples/manifests/manifest_v3.json --nodes samples/nodes.json --workdir run
python -m cfgdist.cli snapshot-save samples/manifests/manifest_v3.json --nodes samples/nodes.json --workdir run
python -m cfgdist.cli inject-port --nodes samples/nodes.json --workdir run

# 6. 测试
python -m pytest -q
```

回环监听/连接默认不受 Windows 防火墙拦截；仅监听 127.0.0.1，不访问任何外部目标。

## 11. 已验证结果

- `pytest -q`：38 passed（清单强校验 10、升级恢复 3、分发收敛 6、状态机 4、
  快照重放 6、计划/运行时 5、CLI 注入 4）。
- 样本端到端：v1（12 个升级步骤，含 limit 中断续跑）与 v2 均升级为 v3，v3 为
  noop；只读备份与逐条轨迹齐全。
- 分发三节点全部收敛，重复提交返回 E_IDEMPOTENT（退出码 14）且无第二份回执。
- 摘要不一致 -> E_CONVERGE；节点降级/版本缺失 -> E_REPLAY；端口占用 -> E_PORT；
  快照版本冲突 -> E_VERSION 且整体替换并归档只读；清单损坏 -> E_FORMAT。
