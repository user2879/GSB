# 实际验证记录

验证日期：2026-11-27（Asia/Shanghai）  
环境：WSL Ubuntu、Linux 6.18、Python 3.11.0rc1、NumPy 1.26.4、pytest 8.2.0  
虚拟环境：项目根目录 `.venv`（`--system-site-packages`，editable 安装 `ckptsim`）

## 自动化测试

命令：

```bash
.venv/bin/python -m pytest -q
```

结果：`23 passed`，约 7.6 秒。

覆盖内容：

- 规格层：重复端口、缺失派生引用、环、事件序号回退、检查点世代号回退、跨文件版本引用；错误均为 `E_SPEC` 且带首个非法条目与行号。
- 数值层：参数摘要、样本摘要、逐批损失摘要在重复计算、样本集合乱序和字典遍历顺序变化时逐位一致。
- 协议层：真实 TCP 批次请求/回执、健康握手、`E_PORT`、真实套接字读取超时 `E_TIMEOUT`、缺失回执字段 `E_FIELD`、worker 侧乐观续约 `E_LEASE`。
- 调度层：全局租约、分片租约、分片并发三重上限；排队等待毫秒记录；基于 `time.monotonic_ns()` 的真实到期；远程校验失败不延长租约。
- 账本层：未提交批次恢复为 `unknown`、租约回收、运行尝试留痕；快照身份不一致时整体替换、只读备份与差异列表。
- 端到端：检查点提交/回滚/版本归属、独立重放、强制杀 worker 后恢复重跑、篡改账本后 `E_VERIFY` 定位、依赖失效子集重算。

## 演示结果

命令：

```bash
.venv/bin/python scripts/demo.py
```

输出目录：`artifacts/demo/`，主摘要：`artifacts/demo/summary.json`。

关键结果：

- 两次独立运行确定性投影 SHA-256 均为
  `d1a8e68998a2db003e559848ff71b2b361fc46a520921294064023d17e840460`。
- 独立账本重放：115 项逐项一致；附加失效报告校验后：117 项一致且 `impact_verified=true`。
- 层编辑 `parameter_version=1 / token.scale_factor: 1000 -> 1300`：
  - 失效批次 4 个：`shard-a/b-a1`、`shard-a/b-a3`、`shard-b/b-b1`、`shard-b/b-b3`；
  - 未失效批次 3 个：`shard-a/b-a2`、`shard-a/b-a4`、`shard-b/b-b2`，摘要与检查点归属保持逐字节不变；
  - 失效检查点 1 个：`cp-base`，其损失基线摘要已重算并变化。
- 强制终止演示真实杀掉正在服务的 `shard-a` 进程；账本留下 4 条 `unknown` 尝试记录，重启后未知批次重跑并全部提交，独立校验通过。强制退出报告字段为 `null`。
- 快照替换演示：`parameter_version_dialect` 与 `content_digest` 差异被记录，账本整体重建，旧账本备份权限为只读 `0444`。

## 手工命令复核

```bash
.venv/bin/ckptsim validate \
  --shards samples/shards.json --layers samples/layers.json \
  --checkpoints samples/checkpoints.json --events samples/events.json

.venv/bin/ckptsim impact \
  --shards samples/shards.json --layers samples/layers.json \
  --checkpoints samples/checkpoints.json --events samples/events.json \
  --edited-layers samples/layers_edited.json

.venv/bin/ckptsim run --fresh --workdir artifacts/run \
  --shards samples/shards.json --layers samples/layers.json \
  --checkpoints samples/checkpoints.json --events samples/events.json \
  --output artifacts/run/run.json

.venv/bin/ckptsim impact \
  --shards samples/shards.json --layers samples/layers.json \
  --checkpoints samples/checkpoints.json --events samples/events.json \
  --edited-layers samples/layers_edited.json \
  --record-ledger artifacts/run/ledger.sqlite

.venv/bin/ckptsim verify \
  --shards samples/shards.json --layers samples/layers.json \
  --checkpoints samples/checkpoints.json --events samples/events.json \
  --ledger artifacts/run/ledger.sqlite \
  --edited-layers samples/layers_edited.json
```

注意：`scripts/demo.py` 的最后一步会故意用另一份层参数声明重开主账本以演示快照整体替换，因此该步骤之后主账本为空并已保留只读备份。若需继续校验原运行，请使用 `artifacts/demo/repeat/ledger.sqlite` 或先重新运行演示。
