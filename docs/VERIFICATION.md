# 实际验证记录

验证日期：2026-11-26（Asia/Shanghai），WSL Ubuntu，Python 3.11.0rc1，pytest 8.2.0。

## 自动化测试

```text
$ python3 -m pytest
..........................
26 passed
```

覆盖：E_SPEC 字段/行号、回环字面量与端口唯一、续约间隔、标签跳跃、重复终态、陈旧标签、双路径逐字节一致、到期与配额作废唯一回收、配额耗尽零下发、路径边界、线程池双配额与等待区间、真实节点协议、端口占用、读取超时、快照整体替换、pending 未知恢复、E_REPLAY 定位、真实双路径及强杀重放。

## 一键验收

```text
$ python3 scripts/run_demo.py
```

关键结果：

```text
pytest                         26 passed
projection file bytes          3538 (canonical JSON 3537 + newline)
projection sha256              953b2326d7bee7f4b6ceaa990ac2176c205d54f2271cc5b3fcef973a6f5e598a
live sequential snapshot ver   13
live batch snapshot ver        13
live node requests             9 (both paths)
max in-flight total            2
crash exit code                137
committed before crash         4 events
unknown pending requests       1 (event seq 5)
recovered snapshot version     5
replay status                  replay_ok
```

独立校验、真实逐事件路径和真实批量路径的投影文件均为同一组 3538 字节（规范 JSON 3537 字节 + 换行）；`results/byte-consistency.json` 中三个比较位全为 `true`。

## 结果文件

```text
results/demo-report.json        总报告
results/verify.json             不读账本的独立双路径校验
results/live-sequential.json    真实子进程/TCP 逐事件运行
results/live-batch.json         真实子进程/TCP 批量运行
results/crash-replay.json       强杀后未知 pending 与重放核对
results/byte-consistency.json   三条投影字节一致性
results/projection*.json        规范投影字节文件
ledgers/live-sequential.sqlite3
ledgers/live-batch.sqlite3
ledgers/crash-recovery.sqlite3
```

## 复核命令

```bash
python3 -m lease_arbiter compare \
  --left results/live-sequential.json \
  --right results/live-batch.json

python3 -m lease_arbiter replay \
  --nodes config/nodes.json \
  --leases config/leases.json \
  --events samples/events.jsonl \
  --ledger ledgers/crash-recovery.sqlite3
```

`compare` 比较完整规范投影字节；`replay` 在打开账本、标记未知 pending 后，仅按完整提交前缀重算并核对。任一步失败分别返回 `E_DIVERGE` 或 `E_REPLAY`。
