# 验证记录

## 环境

- 执行日期：2026-11-19。
- 当前执行环境：WSL2 Ubuntu 22.04，Python `3.11.0rc1`，NumPy `2.4.6`。
- 未使用 Docker、容器、真实设备、账号或外部网络服务。
- 本工作区代码按 Windows 3.11 兼容方式编写，并提供 `scripts/run_windows.ps1` 与 `scripts/run_windows.bat`；当前 Linux/WSL 环境无法直接执行 Windows PowerShell，因此原生 Windows 运行需在目标机执行一次脚本确认。

## 自动化测试

命令：

```bash
PYTHONPATH=src python3 -m pytest -q
```

实际结果：

```text
.................................                                        [100%]
37 passed in 22.76s
```

覆盖点：

- 样本两次生成逐字节一致，已提交样本与生成器一致。
- 四类场景、三类波形、通道数与时间区间真值。
- v1/v2 二进制往返、旧掩码、固定迁移表、迁移幂等、CRC/长度/mask 的 `E_PROTO_FORMAT` 定位。
- 编译逐字节确定性、量化误差界、帧序号连续、批次摘要。
- 量化无效修改零失效；有效修改只重编内容哈希变化帧；与全量编译完全一致。
- 依赖边参数键和协议键；协议切换全帧重编码与旧账本归档。
- 并发结果按序号归并、预算降级无重复/缺口、缺失回执待确认。
- 量化差异、协议字段差异、纯摘要差异三类核对分支。
- 原子提交、未完成快照孤儿化、恢复后增量 = 全量编译。
- 真实子进程强制终止后恢复一致。

## CLI 端到端验证

命令：

```bash
PYTHONPATH=src python3 -m haptic_diag.cli verify \
  --sample cross_slice_boundary \
  --protocol 2 \
  --workspace /tmp/haptic-verify-workbench \
  --output reports/verification.json \
  --batch-size 20
```

实际关键结果：

| 检查项 | 结果 |
|---|---|
| 初始 v2 帧数 | 2400 |
| 初始回环核对 | `confirmed` |
| 量化无效频率修改 | 失效帧 0，复用节点 2525 |
| 有效增量 | 仅相关时间窗/通道帧重编，核对 `confirmed` |
| v2 → v1 迁移 | 2400 帧全部重编码，旧账本归档不可复用 |
| 迁移后回环核对 | `confirmed` |
| 协议往返 | 2400 帧，时间戳差异 0，振幅差异 0，v1 迁移幂等 |
| 最终提交 | `gen-000002-78daf15fd084` |
| 恢复 | 从 `CURRENT` 成功恢复，帧/摘要与提交一致 |

完整 2400 帧逐帧结果见 `reports/verification.json`。精简摘要见 `reports/verification-summary.json`。

## 强制终止与恢复

测试实际启动独立子进程执行 `haptic_diag.cli crash-worker`：

1. 基线快照完整提交。
2. 生成待提交新状态。
3. 并发下发到回环服务，在第 2 个批次后触发终止入口。
4. 模拟写入新 generation 时 `os._exit(137)`。
5. 父进程重启并调用 `recover`。

验证结论：

- 进程非零退出。
- `CURRENT` 仍指向基线 generation。
- 未完成写入目录被归类为 `orphan` 并移动到隔离目录，恢复日志存在。
- 恢复后的帧哈希与基线完全一致。
- 对恢复状态应用同一增量后，帧哈希和批次摘要与全量编译完全一致。

对应测试：

```text
tests/test_persistence.py::test_real_subprocess_forced_termination_then_resume_matches_full
```

## 确定性样本摘要

| 样本 | 用途 | 帧数 | 帧周期 |
|---|---|---:|---:|
| `single_channel_sine` | 单通道 | 2000 | 2000 us |
| `multi_channel_overlap` | 多通道重叠 | 2500 | 2000 us |
| `cross_slice_boundary` | 跨片段边界 | 2400 | 2500 us |
| `long_duration` | 120 秒超长时长 | 6000 | 20000 us |

固定种子：`20261119`、`20261120`、`20261121`、`20261122`。

已提交样本文件 SHA256：

```text
effect_graphs.json   385eac9cce38e2ace2dc8fc959f33614af6ee5443821f849b60ad839822f6cd1
truth_manifest.json  7ebaa1c14f14bfb3b4226be59cb5ec208e16f5db626e6f29454e5699ea78438c
```

## 已知边界

- v1 帧物理上没有频率字段；v2→v1→v2 不可能恢复原始任意频率。实现按协议契约保留时间戳/振幅，并用固定表补齐频率，同时把字段缺失归为协议字段差异。
- 当前 Linux/WSL 验证覆盖完整逻辑和回环进程；Windows 原生原子替换、路径和 PowerShell 脚本需在目标 Windows 主机执行 `scripts/run_windows.ps1` 做平台确认。
