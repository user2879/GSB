# 触觉反馈协议诊断工作台

面向 4 通道触觉设备（左右握把、左右扳机）的本地确定性诊断工作台。项目把参数化效果图编译为 v1/v2 二进制指令帧，维护基元 → 时间轴片段 → 指令帧 → 批次摘要求依赖图，并覆盖量化感知增量失效、协议迁移、并发下发、回执核对和原子提交/崩溃恢复。

所有设备交互只使用本机回环地址 `127.0.0.1`，不依赖硬件、账号、外部素材或容器。代码使用 Python 标准库网络栈和 `pathlib`，可直接在 Windows 3.11 上运行。

## Windows 快速开始

```powershell
cd <workspace>
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m haptic_diag.cli generate
python -m pytest -q
python -m haptic_diag.cli verify `
  --sample cross_slice_boundary `
  --protocol 2 `
  --workspace .\workbench `
  --output .\reports\verification-windows.json
```

也可直接执行 `scripts\run_windows.ps1`（支持 `-SkipInstall`、`-SkipTests`）或 `scripts\run_windows.bat`。离线或固定镜像环境中，由执行环境提供 NumPy 安装源；项目声明的最低依赖为 Python 3.11 和 NumPy 2.x。

## 核心命令

```text
python -m haptic_diag.cli generate     # 生成效果图与真值清单
python -m haptic_diag.cli compile      # 编译单样本并可选输出误差报告
python -m haptic_diag.cli verify       # 编译、回环下发、核对、增量、迁移、提交、恢复
python -m haptic_diag.cli recover      # 仅按 CURRENT 恢复最近稳定提交
python -m haptic_diag.cli crash-worker # 强制终止注入 Worker（供崩溃恢复测试）
python -m haptic_diag.device_server    # 独立回环模拟设备进程
```

`verify` 默认使用 `cross_slice_boundary` 样本，并启动独立 Python 子进程作为模拟设备，只监听 `127.0.0.1`。完整结果写入 JSON 报告，控制台只打印摘要。

## 样本与确定性

固定种子清单位于 `configs/samples.json`：`20261119` 至 `20261122`。生成结果已提交为：

- `samples/effect_graphs.json`
- `samples/truth_manifest.json`

覆盖四类场景：单通道、多通道重叠、跨片段边界、120 秒超长时长。真值清单包含每个基元的通道、通道数、起止微秒与时长。同样配置连续生成两次，图文件和真值文件逐字节一致。

## 协议与量化

两种帧格式均为 `2sBBIQB + 通道体 + CRC32` 的大端二进制：

| 字段 | v2 | v1 |
|---|---|---|
| magic / version / flags | `HF` / `2` / 1B | `HF` / `1` / 1B |
| seq / timestamp_us / mask | u32 / u64 / u8 | u32 / u64 / u8 |
| 每通道 | u16 振幅 + u16 频率 | 仅 u16 振幅 |
| 通道位 | `01/02/04/08` | `01/04/10/40` |
| 尾部 | CRC32 | CRC32 |

振幅量化步长 `1/65535`，频率范围 `0..2000 Hz`、步长 `2000/65535`，声明最大误差均为半个步长。v1 无频率字段；迁移到 v2 时按固定表 `100/120/140/160 Hz` 补齐，迁移幂等。因此 v1 往返保留时间戳和振幅，但原始频率不可表示，诊断器会把缺失频率归为协议字段差异。

无法解析或迁移的数据抛出 `E_PROTO_FORMAT`，并携带 `frame_seq` 与 `field`（如 `channel_mask`、`frequency`、`crc`、`magic`）。

## 增量失效

- 依赖图节点：基元、时间轴片段、指令帧、批次摘要、当前协议版本。
- 依赖边记录触发参数键、涉及通道和 `protocol.version` 键。
- 频率/振幅修改先比较量化后的编码值；编码不变时不产生任何下游失效。
- 编码有效时只探测基元旧/新时间区间与绑定通道覆盖的帧，再用实际内容哈希裁剪失效闭包。
- `UpdateResult` 强制断言失效帧集合与实际重编帧集合完全一致；闭包外节点记录复用来源和版本戳。
- 协议切换必须重编码所有帧并新建失效账本；旧账本归档且标记不可复用，防止跨版本误用。

## 并发、核对与持久化

- 线程池结果按帧序号归并，不按完成顺序拼接；记录重复回执、缺失帧和序号缺口。
- 固定降级优先级：`内存预算(1) → 时间预算(2) → 线程上限(3)`。时间超预算时后续波次降为单线程，且不重发已提交批次。
- 核对器逐帧比较，定位首个不一致帧，区分 `quantization_difference`、`protocol_field_difference`、`transport_missing`；摘要不一致只作为告警，不单独判定传输失败。
- 快照先写入未被引用的 generation 目录，完成全部文件、文件哈希和 `COMMITTED.json` 后，才原子替换 `CURRENT`。
- 恢复只读取 `CURRENT` 指向的完整快照；其他目录归为孤儿或已替代提交，孤儿移入隔离目录并记录到 `state/recovery_log.json`。

## 验证结果

本工作区已实际完成：

- 自动化测试：`37 passed`。
- CLI 端到端：初始 v2 编译与 2400 帧回环核对、量化无效修改、有效增量、v2 → v1 迁移、2400 帧核对、提交与恢复全部成功。
- 真实子进程 `os._exit(137)` 强制终止后，恢复到最后已提交 generation；未提交快照被识别为孤儿；恢复后继续增量与全量重编的帧哈希和批次摘要完全一致。

详细证据见 `reports/verification.json`、`reports/verification-summary.json` 和 `docs/verification.md`。

## 目录结构

```text
configs/                 固定种子样本配置
samples/                 已生成效果图与真值清单
src/haptic_diag/         模型、编译、协议、依赖图、传输、核对、持久化
tests/                   样本、协议、编译、增量、并发、崩溃恢复测试
scripts/                 Windows 启动与验证脚本
docs/                    设计说明与验证记录
reports/                 实际运行报告
```
