# 视频取证增量指纹与依赖失效诊断工作台

本项目在 Windows 本地运行，使用固定种子生成 60 秒、640x360、30fps 的确定性取证视频，并对帧特征、GOP 摘要、相邻 GOP 可疑度和整片结论建立可持久化的依赖 DAG。修改采样区域、分析窗口或水印掩膜后，程序只重算失效闭包内节点，并默认执行逐节点全量对照。

项目只依赖 Python 3.10+ 与锁定版本的 NumPy；不下载真实素材，不使用 Docker，不访问未授权目标。详细设计见 `docs/design.md`。

## Windows 安装

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

也可运行 `powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1`。

`requirements.txt` 固定 `numpy==2.2.6`，该版本提供 Windows CPython 3.10+ 的本地 wheel。Python 本体及来源由使用者自行准备。

## 快速演示

完整演示会生成 60 秒素材，然后执行基线、掩膜变更、采样变更、分析窗口变更、全量对照、重启重载和强制终止恢复。演示工作区必须是新的本地目录：

```powershell
.\.venv\Scripts\python.exe -m forensic_workbench demo --workspace workspace-demo
```

报告写入 `workspace-demo\reports\demo_report.json`。快速小样：

```powershell
.\.venv\Scripts\python.exe -m forensic_workbench demo --workspace workspace-small --width 160 --height 90 --duration 3 --gop-size 10
```

## 常用命令

```powershell
python -m forensic_workbench generate --workspace workspace
python -m forensic_workbench config --workspace workspace
python -m forensic_workbench run --workspace workspace --config workspace\configs\default.json --run-dir workspace\runs\manual-001
python -m forensic_workbench status --workspace workspace
python -m forensic_workbench recover --workspace workspace
```

`run` 默认校验 AVI 与真值文件哈希，并同时执行全量对照；加 `--skip-source-hash-verify` 或 `--no-full-verify` 可分别跳过对应检查。发生业务错误时，命令行输出结构化 JSON 并返回对应业务退出码。

## 输出结构

```text
workspace/
  source/
    source.avi
    ground_truth.jsonl
    source_manifest.json
  state/
    CURRENT.json
    snapshots/
    pending/
    recovery/
    orphans/
  runs/
  recovery-runs/
  reports/
```

`source_manifest.json` 给出帧数、逐帧清单哈希、AVI 字节哈希和像素内容哈希。相同配置在两个不同目录生成时，AVI、逐帧真值和清单均逐字节一致。

## 依赖与失效语义

- 帧特征只依赖源帧像素、实际生效的采样矩形、实际生效的掩膜矩形和特征算法版本。
- GOP 摘要依赖窗口内成员帧、阈值和特征版本。
- 每个相邻 GOP 边界是独立 `xgop` 节点。GOP 2 变化只重算 `xgop:001-002` 与 `xgop:002-003`。
- 结论依赖当前分析窗口内的 GOP 摘要和跨 GOP 节点。
- 闭包外节点复用会记录来源快照版本、来源快照哈希、首次产生版本和内容哈希。
- 每个新增或复用节点都有内容哈希与参数版本戳；快照和当前指针均以原子替换提交。

## 可疑度口径

```text
signature_offset = Hamming(signature, GOP keyframe signature) / 64
tile_mismatch    = differing 4x4 tile hashes / 16
frame_score      = 0.65 * signature_offset + 0.35 * tile_mismatch
overall_score    = 0.70 * max(GOP score) + 0.30 * max(cross-GOP score)
threshold        = 0.35
```

掩膜内像素从内容哈希、签名和色块哈希中排除，并记录掩膜像素数及实际从采样区排除的像素数。所有节点 payload 同时写入归一化公式和阈值。

## 业务错误码

| 错误码 | 含义 |
|---|---|
| `E_FRAME_FORMAT` | AVI/帧/配置结构非法或损坏 |
| `E_REGION_BOUNDS` | 采样区域、分析窗口或有效采样范围非法 |
| `E_DEPENDENCY_STALE` | 缺失依赖节点、节点哈希不匹配或复用节点缺失 |
| `E_CLOSURE_MISMATCH` | 增量与全量不一致，或闭包外复用输入已过期 |
| `E_SNAPSHOT_VERSION` | 版本冲突、快照哈希冲突、孤儿回收 |
| `E_MASK_INVALID` | 水印掩膜越界、尺寸非法或帧范围非法 |

`E_CLOSURE_MISMATCH` 的上下文包含具体 `node_id`、`comparison_item`、上游/下游边及失效边列表，可直接定位跨 GOP 比对项。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖确定性生成、增量闭包、相邻跨 GOP 比较项、增量/全量一致、快照版本冲突、旧缓存禁用、受控退出、强制终止恢复、孤儿回收和六个业务错误码。

## 实际验证记录

工作区中的 `reports/demo_report.json` 是可复核的最近一次完整运行证据，包含素材哈希、每步闭包、全量对照标志、重启节点哈希一致性和崩溃恢复报告路径。最终交付说明记录实际执行结果。

当前环境实际执行结果（Python 3.11.0rc1、NumPy 2.2.6、WSL2；WSL 的 Windows interop 被禁用，因此未执行 Windows 二进制，但项目代码仅使用跨平台标准库 API，并提供 PowerShell 脚本）：

- `python -m unittest discover -s tests -v`：18 项通过，0 失败。
- 完整 `demo`：1800 帧、60 GOP，AVI 1,244,203,432 字节，生成耗时约 6.14 秒，全部场景总耗时约 127.29 秒。
- 视频 SHA-256：`ae558d0ab266aea87bf24f48ffeea770a7da1b7b2f14a4ee06af4392093d6b0d`。
- 像素内容 SHA-256：`a5b3844235e4162652ec16359f81f78aca5a4bed0ac0ee9ec7b7c8be8dc26957`。
- 真值清单 SHA-256：`61c9accace67ae1c98f31615a8c6c03ec04987e59930df52e74c658f52edc7f1`。
- 基线图：1921 个节点；基线全量对照通过。
- 掩膜变更：重算 34 节点，复用 1887 节点，全量对照通过。
- 采样变更：重算 34 节点，复用 1887 节点，全量对照通过。
- 分析窗口变更：重算 4 节点，复用 56 节点，撤销 1861 个窗外节点，全量对照通过。
- 重启重载：版本 4 快照逐节点哈希一致。
- 强制终止：子进程退出码 17，无关闭标记；回收 1 个未引用快照和 1 个待提交摘要，均记录 `E_SNAPSHOT_VERSION`；恢复后提交状态逐节点一致，随后运行重算 0 节点且结论哈希一致。

完整报告：`workspace-final/reports/demo_report.json`（本地生成目录，默认不纳入版本控制）。归档说明见 `docs/verification.md`。
