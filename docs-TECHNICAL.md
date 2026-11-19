# 块存储快照与回滚组件 — 技术说明

面向 Windows 本地环境的教学/排障演示组件，仅依赖 Python 3.11 标准库
（`sqlite3`、`tkinter`、`hashlib`、`unittest` 等），无需网络、GPU 或容器。

## 1. 目录结构

```
block_snap/                可导入 Python 包（领域组件）
  errors.py                业务码 E_INPUT/E_PATH/E_INTEGRITY/E_CHAIN/E_SHAPE/E_STATE + JSONL 结构化日志
  volume.py                固定块镜像、强校验、确定性内容、十六进制变更注入
  bitmap.py                变更块位图（bit/block），长度与块数相容性校验
  storage.py               SQLite 元数据 + 内容寻址块仓（blocks/<blake2b>）
  manager.py               卷生命周期、CoW 快照、清单摘要、断链/完整性校验
  rollback.py              三条回滚路径、暂存区、原子替换、崩溃注入与恢复
  manifest.py              清单格式版本识别、整体替换、只读备份、版本化审计
  service.py               统一门面 API（CLI 与 Tkinter 共用）
  samples.py               三组确定性样本卷参数与工作区清理
  ui.py                    Tkinter 控制台（延迟导入，无显示环境可优雅降级）
scripts/
  blocksnap_cli.py         命令行入口（输出 JSON）
  demo.py                  四个固定场景的可重复演示
  launch_ui.py             Tkinter 界面启动器
tests/                     20 个 unittest 用例
workdata/                  运行期数据（volumes/ blocks/ tmp/ backups/ meta/）
```

## 2. 数据模型与 CoW 语义

- 卷是 `block_count * block_size` 字节的原始镜像；支持块大小集合
  `{512, 1024, 2048, 4096, 8192}`，镜像长度与块数不符抛 `E_SHAPE`。
- 块数据写入内容寻址仓 `blocks/<blake2b-256>`，写一次即不可变；同内容天然去重。
- 快照表保存：单调递增整数 `version_no`、`parent_id`（父快照引用）、变更块位图
  （BLOB，长度 `ceil(block_count/8)`）。块引用表保存 `(snapshot_id, block_no, block_hash)`。
- **基线快照**（version 1）位图为空但持有全部初始块引用；后续快照只保存自父快照
  以来被改写的块引用与位图，不复制整卷。当前活动块状态单独维护于 `current_state`，
  用于计算两次快照之间的 CoW 差分。
- 清单 `content_digest` 由条目集合按 `version_no` 稳定顺序重算（含 id/name/版本/
  父引用/位图/引用映射），任何条目变化都会改变摘要。

## 3. 三条自适应回滚路径

`RollbackEngine.plan_path` 依据显式阈值决策（可被 `force_path` 覆盖）：

| 路径 | 触发条件 | 写入策略 |
| --- | --- | --- |
| `bitmap` 位图差分 | 差异块比例 `< bitmap_threshold`（默认 0.25）且快照跨度 `< chain_span` | 仅写差异块 |
| `full` 全量比对 | 差异比例 `>= threshold` | 逐块比对并重写目标块 |
| `chain` 增量链回放 | 快照跨度 `>= chain_span`（默认 2，即跨多个快照） | 按链重放合成目标全量状态 |

- 链回放从基线起按版本顺序应用每个快照的块引用，后写覆盖，得到目标快照完整状态。
- `cross_check_paths` 用三条路径分别构建暂存镜像并比较整体摘要；出现分歧时
  抛 `E_PATH` 并用二分块扫描给出**首个不一致块号**。
- 同一输入三条路径的目标状态来源一致，产出逐位一致镜像（测试与演示均验证）。

## 4. 原子切换与中断恢复

1. 回滚前先做断链检查（`E_CHAIN`）与逐块摘要校验（`E_INTEGRITY`，给首个坏块号）。
2. 在 `tmp/<job>.staging` 复制当前卷并写入目标块；每块 `fsync`，同时维护
   `tmp/<job>.staging.marker`（JSON，记录 phase/verified/total/wrote）。
3. 全部块在暂存区逐块校验通过后，用单次 `os.replace` **原子替换**卷镜像，再删除标记。
4. 回滚任务持久化于 `rollback_jobs`（RUNNING/COMMITTED/UNKNOWN）。
5. 强制终止后重启调用 `recover`：只能依据持久化位图与暂存区标记判断；RUNNING 任务
   一律标记 `UNKNOWN` 并要求重跑，报告给出临时区域**已验证/未验证块数**。组件不依赖
   也不生成“退出报告”，崩溃用 `crash_after_blocks` 注入 `SystemExit` 模拟。
6. 客户端用 `completion_token(snapshot_id)` 结合清单版本确认回滚是否已 COMMITTED。

## 5. 清单格式版本与整体替换

- 清单携带整数 `format_version` 与 `content_digest`；`manifest_view`/`detect_drift`
  可识别摘要漂移。
- `replace_manifest` 接收**完整新条目集合**（拒绝增量），在单事务内删除全部旧快照/
  位图/引用并写入新集合，随后更新清单版本与摘要，事务失败回滚。
- 替换前旧清单序列化为 `backups/manifest-*.json` 并置为只读；替换报告稳定输出
  被替换条目数、版本变化与新旧摘要差异。

## 6. 审计与版本化结论

`audit` 逐快照校验：位图与引用块集合对应关系、父链可回放至卷初始状态、
快照版本号与清单版本匹配、引用块摘要正确。结论按清单版本写入 `audit_verdicts`；
重启后清单版本不一致时识别差异并整体替换审计结论，输出不一致快照清单与首个失配原因。

## 7. 业务码与结构化日志

| 码 | 触发场景 | 携带上下文 |
| --- | --- | --- |
| `E_INPUT` | 块大小/块数/块号/变更参数/重名快照等非法输入 | 出错字段 `field` |
| `E_PATH` | 三路径结果不一致、暂存校验失败 | 首个不一致块号 |
| `E_INTEGRITY` | 引用块实际摘要与登记摘要不符 | 首个损坏块号 |
| `E_CHAIN` | 父快照/块对象引用缺失、断链 | 断链快照标识 |
| `E_SHAPE` | 镜像长度与块数不符、短读 | 字段 |
| `E_STATE` | 保留的状态类错误 | — |

所有错误写入 `meta/events.jsonl`（每行一个 JSON，含快照标识、块号、清单版本），
`query_logs(code=..., event=..., snapshot_id=...)` 提供查询。

## 8. 三组确定性样本卷

| Profile | 块大小 × 块数 | 变更特征 | 快照数 | 主要验证 |
| --- | --- | --- | --- | --- |
| `small_many_snapshots` | 512B × 16 | 小卷、深链 | 5 | 链回放、审计 |
| `large_low_change` | 4KiB × 256 | 约 3% 稀疏变更 | 4 | bitmap 路径 |
| `large_high_change` | 4KiB × 128 | 约 45–69% 密集变更 | 3 | full 路径 |

内容由 BLAKE2b 计数器模式密钥流（seed+块号+pattern）确定性生成，跨机器/重复运行逐位可复现。

## 9. 使用方式

```bash
# 环境准备（Windows python.org 安装包自带 sqlite3/tkinter）
python --version            # 需要 3.11+

# 一键演示（清理旧数据后顺序执行四个场景）
python scripts/demo.py

# 命令行
python scripts/blocksnap_cli.py init-samples
python scripts/blocksnap_cli.py list small_many_snapshots
python scripts/blocksnap_cli.py plan  <volume> <snapshot_id>
python scripts/blocksnap_cli.py rollback <volume> <snapshot_id> --cross-check
python scripts/blocksnap_cli.py recover <volume>
python scripts/blocksnap_cli.py audit <volume>
python scripts/blocksnap_cli.py logs --code E_INTEGRITY

# 图形界面（需要桌面与 tkinter；缺失时以 rc=3 优雅提示）
python scripts/launch_ui.py

# 测试
python -m unittest discover -s tests
```

## 10. Windows 适配说明

- 路径全部经 `os.path.abspath`/`os.path.join` 处理，无硬编码分隔符；原子替换使用
  `os.replace`，在 NTFS 同目录内为原子操作。
- 暂存区与卷镜像位于同一 `volumes/` 目录树，保证替换不跨卷（跨卷 `os.replace` 非原子）。
- 标记文件与 SQLite 提交均 `fsync`；只读备份用 `stat.S_IREAD` 常量，跨平台有效。
- 当前 Linux/WSL 验证机未安装 tkinter，UI 采用延迟导入并在不可用时返回明确提示，
  包的其余功能、CLI、测试与演示完全不受影响。在 Windows 官方 Python 安装中
  勾选 tcl/tk 即可直接运行 `launch_ui.py`。

## 11. 实际验证结果

- `python -m unittest discover -s tests`：20/20 通过。
- `python scripts/demo.py`：四个场景全部通过，包含三条路径逐位一致、原子切换、
  清单整体替换（只读备份）、强制终止后的 UNKNOWN 恢复报告（已验证/未验证块数）与重跑提交。
- 负向用例覆盖：非法块大小/块号/变更规格（E_INPUT 且零写入）、镜像变形（E_SHAPE）、
  块对象损坏（E_INTEGRITY 首块号）、悬空父引用（E_CHAIN）、路径分歧（E_PATH 首块号）。
