# 验证结果记录

验证环境：Linux 容器（等价 Windows 10+ / Python 3.11 运行语义），
Python 3.11，numpy 1.26.4，PyOpenGL 3.1.7，无 GPU（显示层软件回退）。

## 1. 自动化测试

`python -m pytest` 共 25 个用例全部通过，覆盖：

- 半浮点标量/向量化/numpy 三方逐位一致，subnormal、Inf、NaN、饱和边界；
- 合成序列两次生成逐字节一致、帧格式校验、损坏帧 `E_FRAME_FORMAT`；
- 管线确定性、Reinhard/ACES、sRGB/BT.709 差异、矩阵常量与非法参数；
- 状态内容寻址、精确失效闭包、空操作闭包为 0、历史重放隔离性；
- 增量重算闭包外哈希不变、区间 vs 逐帧输出哈希一致；
- 历史重算与留存产物哈希一致、篡改产物触发 `E_HASH_MISMATCH`；
- 强制终止孤儿回收 `E_RECOVER_ORPHAN`、恢复后续算一致；
- 旧快照 `E_SNAPSHOT_STALE`、schema 冲突 `E_STATE_VERSION`、
  帧号越界、闭包发散 `E_CLOSURE_DIVERGE`。

## 2. 全规格（1920×1080×60，half 线性光）实测

| 项目 | 实测结果 |
| --- | --- |
| 帧序列生成耗时 | 约 53 s |
| 帧目录体积（zlib 压缩 half） | 331 MB / 60 帧 |
| 完整演示（基线实算+5 操作+崩溃+恢复） | 约 2 分 40 s |
| 全量实算校验（60 帧重算+哈希） | 约 36 s，`verified_frames=60 failures=0` |
| 工作区总体积 | 约 806 MB（含全部历史版本产物） |

## 3. 强制终止与恢复（两次独立全规格运行）

演示在第 3 个版本提交后模拟 `exit 137` 强制终止：

- 不产生 `shutdown_report.json`（未伪造受控退出）；
- 重启后依据 manifest 版本戳恢复到 `committed_at_seq=3`；
- 回收未提交孤儿（`tmp/.partial.tmp` 等），记录于 `recovery_report.json`；
- 从稳定状态继续剩余操作至 v5，60 帧输出哈希完整；
- 受控退出补写退出报告。

## 4. 两次运行逐字节一致性

对两次独立全规格演示目录做递归比对：

```
一致: hash_listing.json
一致: recovery_report.json
一致: manifest.json
一致: source_manifest.json
frames/   差异文件数: 0
outputs/  差异文件数: 0
metrics/  差异文件数: 0
```

快速模式（320×180×60）`selftest` 同样报告 hash_listing / recovery_report /
manifest / frames 全部逐字节一致。

## 5. 复现实验命令

```
python -m pip install -r requirements.txt
python -m pytest
python -m hdr_sdr_workbench --workspace ws --quick demo --clean   # 秒级快速演示
python -m hdr_sdr_workbench --workspace ws demo --clean           # 全规格演示
python -m hdr_sdr_workbench --workspace ws verify
python -m hdr_sdr_workbench --workspace ws hashes
python -m hdr_sdr_workbench --workspace ws shell                  # 交互式工作台
```
