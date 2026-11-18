# 确定性列式页缓存、溢出与聚合引擎

本项目面向 Windows 本地文件系统，实现列式页缓存清单、溢出段、受限内存页缓存、整数定点分组聚合、会话快照、重放校验和独立结果验证器。核心规则全部显式编码；缓存淘汰不依赖字典或哈希遍历顺序。

## 交付内容

- `pagecache/`：格式编解码、可恢复升级、确定性 LRU 缓存、压缩溢出、定点聚合、快照/重放、验证器、CLI。
- `samples/generate.py`：固定随机种子的页数据与 v1/v2/v3 清单/溢出段生成器。
- `tests/`：22 项标准库 `unittest` 回归测试。
- `scripts/run_acceptance.py`：一键清理运行产物并执行完整验收。
- `docs/format.md`：二进制格式、升级和确定性规则。
- `docs/verification.md`：验收矩阵与实际结果。
- `artifacts/acceptance/`：最近一次实际验收报告、结构化日志和单元测试日志。

## 环境准备

固定依赖见 `requirements.txt`：`numpy==1.26.4`，要求 Python 3.11 或更高版本。

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

也可运行 `.\scripts\setup.ps1`。无网络时把固定 wheel 放入 `wheelhouse\`，再执行：

```powershell
.\scripts\setup.ps1 -Offline
```

WSL/Linux：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# 或：./scripts/setup.sh
# 离线：./scripts/setup.sh --offline
```

本次执行环境为 WSL2 Ubuntu；构建只使用 `pathlib`、原子 `os.replace`、`zlib` 和 `numpy` 的跨平台能力，目录级 `fsync` 在 Windows 上自动跳过。PowerShell 脚本已提供，但本次没有 Windows 原生执行环境可用于原生运行验证。

## 生成确定样本

```bash
.venv/bin/python -m pagecache generate --out samples/data --seed 20261118
```

输出包含共享页文件和三个独立版本目录：

```text
samples/data/
  pages/p0001.bin ... p0008.bin
  versions/v1/manifest.pcm + overflow/*.seg
  versions/v2/manifest.pcm + overflow/*.seg
  versions/v3/manifest.pcm + overflow/*.seg
  sample_meta.json
```

生成器会先清空目标 `samples/data`，因此同一种子重复生成得到相同页、清单和未压缩段字节。固定压缩段还取决于本机 zlib 实现版本；同一运行环境内重复运行确定。

## 一键验收

```bash
.venv/bin/python scripts/run_acceptance.py
```

该命令会清空 `artifacts/acceptance/`，重新生成样本，并验证：三版本读取、v1/v2 升级、升级中断续做、全常驻/小预算聚合一致性、重复运行确定性、快照重放、独立验证器和三类异常注入。结果：

```text
artifacts/acceptance/acceptance-report.json
artifacts/acceptance/acceptance-summary.md
artifacts/acceptance/structured.log
artifacts/acceptance/unit-tests.log
```

需要仅清理历史运行产物时：

```bash
.venv/bin/python scripts/reset_runs.py --include-acceptance
```

## 常用 CLI

先为可变运行准备独立溢出目录，避免修改只读样本基线。v3 清单引用了样本段，因此示例先复制初始段：

```bash
mkdir -p artifacts/runs/run-small
cp samples/data/versions/v3/overflow/*.seg artifacts/runs/run-small/

.venv/bin/python -m pagecache run \
  --manifest samples/data/versions/v3/manifest.pcm \
  --pages-dir samples/data/pages \
  --overflow-dir artifacts/runs/run-small \
  --budget-bytes 1536 --page-size 512 \
  --output artifacts/runs/run-small/output.json \
  --events artifacts/runs/run-small/events.json
```

自动升级与中断续做：

```bash
.venv/bin/python -m pagecache upgrade \
  --manifest artifacts/runs/upgrade/manifest.pcm \
  --pages-dir samples/data/pages \
  --overflow-dir artifacts/runs/upgrade/overflow

# 模拟完成 2 条记录后进程中止；退出码 75
.venv/bin/python -m pagecache upgrade ... --crash-after-checkpoint 2
# 再次执行同一 upgrade 命令即可从原子检查点续做
```

快照、重放与独立验证：

```bash
.venv/bin/python -m pagecache run ... \
  --snapshot artifacts/runs/run-small/snapshot.json --snapshot-at 5

.venv/bin/python -m pagecache replay ... \
  --snapshot artifacts/runs/run-small/snapshot.json

# 不一致时仍整体替换状态，并稳定列出丢失/新增段
.venv/bin/python -m pagecache restore ... \
  --snapshot artifacts/runs/run-small/snapshot.json

.venv/bin/python -m pagecache verify \
  --snapshot artifacts/runs/run-small/snapshot.json \
  --output artifacts/runs/run-small/output.json \
  --manifest samples/data/versions/v3/manifest.pcm \
  --pages-dir samples/data/pages \
  --overflow-dir artifacts/runs/run-small \
  --report artifacts/runs/run-small/verify.json
```

三类异常注入：

```bash
.venv/bin/python -m pagecache inject --kind manifest_tail_truncate ... --log artifacts/runs/injection.log
.venv/bin/python -m pagecache inject --kind segment_checksum_mismatch ... --log artifacts/runs/injection.log
.venv/bin/python -m pagecache inject --kind memory_budget_zero ... --log artifacts/runs/injection.log
```

对应错误码分别为 `E_FORMAT`、`E_INTEGRITY`、`E_BUDGET`，每次注入追加一行稳定 JSON 结构化日志。

## 关键语义

- **清单版本**：v1 仅承载 `page_id`/`row_count`；v2 增加列摘要和页载荷校验和；v3 增加压缩标记与溢出段引用。所有版本还有记录级物理 CRC 和文件级尾部 CRC，因此 v1 的物理完整性也可逐条验证。
- **升级**：先整体校验旧清单，再建立只读备份和原子检查点；每条升级记录写盘并 `fsync` 后才推进检查点。重启时校验检查点中的工作文件长度、SHA-256 和 CRC，截断未确认尾部并只从 `done_records` 继续。
- **缓存预算**：`capacity_frames = budget_bytes // page_size`。预算为零或不足一帧均为 `E_BUDGET`。
- **淘汰规则**：唯一候选顺序为 `(last_access_clock, page_id)`；每次选最小项，因此不依赖字典/哈希遍历顺序。
- **压缩**：v3 使用 zlib level 9；段头压缩标记、清单压缩标记和实际载荷必须一致，否则 `E_PAYLOAD` 并携带首个不匹配 `page_id`。
- **聚合口径**：`fixed-v1-half-even-trunc-toward-zero`，比例 `1_000_000`。值先按 `Decimal(str(value))` 做 half-even 量化；和用整数定点累加；均值向零截断。每个值量化绝对误差不超过 `1/(2S)`，每组和误差不超过 `count/(2S)`，组均值误差不超过 `3/(2S)=0.0000015`。
- **快照**：保存清单版本集合、清单内容指纹、常驻/溢出索引、LRU 时钟、事件、聚合状态和定点口径。恢复时整体替换缓存和聚合，不合并；丢失、新增和变化的段按文件名稳定排序报告。
- **重放**：从快照位置继续同一访问序列，逐条比较溢出序列、换入序列/次数、常驻索引、缓存状态摘要和聚合中间状态；首个差异报告 `page_id`、`field`、`access_index`。格式版本、清单状态、定点口径变化或必要段缺失时返回 `E_REPLAY` 及 `basis`。

## 测试

```bash
.venv/bin/python -m unittest discover -v
```

最近实际结果：`Ran 22 tests ... OK`，一键验收 8 个阶段全部 PASS；详见 `artifacts/acceptance/acceptance-summary.md`。

## 目录

```text
pagecache/           engine and CLI
samples/             deterministic generator and generated baseline
tests/               unittest suite
scripts/             setup, reset, acceptance
docs/                format and verification notes
artifacts/acceptance acceptance evidence
```
