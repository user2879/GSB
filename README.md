# AQEVAL

面向本地确定性合成语音样本的音频质量评估数据编辑与交换应用。应用统一处理三种工程质量交换版本：

- `v1`：46 字符定宽文本，千分比整数总分、四维整数分和等级码。
- `v2`：JSON，六位小数浮点分值、命名等级和精度标签。
- `v3`：带校验和、帧级特征分区、定点整数分区和旁路元数据偏移表的二进制容器。

核心能力包括查询、编辑、双向转换、按帧重评、降级诊断、精确缓存失效、线程池确定性并发、阶段化快照与中断续跑。

## 快速开始

要求 Python 3.11+、NumPy。无需网络或外部服务。

```bash
python3 -m unittest discover -s tests -v
python3 -m aqeval demo --workspace validation_workspace
python3 -m aqeval generate --workspace sample_workspace
python3 -m aqeval run --workspace sample_workspace
```

Windows 本地命令行：

```bat
py -3 -m unittest discover -s tests -v
py -3 -m aqeval demo --workspace validation_workspace
```

也可运行 `scripts/run_validation.bat`。当前执行容器未启用 WSL/Windows 互操作，实际验证在 WSL2 Ubuntu 的 Bash 中完成；同一命令已在 `.bat` 与 `.sh` 中固化。

## 常用命令

```bash
# 生成确定性的 WAV、配置、等级表、v1/v2/v3 样例和编辑脚本
python3 -m aqeval generate --workspace sample_workspace

# 增量运行并输出 active_v{版本}、旁路元数据、快照和 JSON 诊断
python3 -m aqeval run --workspace sample_workspace

# 查询某个版本或某个文件
python3 -m aqeval query --workspace sample_workspace --version 3 --file sample_boundary

# 双向转换；转换损失写入 bypass 与 degradations
python3 -m aqeval convert --workspace sample_workspace \
  --from-version 2 --to-version 1 \
  --input sample_workspace/eval/v2/results_v2.json \
  --output converted_v1.txt \
  --bypass converted_bypass.json

# 应用 JSONL 编辑序列；默认每步编辑后立即增量运行
python3 -m aqeval edit --workspace sample_workspace \
  --script sample_workspace/edits/engineering_sequence.jsonl

# 中断快照测试
python3 -m aqeval run --workspace sample_workspace --stop-after parse
python3 -m aqeval run --workspace sample_workspace

# 强制串行和快照/缓存重建
python3 -m aqeval run --workspace sample_workspace --serial \
  --rebuild-cache --rebuild-snapshot
```

## 输出

- `output/active_v{1|2|3}.{txt|json|bin}`：当前版本交换产物。
- `output/bypass_active.json`：当前转换无法在目标版本表达的残差、精度标签、命名等级等。
- `reports/diagnostic.json`：规范化结果、偏差、跨版本一致性、降级统计、边界样本、失败清单、注入项、并发与阶段耗时。
- `cache/ledger.json`：按阶段、文件名和产物稳定排序的复用/重算账本。
- `snapshots/manifest.json`：全局版本、配置哈希、内容指纹、版本指纹及四阶段产物版本。

详细格式见 `docs/FORMATS.md`，实现说明见 `docs/ARCHITECTURE.md`，实际验证输出见 `docs/VALIDATION.md`、`validation_output.txt` 与 `validation_tests.txt`。
