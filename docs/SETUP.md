# 环境搭建与运行说明

## Windows 10/11（题目目标环境）

前置：已安装 Python 3.11 或 3.12，并勾选 pip 组件。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install numpy jsonschema
python -m pip install -e .
```

## Linux（本次实际验证环境，系统缺少 ensurepip 时）

```bash
# 1. 用户目录引导 pip（仅系统无 pip 时需要）
curl -sS -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
python3 /tmp/get-pip.py --user
# 2. 无 ensurepip 时创建不带 pip 的虚拟环境，再在环境内安装 pip
python3 -m venv --without-pip .venv
.venv/bin/python /tmp/get-pip.py
.venv/bin/pip install numpy jsonschema
.venv/bin/pip install -e .
```

## 生成确定性样本

```bash
.venv/bin/python scripts/generate_samples.py
# 产物：samples/basic/{seeds,slot_sources}.jsonl、vocab.json、
#       build_config.json、score_params.json、batch_plan.json
```

## 运行命令

```bash
# 增量运行（自动快照续跑），并内置影子全量对照
.venv/bin/sft-diag verify --sample-dir samples/basic --output-dir out/run --up-to-batch 5

# 从空白输出目录一次全量构造（不允许复用已有快照）
.venv/bin/sft-diag full --sample-dir samples/basic --output-dir out/full --up-to-batch 5

# 仅增量运行，不做影子对照
.venv/bin/sft-diag run --sample-dir samples/basic --output-dir out/run --up-to-batch 3 --no-shadow
```

输出目录内容：

| 文件 | 含义 |
| --- | --- |
| `report.json` | 诊断报告（计数、平均分、失效/复用、快照版本、失败清单、影子差异） |
| `candidates.json` | 确定性候选样本序列（含采样序号、模板/槽位来源） |
| `scores.json` | 四维度分数与总分（float64 数值 + 六位有效数字文本） |
| `dedup_groups.json` | 近重复分组、代表样本、剔除记录、边界标注、全部 Jaccard 边 |
| `snapshot/` | 清单与三阶段快照、临时目录 |
| `shadow_report.json` | 影子全量构造报告（verify/默认 run） |

## 测试与验收

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/acceptance_check.py
```

退出码：`0` 成功；`2` 出现 `E_CONFIG_INVALID` 或 `E_CONFIG_MISMATCH`。
