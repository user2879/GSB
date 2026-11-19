# 光谱材质与颜色管理诊断器

这是一个离线、确定性、只依赖 Python 与 numpy 的颜色审核诊断应用。它从 380--780 nm、5 nm 步长的材质反射率和光源 SPD 出发，使用内置的 CIE 1931 2 度 / CIE 1964 10 度观察者做参考积分，再从对应色样对反解 3x3 白点映射矩阵，并对条件数、残差、量化、样本分布、积分口径、异常观测和崩溃恢复给出可复算账本。

应用不访问网络、不使用 Docker、不使用外部光谱数据库，也不允许在病态时静默退化为单位矩阵。

## 快速开始

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\run_demo.py --workspace .\demo_workspace --reset
```

也可以直接运行：

```bat
run_demo.cmd
```

或：

```powershell
.\run_demo.ps1
```

固定入口为 `run_demo.py`、`run_demo.cmd` 和 `run_demo.ps1`。完整结果写入 `demo_workspace/demo-report.json`，人读报告写入 `demo_workspace/VERIFICATION.md`。

测试命令：

```powershell
python -m unittest discover -v
```

## 命令入口

```text
python -m spectral_diag run       --workspace WS [--run-id ID]
python -m spectral_diag generate  --workspace WS
python -m spectral_diag solve     --workspace WS
python -m spectral_diag inject    --workspace WS --kind KIND [--params JSON]
python -m spectral_diag recover   --workspace WS
python -m spectral_diag closure   --workspace WS
python -m spectral_diag demo      --workspace WS [--reset]
```

常用配置参数包括 `--seed`、`--observer`、`--normalization`、`--reference-density`、`--quant-bits`、`--quant-range`、`--cond-threshold`、`--attribution-density`、`--drop-count`。完整运行配置也可通过 `--config-json` 传入。

### 固定序列示例

```powershell
# 正常生成、积分、求解、归因、材料迁移和原子提交
python -m spectral_diag run --workspace .\ws --run-id normal-001

# 注入共线色样，记录拒绝结果
python -m spectral_diag inject --workspace .\ws --run-id bad-001 `
  --kind collinear_sample --params '{}'

# 强制终止后会留下 ACTIVE 与未提交快照
python -m spectral_diag run --workspace .\ws --run-id crash-001 --crash-after-stage

# 恢复最近一次已提交稳定状态，并回收未提交孤儿
python -m spectral_diag recover --workspace .\ws
```

## 关键实现

- 内置 CIE 1931 2 度与 CIE 1964 10 度 5 nm 色匹配函数；高密度参考积分默认 0.5 nm、801 点、复合梯形积分。
- 参考真值来自解析连续高斯混合光谱在 0.5 nm 上的高密度积分，而不是对 5 nm 表再次插值。
- XYZ 归一化默认使用完美漫反射体 `Y=100`；光源/观察者权重先形成，再与反射率相乘。
- 矩阵按 `A @ M = B` 的最小二乘求解；对设计矩阵做 SVD 秩、条件数和逐样本 leverage 诊断。
- 条件数超过 `cond_threshold` 时，按“删除后条件数最小、同值取最小索引”的规则剔除并重解，完整保留剔除轨迹。
- 量化、样本分布、积分口径使用三因子 Shapley 归因；贡献和严格等于 `combined - baseline`，同时保留单独扰动残差和交互分配。
- 观察者/归一化切换只重算依赖闭包；切回旧口径时缓存命中，产物哈希恢复。
- 快照先写暂存目录并 fsync，目录改名为快照，最后原子替换 `CURRENT.json`；`CURRENT.json` 是唯一提交点。
- 强制终止后不生成退出报告。恢复只读取 `CURRENT.json` 指向的已提交快照，把未登记的历史外快照移入 `orphans/` 并记录。

详细公式和协议见 [TECHNICAL.md](TECHNICAL.md)。

## 工作区结构

```text
WS/
  CURRENT.json                   # 唯一提交指针
  committed-snapshots.json       # 已提交历史登记
  snapshots/<snapshot-id>/
    config.json
    scene.json
    samples.json
    matrix.json
    attribution.json
    injection.json
    version.json
    report.json
    CLOSED.json                  # 仅正常关闭
    manifest.json                # 文件哈希和关闭状态
  orphans/<snapshot-id>-<stamp>/ # 崩溃后回收的未提交快照
  runtime/active.json            # 运行中标记，正常关闭后删除
  logs/recovery.jsonl
  logs/last_recovery.json
```

## 错误码

| 错误码 | 含义 |
|---|---|
| `E_MATRIX_SINGULAR` | 色样集秩亏、条件数超限或没有可用剔除子集 |
| `E_QUANT_RANGE` | 量化范围、位深或 XYZ 观测越界 |
| `E_WHITE_POINT` | 光源白点出现零/非正分量 |
| `E_OBSERVER_INVALID` | 观察者标记非法 |
| `E_INTEGRATION_DENSITY` | 参考积分采样过粗 |
| `E_ATTRIB_MISMATCH` | 归因账本不可加、哈希不符或重算闭包不一致 |
| `E_SNAPSHOT_VERSION` | CURRENT/manifest/schema 版本不支持 |

## 实际验证结果

最近一次固定序列（`demo_workspace/demo-report.json`）全部通过：

| 检查 | 结果 |
|---|---|
| 场景两次生成逐字节一致 | PASS |
| 流水线两次运行 `result_sha256` 一致 | PASS |
| 已知 3x3 矩阵精确回解 | PASS，最大元素误差 `1.43e-15`（规范 JSON 12 位舍入后为 `0.0`） |
| 病态样本剔除 | PASS，条件数 `479231.9406348654 -> 19.343766365137`，剔除索引 `[3]` |
| 归因可加 | PASS，`0.004871839248 = 0.004871839248` |
| 观察者切换与恢复闭包 | PASS，恢复时重算集合为空，哈希恢复 |
| 五类异常注入定位 | PASS，首个观测分别为 `(0,0) / (0,1) / (7,1) / (0,0) / (0,0)` |
| 强制终止恢复 | PASS，子进程退出码 `86`，孤儿数 `1`，恢复快照等于崩溃前稳定快照 |
| 25 项单元/集成测试 | PASS |

默认场景的关键数值：

- 场景 SHA-256：`09f68b08ca4ba6c921e2b26d608ffd3407c53693df7451171b74a3e9e8253980`
- 流水线结果 SHA-256：`29507aa02aefa82e8ad421d0cdf7b821d68987e95c76ea1e545e823d13b9430a`
- 矩阵条件数：`20.165631361432`
- 矩阵残差 RMSE：`0.165583012549`
- 基线/全扰动残差：`0.165583012549 / 0.170454851798`

这些值可由 `python run_demo.py --workspace demo_workspace --reset` 重新生成。
