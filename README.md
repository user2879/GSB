# 多视点光度立体 × 可微渲染交替优化组件

仅依赖 Python 3.10+ 标准库与 NumPy 的逆向成像示例：在确定性合成的多视点
线性光观测上，让**独立实现**的光度立体（PS）求解器与可微渲染（DR）优化器
交替运行，逐轮对照两类求解语义、定位像素级冲突，并支持增量/全量渲染核对、
异常注入分类、原子持久化与强制终止恢复。无 GPU、无容器、无外部服务。

## 环境准备（Windows）

```bat
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

POSIX：`python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`。

## 一分钟上手

```bat
REM 生成 256x256 观测集（6 光源 x 2 相机 = 12 视点）
python -m photometric_altopt.cli generate --workspace workspace --size 256
REM 解析梯度有限差分核对
python -m photometric_altopt.cli gradcheck --workspace workspace
REM 交替优化并与真值核对
python -m photometric_altopt.cli run --workspace workspace --run-name baseline --check-truth
REM 强制终止后恢复
python -m photometric_altopt.cli run --workspace workspace --run-name demo --crash-after 2
python -m photometric_altopt.cli resume --workspace workspace --run-name demo
```

完整脚本：`examples/run_example.bat`（Windows）/ `examples/run_example.sh`（POSIX）。

## 工作区布局

```
workspace/
  observations/<obs_seed_size>/   观测栈、真值清单、高度场、口径 manifest
  trajectories/<run>/trajectories 逐轮损失/梯度/收敛原因 JSON
  trajectories/<run>/conflicts    逐轮像素级冲突清单
  trajectories/<run>/snapshots    逐轮 PS/DR 参数与渲染图版本戳（.npy + 哈希）
```

所有产物经“临时文件 + 原子替换 + `.committed` 标记”提交，未完成写入不构成状态。

## 模块与文档

* 设计与数值口径：`docs/DESIGN.md`
* 实际验证结果与性能：`docs/VERIFICATION.md`
* 业务码与异常分类：`photometric_altopt/errors.py`
* 测试：`python -m pytest`（34 个用例）

## 核心业务码

| 码 | 含义 |
| --- | --- |
| `E_NONFINITE_ALBEDO` | 反照率参数出现 NaN/Inf |
| `E_ZERO_LIGHT` | 零向量光源无法单位化 |
| `E_DEGENERATE_VIEWS` | 有效视点少于求解所需 |
| `E_VIEW_SKIPPED` | 管线层渲染跳过指定视点 |
| `E_RENDER_DIVERGE` | 增量与全量重渲染分歧 |
| `E_NO_CHECKPOINT` | 恢复时无已提交快照 |

病态条件数越界为状态型标记（`ill_conditioned`），不中断求解。
