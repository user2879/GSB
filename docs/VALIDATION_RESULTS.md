# 实际验证结果

验证时间：2026-11-27（Asia/Shanghai）。

## 环境

- 工作目录：`/home/zsl/feedback/uidfc5909d3-default-378`
- Python：`Python 3.11.0rc1`
- 虚拟环境：`.venv`
- numpy：`1.26.4`
- Tk：`tkinter.TkVersion = 8.6`，并完成真实 `tk.Tk()` 创建/控件构建/销毁冒烟。

当前机器 `sudo` 需要密码，无法直接 `apt install python3-tk`。已用 `apt-get download` 免 root 获取并解包 `libtk8.6`、`libtcl8.6` 及 Xft/Xss 等依赖到 `.sysroot/root`；`run_demo.sh` 会自动注入动态库和 Tcl/Tk 库路径。标准 WSL 上仍建议按任务要求执行 `sudo apt-get install python3-tk`。

由于基础 Python 的 `ensurepip` 不可用，`.venv` 使用：

```bash
python3 -m venv --without-pip --system-site-packages .venv
```

随后 `python -m pip install -r requirements.txt` 确认 `numpy 1.26.4` 满足要求。

## 复现命令

```bash
bash run_demo.sh
```

原始完整输出保存在：

- `out/full_verification_run.log`
- `out/unittest_verification.log`
- `out/demo_summary.json`

## 自动化测试

命令：

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

结果：

```text
Ran 13 tests in 15.560s

OK
```

覆盖：

- 样本/配置校验业务码与失败顺序。
- 基频/缩放/量化档位会真实改变定点整数与残差。
- 重复全量逐字段一致。
- 相同配置下串行与 4 线程执行逐字段一致。
- 训练长度变更只失效 `stats`、`error`。
- 基频变更只失效指定方案四个阶段。
- 两组包含增删切换/参数/样本变更的命令序列与空白全量重建一致。
- 批次预算、下一批优先和单样本配额恢复。
- 已提交快照恢复和 killed 临时目录忽略。
- 零参考值绝对误差分支、相邻档位翻转贡献。
- Tk 视图模型和真实控件冒烟。

## 关键验收数值

加载校验：

- 有效样本：12 条。
- 异常记录：6 条，全部跳过且未中断加载。
- 业务码计数见 `out/load_validation.json`。

增量与全量：

```json
{
  "canonical_business_fields_equal": true,
  "incremental_task_count": 132,
  "full_task_count": 132,
  "incremental_flip_count": 4,
  "full_flip_count": 4
}
```

主命令流包含 5 个批次检查点，复用任务数依次为 `0、40、0、66、88`：

- 第 2 批：样本替换/删除与方案参数变更后，40 个未受影响的 `linear_ramp` 任务复用。
- 第 4 批：训练长度变更后，3 个保留方案 × 11 个样本 × `position/score` 共 66 个任务复用，仅重算 `stats/error`。
- 第 5 批：新增方案并删除 `linear_ramp` 后，两个未变方案的 88 个任务全部复用，仅执行新方案的 44 个任务。

并发：

```json
{
  "canonical_business_fields_equal": true,
  "same_config_workers": 4,
  "serial_executed_task_count": 144,
  "thread_executed_task_count": 144
}
```

精确失效：

```json
{
  "length_only_invalidates_stats_error": true,
  "base_freq_only_invalidates_named_scheme": true
}
```

Tk 控制器额外手动验证：把 `sinusoidal.base_freq` 改为 `6543.0` 后，一批中 `reuse_task_count=88`、`executed_task_count=44`，正好对应另外两个方案的 88 个阶段任务复用和目标方案的 44 个任务重算。

预算：

- 批次预算 104：第一批完成 `s001,s002`，待办 10 条；第二批优先完成 `s003,s004`，剩余待办 8 条。
- 单样本配额 80：第一批完成 11 条，`s012` 成本 96 触发 `E_POINT_BUDGET_PENDING`；配额提高到 128 后，第二批优先完成 `s012`，待办清零。

恢复：

```json
{
  "recovery": "resumed_committed_snapshot",
  "global_version": 5,
  "killed_temp_directory_ignored": true,
  "current_pointer_exists": true
}
```

三级超差剖面（每剖面 12 个样本）：

- `position_first`：12 个首次停在 `position`。
- `score_first`：12 个首次停在 `score`。
- `weight_first`：12 个首次停在 `weight`。

详见 `out/stage_attribution_profiles.json`。
