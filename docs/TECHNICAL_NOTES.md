# 技术说明与需求映射

## 1. 数据流

```text
固定种子 SampleSpec
  -> synth.build_sample：内置字模、旋转/抖动、PNG、truth
  -> loader.load_workspace：PNG/分辨率/模式/整数/栏区间/序号/配置校验
  -> LeaseExecutor：按 image_id:column_rank 发租约和代次
  -> vision/pipeline：窗口评分 -> 候选掩码 -> 连通域 -> 合并/分裂 -> 定点回归
  -> 阅读排序/真值匹配/冲突决断
  -> CacheStore + 缓存键：窗口/连通域/关键点/阅读顺序
  -> Scheduler：预算、配额、比较上限、ThreadPool、稳定顺序回收
  -> Ledger + atomic snapshot + Tkinter
```

## 2. 关键模块

| 需求 | 实现 |
| --- | --- |
| PNG 容器与像素模式 | `src/vertical_workbench/pngcodec.py` |
| 样本/真值/标注校验 | `src/vertical_workbench/loader.py` |
| 确定性样本与字模 | `src/vertical_workbench/synth.py`, `fonts.py` |
| 租约状态机/真实回收 | `src/vertical_workbench/leases.py`, `clock.py` |
| 滑窗/投影/梯度/CC/合并分裂 | `src/vertical_workbench/vision.py` |
| 定点回归/阅读/冲突 | `src/vertical_workbench/pipeline.py` |
| 缓存键/版本差异 | `src/vertical_workbench/cache.py`, `versions.py` |
| 预算与并发顺序 | `src/vertical_workbench/scheduler.py`, `engine.py` |
| 编辑命令流 | `src/vertical_workbench/studio.py` |
| 原子阶段快照 | `src/vertical_workbench/snapshot.py` |
| Tk 诊断台 | `src/vertical_workbench/gui.py` |
| 异常注入 | `src/vertical_workbench/injection.py`, `configs/injections/` |
| 验收报告 | `src/vertical_workbench/acceptance.py`, `scripts/run_acceptance.sh` |

## 3. 缓存键依赖

- 窗口：`stage=window + 图像像素指纹 + 矩形 + kernel_hash + 量化指纹 + 相交栏租约代次`。缓存命中时不会再次读取或评分该窗口。
- 栏连通域：`stage=connected_component_column + 图像指纹 + 栏号 + 依赖窗口键集合 + kernel_hash + 量化指纹 + 租约代次`。
- 连通域：图像指纹、碎片像素掩码摘要、kernel/quant/租约代次。
- 关键点：连通域键、kernel/quant/租约代次、该图标注文档摘要。
- 阅读顺序：有序关键点键集合、真值摘要、标注摘要、量化和全部栏租约代次。

标注整体替换时先比较全局 `version_id` 与 `content_digest`。若仅版本号变化而摘要不变，判定为 relabel，窗口、连通域、关键点和阅读键均可命中；若文档摘要变化，只有该图的关键点和阅读顺序键失效，窗口与连通域继续复用。

## 4. 稳定性措施

- 所有公开列表按文件名、栏号、`(y,x)` 或显式 ID 排序。
- `ThreadPoolExecutor` 只负责执行，结果通过 `run_ordered()` 按输入稳定顺序 `future.result()` 回收。
- 验收和测试使用固定起点的 `ManualClock`；生产 GUI/CLI 使用 `MonotonicClock`。
- 租约代次按栏区间单调递增；错误代次 CAS 失败时不推进公认代次。
- 预算先在稳定顺序上规划，再提交并发任务，避免线程完成顺序改变预算消耗。
- 快照写入独立临时目录，完成后原子发布目录并替换 `snapshots/current` 符号链接。

## 5. 冲突优先级

相邻预测点先与 48px 内真值做稳定贪心一一匹配。候选冲突按固定优先级选择：

1. `跨栏`，优先级 3；
2. `缺号`，优先级 2；
3. `错序`，优先级 1。

同时保留 `candidate_types` 和具体原因；未匹配真值会单独生成 `缺号` 记录，不静默丢弃。

## 6. 预算规则

- 全局点数预算按栏内真值点数在执行前预留。
- 栏点数大于 `column_point_quota` 时记录 `E_QUOTA_EXCEEDED`，只中止该栏本批。
- 预算不足时从该栏起保留稳定待办顺序，记录 `E_POINT_BUDGET_PENDING`。
- 下一批候选顺序为“历史待办优先 + 新稳定序列”。
- 原始碎片两两比较次数精确等于 `n(n-1)/2`；超过 `max_component_comparisons` 的栏进入比较上限事件和待办。

## 7. 真实验证入口

```bash
python scripts/check_environment.py
.venv/bin/python -m pytest -q
bash scripts/run_acceptance.sh
```

验收产物：

- `reports/acceptance_report.json`：机器可读完整报告。
- `reports/verification.md`：摘要。
- `reports/injection_report.json`：四类注入编号命中证据。
- `reports/pytest_output.txt`：验收器嵌套执行的 pytest 输出。

最近一次开发机验证为 17 个测试全部通过；完整验收的实时布尔结果以新生成的 `reports/acceptance_report.json` 为准。
