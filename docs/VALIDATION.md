# 验证结果（本地实测）

环境：WSL Ubuntu，Python 3.11.0rc1，venv 位于 `.venv/`，仅 CPU；
numpy 2.2.6、Pillow 11.3.0、PyYAML 6.0.4；Tk 8.6 经 `vendor/tk` 提供。
回环 127.0.0.1 可用、临时目录可写，无 GPU、无外部服务。

## 1. 全量测试

命令：`python -B -m pytest tests/ -q`

结果：`48 passed in ~91s`

| 测试文件 | 用例数 | 覆盖验收点 |
| --- | --- | --- |
| `test_fixedpoint_rng.py` | 7 | 定点枚举/量化残差、档位真实改整数、确定性随机源 |
| `test_formats.py` | 6 | v1 定宽契约、v2↔v1 旁路往返、缺失字段、噪声码映射 |
| `test_loader_validation.py` | 5 | OOV、PNG 头/尺寸/模式、运动表形状、配置枚举、坏命令 |
| `test_encoding_kernel.py` | 8 | 词袋命中、融合顺序与残差、潜变量确定性、分块借用守卫 |
| `test_scheduler.py` | 3 | 预算待办、续算优先、不重复扣点、批量上限 |
| `test_pipeline_equivalence.py` | 6 | 样本字节可重复、精确失效、增量=全量、并发无关、账本 |
| `test_snapshot_interactive.py` | 6 | 原子提交、版本冲突拒绝、冷启动恢复、诊断、交互摘要 |
| `test_termination_recovery.py` | 1 | 子进程 SIGTERM 落盘后仅依赖已提交快照恢复 |
| `test_edit_commands.py` | 3 | 图像替换/删除、运动全局失效、非法命令不改状态 |
| `test_tier_and_crc.py` | 3 | 档位跃迁改路径与账本、解码顺序、PNG CRC 篡改检出 |

## 2. 端到端演示

命令：`python scripts/run_demo.py`

实测输出要点：
- `rejected: []`（样本全部通过校验）
- `completed_frames: [0, 1, 2, 3, 4, 5]`
- `consumed_budget: 144`（6 帧 × 6 步 × 4 点）
- 待办事件码集合仅 `E_BUDGET_PENDING`，默认批预算 18 下跨批续算完成
- 同目录产出 `frames.v1.txt`、`frames.v2.jsonl`、`frames.roundtrip.jsonl`

## 3. 双版本往返

- v2→v1：引导浮点 0.7004 四舍五入为千分比 700，残差 0.0004 与精度标签、
  被覆盖字段写入 sidecar；再升回 v2 后内部表示与直接解析 v2 逐字段相等
  （`test_formats.py::test_v2_to_v1_to_v2_field_equality`）。
- CLI：`fxrender roundtrip` 对 6 条帧记录双向转换均成功。

## 4. 增量与全量一致性

施加提示替换、权重变更、运动调整、去噪步数变更、量化档位切换、
交换版本 v2→v1→v2、图像删除的完整命令序列后：
- 逐帧 `pixel_digest`、分块行摘要、最终潜变量整数完全一致；
- 当前有效缓存视图（按 stage/frame/step/tile_row 坐标，匹配活动交换版本）
  的缓存键逐键一致（`test_full_equals_incremental_after_command_sequence`）；
- 并发度 1/2/4 下像素结果一致（`test_worker_count_does_not_change_pixels`）；
- 增量缓存额外保留 superseded 失效历史条目，记录 `edit:*` 失效依据，
  不影响当前有效产物的等价性。

## 5. 预算、待办与档位

- 预算 10、单步 4：首批扣 2 步（8 点），第 3 步记 `E_BUDGET_PENDING`，
  下一批待办优先处理；全部批次扣减键无重复。
- 档位 small/medium/large 的 `batch_steps`、合步策略、解码顺序
  （topdown vs outside-in）真实不同，并反映到点数账本。

## 6. 快照与强制终止

- 版本/配置哈希不匹配时提交被拒（`E_SNAPSHOT_VERSION`），线上快照不被覆盖。
- 子进程收到 SIGTERM 后写出 `snapshot.json` 与 `diagnostics.json`，
  诊断含 signal=15、flushed=true、已完成帧与已消耗预算；恢复仅读取已提交快照。

## 7. 确定性样本可重复性

两次独立 `generate_sample_bundle(..., frame_count=6)` 产出的全部文件
（词表、提示、6 张 PNG、motion.json、render.yaml、edits.json、manifest）
逐字节相同：`byte-different files: []`。

## 8. Tk 支持

`. .venv/bin/tk_env.sh` 后 `import tkinter` 成功（TkVersion 8.6），
可创建并销毁 Tk 根窗口；无显示环境下 UI 自动降级为文本摘要。
