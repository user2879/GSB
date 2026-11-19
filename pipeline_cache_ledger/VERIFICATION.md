# 验证记录

环境：Python 3.11.0rc1、sqlite3 3.37.2；本容器未安装 tkinter 共享库
（无 sudo 权限），工作台自动走等价文本视图，Tk 代码路径在带 tkinter 的
Windows 官方 Python 上直接生效。全部验证仅使用本地确定性样本，无网络访问。

## 1. 单元/集成测试

命令：

```
PYTHONPATH=src python -m unittest discover -s tests
```

结果：`Ran 36 tests ... OK`（36/36 通过）。覆盖映射：

| 业务要求 | 测试文件/用例 |
| --- | --- |
| 1 清单强校验 E_SPEC + 字段路径、整体拒绝 | `test_manifest.py`（10 用例） |
| 2 三版本编解码、长度自洽、偏移、E_CORRUPT、补齐字段 | `test_entries.py`（8 用例） |
| 3 原子升级、崩溃续做、检查点核验 | `test_store_upgrade.py`（4 用例） |
| 4 E_DIGEST 双侧摘要、跨版本一致性 | `test_entries.py`、`test_integration.py` |
| 5 拓扑失效稳定性、E_CYCLE、E_REF 删除保护 | `test_graph.py`（4 用例） |
| 6 双预算不超限、等待时点、交错结论恒定 | `test_scheduler.py`（3 用例） |
| 7 快照摘要校验与整体替换、只读保留 | `test_integration.py::test_snapshot_full_replacement_restore` |
| 8 无头稳定渲染、样本与演示 | `test_integration.py::test_diagnostic_view_renders_stably_without_tk` |

## 2. 三场景演示实际结果

命令：`python run_demo.py`（退出码 0，无 Traceback）。

### 场景 1：冷启动登记与双重预算

- 7 个任务、4 个阶段；全局上限 2、每阶段配额 1。
- 首次运行全部 `MISS/no_existing_entry`；记录排队/开始/结束逻辑时点与槽位，
  每阶段任务均出现等待（配额生效）。
- 以 `global_limit=3, stage_quota=2` 并加入线程抖动重跑，7 个制品全部
  `HIT/digest_refs_version_match`。
- 权威单线程重算与并行结论逐项一致：`authoritative recompute agreement: True`。

### 场景 2：版本更新与失效传播

- `base-lib` 内容变更后稳定失效顺序：
  `base-lib -> codegen -> core-bundle -> api-image -> release-notes -> web-image`
  （与清单遍历顺序无关，由 Kahn + 字典序保证）。
- 重跑计划：未受影响的 `config` 保持 HIT，失效链上制品全部
  `MISS/entry_previously_invalidated`，根制品重新登记。
- 快照比对：`versions_equal=false`，稳定列出 `only_in_snapshot` 与
  `only_on_disk` 摘要差异及状态签名差异；恢复执行整体替换，旧快照转为只读。

### 场景 3：缓存库打开升级（含模拟强杀）

- 打开前磁盘格式：`base-lib=v1, config=v2, codegen=v3`。
- 首次打开在完成 1 个升级后模拟进程终止；检查点
  `processed=["base-lib"]` 已落盘。
- 重开后仅续做剩余条目，本进程升级集合为 `{codegen, config}`，v3 条目不重写；
  核验输出 `all_current=true`、`processed_covers_disk=true`。
- v1 补齐字段：`compressed, refs, size, stage_id`；
  v2 补齐字段：`compressed, refs`。
- 同一 `core-bundle` 分别按 v1/v2/v3 编码后解码，摘要三者完全相同；
  两种调度交错（2/1 与 3/2）在独立工作区上的命中结论集合完全一致。

## 3. 复现实验

```
cd pipeline_cache_ledger
PYTHONPATH=src python -m unittest discover -s tests
python run_demo.py
python pcl_cli.py check
```

Windows 上若使用 python.org 安装包，`tkinter` 默认随 Python 一起安装，
`pcl_cli.py view <workspace>` 会直接打开五标签页图形工作台。
