# 流水线制品缓存与复用诊断工作台

面向 Windows 本地环境、纯 Python 标准库实现的云原生流水线制品缓存诊断系统。
无第三方依赖；`tkinter` 由 Windows 官方 Python 安装器自带，缺失时自动降级为
等价的无头文本诊断视图。

## 目录结构

```
pipeline_cache_ledger/
├── run_demo.py            # 三场景可重复演示（冷启动/版本更新/缓存库升级）
├── pcl_cli.py             # 命令行入口（check/init/run/view/snapshot/restore）
├── src/pcl/
│   ├── errors.py          # E_SPEC/E_CORRUPT/E_DIGEST/E_REF/E_CYCLE/E_SNAPSHOT
│   ├── digest.py          # 确定性摘要规则 sha256:<hex(原始文件字节)>
│   ├── manifest.py        # 清单加载与全量强校验（字段路径报错）
│   ├── entries.py         # v1/v2/v3 条目二进制信封编解码与 v3 归一
│   ├── store.py           # 缓存库、原子写入、打开升级、检查点续做
│   ├── graph.py           # 引用图、成环检测、稳定拓扑失效、删除保护
│   ├── scheduler.py       # 全局+每阶段双预算并行调度（逻辑时钟）
│   ├── state.py           # sqlite3 会话状态（登记/命中/失效/调度）
│   ├── snapshot.py        # 版本化快照、摘要校验、整体替换恢复
│   ├── engine.py          # 编排：编译计划、判定、失效、权威重算、快照
│   ├── samples.py         # 确定性样本清单与制品文件生成
│   ├── report.py          # 稳定顺序文本诊断报告
│   └── workbench.py       # Tkinter 五标签页工作台（无头时文本降级）
├── tests/                 # 36 个 unittest 用例，仅用标准库
├── workspace_main/        # 演示生成：主工作区
└── workspace_legacy/      # 演示生成：含 v1/v2 历史条目的缓存库
```

## 确定性摘要规则

```
digest = "sha256:" + sha256_hex(制品文件原始字节)
```

按字节精确读取，不做换行转换；同一文件在 v1/v2/v3 三种来源格式下摘要恒等。

## 缓存条目磁盘格式（自描述二进制信封）

```
magic b"PCL1" | uint8 格式版本 | uint32 载荷长度 | JSON 载荷
              | b"END1" | uint32 重复长度 | sha256(载荷) 的 hex
```

- v1 载荷：key/version/digest
- v2 载荷：v1 + size + stage_id
- v3 载荷：v2 + refs（排序后的引用三元组）+ compressed
- 双长度字段用于截断自检，载荷摘要用于字节翻转检测；
  任何解码失败均抛 `E_CORRUPT` 并携带条目键与字节偏移。
- 新写入恒为 v3；旧条目在内存归一为 v3 时返回 `backfilled` 补齐字段集合。

## 错误码

| 代码 | 触发场景 | 关键详情 |
| --- | --- | --- |
| E_SPEC | 清单任一校验失败，整体拒绝、不跳过单条 | `paths` 字段路径列表 |
| E_CORRUPT | 条目长度/魔术字/摘要损坏，或检查点不可读 | `key` + `offset` |
| E_DIGEST | 磁盘内容与缓存条目摘要不一致 | 两侧摘要 |
| E_REF | 删除仍被引用的制品 | 引用者键列表 |
| E_CYCLE | 引用关系成环 | 环上制品键 |
| E_SNAPSHOT | 快照格式/长度/摘要不支持或不匹配 | 差异字段 |

## 关键设计

- **双预算调度**：真实线程 + 条件变量；全局槽位上限与每阶段配额同时硬约束，
  逻辑时钟记录排队/开始/结束时点与槽位号；不同交错下命中结论集合恒定。
- **失效传播**：引用边指向上游→下游；Kahn 拓扑 + 字典序保证失效顺序与清单
  遍历顺序无关；传播轨迹持久化到 sqlite。
- **升级续做**：每原子重写一个条目即 `os.replace` 检查点；重启仅依据检查点
  的 `processed` 集合继续，已处理条目不重复升级，`upgrade_verification()`
  可核验磁盘版本与检查点完全一致。
- **快照整体替换**：快照含整数格式版本与载荷摘要；恢复时清空并重建状态库、
  缓存视图与命中状态，绝不增量合并；旧快照标记只读并稳定列出差异版本。
- **UI 一致性**：界面数据来自与权威单线程重算 `authoritative_recompute()`
  相同的纯判定函数 `_evaluate()`，演示中对两者逐项断言一致。

## 运行方式（Windows / Linux / WSL 均可）

```
python run_demo.py                       # 三场景演示 + 总结
python pcl_cli.py check                  # 环境自检
python pcl_cli.py init <workspace>       # 生成确定性样本
python pcl_cli.py run <workspace>        # 并行登记与命中判定
python pcl_cli.py view <workspace>       # 诊断报告（有 tkinter 则开窗）
python pcl_cli.py snapshot <workspace> s1
python pcl_cli.py restore <workspace> s1
python -m unittest discover -s tests     # 测试套件（PYTHONPATH=src）
```

## 验证结果

见同目录 `VERIFICATION.md`（测试 36/36 通过，三场景演示断言全部成功）。
