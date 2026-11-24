# 实际验证结果

验证环境：Python 3.11.0rc1、numpy 1.26.4、无容器、无网络、纯本地样本。

## 1. 自动化测试

命令：`python -m unittest discover -s tests`

结果：`Ran 21 tests ... OK`

| 测试文件 | 覆盖要求 |
| --- | --- |
| `tests/test_loader.py` | 要求 1：标识唯一、端口范围、正整数维度、序号递增、到达非负、批大小、未知工作器 |
| `tests/test_receipts_convert.py` | 要求 2/3：三版本解析、缺失标记、畸形/截断/非有限码、升降级旁路、往返引用一致 |
| `tests/test_attribution_resources.py` | 要求 4/5：覆盖引用、固定裁决顺序、不可用请求、定点指标、失效传播路径 |
| `tests/test_engine_e2e.py` | 要求 5/6/7/8：并发一致、快照版本与复用、快速/权威一致、`E_VIEW_STALE`、注入命中、CLI |
| `tests/test_incremental_equivalence.py` | 要求 7：逐步命令增量 vs 最终输入一次性全量 |

## 2. 端到端验收

### 正常样本（种子 20241125）

```
python -m receipt_ledger --config samples/workers.json --samples samples \
  --commands samples/commands.json --output output/accept-normal \
  --workers 4 --fast-budget-ms 5000
```

实测汇总：

- 请求 12、回执 15、账本条目 12（全部请求成功归属）
- 失效归属条目 1、失效批次 2（worker-b 版本切换 v2→v5、旧批次 v1）
- 失败清单 0；含 1 条 r3→r1 降级（旁路元数据）与还原升级，往返回执引用稳定
- 快速视图与权威账本 `attribution_summary` 逐字段一致

### 异常注入样本

```
python -m receipt_ledger --config samples_faulty/workers.json \
  --samples samples_faulty --commands samples_faulty/commands.json \
  --inject samples_faulty/injection.json --output output/accept-faulty
```

实测命中：`INJ-001/002/003/004` 全部命中；失败码包含
`E_NON_FINITE_DURATION`、`E_RECEIPT_TRUNCATED`、`E_DUPLICATE_REQUEST_ID`，
未知参数版本通过资源失效链（`unknown_param_version:v99`）命中并传播到归属。
故障场景失败总数 3，无命令级误伤。

## 3. 确定性与恢复

- 样本生成器连续两次生成，全部文件 sha256 一致；
- 并发度 1 与 8 产出的 `report.json`、`snapshot_manifest.json` 逐字节一致
  （墙钟字段不进入持久化产物）；
- 同一输出目录二次运行，四个阶段全部标记复用，账本与失败清单逐字段一致；
- 逐步应用 8 条命令（替换回执、新增批次、版本切换、批大小调整、规则修改、
  版本升级、降级旁路、旁路还原）的最终账本与一次性全量运行逐字段一致。

## 4. 静态检查

`python -m pyflakes receipt_ledger tools tests` 无告警。
