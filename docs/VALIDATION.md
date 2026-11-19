# 实际验证记录

执行日期：2026-11-20（Asia/Shanghai）

## 环境

| 项 | 结果 |
|---|---|
| Python | 3.11.0rc1 |
| SQLite | 3.37.2 |
| NumPy | 2.4.6（numpy 快路径） |
| Tkinter | 当前 Linux 沙箱缺失；无桌面兼容测试通过，Windows 官方 Python 自带 |
| 本地端口 | `127.0.0.1:46553` 绑定成功 |
| SQLite 路径 | `artifacts/preflight_final.sqlite` 可写 |

此前 NumPy 未安装时，纯 Python 回退路径同样通过全部测试。

## 自动化测试

命令：

```bash
python -m unittest discover -s tests -t . -v
```

结果：`Ran 41 tests ... OK`；另以 `bucketpack.numeric._np = None` 运行同一测试集，41 项同样通过。覆盖：

- 规格字段/位图/边界/scale/宽度位/超大浮点边界/重复 JSON 键与稳定首错路径；
- 图边、拓扑、稳定环、`E_REF` 引用者与整条依赖链级联删除；
- Decimal 截断、int64 饱和、误差上界、边界落入右桶；
- 包区间连续、行宽/包预算、`E_WIDTH`；
- 全局宽度与每源并发配额、稳定调度；
- 值/空值/边界/删除变更的失效传播、最小重算、全量一致；
- 合并统计与直接统计、`E_DIVERGE` 首差异；
- 账本完整/部分恢复、未知包重放、版本与内容不一致整体替换、`E_LEDGER`；
- CLI 四类注入和真实 `os._exit(17)` 恢复；
- Tk 独立视图模型稳定排序。

## 样例基线

| 项 | 结果 |
|---|---|
| spec 摘要 | `e20fc0a2cf0fc482d205adb0b8b465787e4f839e0842aea338a28bd841bc8688` |
| 内容摘要 | `f50f9112f4f20be52d1e4f7581571780de042e3303aaceab00d50cc10ef87abe` |
| 行数 | 56 |
| 包数 | 11 |
| 最大并发列宽 | 152 |
| 全局列宽预算 | 160 |
| 成本 | 5600 µ |
| 成本预算 | 1,000,000 µ |
| 合并统计 = 直接统计 | true |

## 增量结果

### 值改写 + 空值位图调整

`samples/changes_value.json`：

- 失效桶列：`events.latency_band`、`events.latency_fine`、`events.weight_band`、`orders.region_band`、`orders.spend_band`、`orders.spend_half`。
- 受影响包：`events:p00000`、`events:p00001`、`orders:p00000`、`orders:p00001`。
- 复用包：其余 7 个；包摘要逐字节一致。
- 增量 vs 全量：true。

### 边界调整

`samples/changes_buckets.json`：

- 两个源的相关桶链全部失效；
- 两个数据源 11 个包全部重切；
- 未引用桶列版本保持；
- 增量 vs 全量：true。

### 规则删除

`samples/changes_delete.json` 删除未被引用的 `orders.items_aux`：

- `orders` 行宽从 16 变为 13，重新切成 6 包；
- `events` 4 包完整复用；
- 增量 vs 全量：true。

删除被 `orders.spend_half` 与统计引用的 `orders.spend_band` 得到 `E_REF`，引用者列表包含规则字段 `input.name` 和统计字段 `refs`。

## 账本恢复

- 完整批次：11/11 包验证通过，状态 `COMPLETE`。
- 模拟强制终止：只提交前 2 包；恢复时 2 包通过，其余 9 包标记 `UNKNOWN` 并逐包重放，最终 `COMPLETE`。
- 真实 CLI `--crash-after 2`：进程退出码 17；已提交 `events:p00000`、`orders:p00000` 两包通过校验，`recover` 成功重放其余 9 包并到达 `COMPLETE`。本轮产物为 `artifacts/crash_cli_final_20261120.sqlite` 与 `artifacts/crash_cli_final_20261120_recovery.json`。
- 格式版本篡改：`E_REPLAY`，首个差异包 `events:p00000`，账本整体替换回格式版本 1、11/11 `COMMITTED`。
- 内容摘要篡改：`E_REPLAY`，账本整体替换为待验证快照。

## 异常注入

| 注入 | 业务码 | 关键字段 |
|---|---|---|
| `width` | `E_WIDTH` | 主键 `9665`，行宽 12，预算 1 |
| `bucket` | `E_BUCKET` | `data.rows[0].channel`，int/str 冲突 |
| `cycle` | `E_CYCLE` | `cycle=["injected.cycle"]` |
| `ledger` | `E_LEDGER` | `$.ledger.write`，operation=`commit_pack` |

结构化日志：`artifacts/logs/demo_final.jsonl` 等，均为 JSON Lines。

## 演示报告

`python -m bucketpack demo --spec samples/spec.json --out-dir artifacts/demo` 返回 `ok=true`，全部检查为 true。报告文件：`artifacts/demo/demo_report.json`。
