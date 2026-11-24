# 技术设计

## 模块划分

- `colmv/schema.py`：JSON 强校验、页目录校验、首个非法视图/字段定位。
- `colmv/storage.py`：确定性列式页编码/解码、物理完整性检查。
- `colmv/changes.py`：NDJSON 解析、序号和完整行形状校验。
- `colmv/fixed.py`：向零截断定点量化、定点除法、int64 饱和。
- `colmv/engine.py`：行状态、过滤/分组、增量中间结果、槽位、重扫、路径决策、误差界、规范化结果。
- `colmv/ledger.py`：SQLite 事务、账本链式摘要、快照整体替换、未提交批次恢复、方案陈旧标记。
- `colmv/replay.py`：账本逐字段重放、首差异定位、双路径字节比较。
- `colmv/evolution.py`：视图→分组→页引用传播、整体失效。
- `colmv/cli.py`：命令、错误信封、异常注入、崩溃注入。

## 列式页布局

```text
offset 0   : b"COLMVPG1\n"                 9 bytes
offset 9   : uint32 LE header_length
offset 13  : UTF-8 canonical JSON header
offset 13+H: payload
```

Header 结构示例：

```json
{
  "format_version": 1,
  "page_id": "p000",
  "start_row": 0,
  "end_row": 8,
  "row_count": 8,
  "columns": [
    {
      "name": "row_id",
      "type": "int",
      "dtype": "<i8",
      "null_bitmap_offset": 0,
      "null_bitmap_length": 8,
      "data_offset": 8,
      "data_length": 64,
      "data_sha256": "..."
    }
  ],
  "payload_length": 912
}
```

空值位图每行一字节（1=NULL，0=存在），因此声明长度与行数严格相等。列数据为 `<i8` 或 `<f8`。NULL 位置的编码值为 0；浮点列仍校验所有编码值为有限数。

## 行状态与重扫

基表页文件不可变。引擎在内存中维护：

```text
row_id -> 当前完整行
row_id -> 原始页 id / 虚拟 delta 页
row_id -> 最后变更序号 / 初始行序
group_id -> row_id -> 定点贡献
group_id -> page_id -> 当前引用行数
```

重扫一个分组时：

1. 从分组→页引用表取得稳定排序页集合。
2. 对每个基础页重新调用 `read_page_file`。
3. 用当前行覆盖页中同 `row_id` 的初始行；已删除行跳过。
4. 对 `delta` 页扫描当前插入行。
5. 重新应用过滤、空分组键排除和定点量化，替换整个分组中间结果。

因此重扫确实重新读取本地页文件，而不是仅复制驻留缓存。

## 槽位分配

状态是 `capacity` 个槽位和单调递增的整数 `clock`。触摸分组：

1. 已驻留：更新 `last_touch=++clock`，计命中。
2. 本批已被置换：拒绝重新驻留，进入重扫集合。
3. 有空槽：选最小槽位号驻留。
4. 无空槽：以 `(last_touch, view_id, typed_group_key)` 最小者作为牺牲者；整数和规范化键决定，不依赖哈希或浮点比较。

每批维护 `batch_denied`，保证最终驻留受影响组与重扫受影响组严格互补。

## 定点误差

对浮点值 `v`，定点单位 `q = trunc_toward_zero(v * 10^k)`，`|v*10^k - q| < 1`。整数输入误差为 0。

分组 `sum` 的误差上界是浮点值个数；`avg` 还包含一次除法截断，故有浮点输入时上界为 2 单位，仅整数输入时为 1 单位。全局按误差单位稳定排序，输出占比最高的段。

## SQLite 事务

每批两个事务阶段：

1. 意图事务：`batch_attempts` 写 `inflight`，`attempt_groups` 写计划分组。
2. 完成事务：批次结果、逐组状态、事件、游标、摘要、快照一起提交，并将意图标为 `committed`。

强制终止只可能留下第 1 种状态。恢复将 `inflight` 意图和计划分组标为 `unknown`，游标停在最后完整批次。

内容摘要采用链式 `SHA-256(previous_digest || batch_digest)`。快照 payload 带自己的 SHA-256 和格式版本；打开时摘要/版本/方案任一不匹配即删除整行并从最后提交批次重建。

## 方案演进

方案 fingerprint 由基表摘要、规范化视图、`scale_power` 和预算决定。变化时注册单调递增的新版本：

1. 新方案 `is_current=1`，旧方案 `is_current=0`。
2. 旧 `batches`、`group_states`、`events` 全部 `is_stale=1`。
3. 删除单行快照，当前结论不可见，等待整体重算。
4. 计算受影响引用：视图定义 diff → 当前匹配行 → 非空分组键 → 分组引用 → 行所在页。

演进不逐组修补旧结论。
