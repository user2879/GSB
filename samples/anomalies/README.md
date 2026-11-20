# 异常样本

- `borrow_cycle/borrows.json`：与 `samples/pools.json` 配合，形成 `alpha -> beta -> alpha` 借用环，业务码 `E_BORROW`。
- `missing_close/trace.ndjson`：删除正常轨迹中的 `seq=15` 关闭事件，造成连续序号缺口与未归还借用，业务码 `E_GAP`。
- `version_mismatch/pools.json`：声明版本 `99`，用于与版本 `1` 起步的已有账本做恢复比对，业务码 `E_VERSION`。

异常注入只复制并修改这些输入文件或账本参数，不修改源码。
