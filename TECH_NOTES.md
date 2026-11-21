# 配置中心租约中间件 — 技术说明

面向 Windows 本地环境的纯标准库组件（`sqlite3` / `json` / `threading` /
`tkinter`），无需第三方包，不使用容器。所有数据均为本地文件。

## 1. 目录结构

- `lease_middleware/spec.py`：配置项与订阅客户端声明加载、强校验（`E_SPEC`）。
- `lease_middleware/codec.py`：r1/r2/r3 租约记录编解码、归一与记录级升级。
- `lease_middleware/storage.py`：SQLite 表结构、只追加账本、订阅视图、游标、检查点。
- `lease_middleware/engine.py`：签发/续约/过期/撤销状态机、原子推送、视图整体替换。
- `lease_middleware/upgrade.py`：低版本逐条升级、检查点续做（`E_UPGRADE`）。
- `lease_middleware/replay.py`：从账本恢复并重放，逐项比对（`E_REPLAY`）。
- `lease_middleware/verifier.py`：独立验证器，直接读 SQLite 与声明，不依赖引擎。
- `lease_middleware/inject.py`：异常注入（只改输入文件/参数，不改源码）。
- `lease_middleware/dashboard.py`：可选 Tkinter 仪表盘，无 Tk 时优雅降级。
- `samples/`：确定性声明、变更脚本与 r1/r2/r3 历史租约样本。
- `tests/`：19 个 `unittest` 用例。
- `demo.py`：命令行演示入口。

## 2. 三种租约格式与归一

- r1：`lease_id, config_key, expires_at_ms`
- r2：r1 + `lease_version, client_id`
- r3：r2 + `renewal_count, value_digest, parent_lease_id`

补齐字段使用显式默认值（租约版本默认 1、续约次数 0、父引用 `null`），升级
绝不改写 `lease_version`。r1 缺少版本号时，批量解码会从同一 `lease_id` 的
r2/r3 记录解析权威版本，因此同一租约在三种来源下归一结果逐字段一致
（见 `python demo.py interop`，输出 `identical: true`）。

## 3. 过期判定与状态机

状态 `granted -> renewed -> expired/revoked`，合法迁移在
`storage.LEGAL_TRANSITIONS` 集中声明。过期用 `time.monotonic()` 真实计时
（后台 sweeper 或操作前即时检查），墙钟毫秒仅用于落盘展示。续约带
`expected_version` 做乐观校验，不匹配返回 `E_VERSION` 且不写任何状态。

## 4. 原子推送与游标

每次 `config_push` 在同一 SQLite 事务内：写账本事件 + 推进
`client_cursors` + 更新订阅视图游标。游标只能等于账本中该客户端已记录的
最大推送版本，越过账本立即 `E_CURSOR`；未订阅的键不计入任何客户端。

## 5. 升级与强制终止续做

每条旧记录在内存升级后以 r3 原子重写，原记录存入同列 `backup_json` 作为
只读备份；每完成一条立即原子刷新检查点（含记录 SHA-256）。重启只处理检查点
之外的记录；检查点哈希与账本不符时抛 `E_UPGRADE` 停止，不猜测续做点。
`demo.py upgrade --crash` 用子进程 `os._exit(137)` 模拟强制终止（不生成退出
报告），再由主进程续做。

## 6. 订阅视图整体替换

租约集合、配置版本或客户端集合变化时，`rebuild_subscription_view` 先删除
整张 `subscription_view` 与游标表再整体重算，识别 added/removed/upgraded
客户端并输出旧/新游标。`renew(..., commit=False)` 的续约只存在内存，重建时
标记为 `re_grant_required`（未知），强制终止后不承诺退出报告。

## 7. 重放与独立验证

- `replay.py` 按账本顺序重建租约与客户端，校验状态迁移合法、续约次数严格
  递增、同键无重复推送，并与租约表/游标表逐项比对，差异定位到首个客户端、
  租约与字段（`E_REPLAY`）。
- `verifier.py` 用独立 SQL 逻辑复核同一组不变量，输出稳定排序的 JSON 总结。

## 8. 业务码

| 码 | 触发场景 |
| --- | --- |
| `E_SPEC` | 声明校验失败（整体拒绝，带字段路径） |
| `E_LEASE` | 租约记录非法、状态不可操作 |
| `E_VERSION` | 乐观续约版本不匹配 |
| `E_CURSOR` | 游标越过账本或回退 |
| `E_UPGRADE` | 检查点与账本不符 |
| `E_REPLAY` | 账本不可复现/重放不一致 |

## 9. Windows 运行方式

```powershell
python -c "import tkinter"          # 官方安装包自带，应无输出
python samples\generate_samples.py
python demo.py interop
python demo.py run
python demo.py upgrade --crash
python demo.py rebuild
python demo.py inject spec|lease|cursor
python demo.py verify demo_runtime\run\ledger.sqlite samples\config.json
python -m unittest discover -s tests
```

结构化错误日志写入 `demo_runtime/inject/structured.log`（JSONL，每行含
`code/client_id/lease_id/field`）。
