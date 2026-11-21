# 存储配额仲裁工作台（quota_workbench）

面向本地 Windows 环境的存储配额仲裁工作台：当同一存储卷的配额结论来自多来源
记录（容量声明 / 速率声明 / 派生声明，各自携带整数格式版本）时，按显式仲裁规则
裁决唯一可复核结论，判定其可否迁移到另一来源口径，对不可迁移情形做证据级冲突
归因，并把裁决与归因固化为可整体替换、可重放的 SQLite 账本。

## 环境

- Python 3.11+（Windows 官方发行版自带 Tkinter 与 sqlite3）
- `pip install numpy`
- 不依赖 Docker、网络、真实存储服务或外部账号

## 目录结构

- `quota_workbench/loader.py` — 声明加载与强校验（E_FIELD：未知版本 / 缺字段 / 越界）
- `quota_workbench/normalize.py` — 三类来源多版本编解码与归一（长度字段自洽、默认值补齐、保留来源版本号）
- `quota_workbench/arbitrate.py` — 仲裁规则 arb-v1：来源类型偏置 → 格式版本降序 → 来源标识升序
- `quota_workbench/events.py` — 整数定点令牌桶判定（SCALE=1000，向零截断，E_ORDER）
- `quota_workbench/migrate.py` — 目标来源口径迁移复核（consistent / indistinguishable / conflict）
- `quota_workbench/attribute.py` — 冲突归因：口径差异 / 版本差异 / 默认补齐差异，含可复算中间项
- `quota_workbench/invalidate.py` — 失效传播与结论视图整体替换（未受影响行逐字节校验）
- `quota_workbench/ledger.py` — SQLite 账本：批次事务、E_SNAPSHOT 整体替换恢复、E_REPLAY 差异定位
- `quota_workbench/pipeline.py` — 流水线编排与按卷批次提交
- `quota_workbench/app.py` — Tkinter 操作界面（`python -m quota_workbench.app`）
- `scripts/gen_samples.py` — 固定种子（20261122）确定性样本生成
- `scripts/kill_recovery_test.py` — 强制终止恢复演练（`--kill-after N` 在第 N 批后 `os._exit(42)`）
- `tests/test_workbench.py` — 19 项验收测试

## 样本

`python scripts/gen_samples.py` 生成（重复运行摘要相同）：

- `samples/declaration.json` — 12 卷、44 条来源记录（三类来源各含 v1/v2 两个历史版本）
- `samples/events.ndjson` — 3000 条确定性访问事件
- `samples/rules.json` — 仲裁规则版本说明与目标口径（仅接受 v2+ 记录）

内容摘要（sha256）：

- declaration: `ba8afb283783cba75cf5f67fc971e4215860d79ee8644cda8bf63c13052211c3`
- events: `6fa8dd92950f3de7df1b07835948d1fa4c86e6c9a1af98cd34599fc24a09cfc3`
- rules: `79ada56ec91b9e4898406d7fd585258f238d25cfe2ddeb0cc722fde41669dcf2`

## 运行

```bat
pip install numpy
python scripts\gen_samples.py
python scripts\kill_recovery_test.py            :: 全量提交 + 重放校验
python -m unittest discover -s tests            :: 19 项验收测试
python -m quota_workbench.app                   :: Tkinter 操作界面
```

## 实测验证结果（本机 Linux 容器，Python 3.11）

- 仲裁唯一性：12 卷 × 3 指标均得出唯一结论；8 种随机读取顺序下结论逐字节一致。
- 迁移复核：36 个 (卷, 指标) 中 24 个 consistent、12 个 conflict。
- 证据归因：每个冲突输出 3 类成因贡献与中间项（v_current/v_step1/v_step2/v_step3），
  可复算，残差为 0 或显式标注；排序按贡献降序、指标名升序。
- 事件判定：allow / throttle / over_quota / indeterminate_boundary 四类判定均覆盖；
  时间戳倒退以 E_ORDER 拒绝。
- 强制终止恢复：`--kill-after 5` 退出码 42，游标停在 vol-05（52 行结论）；
  续跑后 12 批全部提交，重放校验通过；只读备份与检查点文件生成于
  `samples/backup/` 与 `samples/checkpoint.txt`。
- 快照整体替换：修改 vol-01-cap1 后重开账本，E_SNAPSHOT 报明 sources_digest
  不一致，整体替换恢复后重放通过，内容摘要与干净运行一致（`fa5e1d57…`）。
- 重放差异定位：篡改账本 arbitration 行 value 字段，E_REPLAY 精确定位到
  首个不一致的卷、指标与字段（field="value"）。
- 全部 19 项 unittest 通过。

注：本容器未安装 Tkinter，界面在 Windows 官方 Python 上运行；核心引擎与界面解耦，
全部业务验证均可经 CLI 与测试完成。
