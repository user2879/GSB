# 实际验证记录

验证环境：Python 3.11.0rc1、NumPy 1.26.4、标准库 sqlite3（SQLite 3.37.2），
单机本地执行，无网络依赖、无容器。

## 1. 自动化测试

命令：`python -m unittest tests.test_all -v`

结果：21 个测试全部通过（`Ran 21 tests ... OK`），覆盖：

1. 身份摘要逐位确定性，以及维度/种子/口径/身份版本任一变化均改变摘要；
2. 四种量化口径与固定四层结构、合成顺序；
3. E_SPEC 首个违规标识与字段定位、重复标识/序号、损坏 JSON；
4. 盖章幂等、每后端独立单调序号、基线验章全通过；
5. 身份摘要演进与口径版本演进分别以 E_STALE + 字段区分；
6. 校验值篡改 => E_IDENTITY；
7. 账本干净重开不替换；摘要不符 => E_SNAPSHOT，整体替换并生成只读备份；
   格式版本不符同样整体替换；
8. 虚拟时钟调度稳定可复算，全局/每后端预算不变量；真实线程池预算执行；
9. 强制终止后只恢复已提交记录，未提交令牌标记 unknown 并补盖；
10. 重放篡改以 E_REPLAY 定位首个令牌与字段；
11. 独立核对工具 E_AUDIT，包括“内容与摘要被同步伪造”的隐蔽场景。

## 2. 端到端演示（python scripts/demo.py，退出码 0）

- 步骤 1：三后端（fp32/int8/int4）生成四层矩阵与逐层摘要、身份摘要，
  并打印矩阵形状、scale 与合成顺序；
- 步骤 2：8 枚令牌盖章；重复盖章 `token-001` 返回 `reused=True`、
  `stamp_seq` 保持 1；alpha 序号 1,2,3，beta/gamma 各自从 1 起；
- 步骤 3：基线 8 枚令牌验章全部 `OK`；
- 步骤 4：身份版本演进 => `E_STALE / identity_digest`；
  口径版本演进 => `E_STALE / token_spec_version`；恢复基线后有效集合仍为 8；
- 步骤 5：16 个任务在全局=2、每后端=1 下的虚拟时钟轨迹，存在排队等待，
  预算不变量校验通过；步骤 5b 真实线程池完成同样 16 个任务；
- 步骤 6：关闭后重开账本，内容摘要一致，替换报告为 `None`；
- 步骤 7：在 `token-004` 提交后模拟强制终止；恢复结果
  committed=token-001..004、unknown=token-005..008，各后端序号
  `{alpha:3, beta:1, gamma:0}`；重放 4 条已提交记录无 E_REPLAY；
  补盖后 unknown 为空、重放 8 条全部一致。

## 3. 异常注入（退出码均为 0，即业务码被正确识别）

| 命令 | 实际业务码 | 定位 |
| --- | --- | --- |
| `demo.py --inject corrupt-spec` | E_SPEC | backend-gamma / hidden_dim |
| `demo.py --inject tamper-identity` | E_IDENTITY | token-001 / checksum |
| `demo.py --inject bad-snapshot` | E_SNAPSHOT | content-digest，并生成只读 `snapshot.backup-001.db` 后整体替换 |

独立核对：

- 干净账本：`audit.py --spec samples/spec.base.json --ledger run/main.db`
  输出 `[AUDIT OK] 共核对 8 条盖章记录，全部一致`；
- 隐蔽篡改（修改验章状态并伪造内容摘要）：手工验证输出
  `E_AUDIT OK: token-003 verification.accepted`，已固化为测试用例。

## 4. 产物

- `run/main.db`：主账本（含 meta 格式版本与内容摘要）；
- `run/crash.db`：崩溃恢复演示账本；
- `run/snapshot.backup-001.db`：只读替换备份（权限 `-r--r--r--`）；
- `run/demo.jsonl`：结构化 JSONL 日志（14 条事件，含 E_STALE/E_IDENTITY/
  E_SNAPSHOT 与恢复事件）。
