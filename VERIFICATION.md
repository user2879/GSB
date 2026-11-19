# 验证记录

## 1. 环境

- 操作系统：WSL2（Linux 6.18，x86_64）；引擎代码仅用标准库，Windows 原生
  Python 3.10+ 同样可运行（锁与原子写已提供 POSIX/Windows 双分支）。
- Python：3.11.0rc1；pytest 8.3.4（离线复制到项目 `.venv`，未访问网络）。
- 引擎运行时零第三方依赖；样本与真值均本地确定性构造，无系统时钟/随机源。

## 2. 复现步骤

```bash
python -m venv .venv            # 离线环境可直接复用本项目 .venv
.venv/bin/python scripts/generate_samples.py
.venv/bin/pytest -q
.venv/bin/python -m audit_engine.cli samples/generated/*.json \
  --checkpoint-dir checkpoints/acceptance \
  --report reports/acceptance.json \
  --concurrency 4 --backend thread --timeout 10 --max-records 1000
```

## 3. 实测结果

### 3.1 pytest 全量

```
63 passed in ~1.0s
```

覆盖矩阵：

| 需求场景 | 测试位置 | 关键断言 |
| --- | --- | --- |
| 路径穿越与越界深度 | tests/test_pathnorm.py、tests/test_engine.py | E_TRAVERSAL、深度 1/2 |
| 符号/硬链接逃逸 | tests/test_pathnorm.py | E_LINK_ESCAPE、link_steps、落点路径 |
| 链接环有界终止 | tests/test_pathnorm.py | 跳数上限内终止 |
| 口径混用与八进制误差 | tests/test_permission.py | truncate/nearest 误差界方向与精确分数 |
| 临界不确定 | tests/test_permission.py、tests/test_engine.py | critical=true 且无确定越权码，eps=189/64 |
| 拒绝覆盖 | tests/test_acl.py、tests/test_engine.py | E_DENY_SHADOW |
| 继承越权 | tests/test_acl.py、tests/test_engine.py | E_INHERIT |
| ACL 排列顺序无关 | tests/test_acl.py | 8 次随机重排裁决完全一致 |
| 策略冲突 | tests/test_engine.py | E_POLICY_CONFLICT + 所涉策略列表 |
| 解析拒绝不中断批 | tests/test_parser.py、tests/test_engine.py | E_PARSE ×3，合法记录继续处理 |
| 超时（真实墙钟） | tests/test_engine.py | E_TIMEOUT、elapsed≥预算、dropped>0 |
| 超记录数预算 | tests/test_engine.py | E_BUDGET、processed=5、dropped=7 |
| 并发等价 | tests/test_concurrency_recovery.py | 并发 1/2/4 与串行裁决签名一致 |
| 进程后端硬终止隔离 | tests/test_cli_and_termination.py | 慢文件 terminate，同批文件 ok |
| 检查点续接 | tests/test_concurrency_recovery.py | 仅完整检查点内文件被跳过 |
| 版本/策略变化重建 | tests/test_concurrency_recovery.py | should_rebuild=True |
| 锁冲突 | tests/test_concurrency_recovery.py | E_LOCK |
| 原子报告 | tests/test_concurrency_recovery.py | 无 .part 残留、JSON 完整 |
| 生成器确定性 | tests/test_generator_determinism.py | 两次输出字节级一致 |
| 真值对照 | tests/test_engine.py | 一致 35、篡改后 E_TRUTH_MISMATCH |
| 退出码 | tests/test_engine.py、tests/test_cli_and_termination.py | 0/1/2/3/4 分支 |

### 3.2 跨并发度与跨后端等价性

对 9 份清单、4 条规范化策略计算每条 `(record_id, policy)` 裁决的
SHA-256 签名：

```
thread-1/2/4/8、serial、process-2 的摘要完全相同 => EQUIVALENT
所有运行退出码一致（=1，存在业务违规）。
```

### 3.3 端到端审计聚合（reports/acceptance.json）

```json
{
  "code_counts": {
    "E_DENY_SHADOW": 4,
    "E_INHERIT": 4,
    "E_LINK_ESCAPE": 7,
    "E_PARSE": 3,
    "E_POLICY_CONFLICT": 12,
    "E_TRAVERSAL": 4
  },
  "dropped_records": 0,
  "files": 9,
  "processed_records": 35,
  "rejected_records": 3,
  "truth_consistent": 35,
  "truth_inconsistent": 0
}
```

业务码按每条策略裁决计数（4 策略），故为记录数的倍数；
E_PARSE 按被拒绝记录计数。CLI 退出码 1（存在违规）。

### 3.4 恢复与强制终止边界实测

- 首批 4 个文件后检查点含 4 条完成记录；第二批续接后报告含全部 9 个文件，
  检查点完成记录变为 9。
- 截断的 JSON 检查点被识别为无效并整体重建，不基于残缺状态续接。
- 进程后端对超时文件执行真实 `terminate()`，同批其余文件状态保持 `ok`。
- 外部 `kill -9` 场景不承诺退出报告；正常路径下报告与检查点均经
  临时文件 + `os.replace` 原子落盘，目录无 `.part` 残留。

## 4. 已知边界与说明

- 引擎不创建真实文件系统链接对象；链接声明以虚拟路径输入参与解析，
  符合任务中“环境不允许真实链接时”的降级约定。
- 八进制误差界采用保守线性求和（最坏同号叠加），临界判定按位翻转距离
  独立评估；临界记录只标记 `critical`，不输出确定性越权结论。
- Windows 原生运行时建议将 `.venv\\Scripts\\python` 加入 PATH；
  文件锁自动切换为 `msvcrt.locking`，原子写依赖 NTFS 的 `MoveFileEx`
  替换语义（Python `os.replace` 已封装）。
