# 实际验证结果

## 环境

- OS：Linux 容器（组件仅使用跨平台 API，Windows 使用 `py -3` 同构运行）
- Python：3.11.0rc1（满足 >= 3.10）
- NumPy：2.4.6（满足 >= 1.24），通过 `get-pip.py` 用户级安装
- 未使用 Docker / 容器，未访问外部业务目标，样本全部本地确定性生成

## 复现命令

```bash
python3 -m pip install --user -r requirements.txt
python3 examples/generate_sample_data.py
python3 -m unittest discover -s tests          # 16 个用例
python3 examples/run_demo.py                  # 增量 vs 全量逐字段对比
```

## 单元 / 集成测试结果

`python3 -m unittest discover -s tests` 实测 `Ran 16 tests ... OK`，覆盖：

| 测试文件 | 覆盖点 |
| --- | --- |
| `test_algorithms.py` | 最大匹配长度/词频/字典序决断、OOV 单字标记、维特比确定性、平滑命中计数、配置与词典校验 |
| `test_consistency.py` | 新增/删除/词频/概率表替换/单句替换完整编辑序列与空白全量逐字段等价、复用句不重算 |
| `test_snapshot.py` | 快照续跑复用、哈希不一致抛 `E_CONFIG_MISMATCH`、报告原子写且含哈希 |
| `test_concurrency_isolation.py` | workers=1/2/4/8 输出逐字段一致、稳定排序键、阶段异常隔离与持久化 |
| `test_signals.py` | 优雅停止后已完整提交快照，重启续跑补齐且复用已提交句 |

## 编辑序列增量失效实测（run_demo.py）

```
[2/6] 新增词条 '训练模型'，失效句子: ['S003']
[3/6] 调整词频 '学习'->200，失效句子: ['S003', 'S008']
[4/6] 删除词条 '野外'，失效句子: ['S006']
[5/6] 替换概率表 v2，失效句子数: 12
[6/6] 替换单句文本 S006，失效: True
=== 逐字段一致性 ===
句子顺序一致: True
分词/词性/得分逐字段一致: True
失败清单顺序与内容一致: True
最终判定: PASS
```

每一步仅重算失效句，其余句子进入复用清单；最终增量工作区与空白全量工作区在词序列、词性序列、局部路径得分、归一化对数概率、平滑命中、低置信度清单、句子顺序、失败清单顺序上完全相等。

## 续跑与配置不匹配实测

- 同输入第二次运行：`reused=11, recomputed=0`，快照版本递增，零重算。
- 词典追加词条后用旧快照续跑：`E_CONFIG_MISMATCH: 快照与当前输入不一致，拒绝续跑: dict_hash`，进程退出码 `2`。

## 异常隔离实测（fault_injection 钩子）

```
首次运行 failures:
  S3 tagging E_STAGE_FAILED InjectedStageError
  S5 oov E_STAGE_FAILED InjectedStageError
成功句子: ['S1', 'S2', 'S4']
二次运行复用: ['S1', 'S2', 'S4']
二次运行重算: []
二次运行失败句子: ['S3', 'S5']
```

隔离句在后续运行中不再产出结果，成功句全部复用、零重算；对隔离句调用 `replace_sentence_text` 可显式解除隔离。

## 样本分词与词性标注实测

完整逐句输出见 `examples/output/sample_segmentation_output.txt`，例如：

```
S004: 乒乓球拍/n 卖完/v 了/d   logprob=-1.608091 smoothing_hits=0
S008: 机器/n 学习/v 算法/n 依赖/v 高质量/a 数据/n
失败清单:
  S001 load E_SENTENCE_INVALID ValueError - 句子 ID 重复
  S013 load E_SENTENCE_INVALID ValueError - 行格式应为 句子ID<TAB>文本 且字段非空
```

## 验收产物清单

- `examples/output/demo_output.txt`：编辑序列与一致性结论
- `examples/output/unit_test_output.txt`：16 个测试详细结果
- `examples/output/exception_isolation_output.txt`：异常隔离实测
- `examples/output/sample_segmentation_output.txt`：逐句分词/标注结果
- `examples/output/demo/`：增量与全量结果 JSON、各次报告、两套快照、最终词典与概率表
