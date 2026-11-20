# tabcluster-workbench

面向表格数据的聚类与密度估计并行调度与资源预算诊断工作台。仅依赖
Python 标准库 + NumPy + PyYAML，不引入容器或本地服务，数据全部由固定
种子程序生成。

## 运行

```bash
pip install -r requirements.txt        # 可用本地 wheel 离线安装
python run.py --config config/experiment.yaml \
              --data config/data_description.yaml \
              --injections config/injections.yaml \
              --workdir out/demo
python -m unittest discover -s tests -v   # 13 个验收测试
```

产物布局：`<workdir>/snapshots/{preprocess,cluster,evaluate,diagnostics}/`
+ `manifest.json`，`<workdir>/cache/*.json`（内容寻址缓存），
`<workdir>/outputs/summary.json`（诊断汇总）。所有写入均为
临时目录/临时文件 + `os.replace` 整体替换。

## 阶段与确定性设计

1. **加载**：YAML 配置 / 数据描述 / 注入清单；校验失败只记业务码不中断。
2. **数据生成**：单一 `RandomState(seed)` 按列描述固定顺序消费；支持
   数值列、类别列、高基数列、冗余列、缺失率与样本量多档放大
   （`scales.small/medium/large`）。
3. **预处理**：缺失值用训练段（前 `train_fraction` 行，固定累加顺序、
   `math.fsum`）统计量填充；类别按字典序编码、众数并列按编码序破缺；
   标准化或分位数变换（并列值取平均秩、线性插值、越界截断，规则
   `average_rank_linear_clip` 写入元数据）。产物携带
   `列描述指纹 + 数据指纹 + 参数指纹` 派生的内容指纹。
4. **聚类**：
   - KMeans：固定初始质心（均值最近点 + 最远点确定性选取，并列取小
     下标），逐轮记录簇内平方和，空簇显式记录并重播种；
   - DBSCAN：行序访问、行序邻域、FIFO 扩展，逐点访问计迭代，全噪声
     显式记录；
   - GMM：由本包确定性 KMeans 播种的 EM，固定收敛规则，协方差奇点按
     `reg_covar` 递增 ×10 吸收（有界）并记录事件，支持
     full/diag/spherical/tied。
5. **调度**：候选按 `方法名 + 超参指纹` 稳定键排序；线程池并发计算
   （纯函数、互不依赖），预算账本在稳定键序上串行结算，因此任意并发度
   与预算耗尽时机下，排名、逐样本簇归属、指标表顺序、扣减序列与失败
   清单顺序都与单线程串行逐字段一致。单候选超配额记
   `E_CAND_BUDGET_EXHAUSTED`；全局预算不足记 `E_ITER_BUDGET_PENDING`
   并保留待办队列。
6. **评估**：轮廓系数、Calinski-Harabasz、密度对数似然（GMM 用模型
   似然，其余用分簇高斯）；噪声点排除并记录；排名按主指标降序、并列
   按方法名与超参字典序。
7. **缓存与失效**：预处理/聚类/评估/诊断四级内容指纹缓存键；改某方法
   超参、预处理方式或替换单条样本时，仅受影响候选及其下游重算，其余
   直接命中缓存。
8. **快照**：四阶段各自先写临时目录再整体替换；清单记录全局版本号、
   配置哈希、注入清单哈希、已完成候选、已消耗预算与各阶段版本；启动
   时哈希不一致记 `E_SNAPSHOT_MISMATCH` 并拒绝复用旧产物。

## 业务码

| 代码 | 含义 |
| --- | --- |
| `E_CONFIG_INVALID` | 未知列/类型不符/结构缺失/并发数非正等 |
| `E_CLUSTER_ZERO` | 簇数为零 |
| `E_BUDGET_INVALID` | 全局预算或单候选配额非正（回退默认值） |
| `E_CAND_BUDGET_EXHAUSTED` | 候选超出单候选配额，已中止 |
| `E_ITER_BUDGET_PENDING` | 全局预算耗尽，候选保留在待办队列 |
| `E_SNAPSHOT_MISMATCH` | 快照版本/配置哈希不一致，拒绝复用 |
| `E_INJECTED_FAULT` | 注入清单触发的候选故障 |
| `E_CANDIDATE_FAULT` | 候选未预期异常（隔离后继续） |

## 验证结果（本机，Python 3.11 + NumPy 1.26）

- `python -m unittest discover -s tests`：13/13 通过，覆盖——两次运行逐
  字段一致；并发度 1/2/8 及预算压力下结果一致；预算账本可审计且待办
  保留；配额中止；配置业务码不中断解析；缓存精确失效（改超参/预处理/
  单样本）；快照不一致拒绝复用；注入故障隔离；增量执行与全量执行一致
  （含逐样本簇归属）；空簇与全噪声显式记录；分位数并列规则元数据。
- 样例配置（small 档 240 行、11 候选、并发 4）：预算消耗 1037/100000，
  注入故障被隔离，其余候选正常评估。
- medium 档（960 行、并发 8、标准化）：预算消耗 3149，运行正常。

> 说明：实现全部使用 `pathlib`/`os.replace` 与相对路径，Windows 与
> Linux 均可运行；目录整体替换采用“临时目录 → 旧目录移开 → 改名 →
> 清理”的顺序以兼容 Windows 的文件锁定语义。
