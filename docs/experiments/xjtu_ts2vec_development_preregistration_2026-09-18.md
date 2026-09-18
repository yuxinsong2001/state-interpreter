# Condition 3紧凑TS2Vec开发实验预注册（2026-09-18）

## 目标

检验官方TS2Vec核心的紧凑适配是否能够学习比65维工程特征更一致的跨bearing健康表示，同时避免进一步强化bearing身份。

## 数据边界

- 开发：B3_1、B3_2、B3_3，执行三折LOBO。
- 验证：B3_4，仅在健康与identity双门槛均通过后允许读取。
- 最终holdout：B3_5，本阶段禁止读取。
- Preflight不接受cache或dataset路径，本步骤未打开任何NPZ。

## 固定输入

每个bearing继续使用前15点进行因果z-score并应用`signed_log1p`。每个训练bearing均匀选择16个长度128的片段，包含生命周期两端，使两个训练bearing贡献相同数量的实例。

## 固定模型

- 输入：65维；
- timestamp representation：64维；
- hidden：32；
- dilated residual depth：6；
- mask probability：0.5；
- instance/temporal contrast权重：0.5/0.5；
- temporal unit：0。

这是保留官方核心目标的紧凑适配，不宣称复现论文默认320维、depth 10的完整benchmark配置。

## 固定训练

- optimizer：AdamW；
- learning rate：0.001；
- batch size：8；
- fixed iterations：200；
- 每次iteration更新SWA模型；
- seeds：20260918、20260921、20260924；
- CPU；
- 最终iteration checkpoint，不用outer-test选模型。

## 因果推理

每个时间点只使用截至当前的最近128步，序列开始处仅在左侧用NaN padding；编码后取最后一个timestamp representation。任何未来上下文均禁止进入state。

## 固定下游评价

在每个outer fold中，TS2Vec只用两个训练bearing进行自监督训练。冻结encoder后，使用两个训练bearing各300个等距时间点拟合alpha=1的Ridge，将64维representation映射到normalized lifetime；第三个bearing只用于评价Spearman。

同时在同一个fold/seed encoder下对三个开发bearing执行五段held-lifetime nearest-centroid identity probe。

## 双重门槛

健康门槛：

- 三个bearing平均Spearman均为正；
- 至少两个bearing平均Spearman≥0.5；
- 每个bearing至少两个seed为正。

Identity门槛：九个fold/seed probe的平均准确率≤0.80，低于原65维特征的0.903基线。

两个门槛必须同时通过，才允许进入B3_4。仅提高RUL相关性但identity仍很高，不足以证明获得了共享健康表示。

## Preflight结果

三个核心工件哈希匹配，8项定向测试、173项全仓库测试通过。状态：`ts2vec_preflight_passed_no_cache_read`。
