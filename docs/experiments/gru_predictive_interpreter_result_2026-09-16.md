# 第一版Predictive GRU State Interpreter探索结果

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：run + validate
- Origin Date：2026-09-16
- Verification Status：ANALYZED
- Version Label：gru_predictive_result_v1

## 实验目的

在停止继续扩大HMM调参后，建立一个最小GRU时序基线，检验学习连续时间动态是否能避免HMM的状态塌缩，并保持或改善设备退化时间排序。

本实验不使用normalized lifetime训练。GRU只通过当前及过去的embedding预测下一时刻embedding。

## 模型和数据权限

模型为单层因果GRU：

```text
bearing-centered z (8D)
→ train-only standardization
→ GRU hidden state (8D)
→ linear head
→ predicted next z (8D)
```

总参数量为504。

| 角色 | Bearing | 用途 |
|---|---|---|
| Training | Bearing1_1–Bearing1_3 | 拟合标准化统计量与GRU参数 |
| Validation | Bearing1_4 | 选择next-step MSE最低的checkpoint |
| Old holdout | Bearing1_5 | checkpoint锁定后评价 |

每个bearing都先减去自身前15个embedding的均值。GRU Level定义为隐藏状态相对前15个隐藏状态均值的距离，并使用与距离基线相同的10点移动平均。

## 训练结果

- 固定随机种子：20260916；
- 训练轮数：300；
- 训练next-step MSE：0.9121降至0.0188；
- 验证next-step MSE：0.8243降至0.0552；
- 最佳checkpoint：epoch 300；
- 未触发patience early stopping。

验证损失在最后一轮仍继续下降，因此当前训练没有显示过拟合拐点，但也不能确认已经完全收敛。本实验不根据结果自动延长训练。

各bearing的next-step MSE为：

| Bearing | Next-step MSE |
|---|---:|
| Bearing1_1 | 0.0246 |
| Bearing1_2 | 0.0200 |
| Bearing1_3 | 0.0128 |
| Bearing1_4 | 0.0552 |
| Bearing1_5 | 0.0313 |

## State Interpreter比较

| Bearing | 距离Level ρ | centered HMM ρ | Predictive GRU ρ |
|---|---:|---:|---:|
| Bearing1_1 | 0.964 | 0.916 | 0.964 |
| Bearing1_2 | 0.994 | 0.831 | 0.999 |
| Bearing1_3 | 0.954 | 0.946 | 0.966 |
| Bearing1_4 | 0.849 | 0.962 | 0.894 |
| Bearing1_5 | 0.950 | 0.896 | 0.932 |
| 五个bearing描述性平均 | 0.942 | 0.910 | 0.951 |

GRU在所有bearing上都产生非恒定Level，没有出现HMM的状态塌缩。

相对centered HMM，GRU在Bearing1_1、1_2、1_3和1_5上更高，在Bearing1_4上低0.068。相对距离Level，GRU在Bearing1_2、1_3、1_4上提高，Bearing1_1基本持平，Bearing1_5低0.018。

五个bearing角色不同，描述性平均值不能作为正式总体性能或显著性结论。

## 平滑性

GRU的Level step标准差在五个bearing上都低于距离基线：

| Bearing | 距离Level step std | GRU step std |
|---|---:|---:|
| Bearing1_1 | 0.1425 | 0.0620 |
| Bearing1_2 | 0.1459 | 0.0564 |
| Bearing1_3 | 0.0846 | 0.0313 |
| Bearing1_4 | 0.1108 | 0.0252 |
| Bearing1_5 | 0.2315 | 0.0959 |

这说明隐藏状态距离产生了更平滑的连续轨迹。GRU的backward-step比例没有在所有bearing上一致下降，因此不能声称它已经完全消除短期反向波动。

## 结论

第一版GRU证明了以下工程与研究可行性：

1. 不使用寿命标签，仅通过下一embedding预测，也能训练出非塌缩的连续状态。
2. GRU整体时间排序与距离Level接近，并在4/5个bearing上优于centered HMM。
3. GRU保留连续变化，相比HMM不需要把轨迹压缩为少量离散阶段。
4. 相比距离Level，GRU轨迹更平滑，但旧holdout Bearing1_5没有改善。
5. 当前证据支持继续研究GRU，但不支持宣布GRU已经优于简单距离基线。

HMM应继续保留为时序对照，而不再作为主要调参方向。Attention目前没有必要加入；应先验证GRU对随机初始化的稳定性。

## 下一步

保持模型结构、训练目标、数据划分和全部超参数不变，使用至少3个随机种子重复训练，报告：

- 每个bearing的GRU Level Spearman ρ均值与标准差；
- next-step MSE均值与标准差；
- 不同seed产生的Level轨迹一致性；
- 是否存在某个seed状态塌缩；
- GRU相对距离Level和centered HMM的胜负是否稳定。

在完成随机种子稳定性后，才决定是否进行leave-one-bearing-out GRU实验或加入Attention。

## 统计与解释边界

- 所有bearing此前均已查看，本实验不是盲测。
- Bearing1_4用于checkpoint选择，不是独立测试证据。
- Bearing1_5是已经查看过的旧holdout，不能恢复严格盲测资格。
- 本轮只有一个随机种子。
- Spearman相关性只评价时间排序，不代表物理损伤、RUL或RL决策收益。
- next-step MSE与健康状态质量不是同一个指标。
- GRU隐藏状态距离仍是模型内部相对量。

## Fallacy Scan（11/11）

| 检查项 | 状态 | 说明 |
|---|---|---|
| Simpson's paradox | 未发现 | 逐bearing报告，未只报告聚合均值 |
| Ecological fallacy | 未发现 | 未由总体均值推断单个bearing |
| Berkson's paradox | 注意 | 只分析一个公开数据集的一个工况 |
| Collider bias | 不适用 | 未加入控制变量 |
| Base-rate neglect | 不适用 | 不是分类或诊断实验 |
| Regression to the mean | 注意 | Bearing1_4因旧实验异常而持续被重点分析 |
| Survivorship bias | 无法确认 | 数据集run筛选过程不由本实验验证 |
| Look-elsewhere effect | 注意 | 三种方法和多个指标均为探索性比较 |
| Garden of forking paths | 注意 | 架构选择发生在已见数据上，必须与后续验证区分 |
| Correlation ≠ causation | 注意 | 时间相关不能证明物理退化因果关系 |
| Reverse causality | 注意 | normalized lifetime只是时间代理 |

## 软件与工件验证

- 新增4项GRU测试；
- 仓库全量测试：79 passed；
- 616个时间步、15行method-bearing指标和5行比较结果完整；
- checkpoint记录输入、配置、seed、标准化统计量和最佳epoch；
- 实验成功执行一次，没有自动重试。

## 工件

- 模型：`src/state_interpreter/gru_temporal.py`
- 配置：`configs/xjtu_gru_predictive_v1.json`
- 脚本：`scripts/run_gru_interpreter_exploratory.py`
- 协议：`docs/experiments/gru_predictive_interpreter_protocol_2026-09-16.md`
- 结果：`results/2026-09-16_gru_predictive_interpreter_exploratory/`

