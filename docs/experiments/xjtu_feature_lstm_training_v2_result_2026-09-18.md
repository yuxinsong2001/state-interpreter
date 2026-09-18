# Condition 3 Feature LSTM开发训练v2结果（2026-09-18）

## Material Passport

- Experiment ID: `xjtu_condition3_feature_lstm_training_v2`
- Type: 3-fold LOBO × 3 seeds
- Status: completed; validation gate failed
- Verification status: `ANALYZED`
- Development data: B3_1–B3_3
- Protected data: B3_4/B3_5未读取

## 方法

每个bearing使用前15点进行因果z-score校准，再应用锁定的`signed_log1p`变换，构造10步窗口。Feature BiLSTM预测窗口末端的normalized RUL，评价时使用`1-clipped predicted RUL`作为health progress。

三折LOBO中每次使用两个bearing训练、一个完全留出bearing评价。运行三个固定seed，每次训练50 epochs，保存最终epoch；测试bearing不参与early stopping或checkpoint选择。

## 结果

| 留出bearing | Mean Spearman | Seed STD | 正相关seed | Mean backward fraction |
|---|---:|---:|---:|---:|
| B3_1 | 0.341 | 0.047 | 3/3 | 0.461 |
| B3_2 | 0.037 | 0.417 | 2/3 | 0.489 |
| B3_3 | 0.394 | 0.192 | 3/3 | 0.485 |

B3_2三个seed的Spearman为0.163、0.472和−0.525，说明结果对初始化高度敏感。九个模型的最终训练Huber loss均较低（约0.00021–0.00041），但LOBO排序仍较弱，表明低训练误差没有转化为可靠的跨bearing状态泛化。

## 预注册门槛

- 三个bearing平均Spearman均为正：通过。
- 至少两个bearing平均Spearman≥0.5：失败（0个）。
- 每个bearing至少两个seed为正：通过。
- 总门槛：**失败**。

因此不允许进入B3_4验证，更不能读取B3_5。该结论不是“Feature LSTM永远无效”，而是当前65维特征、早期相对校准、10步窗口和监督RUL目标的组合，没有在Condition 3开发集上形成足够稳定的跨bearing表示。

## 解释边界与11项谬误检查

- Simpson：主结果按bearing报告，不使用pooled指标掩盖差异。
- Ecological fallacy：评价单位明确为bearing，不推断到现场设备总体。
- Berkson/selection：XJTU-SY是实验室run-to-failure数据，外部适用性有限。
- Collider：未进行因果条件化，不提出因果结论。
- Base-rate neglect：未定义故障事件率，不做事件检测声明。
- Regression to mean：无干预前后设计，不解释为干预效果。
- Survivorship：序列为完整run-to-failure，但实验室样本选择仍构成边界。
- Look-elsewhere：fold、seed、epoch和指标均在运行前锁定。
- Forking paths：稳定缩放在训练前按数值门槛选定，未按LOBO性能选择。
- Causation：时间/RUL只作为监督代理，不等同物理损伤原因。
- Reverse causality：不作因果方向声明。

覆盖：11/11。

## 下一步建议

停止在当前Feature LSTM上继续调seed、epoch或阈值，也不消耗B3_4。下一步应先分析三个方面：特征级跨bearing分布偏移、窗口长度10是否覆盖有效时间尺度，以及RUL监督是否迫使模型学习bearing身份/生命周期长度。根据诊断证据，再决定复现退化起点/phase分解，或转向TS2Vec等自监督时间表示。
