# Predictive GRU多随机种子稳定性实验协议

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：plan
- Origin Date：2026-09-16
- Verification Status：UNVERIFIED
- Version Label：gru_predictive_multiseed_protocol_v1

## 研究问题

第一版Predictive GRU在单个seed上产生非塌缩、较平滑的连续Level。本实验只改变随机初始化，检验该结论是否依赖特定seed。

## 固定因素

- 输入latent CSV与AutoEncoder固定；
- 每个bearing前15点中心化；
- 标准化只使用Bearing1_1–Bearing1_3；
- 单层GRU、hidden dim 8、504个参数；
- next-embedding MSE训练目标；
- learning rate 1e-3、weight decay 1e-5；
- 最多300 epoch、patience 40、gradient clipping 1.0；
- Bearing1_4选择checkpoint；
- 10点滑动平均GRU Level；
- step 15开始统一评分。

唯一变化因素为随机种子：

```text
20260916, 20260917, 20260918
```

## 复现门

seed 20260916将重新训练。相同环境下要求：

- 最佳epoch与上一轮一致；
-模型参数逐元素完全一致；
- 逐bearing GRU ρ与next-step MSE完全一致。

如果不满足，应优先报告复现问题，而不是解释多seed差异。

## 输出和评价

- 每个seed、每个bearing的Level ρ与next-step MSE；
- 每个bearing跨seed均值、样本标准差、最小值和最大值；
- 不同seed Level轨迹的两两Pearson与Spearman相关；
- collapse计数；
- GRU均值相对距离Level和centered HMM的差异；
- 三个独立checkpoint和完整训练历史。

## 解释边界

- 三个seed只能提供初步稳定性证据；
- Bearing1_4用于checkpoint选择，不是独立测试；
- Bearing1_5此前已经查看，不是严格盲测；
- 相同bearing上的多个seed并非独立设备样本；
- normalized lifetime相关性不等于物理损伤准确度；
- 多seed稳定不等于跨bearing或跨工况泛化。
