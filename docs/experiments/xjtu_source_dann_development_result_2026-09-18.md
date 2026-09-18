# XJTU-SY Condition 3 Source-DANN配对开发实验结果

## 结论

Source-DANN相对完全配对的Source-only，在三个LOBO bearing上均提高了平均健康相关性，跨bearing平均Spearman从−0.334提高到−0.107，平均增量为+0.226；identity准确率从0.903降到0.879。

这证明显式身份对抗产生了预期方向的作用，但效果不足且不稳定：`Bearing3_2`和`Bearing3_3`的平均健康相关性仍为负，identity仍高于0.80。因此健康与identity门槛均失败，不允许读取`Bearing3_4`。

## Material Passport

- 实验：`xjtu_condition3_source_dann_development_v1`。
- 数据：XJTU-SY Condition 3开发缓存中的`Bearing3_1`–`Bearing3_3`。
- 保护：`Bearing3_4`和`Bearing3_5`未读取。
- 配置SHA-256：`b0b594a60e04366fefbb8894011ad587eb2e7b1749bdb58d1b6df6baae6cc45d`。
- 设计：3 folds × 3 seeds × 2 paired arms，固定30 epochs。
- 状态：`ANALYZED`；未作第二次独立复跑。

## 健康排序结果

| Bearing | Source-only ρ | Source-DANN ρ | Δρ |
|---|---:|---:|---:|
| Bearing3_1 | 0.126 | 0.412 | +0.286 |
| Bearing3_2 | −0.684 | −0.350 | +0.335 |
| Bearing3_3 | −0.443 | −0.384 | +0.059 |
| 三bearing平均 | −0.334 | −0.107 | +0.226 |

Source-DANN逐seed：

- `Bearing3_1`：0.207、0.535、0.492，3/3为正。
- `Bearing3_2`：0.500、−0.746、−0.803，1/3为正且seed差异很大。
- `Bearing3_3`：−0.298、−0.347、−0.508，0/3为正。

局部backward-step fraction仍大约为0.44–0.51，说明预测轨迹依然存在频繁反向波动。

## Identity结果

- Source-only平均identity准确率：0.903。
- Source-DANN平均identity准确率：0.879。
- 预注册上限：0.80。

identity下降主要来自部分fold，效果并不一致。以`Bearing3_1`作为外层测试时，Source-DANN identity约0.90–0.95；以`Bearing3_3`作为外层测试时，部分seed甚至高于Source-only。因此不能把平均下降解释为已经获得稳定的domain-invariant embedding。

## 门槛判定

- 三个bearing的Source-DANN平均Spearman均为正：失败。
- 至少两个bearing平均Spearman≥0.5：失败。
- 每个bearing至少两个正seed：失败。
- Source-DANN identity准确率≤0.80：失败。
- 联合门槛：失败。
- `Bearing3_4`验证授权：否。

## 解释

与TS2Vec“几乎没有降低identity”相比，Source-DANN提供了更直接的机制证据：身份对抗确实降低了平均identity，并改善了全部三个bearing的平均健康相关性。但全局对齐没有解决健康方向冲突，尤其`Bearing3_3`在全部seed下仍为负。

可能原因是全局domain adversary同时混合了不同健康阶段。两个bearing在相同采样位置不一定处于相同物理退化阶段；若不区分健康进度进行全局对齐，encoder可能在删除身份信息的同时损坏健康结构。该解释是基于结果的研究假设，还不是已验证事实。

## 统计与推理限制

- 只报告三个bearing和三个seed，不进行缺乏足够独立样本的显著性检验。
- bearing而非时间点是泛化单位，不能把数千时间点视为独立样本。
- 所有比较均按预注册配对，避免用不同模型规模解释差异。
- 结果只支持相关性与机制线索，不支持因果或物理损伤量结论。
- 没有根据结果调整GRL系数、epoch或seed，也没有打开保护集。

## 下一步

不在当前结果上事后调节GRL权重。先在B3_1–B3_3内分析identity下降与健康改善是否集中在特定生命周期阶段；随后预注册阶段条件对齐方案，例如按normalized lifetime分段的conditional MMD或阶段条件domain adversary。仍必须包含Source-only对照和健康/identity双门槛。

