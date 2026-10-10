# 固定 q80 方向分辨试验结果

## 目的与边界
复用已归档 Bearing3_3、seed 20260918 状态轨迹，冻结 Autoencoder 和15/10窗口。未训练、未读原始振动或保护集。这是离线规则评价，未将政策安装为接口默认值。

step25–64 的 |Trend| 的80%分位数为 **0.016009069795110582 latent units / measurement step**。绝对值不超过阈值记为 within_resolution；超限时正值为 moving_away、负值为 moving_toward。均相对于初始参考，不表示健康或损伤。

## 结果
| 区间，零起点step | 点数 | 方向覆盖率 | 远离 | 靠近 | 分辨界限内 |
|---|---:|---:|---:|---:|---:|
| 拟合25–64 | 40 | 20.0% | 0 | 8 | 32 |
| 参考核查65–84 | 20 | 0.0% | 0 | 0 | 20 |
| 后续评价85–370 | 286 | 11.2% | 32 | 0 | 254 |

各段相邻且均有方向输出的点之间直接反转数为0；不跨越未分辨点计数，不能解释为全轨迹无转折。

## 参考限制与决定
拟合段 Level–时间 Spearman ρ=−0.913696；前后10点中位数差=−0.277252，MAD=0.128410，差值绝对值/MAD约2.16。固定联合防护需 |ρ|≥0.8 且中位数差超过3×MAD，因此未触发。但强负相关仍说明存在明显方向漂移，**不证明初始参考稳定**。阈值可能吸收早期真实变化而非仅噪声。

report.json 的 policy_activated=true 只表示本次规则通过数值和联合防护条件，从step85应用，非健康阈值认证。保留固定q80，不根据结果进一步降低阈值。后续88.8%点不输出明确方向，不能宣称高覆盖率、准确率或稳定性提升。无逐时健康真值及独立稳定参考，不能估计真实误报率；Movement跨seed问题仍未解决。

当前交付为固定模型的相对状态描述试验。继续时应核查同一规则在其他已用开发记录上的参考代表性与覆盖率，不调分位数、不开放保护集。区间不重叠，但平滑与自相关使其统计上相关。

## 追溯
- 协议：direction_calibration_pilot_protocol_2026-10-10.md
- 脚本：scripts/calibrate_direction_pilot.py
- 结果：results/2026-10-10_direction_calibration_pilot/{report.json,segment_metrics.csv,state_labels.csv}
- checkpoint SHA256：b9eee70d80c5ad69dc3ed55c3bbc0fa7c6265b835b0590bc202b6bdad11ce4c6
- 输入轨迹及checkpoint运行前后hash一致；单个已用开发轴承，不是独立确认。
