# GRU v1梯度平衡：执行前协议

固定同一GRU双头结构、目标、B1_1–1_3训练/B1_4验证、三seed和600轮上限、patience40。唯一更新差异为gru_task_balance的共享等范数贡献；输出头梯度不重加权，保持原clip1与Adam。每轮记录norm/cosine/weights/fallback，验证仍按未经平衡的两项loss之和选checkpoint。

同一GRU归档600轮普通训练作为内部优化参照，不做替代模型选型。报告各seed预测MSE、代理Level误差与固定分段偏差，不仅报告总loss。早停会改变实际预算，不假装等步数比较。无预设效果保证，不按结果继续调权重。新输出目录、不改旧工件；B1_5不实例化，无raw/保护目录读取。
