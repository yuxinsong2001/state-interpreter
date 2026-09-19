# Source-DANN 生命周期阶段诊断

本目录只分析已完成的 Condition 3 Source-only/Source-DANN 开发实验输出，不读取原始数据或 checkpoint，不重新训练。

- `paired_stage_health.csv`：每个 seed、bearing、五等分生命周期阶段的配对阶段内 Spearman。
- `paired_stage_identity.csv`：对应的五段 identity probe 准确率与下降量。
- `stage_summary.csv`：三个 seed 的阶段平均与同向 seed 数。
- `diagnosis_report.json`：输入工件 SHA-256、保护边界和执行状态。

阶段内 Spearman 衡量局部排序，不能代替整个生命周期的 Spearman。三个 bearing 的每段测量数差异很大；尤其 B3_3 每段只有约 74–75 个时间相关测量。本结果属于探索性诊断，不能用于修改先前冻结的门槛，也不能授权读取 B3_4。
