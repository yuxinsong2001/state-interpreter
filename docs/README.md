# State Interpreter 工作文档导航

本目录集中保存 2026 年 7 月下旬开始的 State Interpreter 研究与实现资料。

## 当前阶段状态（2026-09-12）

`Bearing2_5`的一次性联合严格盲测已经完成。Arm A的Level–lifetime Spearman `ρ=0.946183`，Arm B为`ρ=0.964242`，差值为`+0.018059`。两个冻结分支都达到预注册的强支持阈值；Arm B只有小幅额外收益。该bearing从现在起不得再用于模型选择或调参。正式总结见`experiments/joint_blind_evaluation_bearing2_5_2026-09-12.md`。

## 工作主线

```text
理论与文献检索
→ 数据集审计与实验方案
→ 小型 AutoEncoder 生成低维 latent z
→ 分析并解释 latent space
→ 构建 State Interpreter
→ 后续迁移至 VibFM
```

## 目录说明

### `research/`：理论与文献

- `literature_matrix.md`：第一轮正式文献矩阵。
- `rl_state_interpreter_literature_matrix.md`：面向 RL 可用状态的第二轮文献矩阵。
- `teacher_material_theory_guide.md`：老师提供材料的理论导向。
- `juodelyte_2022_reproduction_notes.md`：Juodelyte et al. (2022) 阅读与复现笔记。

### `experiments/`：数据、方案与实验结果

- `autoencoder_state_interpreter_workplan.md`：小型 AutoEncoder 与 State Interpreter 的执行方案。
- `data_audit.md`：XJTU-SY 数据文件结构审计。
- `xjtu_sy_dataset_assessment.md`：XJTU-SY 的适用性评估。
- `gearbox_state_interpreter_data_audit.md`：早期 Gearbox 数据契约审计。
- `autoencoder_baseline_z8_result.md`：首轮 `z=8` AutoEncoder 基线结果说明。
- `latent_analysis_z8_result.md`：首轮 latent 轨迹、健康中心距离、`Δz` 与重建误差分析。
- `health_indicator_comparison_z8_result.md`：全局距离、个体校准距离和轻量时序状态比较。
- `relative_temporal_state_interpreter_v1.md`：第一版在线 State Interpreter 的生命周期、接口和真实重放结果。
- `interpreter_parameter_sensitivity_2026-08-02.md`：12组参数敏感性、稳定区域和 validation-only 参数锁定。
- `locked_holdout_evaluation_2026-08-02.md`：锁定15/10配置下的 Bearing1_5 参数选择后评价。
- `cross_condition_v2_preregistration_2026-08-02.md`：第二工况数据审计、固定划分、严格盲测边界和 v2 跨工况实验预注册。
- `cross_condition_v2_two_arm_amendment_2026-08-02.md`：在盲测前将 v2 修订为“直接泛化 + Encoder适配”双分支联合评价；实际执行以 v2.1 为准。
- `experiment_config_safety_layer_2026-08-02.md`：v2.1只读配置接口、划分/holdout泄漏保护、冻结工件哈希校验及测试结果。
- `arm_a_direct_generalization_dev_result_2026-08-02.md`：冻结v1系统在第二工况 `Bearing2_1–2_4` 上的直接泛化开发结果。
- `arm_b_target_autoencoder_training_result_2026-08-02.md`：分支B在第二工况训练同结构 `z=8` AutoEncoder的训练与验证结果。
- `v0_1_stability_study_2026-09-12.md`：两个工况、5折bearing轮换和3个随机种子的30次稳定性实验与方法选择结论。
- `bearing1_4_diagnosis_2026-09-12.md`：最弱bearing的重建误差、seed一致性和窗口敏感性事后诊断。
- `gru_predictive_interpreter_result_2026-09-16.md`：第一版因果Predictive GRU与距离/HMM基线比较。
- `gru_predictive_multiseed_result_2026-09-16.md`：三个随机种子的GRU稳定性与精确复现结果。
- `gru_readout_comparison_result_2026-09-16.md`：冻结checkpoint后的hidden、predicted-z与residual读出比较。
- `gru_temporal_ranking_protocol_2026-09-16.md`：加入时间排序弱监督的配对实验预设协议。
- `gru_temporal_ranking_result_2026-09-16.md`：三seed配对结果、人工输入诊断、失败门槛与后续决策。

### `meetings/`：组会纪要与周报

- `meeting_notes_2026-07-27.md`：老师调整任务方向后的组会纪要。
- `weekly_report_final_2026-07-27.md`：中文汇报终稿。
- `weekly_report_final_2026-07-27_DE.md`：德语完整版本。
- 其余文件为同次汇报的早期版或专题版。

### `presentations/`：正式汇报材料

- `2026-07-27/`：2026-07-27 组会的德语短报告、PPTX 及检查文件。

## 持续维护文档

以下两份文件仍保留在仓库根目录，方便每次工作后更新：

- [`../../work_log.md`](../../work_log.md)：个人工作思路、任务和问题记录。
- [`../../presentation_notes.md`](../../presentation_notes.md)：未来组会、实习报告和论文汇报素材。

## 边界说明

- `docs/project/` 是更早的 Gearbox-RL 项目学习资料，不属于本轮 State Interpreter 文档。
- `docs/presentation/` 根目录中的旧文件是更早的汇报/学习材料。
- 模型代码、测试和训练脚本位于独立的 `state-interpreter` 代码仓库；checkpoint、CSV、JSON 等机器工件放在该仓库的 `runs/` 中。

## 与代码仓库同步

本目录的项目文档同时同步到 `state-interpreter/docs/`：

- handover 中的本目录保留完整研究与汇报资料，并作为文档整理来源。
- 代码仓库中的 `docs/` 便于结合实现、测试和实验工件阅读。
- `work_log.md` 和 `presentation_notes.md` 属于个人持续记录，只保留在 handover 仓库，不进行双份维护。

最新实验：`experiments/gru_temporal_ranking_result_2026-09-16.md` 记录GRU时间排序目标的三seed配对实验。该固定设置未通过稳定性推进门槛，当前不升级默认Level。

最新表示学习实验：`experiments/xjtu_ts2vec_development_result_2026-09-18.md` 记录紧凑因果TS2Vec的Condition 3三折LOBO结果。健康排序与identity双门槛均未通过，B3_4/B3_5未读取；下一候选转向显式domain/bearing alignment。

最新预注册：`experiments/xjtu_source_dann_development_preregistration_2026-09-18.md` 锁定source-only与Source-DANN配对实验；`research/domain_alignment_method_audit_2026-09-18.md` 记录DANN、MMD与CORAL的比较依据。

最新结果：`experiments/xjtu_source_dann_development_result_2026-09-18.md` 记录Source-DANN在三个bearing上均改善配对相关性并降低平均identity，但健康与identity绝对门槛仍失败，B3_4/B3_5未读取。

最新阶段诊断：`experiments/xjtu_source_dann_lifecycle_diagnosis_2026-09-19.md` 用现有CSV比较五个生命周期区间，发现identity下降与健康改善并不稳定共现；阶段条件对齐仍只是待验证假设。

下一方法审计：`research/conditional_alignment_method_audit_2026-09-19.md` 核对CDAN/DSAN与当前严格LOBO的边界；`experiments/xjtu_source_conditional_mmd_protocol_2026-09-19.md` 是source-only、global MMD、time-bin conditional MMD三分支的**未冻结草案**，目前没有新训练或结果。

条件 MMD 元数据预检：`experiments/xjtu_source_conditional_mmd_metadata_preflight_2026-09-19.md` 确认开发 bearing 的五个时间区间均有至少59个训练 endpoint，但核尺度、梯度与配对采样尚未验证，正式训练仍未开放。

数值预检设计：`experiments/xjtu_source_conditional_mmd_numerical_preflight_design_2026-09-19.md` 固定2 bearing×5时间区间×6 endpoint的配对batch思路和训练来源限定的核/梯度检查；它是未执行的计划，不是MMD实验结果。

数值预检结果：`experiments/xjtu_source_conditional_mmd_numerical_preflight_result_2026-09-19.md` 记录九组开发折×seed的初始化MMD、核尺度与梯度诊断；194项测试通过，尚未训练三分支模型或读取保护集。

三分支开发实验预注册：`experiments/xjtu_source_conditional_mmd_development_preregistration_2026-09-19.md` 锁定source-only/global MMD/conditional MMD同索引比较、λ=0.1、三折三seed与双门槛；配置及入口`--check-only`通过，200项测试通过，尚未正式运行。

三分支开发实验结果：`experiments/xjtu_source_conditional_mmd_development_result_2026-09-19.md` 记录一次性配对运行。Conditional MMD降低identity可识别性，但B3_2/B3_3健康排序仍为负，联合门槛未过；B3_4/B3_5未读取。

条件MMD失败轨迹诊断：`experiments/xjtu_conditional_mmd_failure_diagnosis_2026-09-19.md` 仅用已保存CSV分析五段与逐seed原始曲线：B3_2整体方向偏反，B3_3前中段下降、末端急升；这是探索性诊断，不改变推进门槛。

本周最终步骤：`experiments/joint_evaluation_rehearsal_2026-08-02.md` 记录 Arm A/B 在 Bearing2_4 上共享单次输入物化的联合评估演练；Bearing2_5 仍未读取。

最新组会材料：`meetings/wochenupdate_2026-08-02_DE.md` 为包含方法、双分支结果、限制和下一步的德语周报。
