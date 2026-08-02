# State Interpreter 工作文档导航

本目录集中保存 2026 年 7 月下旬开始的 State Interpreter 研究与实现资料。

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

### `meetings/`：组会纪要与周报

- `meeting_notes_2026-07-27.md`：老师调整任务方向后的组会纪要。
- `weekly_report_final_2026-07-27.md`：中文汇报终稿。
- `weekly_report_final_2026-07-27_DE.md`：德语完整版本。
- 其余文件为同次汇报的早期版或专题版。

### `presentations/`：正式汇报材料

- `2026-07-27/`：2026-07-27 组会的德语短报告、PPTX 及检查文件。

## 文档来源与同步规则

本目录是从 `student_handover 2/docs/state_interpreter/` 同步到代码仓库的项目文档副本。

- 文献、数据审计、实验说明和组会材料在两个仓库中保持同步。
- 个人过程日志 `work_log.md` 和汇报素材库 `presentation_notes.md` 只在 `student_handover 2` 中持续维护，不复制到代码仓库。
- 模型代码或接口变化应首先更新代码仓库；新的研究说明应首先写入 handover，再同步到这里。

## 边界说明

- handover 仓库的 `docs/project/` 是更早的 Gearbox-RL 项目学习资料，不属于本轮 State Interpreter 文档。
- handover 仓库的 `docs/presentation/` 中还有更早的汇报/学习材料，这些内容不在本目录同步范围内。
- 模型代码、测试和训练脚本位于独立的 `state-interpreter` 代码仓库；checkpoint、CSV、JSON 等机器工件放在该仓库的 `runs/` 中。
