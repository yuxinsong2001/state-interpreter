# Condition 3 source-source 条件 MMD 开发协议草案

## 状态

`DRAFT / NO RUN`。这是下一轮实验的可审查设计，不是冻结预注册，也没有结果。只有完成训练集 preflight、数值测试、配置及代码哈希冻结后，才可改为可执行版本；不得覆盖本文件或既有 Source-DANN 配置/结果，后续修改应新建版本。

## 假设

在保持健康回归结构及训练预算不变时，按**归一化时间进度**区间对齐两个训练 bearing，比对其全局表示分布更可能保留跨 bearing 的健康排序。此假设来自[方法审计](../research/conditional_alignment_method_audit_2026-09-19.md)与已有阶段诊断，但不是已证实结论。

## 对照与隔离

- 每折仅用 `Bearing3_1`–`Bearing3_3` 中两个训练 bearing；第三个完整留出。`Bearing3_4`/`Bearing3_5` 不读取。
- 分支 0：原 source-only 健康回归；分支 1：同一 encoder 加 global source-source MMD；分支 2：同一 encoder 加五个归一化时间进度区间内的 source-source MMD。三分支不得额外改变健康头、输入特征、时间上下文或训练样本。
- 原 Source-DANN 作为历史参考，不重训或事后改参数；三分支使用相同 fold、seed、endpoint 和 epoch，固定最后 epoch。任何缓存、归一化及核尺度均不得使用留出 bearing。
- 五等分只是时间代理，不宣称跨 bearing 存在同一物理健康阶段。为防样本数不平衡，区间项按有效区间等权平均，不按测量点数加权；若某区间无法在两个训练 bearing 同时取样，则在训练前判定该 fold 不可运行，而不是临时改分段。

## 待冻结的实现参数

沿用已有 Condition 3 输入、单向 GRU(32)、16 维 embedding、每训练 bearing 300 个 endpoint、30 epochs、三固定 seed 和三折 LOBO。仍需在**仅训练 bearing**的 preflight 中锁定：MMD 核及带宽估计方式、global/conditional 两分支同一正则权重、mini-batch 配对规则、每区间有效样本下限、loss 梯度尺度和随机数控制。preflight 不得查看留出健康指标；预先保存这些数值及哈希后，才允许运行开发折。若无法保证严格配对或数值稳定，本协议停止，不进行结果导向调参。

## 评价与停止规则

使用与 Source-DANN 相同的完整 bearing 门槛：三个留出 bearing 的平均 Spearman 均为正；至少两个均值≥0.5；每个 bearing 至少两个 seed 为正；embedding identity probe 平均准确率≤0.80。报告三个分支逐 fold、逐 seed 的健康与 identity 指标及配对差，不只报告总体均值；必要时报告 Trend/Movement，但不以它们替代预先确定的门槛。

若 conditional MMD 未同时通过全部门槛，不开启 B3_4/B3_5，不事后挑选 bin、seed、带宽或正则权重。若它通过，先进行独立代码/数据泄漏复核，再决定是否按既有验证保护流程读取 B3_4。B3_5 始终保留最终盲测。

## 预期可回答的问题与不能回答的问题

可回答：相对于相同训练条件下的无对齐和全局 MMD，时间条件化是否在开发 bearing 上同时改善健康排序与身份抑制。不能回答：五个时间区间是否等于真实退化阶段、模型能否泛化到未见工况，或该方法是否优于原版 CDAN/DSAN。
