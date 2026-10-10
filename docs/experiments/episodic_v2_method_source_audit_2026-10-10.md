# Episodic v2 方法来源核查

## Material Passport

日期：2026-10-10。范围：当前 RMS/峭度、初始参考、2SD、连续5步进入/退出与趋势窗口的依据。类型：定向来源核查，非系统综述。证据：项目源码与第三方声明、NIST官方手册、论文原始摘要/作者预印本和官方软件文档。本次网页检索不是直接使用Google Scholar，不将“查到摘要”写成“完整精读”。未训练、未改算法、未读取任何新设备数据。

## 总结判定

当前实现应准确命名为“自定义单侧持续超限事件基线”（custom one-sided sustained-exceedance event baseline）。有传统统计监测思想的对应，但没有核实到支持15/2/5/5/10整套组合的论文。本次是事后概念对照，不能倒写为代码最初来自这些论文，也不能宣称正式复现Shewhart、WECO、CUSUM、EWMA或BOCPD。

## 1. 当前实现逐项归属

| 项目设计 | 核查结论 | 边界 |
|---|---|---|
| RMS/峭度特征提取 | 代码来源可追溯：65维特征定义适配 thfmn/xjtu-sy-bearing，固定提交见THIRD_PARTY_NOTICES | 仓库实现依据不等于阈值有效性或健康语义证明 |
| 相对均值/SD偏离 | 对应统计监测基本思想 [S1] | 不是完整标准控制图构建程序 |
| 均值+2SD | 可归入kσ形式的自定义阈值 [S1] | k=2并非该资料所述标准3σ规则；没有实测误报保证 |
| 连续5次超过2SD | 本项目自定义持续性条件 | 与WECO的2/3超过2σ、4/5超过1σ不同 [S1] |
| 连续5次回落才退出 | 自定义事件生命周期规则 | 所查标准报警规则没有为该退出/恢复判定背书 |
| 初始15次校准 | 本项目工程设定 | 所查资料不支持15次自动足够，强调参考稳定性与估计不确定性 [S1,S3] |
| 10步斜率趋势 | 本项目解释性输出 | 不等于EWMA统计量，也不等于WECO趋势规则 |
| UNKNOWN_GAP、record_end删失 | 本项目明确证据缺口/观察终止的工程接口 | 不证明健康状态或恢复，不属于所查检测论文的官方实现 |

源码核查：src/state_interpreter/episodic_weak_event.py 与配置文件；特征源说明：THIRD_PARTY_NOTICES.md。现有规则使用总体SD（ddof=0）；标准差估计与参考样本量仍需独立评价，不修改锁定版本。

## 2. 可核实来源与适用关系

### S1 — NIST：Variables Control Charts

[官方原文](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc32.htm)

核实了kσ控制限、常用3σ及WECO规则。资料提醒小量参考数据估计控制限时实际误报性质可能偏离理论。支持“相对参考监测”思想，不支持我们的参数组合；统计控制限不是物理故障界限。

### S2 — CUSUM：Page (1954) 与NIST实现说明

E. S. Page. Continuous Inspection Schemes. Biometrika, 41(1–2), 100–115. [出版社元数据/DOI](https://doi.org/10.1093/biomet/41.1-2.100)。原论文付费全文未精读。

[NIST可访问方法说明](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm)：累积偏离，分别监测向上和向下移动，参数k/h用于设计。可作为持续小偏移检测候选；目前代码没有CUSUM累积量，不能称复现。

### S3 — EWMA：Roberts (1959) 与NIST实现说明

S. W. Roberts. Control Chart Tests Based on Geometric Moving Averages. Technometrics, 1(3), 239–250. [出版社摘要/DOI](https://www.tandfonline.com/doi/abs/10.1080/00401706.1959.10489860)。1959是卷期年份，不是网页的2012数字化上线日期。

[NIST方法说明](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm)：递归加权历史观测，控制限依赖平滑后统计量的方差；要求代表性参考，经典控制限表有独立/正态假设。不是随便平滑曲线后继续套原始2SD阈值，也不能自动修复不合适的参考。

### S4 — Adams & MacKay (2007)：BOCPD

[作者arXiv原始记录与摘要](https://arxiv.org/abs/0710.3742)。核实在线推断距最近变化点的run-length分布。与固定初始参考不同，输出是概率性的分段证据，不是物理故障标签。当前仅摘要层核查，hazard/观测模型/截断策略待全文核实，尚未复现；作者PDF本次读取失败。

### S5 — Coble & Hines：PHM指标评价

Identifying Optimal Prognostic Parameters from Data: A Genetic Algorithms Approach. [PHM Society官方论文记录](https://www.papers.phmsociety.org/index.php/phmconf/article/view/1404)。论文属于2009会议卷，网页同时标2021上线日期，引用应注明这一差异。

原始摘要定义monotonicity、trendability、prognosability并用于参数选择。它支持评价预后指标适用性，不支持“与时间正相关便是健康真值”；事件状态与连续预后指标也不是同一输出。本次未核对全文公式，不把本项目Spearman称为完整复现这些指标。

### S6 — Duong et al. (2018)：bearing HI

A Reliable Health Indicator for Fault Prognosis of Bearings. Sensors, 18(11), 3740. DOI 10.3390/s18113740。[原始论文摘要及完整作者元数据](https://pubmed.ncbi.nlm.nih.gov/30400203/)；[出版社入口](https://www.mdpi.com/1424-8220/18/11/3740)。

摘要说明DWPT子带、局部加权回归平滑、梯度评价及迭代累积，在两套benchmark数据验证。说明成熟HI构建并非直接把RMS命名为健康程度。出版社限流、PMC验证码，本次仅核实摘要；局部回归是否因果、特征筛选是否使用整条生命周期，必须全文核查后才能用于在线实验。

### S7 — ruptures / Truong, Oudre & Vayatis (2020)

[作者维护文档](https://centre-borelli.github.io/ruptures-docs/)。确认软件为离线变化点分析工具，可用作回顾性结构诊断，不可把使用完整序列得到的分段位置冒充实时报警。文档列出关联论文：Selective review of offline change point detection methods, Signal Processing, 167, 107299 (2020)。当前只核实软件定位与文档记录，未复现论文。

## 3. 核查边界与证据等级

NIST为官方技术指南；S2/S3是原始方法论文元数据及可访问方法说明；S4为作者预印本摘要；S5/S6为原始研究摘要；S7为官方软件文档。这些证据不宜强行套用医学RCT证据等级，也不代表在XJTU-SY上的性能排序。未完成Scopus/撤稿/全部COI独立审计，不能称完成系统性文献质量评估。未检出整套规则的来源，不等于证明世界上不存在相似实现。

## 4. 对后续工作的约束

保留当前基线与结果，新增来源说明，不事后替换引用或参数。后续优先：

1. 读透S1/S3的参考稳定性前提，用开发数据检查参考质量；不是从B3_4选择最优窗口。
2. 若继续事件检测，先按S2正式实现双向CUSUM对照；用合成平稳/偏移/相关序列检查误报和延迟，再在开发设备比较。
3. 明确无故障标签时不能把正常序列以外的报警率直接称为误报率。参考失配仍需单独报告。
4. 若回到embedding-to-state主线，另立问题与接口；CUSUM可以监测合适的表示或残差，但不会自动赋予embedding健康语义。

这里是基于核查的下一步建议，尚未开始CUSUM/EWMA/BOCPD新实验。Bearing3_5保持保护。
