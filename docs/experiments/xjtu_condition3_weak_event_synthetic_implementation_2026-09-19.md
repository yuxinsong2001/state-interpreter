# Condition 3 候选事件检测器：合成测试实现

状态：**仅代码与合成数据测试完成；未冻结真实数据执行协议，未运行 XJTU-SY bearing。**

本步实现了独立的在线检测器 `src/state_interpreter/weak_event_detector.py`，接收按时间到达的单个标量特征、bearing ID 与 step ID。主规则和独立对照分别使用 `rms_threshold_v1` 和 `kurtosis_threshold_v1` provenance；水平/垂直特征均值由 `pair_mean` 计算。前15次观测拟合总体均值与标准差，参考标准差小于 `1e-8` 时拒绝报警；之后严格超过 `μ+2σ` 连续5次才在第5次确认并锁存事件。步号缺口会重置连续计数；更换 bearing 必须显式 `reset`。无效输入在修改状态前拒绝。

输出包含阶段、事件标志、确认步号、确认后才可见的候选首步、标准化变化强度、冻结参考与来源标识。`MONITORING` 仅表示未报警，`ALARMED` 是算法候选事件，都不是物理健康/损伤真值。旧输出不可回填。

测试文件：`tests/test_weak_event_detector.py`。覆盖第五次才确认、严格阈值、缺口、前缀不变性、无报警、无效参考、跨bearing隔离、无效输入和provenance。`pytest -q tests/test_weak_event_detector.py -p no:cacheprovider`：5 passed。全量测试初次因系统默认 pytest 临时目录权限错误出现27项setup error；改用项目内 `--basetemp .pytest_weak_event_full_20260919` 后：**205 passed**。这些错误不是算法失败。未读取 B3_1–B3_5，也未生成任何真实事件指标。

下一关：先审查并冻结真实开发数据执行配置、输入缓存哈希、结果路径及保护集门禁；随后才考虑只在 B3_1–B3_3 上执行弱事件可行性检查。B3_4/B3_5继续保护；若无独立故障锚点，不报告真实onset准确率。
