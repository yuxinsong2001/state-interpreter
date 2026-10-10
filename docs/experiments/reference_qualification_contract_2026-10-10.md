# 相对状态的参考资格契约

## 新接口
ReferenceQualifiedInterpreter 包装现有方向解释器，不改变连续Level / Trend / Movement，不拟合阈值、不认证健康。

默认 unreviewed：保留连续指标，方向输出direction_uncalibrated。显式rejected同样拒绝。只有accepted_for_relative_description声明且阈值政策已配置，才允许原方向标签；窗口未形成仍保持window_warmup。

资格声明绑定episode ID、checkpoint/表示身份、calibration_steps、temporal_window；要求证据引用与理由。身份或设置不匹配直接报错。reset只用于同episode重放；新设备必须创建新实例及新资格。接受声明是外部记录，代码不认证证据文件真实性或科学有效性，不能据此称安全认证。

## 当前真实数据决策
本轮没有给任何真实轴承创建accepted声明。B3_2联合漂移防护已拒绝；B3_3虽未触发门槛仍有强方向漂移；B3_1缺少独立参考资格证据。因此真实方向继续未校准。若需要探索图，可单独标注hypothetical，不能混入有效输出。

## 验证与范围
八项新增测试覆盖默认/未审/拒绝、显式接受、接受但无阈值、身份不匹配、同episode重置、连续指标一致；连同原方向接口13项，共21项。只用合成数据。这里实现的是拒绝输出契约，不是模型性能改进，也未声称解决参考代表性。未改原模块或历史配置，未训练、未读保护集。

下一步若继续，应给出明确的参考证据标准或可控参考数据，才能启用方向；不能通过反复调整已用轴承阈值来制造资格。
