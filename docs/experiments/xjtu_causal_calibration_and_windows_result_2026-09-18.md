# XJTU-SY因果校准与bearing内窗口结果

## 目标

为65维Feature LSTM基线建立在线可用且不跨设备泄漏的数据层，不读取真实XJTU-SY数据。

## 实现

### CausalFeatureStandardizer

- 输入为`[steps, features]`；
- 只使用固定前`calibration_steps`计算feature-wise mean/std；
- 默认设计目标为前15步校准；
- 标准差使用epsilon下限处理常数特征；
- 后期样本只被transform，不参与fit；
- 支持inverse transform用于调试。

### build_bearing_feature_windows

- 输入为测量级feature、bearing ID和step ID；
- 即使输入行交错，也先在每个bearing内部按数值step排序；
- 检查bearing内重复step；
- 生成`[n_windows, window_size, n_features]`；
- 每个窗口保留bearing ID和窗口终点step；
- 短于窗口的bearing不生成窗口；
- 窗口绝不跨bearing边界。

## 测试

定向测试17项全部通过，覆盖：

- 修改校准期之后的数据不会改变mean/std；
- 校准前缀标准化为零均值和单位标准差；
- 常数feature无NaN/Inf；
- 非有限值和不足校准长度被拒绝；
- 交错、乱序数据按bearing/step正确构窗；
- 重复step被拒绝；
- 10步窗口可以直接送入Feature LSTM。

完整仓库回归结果：`143 passed`。

## 数据访问状态

全部测试使用人工Torch tensor。没有读取任何真实CSV，没有训练模型，Bearing3_5未读取。

## 结论

Feature LSTM数据接口已经具备固定早期因果校准和bearing边界保护。下一步不是立即正式训练，而是建立只允许B3_1–B3_3的开发特征物化脚本、缓存格式和预注册LOBO协议；先用开发数据检查65维特征的有限性与尺度，再决定是否启动训练。

## 仓库管理修正

审查Git状态时发现原`.gitignore`中的`data/`会误忽略任意层级的同名目录，包括`src/state_interpreter/data/`。规则已收窄为`/data/`：仓库根目录中的原始数据继续被忽略，但Python数据处理模块现在可以被Git跟踪。
