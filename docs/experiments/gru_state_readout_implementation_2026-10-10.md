# GRU v1双头模型实现

## 已完成
新增GRUStateReadoutInterpreter，复用ResidualGRUStateInterpreter共享GRU，保留零初始化delta预测头，新增Linear(hidden_dim,1)+Softplus标量偏离头。

输入[batch,time,8]（维数可配置），输出hidden_sequence、next_embedding_prediction、scaled_relative_level[batch,time,1]和final_hidden。读出是训练目标尺度下的相对代理，不是健康百分比；目标尺度恢复、因果Trend和延迟评分Movement由后续runner负责，当前模型不自动生成三量。

forward默认从零hidden开始，不跨调用缓存历史；本次不是对新增双头的streaming接口实现。旧OnlineResidualGRU只适配旧Residual类型，不能直接用于新模型。

## 验证
新增11项模型测试与原Residual6项，共17项通过。覆盖非负输出/shape、初始forecast逐值等于输入、两项损失梯度到达各头及共享GRU、非零预测头前缀因果性、独立调用不串episode、非法输入、state_dict内存序列化精确往返。

首次运行16项通过、保存/加载测试setup因临时目录权限失败。改用BytesIO后17项通过，没有训练重试或模型参数更改。保存加载仅验证state_dict，不认证真实checkpoint或文件系统存储。

## 范围与下一步
没有训练、真实数据或保护集读取；不宣称效果。下一步实现GRU v1数据准备与联合损失runner：训练集专用标准化及目标尺度、step>=25有效mask、按episode等权损失、独立配置与输出目录；先小规模smoke训练，再按设计进行三个seed正式开发训练。不覆盖原GRU脚本及结果。
