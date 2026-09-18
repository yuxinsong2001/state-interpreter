# Feature LSTM Condition 3开发实验预注册

## 状态

`preflight_passed_no_condition3_data_read`

本步骤没有接收数据集路径、没有枚举Condition 3目录，也没有读取任何CSV。

## 固定研究问题

65维领域特征与小型时序模型，能否在未见过的Condition 3 bearing上产生一致排序的健康进度表示？

## 数据边界

- 开发：B3_1、B3_2、B3_3；
- 验证：B3_4；
- 最终计算holdout：B3_5。

物化接口只能一次性精确授权前三个开发bearing。B3_4或B3_5出现在开发请求中会在访问数据前失败。

## 外层LOBO

1. B3_2+B3_3训练，B3_1测试；
2. B3_1+B3_3训练，B3_2测试；
3. B3_1+B3_2训练，B3_3测试。

## 固定实现

- 65 features = 37 time + 28 frequency；
- calibration steps = 15；
- window size = 10；
- BiLSTM hidden = 16，Dense = 16，dropout = 0.2；
- PyTorch parameters = 11,169。

## 固定训练

- 忠实基线目标：normalized remaining useful life；
- 汇报的递增状态分数：`1 - clipped predicted RUL`；
- AdamW，learning rate 0.001，weight decay 0.0001；
- Huber loss，delta 0.08；
- batch size 32；
- 固定50 epochs；
- 不使用outer test bearing做early stopping或checkpoint选择；
- seeds：20260918、20260921、20260924。

## 进入B3_4的门槛

- 三个LOBO bearing的seed平均Spearman全部为正；
- 至少两个bearing的平均Spearman不低于0.5；
- 每个bearing至少两个seed为正。

主评价单位是单个bearing，不使用pooling correlation替代逐bearing结果。

## 缓存契约

每个开发bearing单独保存NPZ：`features`、`step_ids`、`source_paths`，并由JSON manifest记录sample count、文件SHA256、工况和实验ID。缓存manifest必须精确包含B3_1–B3_3，不能包含B3_4/B3_5。

## 冻结工件

preflight已验证65维特征、Feature LSTM和因果窗口模块的SHA256。任意工件发生变化后，现有配置将拒绝继续执行，必须显式建立新版本。

## 测试

- 协议/窗口定向测试：14 passed；
- 最终协议定向复核：7 passed；
- 完整回归：150 passed。

## 下一步

实现独立的development-only物化脚本。脚本必须先调用本协议授权，再读取B3_1–B3_3、生成缓存和质量报告；本次尚未执行该步骤。
