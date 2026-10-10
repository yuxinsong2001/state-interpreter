# PCGrad双头GRU短训练接入核验

状态：三seed各5轮完成；不是正式效果比较。

## 固定范围

新脚本scripts/train_gru_state_v1_pcgrad.py复用既有prepare/loss。新配置configs/gru_state_v1_pcgrad_smoke_2026-10-10.json独立保存。仅B1_1–1_3训练、B1_4验证的归档z8；hidden8、参考15、窗口10、step25评分、Adam 0.001、weight decay 1e-5及clip1保持原设定。没有改旧脚本、配置或结果。

## 接入结果

| seed | epoch1验证联合损失 | epoch5 | 实际投影次数 | 零范数跳过 |
|---|---:|---:|---:|---:|
| 20260916 | 0.337396 | 0.312505 | 0 | 1 |
| 20260917 | 0.187863 | 0.173043 | 0 | 1 |
| 20260918 | 0.518887 | 0.471629 | 3 | 1 |

三seed最佳均epoch5。第三个seed第2–4轮触发投影；前两者无负冲突，保留普通和梯度。首轮预测头零初始化使预测共享梯度为零，跳过投影是预期行为。损失下降只是短训练接入现象，不能与600轮结果比较并宣称提升。

## 核验

三权重重载后的两头输出完全相同；所有四bearing两头35步前缀与全序列一致（atol1e-6、rtol1e-5）；预测[24:-1]对应目标[25:]及Level step25起损失与共用函数一致。将验证z乘100不改变训练mean/std/代理Level尺度。输入及配置hash前后不变。

源CSV SHA256：1fbff8151767b124b68528ea5228f6e76cc391cc9b1ead333080d68e01fb9ec0。共享CSV扫描排除行，但未实例化B1_5 embedding；无原始振动或保护目录读取。验证选择checkpoint，结果均为开发证据。

## 工件与后续

results/2026-10-10_gru_state_v1_pcgrad_smoke包含三个checkpoint、15行training_history.csv、12行metrics.csv及report.json。报告记录runner/module/config/input哈希与上述检查。

下一步另立600轮正式配置，保持三seed及patience40，对照已归档普通600轮；汇总两任务、分段偏差和逐轮梯度证据。当前普通600轮仍是保留版本，短训练不作策略采用决定。正式runner执行前应补充运行超时保护，不能把当前5轮脚本直接当已锁定正式协议。
