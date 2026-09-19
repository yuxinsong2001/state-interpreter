# Condition 3 弱事件候选规则：开发运行结果

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run and descriptive result check
- Origin Date: 2026-09-19
- Verification Status: DEVELOPMENT COMPLETED / PROTECTED SETS UNREAD / NO PHYSICAL ONSET VALIDATION
- Command: `python scripts/run_weak_event_condition3_development.py --execute-development`
- Config SHA256: `5e72b9646d4bcbfb11994b9ff346628ac61eb95145ba6c13ad103edff0b353b0`
- Result report SHA256: `278984eb2559f06d9a569d974c927dc45909f8c97751fef950c96d7e0237a329`

本次按此前锁定的配置和代码，只运行一次B3_1–B3_3开发缓存。完整输出位于`results/2026-09-19_condition3_weak_event_development_v1/`，独立状态记录位于`records/xjtu_condition3_weak_event_development_v1/evaluation_record.json`。输入、detector、evaluator和runner哈希均在解析NPZ前通过；退出码0。两个规则各一条逐步轨迹，合计10,810行；六行汇总齐全。三条bearing两规则的参考均有效，测量步号缺口均为0。B3_4/B3_5未读取。

| Bearing | 测量次数 | RMS确认步 | 峭度确认步 | 事后观察 |
| --- | ---: | ---: | ---: | --- |
| B3_1 | 2538 | 24（全程约0.9%） | 2352（约92.7%） | 两规则严重分歧；RMS近开头即报警 |
| B3_2 | 2496 | 1235（约49.5%） | 无报警 | 同一run只有RMS触发 |
| B3_3 | 371 | 345（约93.2%） | 343（约92.7%） | 两规则在末段接近 |

表中比例仅为确认位置除以`N−1`的**事后描述**，未进入在线阈值或推理。B3_1的RMS参考标准差约0.0090，阈值约0.5265；早期第24步触发值得重点核查，但由于缺乏独立物理故障起点，不能直接判定为假阳性。B3_2峭度无报警也不能判定为漏检。B3_3两个规则一致也不是正确性证明，因为它们都来自同一振动系统。

当前只能得出：规则在开发数据上可执行、参考有效、报警时刻具有明显bearing/特征依赖性。**没有证据支持把候选事件当真实损伤起点，亦不能以此建立准确率或宣布方法稳定。**不依据这些结果更换阈值、择优保留规则或打开B3_4/B3_5。

下一步先对现有保存轨迹做只读诊断：核查B3_1早报警前后超阈持续性和参考尺度，B3_2峭度无报警的阈值/轨迹关系，以及B3_3末段两规则接近是否仅由共同终点驱动。诊断是探索性的；若提出新规则，必须另建版本和独立评价契约，不能改写本次结果。
