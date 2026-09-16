# H1语言模型验证阶段结果

H1完成1024 step/2 epoch，训练耗时3417.3315s，现进入stage3生成。HV/LF完整验收已通过，B2尚待运行；整体未完成。

训练revision `70b82db195f600e3271030ddb5e6c228e4915f31`、pilot配置、seed42、冻结8192/1024划分不变。H1为共享point-fusion encoder与标准RQ，辅助前缀深度1/2/4；LF为独立encoder融合latent后标准RQ，同样辅助深度1/2/4。两者同时涉及独立分支与参数量变化，不能单独归因为参数隔离。完整样本、特征/单位、encoder、量化器、decoder、初始化、loss、超参与评估继承spec.md，未对运行作任何改动。

1024验证轨迹，5996事件/层，52064监督token：teacher-forced SID NLL 0.773352364；四层NLL 1.087083114/0.791486813/0.624771666/0.590067862，准确率0.553202135/0.668612408/0.751167445/0.747498332；sampling准确率0.721679688、NLL0.788980673，整序列精确率0/1024。完整tuple准确率待run完成后补评，不能用逐层或整序列指标替代。

H1 SID NLL低于LF的0.838591656、高于HV的0.568885164；相较LF，H1第一层准确率略低，第二至四层更高。但码分布与预测难度不同，尚不能据此判定独立latent表示或最终运动生成退步。待码本频率基线、四G、probe及tuple统一分析。

来源：`/workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/pilot/porto/42/H1/attempt_001/stage2/stage2_metrics.json`，SHA256 `a9f9c7ef6d4d3987cd4e2e08e0c080bd77b3208296fe9b8df51bef39c2a9fd1f`；本地归档h1_stage2_metrics.json。执行命令继承pilot_launch.json。本次仅归档已完成指标，无新训练/优化器/损失/初始化或GPS解码变更。下一步等待H1生成、解码与诊断后验收、补评tuple，再完成B2及最终配对分析。
