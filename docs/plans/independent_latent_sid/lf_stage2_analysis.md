# LF语言模型验证完成

状态：LF stage2完成，stage3生成中；HV完整run已验收，H1/B2尚待运行。不是四组最终结论。

沿用训练revision `70b82db195f600e3271030ddb5e6c228e4915f31`、pilot配置、seed42、冻结8192/1024划分。样本单位、独立视图、量化器、decoder、损失、初始化及超参均继承spec.md，未改变训练。LF完成1024 step/2 epoch，训练耗时3348.4784s；验证1024轨迹、5996事件/层、52064 supervised tokens。

|teacher-forced指标|LF|HV|
|---|---:|---:|
|SID NLL (nats/token)|0.838592|0.568885|
|第一层准确率|0.568045|0.946965|
|第二层准确率|0.633089|0.790193|
|第三层准确率|0.661107|0.626418|
|第四层准确率|0.716311|0.735490|
|sampling准确率|0.721680|0.720703|
|整序列精确率|0/1024|18/1024|

LF各层NLL为1.056831/0.818725/0.791607/0.687204；sampling NLL为0.816695。逐层准确率不等于完整四码tuple准确率，整序列精确率也不是tuple准确率；LF tuple补评须待整个run完成后执行。

LF第一层准确率56.80%超过其验证集最大码占比42.85%，但这不是训练unigram基线检验；HV第一层94.70%几乎等于其最大码占比94.68%。码分布不同使原始NLL和准确率不可直接用于判断表示质量。待完整产物统一计算train unigram(alpha=.5)在validation的交叉熵，结合四G、probe及tuple分析。LF第三层准确率高于HV，但不足以推出整体SID或生成改善。

原始产物为 `/workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/pilot/porto/42/LF/attempt_001/stage2/stage2_metrics.json`，SHA256 `acc1df7f857a1cd190b456773086461bc336d2499f6bcbd72d3bcf534f8626f3`；本地归档lf_stage2_metrics.json。运行命令保持原四组pilot命令（见pilot_launch.json），本次仅复制与读取JSON，无新代码、训练或评估；新增loss、初始化、optimizer和GPS重建变更均不适用。

下一步：待LF生成/解码/诊断完成后验收全部阶段及SHA，再运行tuple补评；继续原定H1/B2，同一验证人口进行最终配对分析。
