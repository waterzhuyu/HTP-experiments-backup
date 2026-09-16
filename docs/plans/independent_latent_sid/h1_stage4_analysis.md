# H1生成与四条件评估阶段结果

状态：stage4完成，diagnostics运行中；H1尚未通过完整run验收，B2待运行。此文为单seed阶段观察，不是最终方法结论。

训练revision `70b82db195f600e3271030ddb5e6c228e4915f31`，pilot配置、seed42、冻结8192/1024 train/validation划分不变。H1采用共享point-fusion encoder、标准RQ、辅助深度Grid/Road/GPS=1/2/4。具体样本、视图对齐、特征单位、encoder、码本、decoder、初始化、loss和超参继承spec.md与原pilot命令；本次无新训练或实现修改。

生成1024条，解析失败0，生成耗时3664.2029418637976秒。G00/G10/G01/G11均成功解码1024条，无失败。

|条件|H1|LF|HV|
|---|---:|---:|---:|
|G00|0.01665292334991235|0.01764743359180145|0.018242598024157092|
|G10|0.020242912183963797|0.021536851060222736|0.02221099827124816|
|G01|0.016822412976542866|0.017705329077129614|0.01934921204516262|
|G11|0.0200275766271018|0.021205848841167024|0.021796897452127824|

指标为全1024轨迹四项km统计量abs-log1p差均值，invalid惩罚1，无量纲，不是ADE或相对误差百分比。H1四条件点估计均低于LF/HV，但尚未计算完整四组配对区间，不能据此宣称统计或跨seed优势。HV后三条件包含同一条截断失败，H1/LF无失败，必须报告惩罚10敏感性及共同成功集GPS误差。

H1与LF不仅改变分支是否独立，还改变参数量；H1与HV还涉及量化目标、ST路由等。因此暂不能把点估计差异单独归因于独立latent。所有条件使用oracle路线，不构成自由路线生成或真实运动语义改善证据。下一步完成H1 probes与产物SHA验收、完整SID tuple补评，继续B2，再统一分析unigram收益、码本健康、配对差及SID/采样分解。

来源：`/workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/pilot/porto/42/H1/attempt_001/stage4/stage4_metrics.json`；SHA256 `41d5da1ede3b3b37cfdcd5d3e812c80eaedb556c9acaa89948b2f756f18724ee`。本地精简归档`h1_stage4_summary.json`，未修改生成输出或原始指标。
