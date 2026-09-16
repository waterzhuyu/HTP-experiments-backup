# LF生成与解码阶段结果

状态：stage4已完成，diagnostics待完整验收；HV已完整验收，H1/B2待运行。此处只报告点估计，不作显著性或最终方法收益判断。

沿用训练revision `70b82db195f600e3271030ddb5e6c228e4915f31`、pilot配置、seed42、冻结8192/1024划分，方法与超参完整继承spec.md及原pilot命令。无代码、训练、初始化、损失或GPS恢复变更。LF生成1024条，解析失败0，耗时3544.1927598472685s。G00/G10/G01/G11均成功解码1024条，无失败。

|全体1024轨迹平均惩罚误差|LF|HV|
|---|---:|---:|
|G00（真实SID/真实采样）|0.0176474336|0.0182425980|
|G10（生成SID/真实采样）|0.0215368511|0.0222109983|
|G01（真实SID/生成采样）|0.0177053291|0.0193492120|
|G11（生成SID/生成采样）|0.0212058488|0.0217968975|

指标为四项km统计量abs-log1p误差的均值，失败惩罚1，无量纲，不是ADE或相对误差百分比。所有条件使用oracle路线。HV的G10/G01/G11各有同一条生成截断失败，LF无失败，因此不能把差值全部解释为成功轨迹上的运动质量差异。待四组完整验收后报告既定配对bootstrap、惩罚10敏感性、共同成功集GPS误差和SID/采样分解；目前尚未计算正式比较区间。

LF原始stage4来源：`/workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/pilot/porto/42/LF/attempt_001/stage4/stage4_metrics.json`，SHA256 `b42445a20cbdcffaf81057d5881b35593102fcd1a173c69adeeac7f2c0130527`；本地精简汇总 `lf_stage4_summary.json`。原始逐轨迹JSON保留在远端，临时副本仅用于提取摘要。没有重跑或修改失败输出。下一步验收LF诊断和全部产物，完成tuple补评，并继续H1/B2。
