# H1完整验收、冻结probe与SID元组补评

`results_snapshot_004.json`已验收HV/LF/H1九阶段、训练预算和各85项产物SHA。B2正在stage1，整体仍incomplete。H1源manifest SHA为`a7945c41efd35cb983ee266c1a59d7ee92988ba684cba9879787c7c54bcf6aea`，训练revision仍`70b82db195f600e3271030ddb5e6c228e4915f31`，seed42、冻结8192/1024划分不变。

24项probe均10epoch/1860step，AdamW lr=.001、wd=.0001、batch_events256、seed42；冻结单层或累计码向量加8维offset输入，目标和单位继承spec，按有效点逆轨迹长度加权。前缀1→4：GPS MSE .03021437/.02971047/.02954960/.02943039；Grid准确率.01165663/.01339850/.01512540/.01526792；Road准确率.01555493/.01885176/.02146407/.02219877。Grid CE 7.163556/7.088467/7.053474/7.039858；Road CE 7.455053/7.364761/7.328942/7.304480。均随深度改善，未观察到已测任务的前缀退化。

最深前缀LF的GPS probe MSE .02814978低于H1的.02943039，Grid/Road准确率也略高，但H1的G11轨迹统计误差点估计更低。冻结probe与端到端轨迹质量衡量不同环节，不能用其中一个替代另一个，也不能由此单独推断decoder因果机制。

补评分析revision `8efb4ec813f08849ccd0a2f6a7741c8a56fb6615`；GPU3，保存adapter冻结、无训练、teacher batch4。命令：

```bash
CUDA_VISIBLE_DEVICES=3 uv run python -m hierarchical_rqvae.audit_tuple_accuracy --root /workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917 --cells H1 --output /workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/analysis/tuple_h1_001
```

1024轨迹/5996事件，teacher-forced完整四码正确1033/5996=17.22815%，free-running 777/5996=12.95864%。LF分别15.49366%/12.70847%，HV为36.97465%/28.10207%，不同码空间的准确率不可直接作为生成质量排名。H1自由生成四层准确率46.0474%/52.9186%/46.5977%/48.9326%；0条解析失败、额外43事件。按目标event序号对齐、缺失和失败计错、额外事件单列，不是时间或语义对齐。

恢复SID NLL .7732869420073842，与原评估差.00006542184338553803，小于.005；每层5996、总token52064分母一致。原评估不覆盖。运行90.04697秒（不含模型加载）。本次无新训练、loss、初始化、量化器或GPS重建修改，完整方法继承spec与既有补评协议。

本地归档`h1_probe_metrics.json`、`h1_tuple_metrics.json`及验收快照。下一步等待B2完成、同样补评tuple，随后正式进行四组配对统计、unigram收益、码本健康与最终建议；当前不宣称跨seed或泛化收益。
