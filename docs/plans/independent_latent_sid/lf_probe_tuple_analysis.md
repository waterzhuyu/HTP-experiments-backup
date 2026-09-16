# LF完整验收与probe、tuple补评

`results_snapshot_003.json`验收HV/LF九阶段、预算与各85项产物SHA通过，整体仍incomplete，H1进入stage1、B2待运行。训练revision70b82db195f600e3271030ddb5e6c228e4915f31不变。LF源manifest SHA为98e1afb6d26f298710aea5973a5fdcf3c68184068ce7b226fa0364f213c121a2。

## 冻结probe

24项probe均10epoch/1860step，seed42，AdamW lr=.001、wd=.0001、batch_events256；输入单层或累计码向量加点内offset，目标和单位继承spec，按有效点逆轨迹长度加权。没有更改tokenizer、LLM或GPS恢复。

|LF前缀深度|GPS MSE|Grid准确率|Road准确率|Grid CE|Road CE|
|---|---:|---:|---:|---:|---:|
|1|.0298464|.0115312|.0161401|7.15472|7.45193|
|2|.0296558|.0138647|.0186804|7.10554|7.38752|
|3|.0291906|.0155006|.0218846|7.05078|7.34128|
|4|.0281498|.0169535|.0229153|7.02225|7.30727|

LF三视图随前缀深度总体改善，没有已测任务前缀增加后退化的证据。与HV相比，LF各深度Grid/Road准确率均更高，但深度2–4的CE更高；GPS MSE在深度1/4更低、2/3略高。不能宣布信息指标全面占优，准确率与对真类赋予的概率需分别解释，也不能据此直接证明校准问题。绝对信息量仍不能从这些probe直接读出。

## 完整SID元组

分析revision7f98a2699cd14ed6dc6a42b5dec676730122aa45，冻结已保存adapter，无训练，teacher batch4，1024轨迹/5996事件；同一补评方法继承HV。

```bash
CUDA_VISIBLE_DEVICES=3 uv run python -m hierarchical_rqvae.audit_tuple_accuracy --root /workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917 --cells LF --output /workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/analysis/tuple_lf_001
```

LF teacher-forced完整四码正确929/5996=15.4937%，free-running为762/5996=12.7085%；HV分别36.9746%/28.1021%。LF恢复SID NLL .838312245，与原评估差.000279411，小于.005，所有分母一致。不覆盖原始评估指标。

自由生成按目标event序号对齐，缺失/失败计错，额外46个event单列；0条解析失败不意味着event数完全匹配。不是时间或语义匹配。LF自由生成四层准确率51.3843%/48.7492%/42.9286%/40.9606%。不同码空间的tuple难度不同，较低tuple准确率不能单独证明运动生成更差；LF G11惩罚误差点估计较低也不能反向证明SID更好。需结合unigram基线、共同成功集GPS及完整四组配对结果。

来源LF `diagnostics/probes/metrics.json`及 `analysis/tuple_lf_001/LF/metrics.json`；本地归档lf_probe_metrics.json与lf_tuple_metrics.json。后者含分析/训练revision、源manifest SHA、逐轨迹记录、恢复评估一致性。无新增loss、初始化、优化器或decoder变更，均不适用；完整方法继承spec。下一步继续H1/B2，最终统一统计与排序建议。
