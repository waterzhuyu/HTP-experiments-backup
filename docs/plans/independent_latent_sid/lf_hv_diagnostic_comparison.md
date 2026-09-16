# LF与HV tokenizer阶段诊断

状态：阶段性机制证据；HV完整run已验收，LF语言模型训练中，H1/B2待运行。没有组间生成收益结论。

训练revision均为70b82db195f600e3271030ddb5e6c228e4915f31，seed42，8192/1024冻结train/validation，20epoch/1280step。完整样本、视图、单位、encoder、码本、decoder、初始化、loss、超参及评估定义继承spec.md。LF与HV不仅改变残差目标，也改变ST梯度路由，不能归因于单一因素。

LF采用三个独立latent相加除sqrt(3)再标准RQ，重构和各辅助任务的梯度均可流向三个分支。HV逐层拟合Grid、Road和GPS的局部残差，重构ST只到GPS、各辅助ST只到所属分支。两者均为严格离散decoder输入、相同K=32/64/128/256和维度64。

## 全验证集5996事件

|指标|LF|HV|
|---|---|---|
|Grid/Road/GPS有效秩|4.725/4.270/7.166|1.186/1.115/2.821|
|四层perplexity|4.158/4.198/4.568/5.037|1.378/2.942/3.988/4.631|
|四层最大码占比|42.85%/34.99%/36.27%/28.72%|94.68%/65.41%/30.40%/35.34%|
|四层active codes|8/9/8/12|29/27/6/7|

train上的秩与分布趋势相同。LF有更少的active code却有更高前两层perplexity，表明不能以“曾使用过多少码”判断有效容量。两者后层有效用量仍很低；LF不是已解决码本利用问题的方案。不同模型的latent坐标和尺度可变化，有效秩提高也不直接证明语义内容更丰富。

## 梯度方向

两组同一方法：固定最终checkpoint，连续前8个train batch×128，FP32 train模式，dropout seeds42–49，无optimizer和参数更新。只测三分支projection weight上实际加权损失的梯度cosine，不能代表全参数或训练历史。

LF的VQ与所属辅助项cosine均值：Grid −0.721571，Road −0.309758；HV分别−0.999877、−0.999620。LF重构与VQ在Grid/Road/GPS为−0.177634/−0.453883/−0.043469，HV的Grid/Road无重构路径、GPS为−0.646182。LF首批重构梯度norm在Grid/Road/GPS为.17104/.29766/.12179；HV仅GPS为.81057。单batch尺度不可直接用于调权。

HV近完全反向的局部梯度平衡，在LF中明显减弱，与其更高latent有效秩相伴出现。但稳定点本可发生损失梯度抵消，且两组同时改变多个机制；当前结果不能证明梯度冲突导致塌缩。

## 对改进优先级的影响

先完成四组生成、重构、probe和tuple对照。若HV端到端也劣于LF，优先做保持HV前向量化目标不变、仅改变最终重构ST路由的消融，以分离目标与梯度隔离；明确其改变原梯度隔离性质。连续预训练后冻结encoder拟合码本仍是候选，但应与仅数据初始化分开。后层低利用是共同问题，可另做码本初始化/更新机制控制。不能把所有修复同时叠加后归因于其中之一。

HV已完成的prefix probe总体随层数改善，当前不支持“后续层造成已测任务信息退化”；不因理论上存在覆盖风险就优先更换为拼接接口。

## 复现与产物

LF诊断revision 3891560677d968ffad44b3bf36a8ac9fc73a597a；HV几何诊断869a8ef、梯度诊断b087e2a。源JSON记录完整revision及checkpoint/config/preflight SHA。无新训练，无test访问，无新GPS解码。

在分析worktree执行（GPU3）：

```bash
CUDA_VISIBLE_DEVICES=3 uv run python -m independent_latent_sid.inspect_latents --config stage_1/configs/independent_latent_sid/pilot.json --cell LF --run-dir /workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/pilot/porto/42/LF/attempt_001 --output /workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/analysis/lf_latent_001.json
CUDA_VISIBLE_DEVICES=3 uv run python -m independent_latent_sid.gradient_conflict --config stage_1/configs/independent_latent_sid/pilot.json --run-dir /workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/pilot/porto/42/LF/attempt_001 --output /workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/analysis/lf_gradient_conflict_001.json
```

本地同名JSON归档；诊断代码未修改，沿用已通过测试版本。所有解释均属描述性单seed证据，最终建议待完整对照。
