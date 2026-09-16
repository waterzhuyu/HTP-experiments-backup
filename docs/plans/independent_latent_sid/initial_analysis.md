# 独立 latent SID：HV tokenizer 初步诊断（2026-09-17）

状态：阶段性分析，四组端到端结果未齐。不得以本报告替代HV−LF/H1/B2的完整结果比较。当前正式训练PID2602619仍存活；HV已完成20epoch/1280step tokenizer，LLM正在1024step预算内训练。后续LF/H1/B2尚待执行。上一目标回合分类：progress（实现并启动实验）；本回合新增冻结模型诊断证据，非单纯等待。

## 已确认事实

验证人口1024条轨迹、5996 events。四层统计：

|层|用过码字|perplexity|最大码占比|
|---|---:|---:|---:|
|Grid q1|29/32|1.3782|94.68%|
|Road q2|27/64|2.9420|65.41%|
|GPS q3|6/128|3.9875|30.40%|
|GPS q4|7/256|4.6310|35.34%|

第一层“用过29个码字”掩盖了强烈集中。训练集8192条/47533events相同：q1最大码占比94.40%，perplexity1.4004；不是仅验证集分布造成。HV当前已不满足原计划的健康门槛，但不终止后续生成或丢弃这组，以保留负面结果。

验证latent协方差的entropy effective rank：Grid1.1860、Road1.1150、GPS2.8206；训练为1.1739/1.1116/2.8050。这里effective rank=exp(H(协方差非负特征值占比))，是连续变化的维度指标，不代表语义类别数量。

验证平均event L2范数：Grid4.3771、Road2.7362、GPS9.8098；累计码字s1/s2/s4分别1.0212/1.5917/1.9670。z_gps与s4的逐维MSE1.8423。三视图的量化尺度/误差明显不同，但尚不能证明尺度错配是坍缩的唯一原因。

第一批128条训练轨迹、FP32 train模式/dropout seed42的projection weight梯度：

|损失项（已加实际权重）|Grid|Road|GPS|
|---|---:|---:|---:|
|轨迹重构|无路径|无路径|0.81057|
|总VQ|0.04602|0.02022|0.24142|
|0.1 Grid辅助|0.04554|无路径|无路径|
|0.1 Road辅助|无路径|0.02038|无路径|
|0.1 GPS辅助|无路径|无路径|0.02097|

这是结构规定的信息流，不是测得梯度很小就推定缺失；不能用一个batch的norm比值决定新loss权重。Grid/Road辅助梯度与VQ在该batch量级接近，并无证据支持“辅助梯度完全消失”。

q2与s1的平均cosine=+0.4149，q3与s2≈−0.0153；没有普遍反向抵消的证据。Grid/Road latent平均cosine0.8490，二者与GPS分别0.1540/0.1672。它们由相同初始encoder复制但独立更新；高cosine不等于语义一致。

## 机制解释与改进候选（尚未确证）

1. **独立编码不自动产生有用分工。** 当前Grid/Road连续表示已高度低秩，并且无法从最终轨迹重构获得梯度；这与“分支只解决弱辅助任务，离散编码大量集中”的解释一致。先比较LF与HV同模块、相同初值的结果和梯度，再决定是否改变梯度路由。候选小消融：保持HV前向与码本目标，只改变最终ST代理，使解码重构梯度能到达三个encoder；同时报告这会削弱当前梯度隔离性质，不能直接称为同一方法。
2. **量化器跟不上latent尺度是可检验的候选。** GPS latent与量化和的范数差距很大；当前码本uniform初始化幅度随层K递减且没有latent归一化。先在冻结训练latent上检查最近邻残差分布和训练期码本/latent范数轨迹，再比较数据初始化或规范化。候选：保持同K和架构，分别做训练数据k-means初始化、EMA更新/死码重置或规范化；每次只改变一个机制。不把usage正则直接叠加成多因素“修复”，也不把码本变均匀当成语义收益。
3. **局部残差目标不保证信息递增。** 当前s2拟合z_road、s4拟合z_gps，并非拟合包含所有前视图的联合表示；这是公式层面的限制。需用各前缀的三视图冻结probe比较“加入下一层后是否保留前面信息”，而不是按命名认定Grid→Road→GPS成立。若实测前缀信息退化，再比较“保留独立分组码字并拼接给decoder”的方案；该方案改变decoder接口，必须另列对照，不能和尺度修复合并归因。
4. **辅助目标可能过细。** 32个Grid码字拟合数千绝对网格ID，既有容量也有语义粒度问题。仅当LF/HV probe表明绝对位置任务主导且缺少运动信息时，再比较多尺度区域/相对位移/道路转移目标；保持同码率并同时检查位置保持和运动保持，避免靠放弃任务让CE变低。

优先级在四组结果齐全前是暂定的：先定位LF和HV之间的差异，再选最小消融；不启动多种未经筛选的新训练。完整判断必须同时看G11主误差、G00重构、有效率、前缀probe、tuple熵及相对unigram的NLL收益。若HV主误差未改善且码本集中，结论应是当前实现不支持假设，而不是“独立latent方向已被证伪”。

## 诊断方法与复现

继承样本、预处理、特征单位、三独立HTP encoder、四层条件残差公式、码本、初始化、loss、训练配置、GPS解码语义与差异表均见spec.md，未重新训练或更改tokenizer。新增诊断对完整train与validation执行FP32 eval，按event等权汇总；不是原主误差的按轨迹平均。latent均值/协方差在每个split各自计算，只用于描述，不用于训练、初始化或阈值调节；无test访问。

输出：latent均值范数、event范数、中心化RMS、逐维std均值、有效秩；每层码本用量和前后缀范数；所有四前缀与三个latent的逐维MSE；跨视图及残差与前缀的cosine。无新增GPS解码。梯度诊断只在第一批128训练轨迹做前向/autograd.grad，无optimizer、无参数更新、无checkpoint写入；FP32、train模式、dropout seed42，记录projection weight梯度范数，无正负样本构造。本次不改变任何训练模块或目标。

|组成|训练run|本次只读诊断|
|---|---|---|
|样本/视图/特征/单位|冻结Porto|相同|
|encoder/token支持/quantizer|HV实现|相同，加载最终epoch checkpoint|
|decoder/GPS锚点|原HTP|未执行新GPS解码|
|初始化/冻结|训练run定义|参数固定；梯度测量不更新|
|重构/VQ/辅助目标|原公式|只测量实际加权项|
|训练数据/超参|8192/1024，seed42|无训练；8192/1024全量描述+首128梯度|
|评估对齐/指标|端到端及probe|追加event等权几何诊断，不能替代主指标|

权威训练revision `70b82db195f600e3271030ddb5e6c228e4915f31`；分析revision `869a8ef`（完整hash在JSON中）。独立分析worktree `/workspace/guozuyu/HTP-independent-latent-sid-analysis`，branch `analysis/independent-latent-sid`。uv sync --frozen，2项诊断测试通过（7.55s）：常量/秩一协方差、全单码与已知残差抵消合成案例。

```bash
CUDA_VISIBLE_DEVICES=3 uv run python -m independent_latent_sid.inspect_latents --config stage_1/configs/independent_latent_sid/pilot.json --cell HV --run-dir /workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/pilot/porto/42/HV/attempt_001 --output /workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/analysis/hv_latent_001.json
```

运行前核验stage1所有产物SHA、cell/config身份；run JSON记录checkpoint SHA、preflight SHA、训练与分析revision、完整命令。科学结论状态：incomplete，待四组完整端到端结果与probe。没有因当前不利结果更改原运行目标、超参或队列。

## 八批梯度方向补查

只读加载同一最终HV checkpoint，取冻结train按原顺序前8批×128=1024条轨迹，FP32 train模式、dropout seeds42–49，无optimizer、不更新checkpoint。计算每个视图projection weight上实际加权loss梯度的展平向量cosine；缺失/零梯度返回N/A。分析代码提交 `b087e2a`，1项测试通过（2.67s），验证反向/正交/无梯度的已知例子。原模型、数据、单位、事件支持、目标权重及其他方法组成均不变；无新增GPS解码。方法与限制写入输出JSON。

|梯度对|8批平均cosine|范围|
|---|---:|---|
|Grid：VQ vs 0.1 Grid辅助|−0.999877|−0.999926 至 −0.999815|
|Road：VQ vs 0.1 Road辅助|−0.999620|−0.999783 至 −0.999473|
|GPS：重构 vs VQ|−0.646182|−0.772049 至 −0.470429|

**解释边界：**这表明最终checkpoint的projection weight上，两种目标的局部方向几乎对立；结合前次量级相近，可解释为何梯度存在但未形成丰富表示。然而在目标加权和的稳定点，各项梯度本来就可能相互抵消。这不是训练过程追踪，不能单凭cosine断言冲突导致坍缩，也不能直接据此采用梯度投影算法。需要在LF上相同方法对照，并检查新初始化/训练阶段安排能否改善真实code信息和生成。

优先候选补充：**先训练独立连续表示，再引入量化**。先用相同视图head/目标检验连续latent能否保持足够信息与维度，然后冻结encoder、在仅训练集latent上初始化/拟合码本，再训练严格只读取离散码字的decoder；最后可选择小学习率联合微调。该候选同时改变初始化和训练日程，须拆出“仅数据初始化”“冻结encoder量化”“连续预训练+量化”对照，且不能让最终decoder使用连续latent旁路。它目前是待比较LF结果后确定的改进意见，尚未执行新训练。

```bash
CUDA_VISIBLE_DEVICES=3 uv run python -m independent_latent_sid.gradient_conflict --config stage_1/configs/independent_latent_sid/pilot.json --run-dir /workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/pilot/porto/42/HV/attempt_001 --output /workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/analysis/hv_gradient_conflict_001.json
```

输出含完整分析/训练revision、checkpoint/config/preflight SHA、命令、逐batch记录。不能将本次诊断当成四组端到端分析已完成。
