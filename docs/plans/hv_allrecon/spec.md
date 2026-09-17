# HV-AllRecon：重构梯度路径消融（2026-09-17）

状态：planned（正式启动前记录）；单 seed tokenizer-only 消融，不是完整生成收益实验。

## 假设、范围与门槛

问题：保持条件残差 SID 前向和局部量化目标不变，仅允许 Grid/Road 编码器获得轨迹重构指导，能否改善码集中与 G00 重构？此改动同时增加接收重构梯度的分支和整体梯度尺度；不宣称纯粹解决冲突。新增参数为0。对照为已完成 HV seed42，不重复已完成训练、不覆盖旧产物。对照 revision 70b82db195f600e3271030ddb5e6c228e4915f31，manifest 路径/SHA 固定在新配置中。

原 HV：u=z_gps。HV-AllRecon：u=z_gps+(z_grid−sg(z_grid))+(z_road−sg(z_road))。两者 decoder_input=u+sg(s4−u)。前向都是s4；后者对三个latent的重构Jacobian均为单位阵。码本不接收重构梯度；辅助梯度仍只到所属分支。sg表示停止梯度。真实SID四层码向量求和，无连续旁路。

先运行32 train/16 validation、1 epoch、batch8的smoke，再正式8192/1024、20epoch/1280step、batch128。smoke必须完成初始梯度诊断、训练、tokens、全16条G00恢复、24个probe各1epoch及码本/采样/秩诊断，产物SHA与代码revision匹配后才允许pilot。独立目录保存smoke，不参与科学结论。

正式阶段仅运行tokenizer、G00及诊断，8 GPU小时/20GiB上限、单GPU、seed42。不执行LLM训练/生成，G10/G01/G11和SID模型NLL为N/A。LLM后续门槛：G00主误差≤原HV的1.10倍，有效率≥99%且下降≤1pp，每层active/K≥.10、perplexity/K≥.05、最大占比≤.50。另描述性检查Grid/Road有效秩及前缀probe，不编造其预设效应阈值。结束后人工分析门槛，不自动启动LLM；若未来执行LLM仍需相对匹配基线G11改善≥5%且配对区间支持。seed重复/跨城/封存test不执行。

## 方法变更表

|组成|原HV|HV-AllRecon|变化|
|---|---|---|---|
|样本物理单元|整条轨迹20–200点|相同|无|
|视图构造/覆盖|GPS/Grid/Road按traj_id和原点索引一一对齐|相同|无|
|输入/单位|GPS归一化经纬度，100m Grid ID，Road ID|相同|无|
|encoder/downsampling|三个参数独立encoder；3次stride2|相同|无|
|token支持|ceil(T/8)事件，标签支持8j至min(8j+7,T−1)，感受野更宽|相同|无|
|quantizer/codebooks|局部条件残差，64维，K32/64/128/256|相同|无|
|decoder/输出锚点|route-conditioned，percent+dxy恢复原点GPS|相同前向|仅输入ST反向不同|
|初始化/冻结|复制相同encoder初值，从头训练；Road N2V初始化可训练|相同|无|
|reconstruction/VQ|重构只回GPS，局部VQ四层平均|重构回三分支，VQ完全相同|唯一训练干预|
|辅助loss/权重|Grid/Road/GPS深度1/2/4，各.1且只回自身|相同|无|
|数据/优化器/预算|8192/1024，seed42，tokenizer1280step|相同tokenizer预算|本轮不运行LLM|
|评估|全人口G00及诊断|同G00/码本/probe；另记梯度和秩|G10/G01/G11暂N/A|

## 可复现方法

数据：Porto官方train内部冻结8192 train、1024 validation整轨迹，split_seed20260907，GPS/time哈希去重，无跨split轨迹或视图混入。车辆/道路/日期可重叠，非OOD评估；不读取封存test。代码内部test文件名指上述validation。preflight=/workspace/guozuyu/HTP-hierarchical-rqvae-artifacts/hierarchical_rqvae_gps_grid_road_v1_20260907/preflight，manifest SHA=a8857cd044076731d133841b1f4ba45fe7e44e7198c2242c939239a25bbaad28。具体源数据文件和hash由preflight冻结。

输入GPS[B,T,2]、Grid/Road[B,T]、有效length/mask；GPS经度[-8.69,-8.55]、纬度[41.13,41.19]继承MBR归一化；dxy原单位km、road percent使用train z-score，time秒仅诊断。padding屏蔽并在embedding查表前清理。GPS Linear2→64+LN；Grid Embedding64+LN；Road可训练Node2Vec128→Linear64+LN。每分支HTP encoder hidden64、ch_multi[1,2,2,4]、每级2残差块、dropout.1、三次stride2；256→64 projection；encoder初值复制、参数独立。总参数与HV相同19,836,061。

q1=Q1(z_grid)，q2=Q2(z_road−sg(s1))，q3=Q3(z_gps−sg(s2))，q4=Q4(z_gps−sg(s3))，s_l=sum(q1..q_l)，最近邻平方欧氏距离。量化FP32。L_vq=mean_l[MSE(q_l,sg(target_l))+.25MSE(target_l,sg(q_l))]。码本随机初始化与原HV完全相同，无EMA/重置/额外初始化；每层码本仅由本地VQ更新。

L_total=L_percent+L_dxy+L_vq+.1L_grid+.1L_road+.1L_gps。percent/dxy为batch有效元素MSE；辅助为64维累计码+8维one-hot offset→64SiLU→目标，GPS归一化MSE，Grid/Road CE除log(train词表大小)，按有效点逆轨迹长度加权再batch均值。辅助使用自身latent的ST，无新一致性、usage、covariance或采样训练损失。

decoder将四码和Linear64→256，真实route条件，预测percent+dxy，通过道路几何与反归一化恢复GPS；真实sampling的三bit恢复点数与原观测索引。无时间窗改动、无连续解码旁路。训练AdamW lr1e-4/wd1e-4/betas(.9,.999)/eps1e-8；clip1，20epoch，batch128，warmup10epoch倍率.01后cosine到.1；BF16，quantizer/CE FP32，num_workers0，无early stopping。seed42、cudnn deterministic true/benchmark false，与baseline一致，不保证所有GPU算子完全确定性。

## 评估与诊断

G00：真实SID、真实sampling、真实route。主误差为四项km统计量abs(log1p(pred)−log1p(real))均值，所有1024轨迹含失败惩罚1；另需报告失败惩罚10敏感性。GPS ADE/FDE(m)在100个观测索引进度锚点线性插值，非时间对齐；成功集与全人口分开。后续对比使用配对trajectory bootstrap2000次/seed271828，单seed区间不代表训练随机性。

码本逐层active/perplexity/max share，完整tuple熵/碰撞/unique tuple；冻结probe使用train events训练、validation评估，输入单层或累计码加8维offset，MLP64，AdamW.001/wd.0001，batch_events256，10epoch，seed42，共24项，与原HV一致。常量probe N/A，不将低CE解释为信息独占。

采样诊断仅原点0::2/0::3保留末点，再按真实time插值回原网格；GPS线性、Grid/Road最近时间（平局较早），长度/mask不变，同事件比较逐层ID/tuple一致率和AMI。不声称删点后的采样鲁棒性。

新增只读诊断：初始化和最终checkpoint各取固定首个validation batch，train-mode BF16、seed42，分别用gps_only/all_views重算reconstruction_loss并反向，不执行optimizer，记录三个encoder与projection的clip前L2梯度范数及码本梯度（应0）。仅是固定批次诊断，不当成完整训练轨迹的梯度统计。最终latent有效秩为全部validation事件中心化latent矩阵奇异值归一化后的exp(entropy)，逐视图报告；若与旧报告比较必须先核对定义。

## 版本、命令与产物

新worktree=/workspace/guozuyu/HTP-hv-allrecon；branch=experiment/hv-allrecon-20260917；环境uv sync --frozen，依赖未修改。训练必须先提交，实际Git SHA由run_manifest记录。

配置stage_1/configs/independent_latent_sid/hv_allrecon.json；执行命令：

```sh
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 uv run python -m independent_latent_sid.recon_run --config /workspace/guozuyu/HTP-hv-allrecon/stage_1/configs/independent_latent_sid/hv_allrecon.json --preflight /workspace/guozuyu/HTP-hierarchical-rqvae-artifacts/hierarchical_rqvae_gps_grid_road_v1_20260907/preflight --phase pilot --cells HVAllRecon --seed 42
```

smoke同命令仅phase=smoke；验证命令uv run pytest stage_1/tests/test_recon_route.py stage_1/tests/test_independent_latent_sid.py -q。正式产物根/workspace/guozuyu/HTP-independent-latent-sid-artifacts/hv_allrecon_20260917/pilot/porto/42/HVAllRecon/attempt_001。保存run_manifest/完整解析配置、stage1日志与checkpoint、tokens、oracle/G00、diagnostics/probes、initial/final_route_audit及每阶段SHA。结果尚未产生，不能声称改善。完成后将事实、解释和gate判定分开记录；失败保留原attempt并记原因。

验证记录：19项模型专项测试通过；完整仓库317项测试通过（57条既有Transformer warning），27.02秒。尚待GPU端到端smoke。
