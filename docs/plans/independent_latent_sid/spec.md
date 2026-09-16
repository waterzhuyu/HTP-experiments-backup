# 独立 latent → 分层 SID：探索实验（2026-09-17）

状态：实现验证；科学结论 N/A。来源：[Research 9.3–9.10 的分析及 TODO](https://app.notion.com/p/2a4c24a261be80e58c50dbc95944735b)。9.10–9.17 最新区段记录容量消融进度，本实验依据前一周的具体方法分析。旧 H1 相对 B2 主误差改善 0.10%，CI −2.74% 至 +3.11%，不能主张有效；NLL 降低伴随 tuple 熵降低、指定前缀 probe 未改善。

## 问题与对照

独立编码后按 Grid → Road → GPS 构造有向条件 SID，能否改善前缀信息和端到端生成，而不是仅缩小编码熵？这是新的探索性实现选择，周报并未规定量化公式。独立指参数和输入通道独立，不声称统计独立或视图信息互斥。没有加入 AttnRes/mHC。

- B2：原逐点融合，共享 encoder，四层标准 RQ，辅助前缀 4/4/4。
- H1：原逐点融合，共享 encoder，四层标准 RQ，辅助前缀 Grid/Road/GPS=1/2/4。
- LF：三个独立 encoder，得到三个 latent 后求和除 sqrt(3)，标准 RQ，辅助前缀 1/2/4。控制独立 encoder 带来的参数增量。
- HV：与 LF 完全相同的模块、初始参数、码本和预算，改为视图条件残差量化及各视图的局部梯度路由。

主要比较 HV−LF；LF−H1 衡量编码位置与参数量的联合变化；HV−H1 为总体方法变化；H1−B2 复跑旧比较。HV−LF 同时改变量化目标与梯度路由，不能归因为单一监督位置。单 seed 42 pilot，不作确认性结论；后续 seed 314159、20260907 的重复在本配置中禁用，避免自动扩大预算。

## 数据与对齐

沿用冻结 Porto preflight，官方 train 中 8192 train / 1024 validation，20–200 原始观测点整条轨迹，按 GPS/time 哈希去重，split seed20260907。文件名 test 在继承管线中指 validation_from_official_train；封存 official test 不读取。相同轨迹三视图必在相同 split。车辆、道路、日期可重叠，只支持同城 held-out-trip。数据源、split、归一化文件 SHA 由原 preflight manifest 固定；不重新抽取更有利子集。

样本为整条轨迹，不是独立 8 点窗口。输入 GPS B×T×2，Grid/Road B×T，由 traj_id/point_index 对齐，原点索引全部保留；不删点、不插值、无跨采样训练损失。GPS 经度范围 −8.69 至 −8.55、纬度41.13至41.19按继承 MBR 归一化；Grid 为100m网格 ID；Road 为逐点地图匹配道路 ID。真实 route 是所有组相同的额外 decoder/LLM 条件。dxy（原 km 单位）及 road percent 用训练集统计 z-score。time 为秒，仅用于诊断不进入 encoder；诊断采样沿用原管线，不能据此宣称采样鲁棒。

## 编码与量化

GPS Linear2→64+LayerNorm；Grid Embedding→64+LayerNorm；Road 128维 Node2Vec 初始化的 embedding→Linear64+LayerNorm。重要：实际继承代码中 point-road embedding 和 route embedding 都可训练，并未冻结；Node2Vec 只是初始化，原来源缺独立生成 manifest 的限制沿用。三组保留相同此设置。

三个参数独立的 HTP Encoder：hidden64、ch_multi=[1,2,2,4]、每级2残差块、dropout0.1、三次 stride2、256→64 projection。每轨迹 ceil(T/8) event；event j 的监督点为8j至min(8j+7,T−1)，但卷积/注意力有更大上下文感受野。三视图 lengths、sampling三bit、mask 相同；padding在embedding前屏蔽。Encoder/projection 从同一随机初值复制，权重不共享；LF/HV全 state 初值逐tensor一致。

HV 定义 z_x=Grid latent、z_r=Road latent、z_g=GPS latent，每个64维。s0=0：

```text
t1 = z_x                 q1 = nearest(C1, t1)       s1 = q1
t2 = z_r - stopgrad(s1)  q2 = nearest(C2, t2)       s2 = q1+q2
t3 = z_g - stopgrad(s2)  q3 = nearest(C3, t3)       s3 = q1+q2+q3
t4 = z_g - stopgrad(s3)  q4 = nearest(C4, t4)       s4 = q1+q2+q3+q4
```

K=[32,64,128,256]，欧式最近邻，64维码字，四个 SID a/b/c/d，26名义bits/event；无连续 private bypass、无新增词表。第一层不能读Road/GPS；第二层不能读GPS。共享的向量空间由累计重构及视图目标联合学习；不假设不同 encoder 坐标系天然对齐。

HV量化损失 `L_vq=mean_l[MSE(q_l,stopgrad(t_l))+0.25 MSE(t_l,stopgrad(q_l))]`。前缀作为残差条件被 detach，后层VQ不反向更新前层码本；各层码本只受自身局部量化损失更新。辅助 ST：Grid用 `z_x+stopgrad(s1-z_x)`，Road用 `z_r+stopgrad(s2-z_r)`，GPS用 `z_g+stopgrad(s4-z_g)`，各 head 的梯度只回本视图 encoder。解码用 `z_g+stopgrad(s4-z_g)`，GPS重构梯度只回GPS encoder；Grid/Road encoder由各自辅助+commitment更新。

LF 使用 z=(z_x+z_r+z_g)/sqrt(3)，标准原 RQ：每层拟合同一个 z 的累计前缀，辅助和 decoder ST 都回到 z。B2/H1 保持原定义。

三辅助 head 均为[64维前缀+8维点内offset onehot]→64→SiLU→目标，Grid/Road CE除log(max(训练出现类别数,2))，GPS归一化坐标MSE。有效点先每轨迹平均，再跨batch轨迹平均；末event mask。总损失 `L = MSE(percent_pred,percent_zscore)+MSE(dxy_pred,dxy_zscore)+L_vq+0.1 L_grid+0.1 L_road+0.1 L_gps`；前两项按batch有效点元素平均。无usage/一致性/独占/covariance正则；正负样本选择不适用。

## 解码、训练与评估

四码查表求和→Linear64→256→继承 HTP route-conditioned Decoder→percent+dxy→真实道路几何和反z-score恢复GPS。sampling三bit恢复观测点数；锚点为原观测进度，不是固定真实时钟。GPS解码逐轨迹执行，禁用旧batch干预路径。

所有模块从头（Road embedding用既有初始化）；无旧checkpoint；冻结模块在tokenizer训练阶段无额外冻结。AdamW lr1e-4/wd1e-4/betas(.9,.999)/eps1e-8，clip1，20epoch×64step=1280step，batch128，warmup10epoch起始.01后cosine到.10，BF16、量化和CE FP32，num_workers0，无early stop，最终epoch比较。seed42，cudnn deterministic=true/benchmark=false；相同数据顺序。LF/HV模块数相同但相对H1增加两个encoder/projection，实际参数量写入metrics。

冻结tokenizer，Qwen3-1.7B LoRA r16/alpha32/dropout.05、q/k/v/o/gate/up/down，base冻结、新token行训练；2epoch/1024steps，microbatch2×accum8，lr5e-4/wd1e-4/clip1、linear warmup.05、BF16+gradient checkpointing。序列为a/b/c/d+sampling，assistant-only监督，maxlen2048。每组独立训练adapter；保存协议及config SHA，严格恢复。

生成greedy、batch1、max_new_tokens256、不重试失败。G00真实SID/真实sampling，G10生成SID/真实sampling，G01真实SID/生成sampling，G11两者生成；保留完整1024人口和失败。主指标G11四项km统计量的abs(log1p(generated)−log1p(real))均值，无效固定1，敏感性10。报告逐层SID NLL/accuracy及完整tuple、unigram校正、used codes/perplexity/max share、tuple熵/碰撞、G00→G11误差、100个observation-progress锚点ADE/FDE(m)。冻结probe在训练events训练、验证events评估，单层及累计前缀预测三视图，MLP64、AdamW .001、10epoch、batch256、seed42。不把低CE或高ID一致性解释成视图独占。

探索门槛：HV对LF主误差相对改善≥5%，配对trajectory bootstrap 2000次/seed271828的区间支持改善，G00恶化≤10%，valid≥99%且下降≤1pp；每层active≥.10、perplexity/K≥.05、max share≤.5，同时检查Grid s1/Road s2 probe及tuple熵。门槛均为探索性预设；不以NLL降低单独过关，未显著不等价。单seed bootstrap只刻画轨迹不确定性，不能替代跨seed结果；任何失败完整保留。

## 方法变更表

|组成|H1→LF|LF→HV|
|---|---|---|
|样本/切分|相同|相同|
|视图构造/覆盖|相同|相同|
|输入特征/单位|相同|相同|
|encoder/downsampling|1个共享→3个独立，stride仍8|相同|
|token支持|相同ceil(T/8)与上下文感受野|相同|
|quantizer/codebook|同K与标准RQ，输入改为后融合latent|改为视图条件局部残差VQ，K不变|
|decoder/时间锚点|相同|相同数值解码，ST路由只回GPS|
|初始化/冻结|独立复制encoder，参数增多|逐tensor相同，无额外冻结|
|重构/VQ loss|公式相同|局部残差目标与stopgrad改变|
|辅助loss/权重|前缀1/2/4和0.1不变|目标不变，ST只回本视图|
|训练数据/超参|相同|相同|
|评估对齐/指标|相同|相同|

## 执行与复现

远端权威工作区 `/workspace/guozuyu/HTP-independent-latent-sid`，branch `exp/independent-latent-sid`，fork `52b6666e6a65eeee7fbe7ecc7cae8b84548f7ea3`；upstream `1294275d2bed5f9de86269ab285ff5e23828b947`。继承容量消融的运行/协议修复，旧容量实验工作区不修改。

配置 `stage_1/configs/independent_latent_sid/pilot.json`；模块 `stage_1/src/independent_latent_sid`；本文件同步至 `docs/plans/independent_latent_sid/spec.md`。uv sync --frozen；Python3.12.3；pyproject.toml/uv.lock不变。实验修改在运行前提交；每run manifest记录revision、完整解析配置、seed、数据/模型SHA、产物SHA、elapsed、失败。

```bash
CUDA_VISIBLE_DEVICES=2 uv run python -m independent_latent_sid.run --config stage_1/configs/independent_latent_sid/pilot.json --preflight /workspace/guozuyu/HTP-hierarchical-rqvae-artifacts/hierarchical_rqvae_gps_grid_road_v1_20260907/preflight --phase smoke --cells LF HV --seed 42
```

Smoke：32 train/16 validation、tokenizer1epoch/batch8=4steps、LLM2steps、8生成轨迹、probe1epoch；只验证实现。pilot把phase改为pilot、cells改为HV LF H1 B2，四组总24 GPU小时、40GiB产物，单进程串行；单seed42。产物根 `/workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/{phase}/porto/42/{cell}/attempt_NNN/`，拒绝dirty运行、覆盖或重复已完成cell。正式pilot在LF/HV完整smoke通过且全部产物SHA、四G各8条唯一轨迹、G00全部解码成功后启动。启动器 `uv run python -m independent_latent_sid.launch --config stage_1/configs/independent_latent_sid/pilot.json --gpu 2` 独占创建启动记录，后台串行运行HV LF H1 B2；记录PID、日志、GPU、完整命令与Git revision。单seed探索不代表正式确认。
