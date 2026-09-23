# I3：GPS-only 重构梯度尺度对照（2026-09-24）

状态：预注册，尚未运行。与已完成的 HV 和 HV-AllRecon seed 42 tokenizer 比较；本次只改变 decoder 重构对 GPS encoder/projection 的反向梯度倍数。探索性单训练 seed，不作端到端生成结论。

## 假设与干预

HV-AllRecon 的重构梯度回到 Grid/Road/GPS 三分支，前层码集中缓解、G00 ADE 改善，但它也增加了 encoder 总重构梯度范数。本对照检验只增加梯度尺度能否产生相似效果。前向 `s4`、四层真实 SID、码本、decoder、辅助损失与 HV 完全相同。仅把 decoder 入口的直通估计器从 `u=z_gps; st=u+sg(s4-u)` 改为 `u=α z_gps; st=u+sg(s4-u)`；前向仍严格等于 `s4`，重构对 GPS encoder/projection 的梯度乘 α，Grid/Road 仍无重构路径，VQ 与辅助梯度不乘 α，decoder 本身梯度不变。α=1.2593601909739016。

α 在读取本轮结果之前由已归档 AllRecon 初始固定验证 batch（128 条，seed 42，BF16 train mode，无 optimizer）的 encoder 重构梯度范数确定：`sqrt(1.1366103356²+1.4626972335²+2.4198538830²)/2.4198538830`。这只匹配 encoder 三分支的初始联合 L2 范数，不保证训练中或整个模型的梯度尺度匹配。来源 `/workspace/guozuyu/HTP-independent-latent-sid-artifacts/hv_allrecon_20260917/pilot/porto/42/HVAllRecon/attempt_001/initial_route_audit.json`。训练日志必须记录每步 clip 前总范数，报告 clip>1 比例，并以初始/最终同批次审计确认梯度路由。

## 不变项与评价

数据为冻结 Porto 官方 train 内 8192 train /1024 validation 整轨迹，GPS/time 指纹去重，split seed 20260907；封存 official test 不使用。GPS/Grid/Road 按 `traj_id/point_index` 对齐；GPS 为继承 MBR 归一化经纬度，Grid 100 m ID，Road 零起点道路 ID。dxy 原 km、road percent 经 train z-score；time 秒仅诊断。输入由独立三分支 HTP encoder 处理：GPS Linear2→64+LN、Grid embedding64+LN、Road Node2Vec128 初始化且可训练→Linear64+LN，三次 stride 2，256→64 projection，每轨迹 `ceil(T/8)` 个 event；event `j` 的监督锚点是 `[8j,min(8j+8,T))`，感受野更宽。无额外采样视图或正负样本。

量化为 `q1=Q1(z_grid)`、`q2=Q2(z_road−sg(s1))`、`q3=Q3(z_gps−sg(s2))`、`q4=Q4(z_gps−sg(s3))`，`s_l=Σ_{i≤l}q_i`，64维，K=[32,64,128,256]，平方欧氏最近邻。`L_vq=mean_l[MSE(q_l,sg(t_l))+.25MSE(t_l,sg(q_l))]`。Grid/Road/GPS 辅助 head 用 s1/s2/s4 加块内 8 位 one-hot offset，loss 权重各 .1；各自 ST 梯度只回对应 encoder。主损失为 z-score percent MSE+dxy MSE+VQ+三辅助损失。训练模块全可训练，无新增参数、冻结、EMA、初始化或 loss。四码之和经 Linear64→256、真实 route 条件 decoder 输出 percent/dxy，正式逐轨迹路径恢复 GPS；sampling 三位用真实值恢复点数。

AdamW lr1e-4/wd1e-4、beta(.9,.999)、eps1e-8、clip1；20 epoch×64 step=1280 step，batch128，前10 epoch warmup 从 .01 倍到 1，后 cosine 到 .1；BF16、量化和 CE FP32，num_workers0，seed42，cudnn deterministic=true/benchmark=false。与 HV/HV-AllRecon 匹配。先 smoke 32 train/16 validation、1 epoch/4 step；正式 tokenizer-only 8 GPU 小时和 20 GiB 上限，单 GPU。G00 全 1024 验证轨迹，真实 SID/sampling/route，四项 km 统计 abs(log1p 差)均值，失败罚1并检查罚10；100 个 observation-progress 锚点的 ADE/FDE 单位 m。逐层 active/K≥.10、perplexity/K≥.05、max share≤.5、G00 不恶化超过 HV 的10%、valid≥99% 且下降≤1pp 为探索门槛；并看全人口 tuple 熵、冻结 Grid/Road/GPS probe、latent rank。门槛不通过仍保留失败。只有 tokenizer 门槛通过且单独记录决策后才进入 LLM；本配置不执行 LLM、G10/G01/G11，故不能宣称端到端改进。

## 版本与产物

工作区 `/workspace/guozuyu/HTP-hv-gps-scale`，基于 AllRecon revision `b2ddac5b7ef7b8198a66301594e04daf65cfe8e2` 的独立 worktree，upstream `1294275d2bed5f9de86269ab285ff5e23828b947`，原 HV 训练 revision `70b82db195f600e3271030ddb5e6c228e4915f31`。配置 `stage_1/configs/independent_latent_sid/hv_gps_scale.json`，运行前提交并从 manifest 记录完整 40 位 revision、config/preflight/baseline SHA、命令、seed、输出与每阶段产物 SHA。`uv sync --frozen`，不变更依赖。输出 `/workspace/guozuyu/HTP-independent-latent-sid-artifacts/hv_gps_scale_20260924/{smoke,pilot}/porto/42/HVGpsScale/attempt_001`。
