# 分层视图 RQ-VAE → LLM 实验 spec

- Experiment ID：`hierarchical_rqvae_gps_grid_road_v1_20260907`
- 状态：**planned / spec-ready；未实现模型扩展、未启动 smoke 或训练，无实验结果。**
- 日期：2026-09-07。
- authoritative worktree：`3090s:/workspace/guozuyu/HTP-hierarchical-rqvae`
- branch：`exp/hierarchical-rqvae-gps-grid-road`
- 分叉 revision：`da3b80e3d9dffe525c67a7722e99387cdb345347`（已包含集成后的 staged-attribution、紧凑 adapter 与进度记录）。
- upstream reproduction revision：`1294275d2bed5f9de86269ab285ff5e23828b947`。历史结果不能直接充当本次相同预算的 baseline。
- 本轮交付：实验设计、实现契约、预算配置和记录；下一阶段先实现与测试，再做 smoke，再按门槛开展 pilot。

## 1. 研究问题和主张边界

问题：在相同输入信息、RQ 容量、token 数、LLM 与训练预算下，给 RQ 的不同前缀施加 GPS/Grid/Road 监督，能否产生更易被 LLM 学习、且最终轨迹生成更好的离散表示？

HTP 原文使用 coarse-to-fine RQ token 后接 LLM SFT；这里增加“前缀的视图职责”，属于独立方法扩展，不是原论文复现本身。论文来源：[From GPS Points to Travel Patterns](https://arxiv.org/abs/2605.30014)。当前代码事实以分叉 revision 的 `staged_attribution/model.py`、`quantizer.py`、`token_protocol.py` 为准。

- H1：分层视图监督相比同目标的末层监督降低 held-out LLM 的 SID NLL，并改善 G11 生成误差。
- H2：Grid 前缀能恢复粗空间占据，增加 Road 层提高道路重构，增加 GPS 层提高精细坐标重构。必须用前缀 probe、截断和层替换诊断证实，不能靠名称判断。
- H3：收益不能仅由额外输入视图、辅助监督或增加 token 数解释。
- GPS/Grid/Road 是同源轨迹的三种表示；不声称独立模态、因果解耦或道路身份不变的运动原语。
- 所有 cell 的 LLM prompt 和主 GPS decoder 都使用相同真实 route。这是 **给定 route 的运动轨迹生成**，不能据此声称改善路线选择。
- 第一版保留标准 residual quantization；不把不同坐标系的三个 encoder 输出直接相减。所有量化残差都在同一个 64 维 latent 空间中计算。

## 2. 首版层分配：Grid → Road → GPS（1/1/2）

保留四层码本 `[32,64,128,256]`，每个 event 固定四个 token，最大名义组合容量 `32×64×128×256=67,108,864`（26 bits；实际熵另报）。

| RQ 层 | 监督入口 | 预期作用 |
|---|---|---|
| L1，32 codes | Grid head 读取累计量化 s1 | 粗空间网格信息 |
| L2，64 codes | Road head 读取 s2=q1+q2 | 在前缀基础上补充道路信息 |
| L3、L4，128/256 codes | GPS head 和原主 decoder 读取 s4 | 补充精细 GPS / 路线上运动信息 |

```mermaid
flowchart LR
  V["逐点 GPS / 100 m Grid / Road"] --> E["相同 fusion + HTP encoder"]
  E --> Z["z: 64 维"]
  Z --> Q["四层残差量化"]
  Q --> S1["s1 → Grid head"]
  Q --> S2["s2 → Road head"]
  Q --> S4["s4 → GPS head / 主 decoder"]
  Q --> T["每 event: a,b,c,d"]
  T --> L["Qwen3-1.7B SFT"]
  L --> D["生成 SID + sampling → GPS"]
end
```

“L2 表示 Road”的严格含义是：**累计前缀 s2 有 Road 解码目标**；不保证 q2 单独携带全部 Road 信息，亦不保证早层不含 GPS。encoder 从开始就看到三个视图。强制模态信息隔离需要另一个 routed-encoder 方法，首版不混入。

Grid/Road 不存在普遍严格的空间粗细顺序，因此加入 Road→Grid→GPS 顺序对照。3 层 1/1/1 和 6 层 2/2/2 会改变 token 长度、码率和 LLM 难度，延期作为容量实验，不混入首轮。

## 3. 数据、样本和冻结划分

### 3.1 来源与物理单元

首轮 Porto；确认阶段再扩展 Chengdu。复用以下只读数据路径：

- 原始处理数据：`/workspace/guozuyu/HTP-main/traj_dataset`。
- Porto aligned：`/workspace/guozuyu/HTP-main/reproduction/runs/multiview_vq/porto/data`。
- Chengdu aligned（后续）：`/workspace/guozuyu/HTP-staged-attribution-artifacts/aligned/chengdu`。
- 模型：`/workspace/guozuyu/HTP-main/models/Qwen3-1.7B`。

一个样本是一条 20–200 个 observation 的轨迹，不切独立滑窗；沿用 processed trajectory 的筛选，再核实长度及字段。以 `traj_id + point_index` join GPS/Grid/pointwise Road/time/dxy/percent。route path 是另一个变长序列，不能按 observation 索引误对齐。

输入：GPS `[B,T,2]`，MBR 归一化 lon/lat（无量纲）；Grid/Road `[B,T]` 整数；时间 Unix seconds，仅用于诊断和时间对齐；lengths `[B]`，padding 不参与损失。Porto MBR 为 lon `[-8.69,-8.55]`、lat `[41.13,41.19]`。距离评估用米。Grid 沿用固定 local-equirectangular 100 m 定义、原点、映射和 SHA256，不临时重建不同网格。

训练前必须冻结 source hashes、完整 processed/aligned 路径、MBR、Grid 原点/尺寸、Road ID 词表与几何、road embedding 来源、归一化统计来源。z-score 若由完整官方 train 计算但含本轮 validation，应改为仅本轮 training population 计算，并对全部 cell 统一；保留原统计只用于复现参考。缺失地图匹配或非法 ID 统一剔除并统计，不按 cell 分别筛样本。不存在的字段写 N/A。

### 3.2 split 构造

1. 首先锁定官方 train/test ID，检查零交集；任何视图和重复轨迹必须跟随所属轨迹。
2. 官方 train 内按 canonical GPS/time 内容指纹做重复分组；按 `SHA256("20260907:"+group_id)` 稳定排序，取独立组构成 1,024 条 validation 和 8,192 条 pilot train。若分组导致不足，preflight 失败，不静默替换规则。更大规模 train 排除相同 validation 及其重复组。
3. 独立 test 保持封存；pilot 只对 validation 生成并选候选。冻结候选后按同样 hash 规则从官方 test 取 1,024 条评估；确认使用 full test。若 test 与 train 有内容重复，使用统一排除 manifest 并报告数量。
4. 所有 cell 共用同一份 split manifest。模型随机 seed 不改变数据成员。smoke 的 32/16 是上述 train/validation 的固定子集。
5. 记录车辆、道路、时间重叠审计。若车辆标识不可用则 N/A；首轮允许道路和时间重叠，仅声称同城 held-out-trip 效果，不能声称未见车辆/道路泛化。后续需要独立 vehicle/road/time-disjoint 协议才能作迁移结论。

不得用历史实验已经用于选择的 test subset 充当新的无偏探索集。数据数量、hash 目前尚未生成，属于实现前置工作，不伪造已冻结的样本 ID。

### 3.3 event 与扰动视图

HTP 三次 stride-2，`M=ceil(T/8)`。event j 的监督锚块为 `[8j,min(8j+8,T))`，不是 encoder 的完整感受野；末尾块显式 mask，不能重复末点凑目标。

GPS/Grid/Road 主训练全部使用同一 native point index；不加采样一致性损失。诊断 2×/3×：保留 `0::k` 并强制首尾，对 GPS 在原 timestamp 上线性插值恢复 T 点，Road/Grid 用保留点的最近时间邻居恢复，tie 选较早点；原目标、event 数和起止时刻不变。重复或非递增 timestamp 的处理记录在审计中，不靠插值隐式修复。该诊断是“缺失后恢复到同网格”，不声称是真实低频输入的原生 token 不变性。

## 4. 模型与损失实现契约

### 4.1 共享模块

复用 `ViewFusion`（GPS Linear 2→64；Road 128 维 embedding→64；Grid embedding 64，各自 LayerNorm，active normalization 和 null 语义沿用代码）、HTP Encoder（hidden 64、ch_multi `[1,2,2,4]`、每级 2 residual blocks、dropout .1、三次 stride-2）、pre-quant Linear 256→64、post-quant 64→256、原 route-conditioned decoder。所有 cell 从相同 seed 初始化，road embedding 采用同一预处理来源，不加载旧 tokenizer 权重。必须审计该 embedding 是否用到测试轨迹；仅道路静态几何可跨 split 使用。

新增三个 head 在所有 cell 中都实例化：每个位置使用 `[s_prefix(64), offset_onehot(8)]`，MLP `72→64→target_dim`，SiLU，无额外 dropout。Grid/Road 输出各自固定词表的分类 logits；GPS 输出该 offset 的归一化 lon/lat。目标是块内 **全部有效 observation**，不是多数 ID 或只取首点。Grid/Road 完整 softmax 按最多 256 个有效位置分块累积 CE，避免创建巨大的 `[B,M,8,V]` 张量；禁止用 sampled-softmax 偷换目标。所有 head 都只读量化前缀和 offset，不读真实 GPS/route 输入或 encoder skip features。

`L_grid`、`L_road`：先按每条轨迹有效点平均 CE，再在 batch 求均值，并分别除以 `log(max(V_train_observed,2))`；分母训练集固定。`L_gps`：相同轨迹平均的归一化坐标 MSE。主 HTP percent/dxy loss 保持 inherited reduction，不把辅助 head 误当作主 GPS decoder。

### 4.2 标准 RQ 及梯度

对每个有效 event，`r0=stopgrad(z)`；依次执行：

```text
c_l = argmin_k ||r_(l-1) - C_l[k]||²
q_l = C_l[c_l]
s_l = sum(q_1,...,q_l)
r_l = r_(l-1) - q_l
L_RQ = mean_l [MSE(stopgrad(z), s_l) + .25*MSE(z, stopgrad(s_l))]
st_l = z + stopgrad(s_l-z)       # train; eval 使用实际 s_l
```

沿用当前 quantizer 的累计前缀 VQ loss、梯度更新码本和初始化；不是 EMA 更新，虽然原 YAML 有 decay 字段。辅助 head 读取 `st_l`，因此 head loss 通过 straight-through 更新 encoder，codebook 由 `L_RQ` 更新；不得误用 `detach(s_l)` 导致 encoder 得不到语义梯度，也不得偷偷加入直接传向所有码本的不同 estimator。返回真实 `q_l/s_l` 供离线 probe，不以连续 z 冒充离散前缀。

```text
L_base = MSE(percent_hat, percent_target)
       + MSE(dxy_hat, dxy_target) + L_RQ
L_total = L_base + .10*L_grid + .10*L_road + .10*L_gps
```

没有 cosine、usage、variance/covariance 或其他辅助损失，权重全部 0；无正负样本配对、无多视图平均损失，N/A。首版三项权重 .10 为探索性预注册值，不声称已优化；若失败，记录后另建版本，不在同一结果中调参。

### 4.3 主 decoder 和 GPS 语义

完整 s4、生成/真实 sampling length type 和所有 cell 相同的 route context 进入 HTP decoder，输出逐点 standardized percent increment 与 dxy。使用冻结 z-score 反变换，沿用 cumulative percent 的 abs/clip 和 route interpolation 后加 dxy 复原 GPS。保持现有 `decode_stage3.py` 语义并做 round-trip 对照；全程记录输出点数与来源。

原生重构在相同 observation 时间索引评估。生成 sampling 长度不等时不做逐数组下标比较：将两条序列分别按均匀 `[0,1]` observation progress 线性插值到 100 个锚点，计算 matched-progress ADE/FDE（米），明确它不是真实时钟 ADE；同时报告 DTW 和长度误差。主终点沿用同一 population-aware penalized error；GPS head 的 auxiliary MSE 仅是中间指标。

## 5. 对照矩阵与归因

全部六个 cell 在首轮均进行 tokenizer + LLM + G 2×2 + GPS/Grid/Road 评估，不用重构结果替代下游效果。

| ID | 输入 | Grid/Road/GPS head 读取层 | 辅助权重 | 作用 |
|---|---|---|---|---|
| B0 | GPS | 4/4/4 | 全 0 | 相同预算 GPS-only reference |
| B1 | GPS+Grid+Road | 4/4/4 | 全 0 | B1−B0：输入信息增量 |
| B2 | GPS+Grid+Road | 4/4/4 | 各 .10 | B2−B1：辅助监督增量 |
| H1 | GPS+Grid+Road | 1/2/4 | 各 .10 | **H1−B2：主要分层效应** |
| H2 | GPS+Grid+Road | 2/1/4 | 各 .10 | 顺序：Road→Grid→GPS |
| H3 | GPS+Grid+Road | 2/3/4 | 各 .10 | 分配：Grid/Road/GPS=2/1/1 |

H2 同时改变视图对应的前缀容量（32 vs 2048 combinations），不能宣称纯顺序因果效应；H3 同理是层预算探索。所有 cell total 参数量相同，但 B0/B1 head 不参与梯度，effective trainable count 不同，分别报告。不为配平而给 baseline 增加额外训练信息。

三套独立 tokenizer 拼接是另一种表示组织 baseline，但会增加码率和序列长度，首轮不加入；若后续加入必须提供 matched-total-bits 和 matched-total-token 两种比较，不与 H1 混作单因素结论。

### 强制方法变更表（H1 对 B2）

| 方法组成 | B2 | H1 | 是否变化 |
|---|---|---|---|
| 样本物理单元 | 整条轨迹 20–200 点 | 相同 | 否 |
| 视图构造与覆盖 | native 同索引三视图 | 相同 | 否 |
| 输入特征与单位 | MBR GPS + 100m Grid + Road | 相同 | 否 |
| encoder/downsampling | fusion + HTP /8 | 相同 | 否 |
| token 支持 | ceil(T/8)，上下文感受野 | 相同 | 否 |
| quantizer/codebook | 4 层，32/64/128/256，64维 | 相同 | 否 |
| decoder 与输出时间锚点 | HTP route-conditioned | 相同 | 否 |
| 初始化与冻结 | 同 seed，从头；LLM 阶段冻结 tokenizer | 相同 | 否 |
| reconstruction/VQ loss | 原 L_base | 相同 | 否 |
| 新增 loss 与权重 | 三 head 各 .10，读 s4 | 同权重，读 s1/s2/s4 | **是，仅监督位置** |
| 训练数据与超参数 | 冻结 split/budget | 相同 | 否 |
| 评估对齐与指标 | 固定同 population | 相同 | 否 |

## 6. LLM 输入、训练和生成

保持 Qwen3-1.7B、现有 prompt 和 wire format：每 event `<a_i><b_j><c_k><d_l>`，分隔符和 sampling `<t_n>` 按原 `token_protocol.py`。不新增模态标签 token，不把完整 tuple 当作一个巨型词表项。层语义保存在 config/metadata 中。预计最长 200 点→25 events→100 RQ token，实际总长度必须测量，含 route prompt、sampling 和特殊 token。

SFT 中这些 token 是 assistant target，也是 teacher forcing 下的自回归输入；部署时只有原条件 prompt，没有测试轨迹的真实 SID/GPS/Grid/pointwise Road。生成的 SID 经相应 cell 的冻结 tokenizer decoder 输出 GPS。每个 cell **独立** SFT，绝不复用另一个 tokenizer 对应的 LLM adapter。

- Tokenizer checkpoint 冻结；LLM base 冻结；训练 LoRA 与新增 token embedding/output rows，使用继承的 compact-adapter 存储路径及 tied-weight 处理，验证加载前后 logits 一致。
- LLM：AdamW lr=5e-4、weight_decay=1e-4、clip=1、linear scheduler、warmup_ratio=.05、BF16、gradient checkpointing；LoRA r=16、alpha=32、dropout=.05，targets=q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj（已核实 inherited 实现）。继承 runner 存在硬编码 full-test eval，必须改为显式 validation manifest，不能在 pilot 偷看 test。
- assistant-only labels，prompt/padding `-100`，无样本截断；max sequence length 2048，超长按统一 cohort 排除并记录，再重建全部 cell 相同的 manifest。不得某 cell 单独丢样本。
- G00：真实 SID/真实 sampling；G10：生成 SID/真实 sampling；G01：真实 SID/生成 sampling；G11：生成 SID/生成 sampling。真实 route 四组相同，使用同 traj_id，生成一次记录同时派生各组。
- 主生成已核实为 greedy：do_sample=false、max_new_tokens=256、batch_size=1、原 eos/pad ID，无额外 grammar约束；temperature/top-p=N/A；不因为某 cell 无效率高临时添加约束。解析错误、缺失、越界、截断计入原始 population；不静默重采样直到成功。

Tokenizer 优化：AdamW lr=1e-4、weight_decay=1e-4、betas=(.9,.999)、eps=1e-8、clip=1；linear warmup 10 epochs（ratio .01）后 cosine 到 .10，smoke 只走对应首epoch；BF16 autocast，RQ距离与CE强制FP32；全部 tokenizer模块可训练，无early stopping，最终epoch checkpoint用于比较。Python/NumPy/Torch/CUDA seed一致；DataLoader独立 generator，num_workers=0、shuffle=true；cudnn benchmark=false、deterministic=true，记录不支持确定性的kernel，不声称bitwise复现。两阶段均无超参搜索；新增头采用独立 RNG初始化，避免额外初始化消耗改变共享模块的seed。

## 7. 规模、算力和分阶段门槛

| 阶段 | 城市 / seed / cells | Tokenizer | LLM | 生成评估 |
|---|---|---|---|---|
| 实现与单测 | CPU，全部结构 | 合成数据 | mock + adapter 契约 | 不产生科研结果 |
| smoke | Porto / 42 / 六 cells | train32、val16；1 epoch；batch8 | 每 cell 2 optimizer steps，microbatch1、accum1 | 每 cell 8 val，四 G |
| pilot | Porto / 42 / 六 cells | train8192、val1024；20 epochs；batch128 | 相同8192条，2 epochs；microbatch2、accum8，有效16 | 每 cell 1024 val，四 G |
| repeat | Porto / 314159、20260907 / B2、H1、一个按 val 选定的 H2/H3 | 与 pilot 相同 | 与 pilot 相同 | 同 val；之后冻结候选 |
| held-out check | Porto / 已训练三 seeds / B2、H1、选定候选 | 不重训 | 不重训 | 冻结 test1024，四 G |
| confirmation | Porto+Chengdu / 42、314159、20260907 / B0、B2、最终 H | full train 减 val；100 epochs；batch128 | 相同 full train，2 epochs | full test，四 G |

pilot 每 cell Stage1 `8192/128×20=1280` optimizer steps，Stage2 `8192/16×2=1024` steps；六 cell 总计 7,680 和 6,144 steps（尚未执行）。重复阶段最多再六组 full-pipeline run，confirmation 18组；这三部分分开记账，**不默认立即启动 confirmation**。

首批资源硬上限：smoke 合计 2 GPU-hours；pilot 合计 36 GPU-hours、artifact 20 GiB，单 GPU 顺序运行；内存目标峰值≤22 GiB，preflight free≥23 GiB，避免和别的实验抢卡。达到预算时在 checkpoint 边界停止，标记 budget_exhausted，不把缩短训练的结果当作原计划完成。GPU-hours 是预算，不是实测工期。

完成 smoke 后使用每 cell 训练/teacher-forced eval/生成实测耗时估计整轮用量；若六 cell 预测超 36h，则先完整执行 B2/H1 配对，其余保持 pending，不能悄悄缩小样本后保留原实验名。设备空闲快照不代表已预留。2026-09-07 检查时四张 GPU 空闲约 24,124 MiB；workspace 剩余约156 GiB，属瞬时信息。大 checkpoint、模型和数据不提交 Git，禁止复制 Qwen 全模型到每个 attempt。

## 8. 评估、选择和停止规则

### 8.1 主要终点与基线

主比较 H1−B2；B0/B1 负责解释信息/辅助监督增量。主要终点是 G11 `population_mean_penalized_error`（越低越好），已核实 inherited 公式：有效轨迹 e_i 为四个统计量 travel_distance_km、mean_segment_distance_km、radius_km、mean_point_to_road_km 上 abs(log1p(generated)-log1p(real)) 的平均；无效轨迹 e_i=1.0；主指标 mean_i(e_i)，所有原定轨迹入分母。该指标不是坐标ADE，可能对不同形状不敏感，必须联合ADE/DTW与三域指标解释。另报 invalid_penalty=10 的敏感性分析，因为有效极端误差可能超过1.0；不按敏感性结果事后替换主终点。在任何模型训练前，preflight 把公式、距离尺度、缺失惩罚和源文件 SHA 写入 metric manifest；不得对不同 cell 使用不同失败惩罚。若无法核实该指标，以 spec_v2 修改后再运行，不能只凭名称比较。

次要：GPS travel/segment-distance/radius JSD；Road density cosine、top100 pattern F1、GPS-to-road-distance JSD；100m Grid density cosine、top100 pattern F1；matched-progress ADE/FDE、DTW、长度误差；G00→G11 gap 与 G 的 2×2 Shapley。重新采样 bootstrap 时重新计算聚合 JSD/cosine，不对不存在的逐轨迹 JSD 求均值。

LLM：validation SID token NLL（排除 sampling/标点）、各层 NLL/accuracy、完整 tuple accuracy、sampling accuracy、首次错误层及后续错误传播、free-running invalid rate；不同 cell 的 token NLL 不是公共 GPS 空间的统一概率密度，只用作可学习性诊断，不能独立证明生成更优。

### 8.2 分层语义与防坍缩

- 每层 active fraction、perplexity/K、max share、tuple collision、条件熵；以有效 event 为分母，逐层和完整 tuple 均报告。
- 所有 s1/s2/s3/s4 冻结后，在同 train/val 上拟合同结构的轻量 probe（同 head MLP，AdamW lr1e-3，10 epochs，batch256 events，seed42），预测三视图目标，防止只用训练时指定 head 的优势循环论证。
- 检验 s2 比 s1 的 Road CE、s4 比 s2 的 GPS MSE 是否降低；报告 q_l 单独 probe 与累计前缀 probe，不以二者替换。
- 截断：缺失后层向量置零，对主 decoder 测 oracle reconstruction；标记 OOD 诊断，不当作训练过的可变码率模型。
- 层交换：在 validation 中按 route长度与点数分桶，用 hash 排序循环移位得到 donor；分别替换一个层/层组，测三域变化。不存在合格 donor 则 N/A。这是关联诊断，不声称因果隔离。
- 扰动视图：在同 traj_id/event 上逐层 exact agreement/AMI 与 tuple exact agreement，报告真实总分母；不跨训练 seed 直接比较 code ID。

### 8.3 预设门槛（探索性阈值）

smoke：梯度有限、round-trip 通过、无 split 漏泄、六 cell wire format 均有效；不要求 tiny-data codebook 使用率。

pilot 候选先满足：每层 active/K≥.10、perplexity/K≥.05、max_share≤.50；G00 GPS error 相对 B2 不恶化>10%；G11 valid rate≥.99 且相对 B2 不降低>0.01。失败完整记录，不作 winner。

进入 repeat 的信号：H1 相对 B2 的主要终点改善≥5%，SID NLL不恶化>2%，并通过上述门槛；如果主终点改善<5%但95%CI支持改善，仅记弱信号，先分析，不能宣称成功。H2/H3 按 validation 主终点最低选择一个，tie≤1%选H2，不依据 test 改选。H1即使未胜出仍保留为预注册主假设结果。

最终“有效”要求：冻结 test 上三 seeds 的主差异方向一致、均值改善≥5%、逐轨迹配对 bootstrap 2000次（seed271828）的95%CI不跨0，同时报告 seed均值/标准差/范围；bootstrap不能替代独立seed重复。三个域的主要指标不超过预定容忍（JSD +.01、cosine/F1 −.01），否则写明质量权衡。多项次要显著性使用 Holm 修正，不从众多指标中挑一个显著结果。跨城结论要两城分别支持；pilot不支持论文级“优于HTP”。

## 9. 实现任务与验收

新增 `stage_1/src/hierarchical_rqvae/`、`stage_1/tests/hierarchical_rqvae/`、`stage_1/configs/hierarchical_rqvae/`。复用现有读取与评估组件，扩展不直接覆盖 upstream `model/`。

1. `config.py / data.py`：解析独立 config、构造训练/验证/test manifest、训练统计、8-offset target mask；只读数据；非法 ID、越界、重复 ID、跨 split 重复失败关闭。
2. `quantizer.py`：在兼容量化器上暴露真实 layer_vectors/prefixes 和 prefix straight-through；复用旧参数命名，B1 与旧 E6/Q0 同输入/同 state 的 codes、recon、quant_loss 一致。
3. `model.py / losses.py`：三 head、六 cell routing、分块 exact CE；测试 head 没有读未来前缀、route或z旁路，检查梯度流与padding不变性。
4. `train.py / export.py`：相同初始化 RNG/data order；导出 codes `[N_events,4]`、offsets、traj_id、sampling、checkpoint hash；只接受本 cell 的 codec/version hash。
5. `llm.py / run.py`：按 cell 注入现有 SFT/compact adapter/prompt 接口，显式 manifest 替代旧 full-test默认；校验每条 prompt、label mask、长度、step budget；禁止调用旧 experiment watcher 误启动60-cell任务。
6. `evaluate.py / report.py`：四 G 同 population、失败入分母、冻结 probes、bootstrap、资源统计；输出 JSON + Markdown + Notion。metrics schema 不容忍缺失被写成0。

必须测试：残差求和与 argmin、前缀梯度、全量CE与chunk CE等价、末块mask、变长/重排/视图对齐、旧模型兼容、token序列化round-trip、generated sampling不同长度、parser失败保留population、tokenizer/LLM hash错配拒绝、compact adapter logits一致、dirty revision拒绝、预算耗尽保存状态；之后跑继承的整套 pytest。单测与真实 smoke 分开记录。

## 10. 运行契约和记录

当前 `plan.json` 是设计配置，`execution_enabled=false`；新增 runner 尚不存在。下面是 **待实现 CLI 契约，不是已经可以运行的命令**：

```bash
cd /workspace/guozuyu/HTP-hierarchical-rqvae
uv sync --frozen
uv run python --version
uv run pytest
uv run python -m hierarchical_rqvae.run --config stage_1/configs/hierarchical_rqvae/plan.json --phase preflight --validate-only
uv run python -m hierarchical_rqvae.run --config stage_1/configs/hierarchical_rqvae/plan.json --phase smoke --cells B0 B1 B2 H1 H2 H3 --city porto --seed 42
uv run python -m hierarchical_rqvae.run --config stage_1/configs/hierarchical_rqvae/plan.json --phase pilot --cells B0 B1 B2 H1 H2 H3 --city porto --seed 42
```

实现完成后，在专门提交中更新 execution_enabled 与执行 revision；启动前拒绝 dirty tree 和 pending 必需配置。每次 run 使用独立 `attempt_NNN`，不覆盖失败。

输出根：`/workspace/guozuyu/HTP-hierarchical-rqvae-artifacts/<experiment_id>/<phase>/<city>/<seed>/<cell>/attempt_NNN/`；配置根与旧实验分离。保存 run_manifest、resolved_config、data/metric/model hashes、训练/生成进度、tokenizer、compact LoRA、新 token rows、tokens、G00/G10/G01/G11、metrics、bootstrap、report、资源耗时/峰值和失败原因。训练/评估前先提交代码；run记录运行时完整40位Git SHA，不用文档提交SHA冒充未来实现SHA。

Notion 新建独立 planned 页，链接本 spec 和后续 run 记录；无 metrics/checkpoint/训练命令结果时明确 N/A。本轮没有运行实验，所以不填写虚构准确率或改善值。当前下一动作：按本 spec 实现模块和必需 preflight，再验证 smoke；不是启动完整训练队列。
