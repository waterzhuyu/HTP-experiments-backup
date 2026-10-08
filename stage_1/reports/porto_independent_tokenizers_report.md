# Porto GPS / Grid / Road 独立 Tokenizer 跨视图 Pattern 实验

> **Gate 决策：`do_not_advance_to_shared_private_design`。** Gate 0=True，H1=True，H2=False，H3=False。POI 未纳入本实验。

## 方法冻结

- 数据：Porto 官方 train/test 索引；537,993/59,778 条轨迹。GPS、100 m Grid、逐点 map-matched `opath` Road 共享轨迹边界与点索引。
- 样本单元：三层 stride-2 后的一个 latent event，输出长度 `ceil(T/8)`；末尾不足 8 点保留。Transformer event 具有全轨迹注意力上下文，GRU event 具有双向递归上下文，因此“8 点”仅是下采样锚定而非严格局部感受野。
- 独立性：每个 view × encoder × seed 拥有独立 input adapter、encoder、K=256 spherical VQ codebook 和 decoder；无共享参数、无共享码本、无跨视图 loss。
- Encoder：3×(Transformer 或双向 GRU + stride-2 Conv1d)，hidden=64，latent=64。Road 输入使用冻结的 128 维 Node2Vec，再投影到 64 维。
- Decoder：latent 重复 8 次；GPS 重构归一化经纬度，Grid 重构网格中心，Road 重构 L2-normalized Node2Vec。未执行 GPS 轨迹拼接式解码。
- 损失：`L = L_recon + L_codebook + 0.25 L_commit + 0.05 L_soft_usage`。AdamW 2e-4、weight decay 1e-4、batch 256、3 epochs、BF16、clip=1.0。
- Seeds：42、314159、20260819。Encoder families：Transformer、GRU。
- 条件打乱：在 2 km origin 区域、8方向 OD、轨迹长度 bin、latent 相对位置 quartile 内打乱 event 配对；bootstrap 以完整轨迹为重采样单位。

## 方法变更表

| 方法组成 | 共享 VQ 先前基线 | 本实验 | 是否变化 |
|---|---|---|---|
| 样本物理单元 | 同轨迹的 ceil(T/8) latent 位置 | 相同 | 否 |
| 视图构造与覆盖范围 | GPS/Grid/Road 逐点对齐 | 相同；POI 跳过 | POI 不适用 |
| 输入特征与单位 | normalized GPS / grid ID / road Node2Vec | 相同 | 否 |
| encoder/downsampling | 三个 Transformer encoder | 独立 Transformer 与 GRU 控制 | 是 |
| token 时间/空间支持 | ceil(T/8) | 相同 | 否 |
| quantizer/codebook | 三视图共享 K=256 | 每个 tokenizer 私有 K=256 | 是（主变量） |
| decoder 与输出时间锚点 | view-specific，repeat ×8 | 相同 | 否 |
| 初始化与冻结模块 | road embedding 冻结 | 相同；每 run 从头初始化 | 否 |
| reconstruction/VQ loss | 三视图损失求和 | 单视图独立优化 | 是（独立训练必需） |
| 新增 loss 与权重 | alignment 可选 | 无 alignment | 是 |
| 训练数据与超参数 | full train, 3 epochs, seed 42 | full train, 3 seeds | 是 |
| 评估对齐与指标 | exact/NMI/shuffle | 匹配率、双向条件熵、VI、AMI、global/conditional shuffle、probe、bootstrap | 是 |

## 码本有效性

| Encoder | Seed | View | Active | Perplexity | Max share | Recon | Pass |
|---|---:|---|---:|---:|---:|---:|---|
| transformer | 42 | gps | 190 | 137.527 | 0.0187 | 0.000810 | True |
| transformer | 42 | grid | 228 | 146.480 | 0.0223 | 0.000904 | True |
| transformer | 42 | road | 241 | 213.203 | 0.0108 | 0.002976 | True |
| transformer | 314159 | gps | 201 | 140.254 | 0.0222 | 0.000818 | True |
| transformer | 314159 | grid | 221 | 152.920 | 0.0193 | 0.000909 | True |
| transformer | 314159 | road | 249 | 229.255 | 0.0091 | 0.002940 | True |
| transformer | 20260819 | gps | 186 | 127.079 | 0.0197 | 0.000839 | True |
| transformer | 20260819 | grid | 207 | 131.523 | 0.0249 | 0.000968 | True |
| transformer | 20260819 | road | 202 | 179.037 | 0.0133 | 0.003058 | True |
| gru | 42 | gps | 207 | 144.690 | 0.0203 | 0.000833 | True |
| gru | 42 | grid | 222 | 164.156 | 0.0181 | 0.000946 | True |
| gru | 42 | road | 248 | 228.305 | 0.0084 | 0.002976 | True |
| gru | 314159 | gps | 210 | 148.705 | 0.0188 | 0.000822 | True |
| gru | 314159 | grid | 226 | 164.377 | 0.0174 | 0.000931 | True |
| gru | 314159 | road | 232 | 208.257 | 0.0108 | 0.003012 | True |
| gru | 20260819 | gps | 198 | 131.466 | 0.0216 | 0.000919 | True |
| gru | 20260819 | grid | 233 | 165.887 | 0.0181 | 0.000947 | True |
| gru | 20260819 | road | 231 | 198.141 | 0.0100 | 0.003038 | True |

## 跨视图结果（同 seed）

| Encoder | Seed | Pair | AMI | Cond-shuffle AMI | Lift | Match | H(R given L) | H(L given R) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| transformer | 42 | gps-grid | 0.7889 | 0.1680 | 0.6209 | 0.5827 | 0.1925 | 0.1811 |
| transformer | 42 | gps-road | 0.7845 | 0.1705 | 0.6139 | 0.4569 | 0.2371 | 0.1580 |
| transformer | 42 | grid-road | 0.7458 | 0.1687 | 0.5771 | 0.4275 | 0.2682 | 0.2005 |
| transformer | 314159 | gps-grid | 0.7810 | 0.1671 | 0.6139 | 0.5479 | 0.2027 | 0.1871 |
| transformer | 314159 | gps-road | 0.7829 | 0.1704 | 0.6126 | 0.4386 | 0.2448 | 0.1562 |
| transformer | 314159 | grid-road | 0.7405 | 0.1679 | 0.5726 | 0.4097 | 0.2780 | 0.2050 |
| transformer | 20260819 | gps-grid | 0.7781 | 0.1695 | 0.6086 | 0.5518 | 0.1960 | 0.1898 |
| transformer | 20260819 | gps-road | 0.7778 | 0.1721 | 0.6057 | 0.4467 | 0.2300 | 0.1682 |
| transformer | 20260819 | grid-road | 0.7373 | 0.1706 | 0.5667 | 0.4282 | 0.2640 | 0.2084 |
| gru | 42 | gps-grid | 0.7546 | 0.1649 | 0.5897 | 0.5020 | 0.2319 | 0.2091 |
| gru | 42 | gps-road | 0.7785 | 0.1701 | 0.6084 | 0.4572 | 0.2463 | 0.1640 |
| gru | 42 | grid-road | 0.7193 | 0.1665 | 0.5528 | 0.3953 | 0.2926 | 0.2331 |
| gru | 314159 | gps-grid | 0.7598 | 0.1650 | 0.5948 | 0.5029 | 0.2255 | 0.2074 |
| gru | 314159 | gps-road | 0.7816 | 0.1707 | 0.6109 | 0.4712 | 0.2315 | 0.1708 |
| gru | 314159 | grid-road | 0.7275 | 0.1673 | 0.5602 | 0.4116 | 0.2746 | 0.2319 |
| gru | 20260819 | gps-grid | 0.7379 | 0.1684 | 0.5695 | 0.4676 | 0.2546 | 0.2127 |
| gru | 20260819 | gps-road | 0.7547 | 0.1748 | 0.5799 | 0.4122 | 0.2594 | 0.1854 |
| gru | 20260819 | grid-road | 0.7224 | 0.1670 | 0.5554 | 0.4124 | 0.2731 | 0.2410 |

## Gate 判定

- Gate 0（同视图跨 seed AMI 均值 ≥0.50 且无坍塌）：**True**。
- H1（三对在两种 encoder 下的 AMI 相对条件打乱提升均值 ≥0.05）：**True**。
- H2（至少一对不是近似一一置换：match<0.80 且方向条件熵≥0.30）：**False**。
- H3（AMI seed std≤0.05、方向一致、跨 encoder 熵排序 Spearman≥0.70、块结构相似）：**False**。
- Transformer/GRU 双向条件熵排序 Spearman：0.9429。
- 同 family 跨 seed 共现块 cosine：0.7396；跨 encoder：0.7318。

## 打乱对照与 trajectory bootstrap

- 全局随机打乱的 AMI 为 -0.00010–0.00014，接近 0；真实配对 AMI 为 0.7193–0.7889。
- 条件打乱保留 2 km origin、OD 方向、轨迹长度和 latent 相对位置，AMI 为 0.1649–0.1748。真实配对相对条件打乱的 AMI lift 为 0.5528–0.6209，六个 `encoder × pair` 组均远高于预注册的 0.05 门槛。
- 条件打乱共有 1,936 个 strata、42 个 singleton events；打乱后仍来自同一轨迹的 event 比例为 0.001906。全局打乱的同轨迹残余比例为 0.0000115。
- 200 次 bootstrap 以完整测试轨迹为单位。全部 18 个跨视图比较均保存 NMI、最优匹配率和双向条件熵的 95% 区间。例如 Transformer seed 42 的 GPS–Grid optimal match 为 0.5827，95% CI [0.5813, 0.5843]；GRU seed 20260819 的 Grid–Road 为 0.4127，95% CI [0.4109, 0.4141]。区间很窄，说明主要结论不是由少量轨迹驱动。

## 连续 latent probe

冻结 source encoder 的连续 latent，使用固定的 20,000 条训练轨迹（115,993 latent events）训练 5 epochs 线性分类器，预测另一 view 独立 tokenizer 的 code；每个方向最多评估 120,000 个测试 events。下表是三 seeds 的 accuracy 均值，括号内是 seed 范围；所有结果同时报告 majority baseline，完整 36 个 probe 保存在 `analysis.json`。

| Encoder | Direction | Accuracy mean (range) | Mean lift over majority |
|---|---|---:|---:|
| Transformer | GPS→Grid | 0.3512 (0.3300–0.3620) | 0.3287 |
| Transformer | Grid→GPS | 0.3810 (0.3548–0.4003) | 0.3612 |
| Transformer | GPS→Road | 0.3031 (0.2886–0.3222) | 0.2921 |
| Transformer | Road→GPS | 0.7220 (0.7126–0.7271) | 0.7022 |
| Transformer | Grid→Road | 0.3025 (0.2769–0.3322) | 0.2915 |
| Transformer | Road→Grid | 0.6485 (0.6305–0.6580) | 0.6260 |
| GRU | GPS→Grid | 0.3421 (0.3238–0.3532) | 0.3244 |
| GRU | Grid→GPS | 0.3761 (0.3437–0.4015) | 0.3561 |
| GRU | GPS→Road | 0.3226 (0.3053–0.3487) | 0.3127 |
| GRU | Road→GPS | 0.6928 (0.6873–0.6957) | 0.6728 |
| GRU | Grid→Road | 0.3019 (0.2845–0.3230) | 0.2920 |
| GRU | Road→Grid | 0.5868 (0.5830–0.5931) | 0.5691 |

Road latent 对 GPS/Grid code 的预测显著强于反方向，这与双向条件熵的不对称方向一致：Road 表示保留了较多可用于推断几何/网格划分的信息，而 GPS/Grid latent 不能同样准确地恢复更细的 Road code。该 probe 是诊断证据，不单独决定 H1–H3。

## 共现块稳定性与语义审计

- 同 family 跨 seed 的 Hungarian 对齐后共现块 cosine 为 0.7033–0.7782，均值 0.7396，低于预注册 0.80。
- Transformer–GRU 的跨 encoder 共现块 cosine 为 0.7066–0.7794，均值 0.7318，高于预注册 0.60。
- 两种 encoder 的六个方向性条件熵排序 Spearman=0.9429；各 `family × pair` 的 AMI seed std 为 0.0028–0.0120，条件熵方向在 3/3 seeds 一致。因此 H3 失败集中在**同 family 跨 seed 的细粒度共现块未达到 0.80**，而不是关系方向或总体关联强度不稳定。
- 语义审计为 Transformer/GRU × GPS/Grid/Road 各自 seed 42 的 10 个最高频 code 保存 path length、displacement、mean absolute turn、Grid/Road transition rate、trajectory length 和 5 条代表轨迹索引。示例：Transformer Road 高频 code 170（3,738 events）对应 path length 均值 1,048.6 m、displacement 994.8 m、mean turn 0.134 rad，表现为长且较直的事件；GRU GPS 高频 code 84（7,063 events）对应 path length 386.5 m、displacement 267.6 m、mean turn 0.732 rad，表现为更短且转向较多的事件。不同 tokenizer 的 code ID 不可直接按编号比较。

## 观察、解释与 Gate 决策

1. **观察事实**：18/18 run 无坍塌；同视图跨 seed AMI 均值为 0.7284–0.8767；真实跨视图 AMI 为 0.7193–0.7889，而全局打乱约为 0、条件打乱约为 0.17。H1 明确通过。
2. **观察事实**：最优匹配率仅为 0.3953–0.5827，明显不是 code ID 的简单置换；但最大 normalized directional entropy 为 0.2926，距离预注册 H2 的 0.30 尚差 0.0074。因此结构上存在非一一迹象，但**不能按冻结确认规则宣称 H2 通过**。
3. **观察事实**：seed 间 AMI 与熵方向稳定、跨 encoder 排序稳定，跨 encoder 块 cosine 也通过；同 family 跨 seed 块 cosine 均值 0.7396 未过 0.80。H3 的正式判定为失败。
4. **解释**：GPS/Grid/Road 的独立 tokenizer 确实捕获强且可预测的同源跨表达结构，这一关系不能仅用位置、OD、长度或 event 位置捷径解释；但当前证据更接近“高关联、近似细粒度重编码”，不足以确认稳定的共享核心—私有残差层次。
5. **决策**：维持 `do_not_advance_to_shared_private_design`。下一轮若继续，应预先设计更直接的多对一/多对多块结构检验或改变 event scale；不能事后把 0.30 门槛降低来改写本次结论。

## 运行完整性审计

- 18/18 `resolved_config.json`、18/18 checkpoints、18/18 `metrics.json`、18/18 test representations、18/18 train-probe representations 均存在。
- 全部 run 的 Git revision 均为 `03d6d7fb3567e6c838910b37d8cf3e05003a2fb5`，`train_limit=test_limit=0`，测试均为 59,778 trajectories / 347,415 latent events。
- 18 runs 总运行时 5,483.35 s；单 run 165.14–453.33 s。正式日志无 traceback、error、NaN 或 Inf；队列结束标记为 `INDEPENDENT TOKENIZER FORMAL COMPLETE`。
- 机器可读分析覆盖 18 validity rows、18 same-view seed comparisons、18 cross-view comparisons、36 directed probes、21 block-stability comparisons 和 6 semantic-audit groups；K×K 共现矩阵保存在 `cooccurrence_matrices.npz`。

## 解释边界

GPS、Grid、Road 是同源派生视图，本实验验证的是同一移动事件的跨表达稳定性，不是统计独立模态的一致性。共现与 probe 只证明可预测关系；语义审计使用几何、转向、grid/road transition 和轨迹长度，未使用 POI。Road reconstruction 是冻结 Node2Vec MSE，不等价于 road ID 分类准确率。

## 可复现信息

- Git revision：`03d6d7fb3567e6c838910b37d8cf3e05003a2fb5`；branch：`exp/porto-independent-gps-road-grid`。
- 完整 resolved config、checkpoint、test/probe representations：`/workspace/guozuyu/HTP-multiview-independent/stage_1/runs/independent_tokenizers/porto`。
- 机器可读汇总、共现矩阵与报告：`/workspace/guozuyu/HTP-multiview-independent/stage_1/runs/independent_tokenizers/porto/analysis`。
- Bootstrap replicates：200；probe 使用冻结 latent 的线性分类器，5 epochs，最多 120k train/test events。
