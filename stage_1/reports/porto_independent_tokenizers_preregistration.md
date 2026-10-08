# Porto GPS / Grid / Road 独立 Tokenizer 预注册

实验 ID：`porto_independent_tokenizers_v1_20260819`。状态：planned。实验类型：冻结确认实验。POI 明确不在本轮范围。

## 问题与假设

- H1：同一 latent event 的 GPS/Grid/Road 独立离散 code 关联强于全局及条件打乱。
- H2：至少一组跨视图关系不是 code ID 的一一置换，而具有稳定的一对多、多对一或多对多结构。
- H3：H1/H2 的方向性条件熵与共现块结构对随机种子和 Transformer/GRU encoder family 稳健。

Gate 0 先检查同一 view 的跨 seed 稳定性。若任一 run 坍塌或同视图 AMI 均值低于 0.50，则不解释跨视图多对多关系。

## 数据与事件对齐

使用 Porto 官方 `train_index.npy`/`test_index.npy`，分别包含 537,993/59,778 条轨迹。三个视图来自同一批 GPS 轨迹：GPS 为 MBR 归一化经纬度，Grid 为约 100 m 网格 ID，Road 为发布的逐 GPS 点 `opath` road ID。视图共享轨迹边界、点数与点索引；先 split 再构造视图。

一个评估 event 是三层 stride-2 下采样后的一个 latent 位置，序列长度 `ceil(T/8)`。位置 `j` 的 decoder 锚定原始点索引 `[8j, min(8j+8,T))`，末尾不足 8 点保留。Transformer 在每层下采样前使用全序列 self-attention，GRU 使用双向递归，所以 event 的下采样锚点约为 8 点，但编码器感受野包含更大的轨迹上下文；报告不得把它称为严格局部 8 点块。

## 模型与损失

每个 `view × encoder_family × seed` 从头训练一套完全独立的 input adapter、encoder、K=256 spherical VQ、decoder。不存在共享参数、共享 codebook 或跨视图 loss。

- GPS：2→64 Linear；重构 normalized `(lon,lat)`。
- Grid：7,906×64 embedding；重构 normalized grid center。
- Road：冻结 10,537×128 Node2Vec，128→64 Linear；重构 L2-normalized Node2Vec。
- Encoder：3× contextual block + Conv1d(kernel=3,stride=2,padding=1)，hidden=64，latent=64。主族为 Transformer（4 heads, FFN=256）；稳健性族为双向 GRU（每方向 32）。
- Decoder：`64→64→target_dim`，每个 latent 重复 8 次后按原长度裁剪。未执行轨迹级 GPS 反归一化与拼接解码。
- Quantizer：spherical Euclidean distance，straight-through，K=256，commitment beta=0.25。

损失为：

```text
L_total = L_reconstruction + L_codebook + 0.25 L_commitment
        + 0.05 L_soft_usage
```

`L_soft_usage = 1 - H(mean soft assignment)/log(256)`，temperature=0.2。不同 view 的 reconstruction 数值不横向比较。

训练使用 full train split、AdamW、learning rate 2e-4、weight decay 1e-4、batch 256、3 epochs、gradient clip 1.0、BF16、单 GPU；seeds 为 42/314159/20260819。

## 指标、对照与阈值

每个 run 必须满足 active codes≥32、perplexity≥16、max token share≤0.20、reconstruction finite。

Gate 0：同 view 同 family 的三个 seed pair 报告 K×K 共现、Hungarian 最优匹配率、双向条件熵、VI、AMI。各 `family × view` 的跨 seed AMI 均值须≥0.50。

Gate 1：每个 family/seed 分析 GPS–Grid、GPS–Road、Grid–Road，报告相同指标。全局打乱随机置换 latent events。条件打乱在 2 km origin region、8方向 OD（另加 stationary）、轨迹长度 bin 与 event 相对位置 quartile 内进行。打乱尽量禁止来自同一轨迹的配对，并记录残余比例。

- H1 通过：Gate 0 通过，且三组 pair 在两种 encoder family 下的 `AMI(actual)-AMI(conditional shuffle)` 跨 seed 均值均≥0.05。
- H2 通过：H1 通过，且至少一组 pair 的跨 seed 平均 optimal match<0.80，同时至少一个方向 normalized conditional entropy≥0.30。
- H3 通过：H2 通过；每个 `family × pair` 的 AMI seed std≤0.05、条件熵方向符号至少 2/3 seeds 一致；两 encoder 的六个方向性熵 Spearman≥0.70；Hungarian 对齐后的共现块 cosine 同 family 跨 seed 均值≥0.80、跨 encoder 均值≥0.60。

连续 latent probe 使用固定的 20,000 条 train 轨迹抽样（seed 20260819），从 source latent 线性预测 target code，5 epochs，最多 120k train/test events；报告 accuracy、majority baseline 与 lift。它是诊断，不单独决定 H1–H3。

置信区间以完整 test 轨迹为 bootstrap 单位，200 replicates，覆盖 NMI、最优匹配率和双向条件熵；AMI 的超几何期望只在完整样本上计算，避免对每个 K=256 bootstrap replicate 重复高代价 expected-MI。语义审计只使用 path length、displacement、turn、Grid/Road transition rate 和轨迹长度；不使用 POI。

## 停止规则

任一 run 坍塌、Gate 0 失败或 H1 不超过条件打乱时，停止“共享离散 pattern”解释。只有 H1–H3 全部通过，才建议进入 shared-core/private-residual 设计；失败结果原样保留，不通过事后改阈值改写。
