# GPS / Road / Grid 分阶段归因框架设计

日期：2026-08-26
状态：已冻结，进入实现
分支：`exp/staged-attribution-gps-road-grid`
worktree：`/workspace/guozuyu/HTP-staged-attribution`

## 1. 目标、范围与证据边界

目标是在 HTP 的真实生成链路中，把结果差异拆成三个阶段的受控效应：

1. **E（encoding）**：tokenizer encoder 看见哪些轨迹视图；
2. **Q（quantization）**：离散化目标和约束如何改变 Semantic ID；
3. **G（generation）**：LLM 生成 SID 与 sampling type 分别引入多少误差。

正式结论只覆盖 GPS、Road、100 m Grid。POI 不进入输入、辅助标签、分层抽样或结论。GPS、Grid、Road 都由同一条轨迹派生，因此“跨视图一致性”只能解释为同源表达的可预测关系，不能声称是独立模态间的因果一致性。

本工作从 `exp/porto-independent-gps-road-grid` 的 `96f65de` 创建，继承其环境、数据适配器和已完成的独立 tokenizer 证据。上游 HTP 基线 revision 为 `1294275`。已有独立 tokenizer 实验是设计先验和外部诊断，不代替本轮 Stage 1→2→3/4 端到端实验。

## 2. 为什么采用 HTP-compatible 设计

所有正式单元必须保持下列接口不变：

- 三次 stride-2 下采样，latent 长度为 `ceil(T/8)`；
- 四层 residual code，词表大小固定为 `[32, 64, 128, 256]`；
- 每个 latent event 序列化为 `<a_i><b_j><c_k><d_l>`；
- sampling type 仍为三位长度类型 token `<t_0>...<t_7>`；
- decoder 始终使用真实 route road sequence 作为受控条件；
- Stage 2 使用相同的 Qwen3-1.7B、LoRA 目标层、batch、训练 token 数和优化预算；
- Stage 3 使用相同的提示、解码参数、生成集合与 seed；
- Stage 4 使用相同的原始轨迹、道路网络、网格定义和统计实现。

不采用“三个独立 tokenizer code 直接拼接”，因为它会同时改变序列长度、词表、信息容量和 Stage 2 难度。不把 120 秒 fixed-time pilot 直接作为主线，因为它会同时改变样本物理单元、decoder target 和 Stage 2 协议。

## 3. 数据、样本和泄漏控制

### 3.1 城市与数据源

主实验覆盖 Porto 和 Chengdu；每个城市使用官方 train/test index，先划分再构造视图。正式输入为：

- GPS：MBR 归一化 `(lon, lat)`；
- Grid：固定 100 m local-equirectangular grid ID；
- Road：发布的逐 GPS observation map-matched road ID；
- route condition：去重后的道路 path，仅提供给所有单元共同的 decoder 和 LLM prompt，不计入 E 的变量。

Porto 已有逐点对齐数组位于 `/workspace/guozuyu/HTP-main/reproduction/runs/multiview_vq/porto/data`。实现必须用 manifest 配置解析绝对数据根，不能依赖 worktree 当前目录中的隐式软链接。Chengdu 用同一 preprocessing schema 生成并保存 source hashes。

### 3.2 对齐

一个 latent event 是下采样后的一个序列位置；位置 `j` 锚定原始 observation 范围 `[8j, min(8j+8,T))`，但 encoder 的卷积与上下文模块带来更大感受野，因此报告中不得称其为严格独立的 8 点块。

所有模态共享 `traj_id`、轨迹边界、点数和点索引。loader 必须检查：

- aligned array 的 `traj_id` 与 HTP processed trajectory 一一对应；
- GPS 长度等于 Grid、pointwise Road、`dxy`、`percent_dist` 和 `time_seq` 长度；
- route path 可以与 pointwise Road 长度不同，这不是错误；
- train/test 之间 `traj_id` 不重叠；
- 固定 test subset 只由 test split 采样，不参与模型选择之外的训练。

### 3.3 正式预算层级

所有核心消融单元都必须完整跑 Stage 1、Stage 2、Stage 3 和 Stage 4，不允许用 tokenizer proxy 取代下游结果。

- smoke：每个单元 32 条训练轨迹、16 条测试轨迹，Stage 2 最多 2 step，生成 8 条；只验证接口，不形成科学结论；
- controlled full-pipeline：full train split，Stage 1 固定 20 epochs，Stage 2 固定 2 epochs，固定分层抽取 1,024 条 test trajectories 完成生成和四种 G 干预；城市 × seeds `42/314159/20260819`；
- confirmation：原始 HTP、controlled 阶段的非支配候选和最终候选使用 Stage 1 原论文级 100 epochs、Stage 2 2 epochs、full test split；任何“最终优于基线”结论只由 confirmation 支持。

若资源不足，允许排队等待，不允许把未完成的 Stage 2/3 单元记为完成或用 Stage 1 排名补齐。

## 4. E 阶段：输入视图归因

在 `Q0` 与固定 G 配置下训练七个单元：

| ID | GPS | Road | Grid |
|---|---:|---:|---:|
| E0 | 1 | 0 | 0 |
| E1 | 0 | 1 | 0 |
| E2 | 0 | 0 | 1 |
| E3 | 1 | 1 | 0 |
| E4 | 1 | 0 | 1 |
| E5 | 0 | 1 | 1 |
| E6 | 1 | 1 | 1 |

模型始终实例化三个等宽 adapter：GPS Linear、Road embedding+Linear、Grid embedding。未启用视图用 learned null vector 替换。三个 adapter 经 LayerNorm 后按 active mask 求归一化和，再进入同一个 HTP encoder。所有 E 单元保持相同 hidden/latent 宽度、encoder、decoder、RQ codebooks、token 数和模块参数量；resolved config 同时报告 total/trainable/effectively-used parameter count。

Road route condition 在七个单元中完全相同。E1/E3/E5/E6 衡量的是“pointwise Road 进入 tokenizer encoder”的增量，而不是“系统是否知道 route”。

受控效应以 E0 为 reference，按同城市、同 seed、同测试轨迹、同 G 干预做 paired contrast。还报告 pair 对最佳 single 的增量和 E6 对最佳 pair 的增量。

## 5. Q 阶段：量化目标归因

Q 阶段固定 E6，使用逐项增加的阶梯：

- Q0：HTP reconstruction + 原始 residual VQ codebook/commitment loss；
- Q1：Q0 + 每层 soft marginal usage entropy regularization；
- Q2：Q1 + sampling-view latent consistency，并包含 variance/covariance anti-collapse bundle；
- Q3：Q2 + trajectory-native auxiliary heads，预测 speed state、turn state、Road transition、Grid transition。

所有 Q 单元都实例化相同 auxiliary heads；Q0–Q2 将相应 loss weight 设为 0，以保持参数结构一致。初始冻结权重为：

- `lambda_usage = 0.05`；
- `lambda_invariance = 0.05`；
- `lambda_variance = 0.05`；
- `lambda_covariance = 0.005`；
- `lambda_aux = 0.10`。

sampling view 通过对原 observation 序列随机稀疏采样后，GPS 线性插值、Road/Grid 最近邻恢复到原时间索引构造；因此两个 view 保持相同 target、latent 长度和 event index。不得随机跨轨迹配对。Q2 的一致性只比较相同 `traj_id/event index`，并报告原 view 和扰动 view 的 reconstruction，防止只优化 agreement。

Q3 标签只从训练 split 的 GPS/time、Road 和 Grid transition 派生。速度阈值由每个城市 train split 冻结并写入 manifest；test 不参与阈值估计。

Q 的主对比是相邻阶梯 `Qj - Q(j-1)`；同时报告 `Qj - Q0`。E6/Q0 与 E 阶段 E6 是同一个单元，不重复训练。

controlled full-pipeline 后按预先规则选择最佳 single-view E 和最佳 pair-view E：先过有效性 gate，再以 GPS/Road/Grid 三域 Pareto 排序；各自与最佳 Q 组合，增加两个 interaction cells。interaction 用 difference-in-differences 判断 Q 改善是否依赖 E6，而不是把自适应单元当作新的无条件主效应。

## 6. G 阶段：SID 与 sampling 的 2×2 归因

当前 HTP pipeline 的 road sequence 来自 prompt 中的真实 path，Stage 3 真正生成的是 SID 与 sampling type。因此每个 E/Q 单元建立四个 paired decode 条件：

| 条件 | SID | sampling type | Road |
|---|---|---|---|
| G00 | oracle Stage 1 | oracle Stage 1 | oracle/prompt |
| G10 | generated | oracle Stage 1 | oracle/prompt |
| G01 | oracle Stage 1 | generated | oracle/prompt |
| G11 | generated | generated | oracle/prompt |

四个条件必须包含完全相同的 `traj_id` population；解析失败不能静默删除，必须作为 invalid generation，并在主分析中按失败计入，同时提供 successful-only 敏感性分析。

对每个“越小越好”的误差指标 `f`，使用二因素 Shapley 分解：

- SID contribution = `0.5[(f10-f00) + (f11-f01)]`；
- sampling contribution = `0.5[(f01-f00) + (f11-f10)]`；
- 总生成 gap = `f11-f00`；
- interaction = `f11-f10-f01+f00`。

对“越大越好”指标先转成 regret 再分解。所有差值使用 trajectory-level paired bootstrap；不得把四组不同 population 的点估计相减。

## 7. 指标与 gate

### 7.1 Stage 1 / tokenizer

- GPS reconstruction：normalized MSE、反归一化 ADE/FDE；
- road-conditioned reconstruction：percent/dxy MSE、oracle decode GPS metrics；
- RQ：每层 active codes、perplexity、max share、prefix conditional entropy、tuple collision；
- sampling robustness：逐层 agreement/AMI、tuple agreement、continuous latent cosine；
- native semantics：speed/turn/road-transition/grid-transition probe；
- collision audit：同 tuple 内速度、转向、Road/Grid transition 的冲突率。

有效性 gate：所有 loss finite；每层 active fraction ≥0.10，perplexity fraction ≥0.05，max share ≤0.50；oracle decode 不得比同 seed E0/Q0 的 GPS error 恶化超过 25%。gate 失败的单元仍跑完已开始的链路并保留失败，但不参与 winner 选择。

### 7.2 Stage 2 / 3

- teacher-forced token NLL、逐 RQ 层 accuracy、sampling token accuracy；
- free-running exact block、prefix accuracy、edit distance；
- invalid syntax、invalid code range、empty road、decode failure；
- 按首个错误层统计后续 token 错误传播。

### 7.3 Stage 4 GPS / Road / Grid

- GPS：travel-distance JSD、segment-distance JSD、radius JSD；
- Road：road-density cosine、top-100 pattern F1、GPS-to-road-distance JSD；
- Grid：100 m grid-density cosine、top-100 pattern F1；
- 系统：valid-generation rate、decoded population coverage。

所有指标输出 machine-readable JSON；现有只写日志的 Stage4 实现不作为最终记录接口。

## 8. 选择规则与统计归因

不把异质指标事后手调权重压成单一分数。候选必须：

1. 通过 tokenizer 和 generation validity gates；
2. 在 GPS、Road、Grid 三域均不出现相对 E0/Q0 的显著劣化；
3. 至少在一个 GPS 指标、一个 Road 指标或一个 Grid 指标上有 paired-bootstrap 95% CI 不跨 0 的改善；
4. 先按跨 seed/city Pareto dominance 选非支配集合；若仍有多个，按“改善域数量→最差域 standardized regret→总训练/生成失败率”排序。

报告均值、标准差、trajectory bootstrap 95% CI、seed-level effect 和 city interaction。Porto 与 Chengdu 方向不一致时，不给出无条件通用结论，而分别报告城市结论及 pooled heterogeneity。

E/Q 结论是阶段内受控 contrast，不声称全局因果 Shapley；G 因具备完整 2×2 counterfactual，允许报告二因素 Shapley。interaction cells 只用于检验阶段结论是否依赖输入组合。

## 9. 运行、记录和提交约束

每个 run 必须保存：

- experiment/cell/city/seed/status；
- Git revision、dirty flag、upstream revision；
- 原始命令和 resolved config；
- 数据/split/subset manifest 与 SHA256；
- 输入单位、normalization、view mask 和 sampling-view recipe；
- 模型结构、参数量、初始化/冻结模块；
- 完整 loss 公式与实际权重；
- 日志、checkpoint、Stage 1/2/3/4 JSON、runtime 和 GPU；
- observation、interpretation、gate、failure、next action。

目录约定：

- 代码：`stage_1/src/staged_attribution/`；
- 单元测试：`stage_1/tests/staged_attribution/`；
- 冻结配置：`stage_1/configs/staged_attribution/`；
- 运行 manifest：`stage_1/runs/staged_attribution/<experiment_id>/`；
- 汇总报告：`stage_1/reports/staged_attribution/`；
- 大 checkpoint/模型留在远端 ignored artifact root，不提交 Git。

每次正式运行必须来自 clean commit。实现、预注册、smoke 修复和正式结果分开提交。失败不得删除。Markdown 报告与 Notion 实验记录使用同一 experiment ID；Notion 不替代本地 machine-readable evidence。

## 10. 完成定义

只有以下证据全部存在时才算完成：

- 新 branch/worktree 可定位且 clean；
- 数据对齐/preprocessing、E/Q model、Stage 2/3 runner、G 2×2、Stage4 JSON、统计汇总均有测试；
- 七个 E 核心单元、四个 Q 阶梯（E6/Q0 复用）和两个 interaction cells 在两城市三 seeds 完整通过 controlled Stage 2/3；
- baseline 与最终候选完成 confirmation；
- 所有机器记录、日志、配置、commit、失败和 Notion 条目齐全；
- 最终报告明确区分观察、归因估计、解释、限制和适用于 GPS/Road/Grid 轨迹上下文的结论。
