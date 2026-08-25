# GPS / Road / Grid 分阶段归因实验预注册

- 实验 ID：`gps_road_grid_staged_attribution_v1_20260826`
- 预注册日期：2026-08-26
- 状态：frozen，尚无 controlled/confirmation 结果
- 代码分支：`exp/staged-attribution-gps-road-grid`
- 设计依据：`docs/plans/2026-08-26-staged-attribution-gps-road-grid-design.md`
- 实现计划：`docs/plans/2026-08-26-staged-attribution-gps-road-grid-implementation.md`
- 上游 HTP revision：`1294275d2bed5f9de86269ab285ff5e23828b947`

## 1. 研究问题、范围与假设

目标是把 HTP 生成轨迹的误差拆到三个受控阶段，而不是只用 tokenizer proxy 排名：

1. E：pointwise GPS、Road、100 m Grid 的 encoder 可见性如何改变完整 Stage1→2→3/4 结果；
2. Q：usage、sampling consistency/anti-collapse、轨迹原生辅助目标逐项加入后如何改变结果；
3. G：LLM 生成 SID 与 sampling type 各自造成多少误差，以及二者是否交互。

正式结论不包含 POI。所有 cell 使用 oracle/prompt route condition；Road 是否进入 E 只指逐 observation Road ID 是否进入 tokenizer encoder。

预注册假设：

- H-E：至少一个多视图 E cell 在不降低 validity 的前提下，相对 E0/Q0 改善至少一个 GPS、Road 或 Grid 下游指标；
- H-Q：Q1→Q3 中至少一个相邻阶梯提高码本/采样稳健性且不显著损害下游 Pareto 指标；
- H-G：G11 与 G00 的 gap 可由 SID 与 sampling 的完整 2×2 配对 Shapley 分解，且 population 不因解析失败改变；
- 若城市方向不一致，则拒绝无条件通用结论，分别报告 Porto/Chengdu 与 heterogeneity。

## 2. Cell、变量与控制

E 固定 Q0：

| Cell | Encoder GPS | pointwise Road | 100 m Grid |
|---|---:|---:|---:|
| E0 | 1 | 0 | 0 |
| E1 | 0 | 1 | 0 |
| E2 | 0 | 0 | 1 |
| E3 | 1 | 1 | 0 |
| E4 | 1 | 0 | 1 |
| E5 | 0 | 1 | 1 |
| E6 | 1 | 1 | 1 |

Q 固定 E6，并逐项累加：

| Cell | usage | invariance | variance | covariance | auxiliary |
|---|---:|---:|---:|---:|---:|
| Q0 | 0 | 0 | 0 | 0 | 0 |
| Q1 | .05 | 0 | 0 | 0 | 0 |
| Q2 | .05 | .05 | .05 | .005 | 0 |
| Q3 | .05 | .05 | .05 | .005 | .10 |

核心 cell 为 E0..E6/Q0 与 E6/Q1..Q3，共十个；E6/Q0 不重复训练。controlled 后按冻结 Pareto 规则选择最佳 single 与 pair，各增加一个与最佳 Q 的 interaction cell；它们不作为无条件主效应。

保持不变：HTP latent 长度、四层 RQ 词表、decoder、oracle route、Qwen3-1.7B、LoRA target modules、prompt、Stage3 解码参数、test ID、Stage4 实现和每个对比内的 seed。

## 3. 数据、样本单元与泄漏

- 城市：Porto、Chengdu；官方 `train_index.npy/test_index.npy`；train/test `traj_id` 必须零交集；
- 一个样本是一条完整轨迹，20–200 个 GPS observation，不切滑窗；
- GPS 是 MBR 归一化 `(lon,lat)`；时间为 Unix seconds；
- Grid 是 local-equirectangular 100 m cell ID；
- pointwise Road 是发布的逐 GPS observation map-matched road ID；
- route path 可与 observation 数不同，只进入所有 cell 相同的 decoder/prompt；
- GPS/Grid/Road/time/dxy/percent 以 `traj_id + point index` 对齐，不按未校验行号 join；
- Porto aligned root：`/workspace/guozuyu/HTP-main/reproduction/runs/multiview_vq/porto/data`；
- Chengdu aligned root：`/workspace/guozuyu/HTP-staged-attribution-artifacts/aligned/chengdu`；
- 轨迹、用户、道路或采样 view 不跨 split 配对；test 不参与速度/转向阈值估计。

预算：

| budget | Stage1 train/test | Stage1 | Stage2 | Stage3/4 |
|---|---|---|---|---|
| smoke | 32/16，ID hash 固定 | 1 epoch，batch 8 | 1 epoch，最多 2 step | 固定 8 test IDs；仅接口验证 |
| controlled | full/full | 20 epochs，batch 512 | 2 epochs，full train | 固定 1,024 test IDs；两城×三 seed |
| confirmation | full/full | 100 epochs，batch 512 | 2 epochs，full train | full test；baseline/非支配候选/最终候选 |

seeds：`42, 314159, 20260819`。smoke 只跑 Porto/42；不得形成科学结论。

## 4. 输入、表示、token 与解码

三个 adapter 始终实例化且均可训练：GPS `Linear(2,64)`、Road `Embedding(128)+Linear`、Grid `Embedding(64)`；各自 LayerNorm。禁用视图用 learned null vector 替换；active 表示求和后除以 `sqrt(active_count)`。因此各 E cell total/trainable parameter count 相同。

HTP encoder 三次 stride-2，下采样后 event 数为 `ceil(T/8)`。event `j` 锚定原 point 范围 `[8j,min(8j+8,T))`，但因卷积感受野不声称严格独立八点块。RQ 是四层 residual VQ，词表固定 `[32,64,128,256]`；每个 event 线协议为 `<a_i><b_j><c_k><d_l>`，sampling type 为三位长度类型 `<t_0>...<t_7>`。

decoder 输入为量化 latent、sampling length type、oracle route embedding 与归一化 route geometry；输出逐点 standardized `percent_dist increment` 与 standardized `(dx,dy)`。反变换后对 percent increment 累积、绝对值并 clip 到 `[0,.9999999]`，沿 oracle route 插值后加反标准化 dxy 得到 GPS。所有 G condition 的 Road 都相同。

Stage2 answer 精确为：

```text
<t_n>, <|t_begin|><|p_begin|><a_i><b_j><c_k><d_l><|p_end|>, ...<|t_end|>
```

assistant-only labels 从 think block 之后开始；system/user/padding 为 `-100`。跨 cell 必须同 `traj_id` 保持相同 prompt/answer token budget。

## 5. Sampling view 与 Q 损失

对每条原 observation 序列，以 cell seed + global step 生成 Bernoulli keep mask，keep probability `.65`，强制首尾点。GPS 在原 time index 上线性插值恢复；Road/Grid 以最近邻 time index 恢复；target、轨迹边界、latent event 数不变。

总损失：

```text
L_base = MSE(percent_hat, percent) + MSE(dxy_hat, dxy) + L_RQ
L_total = L_base
        + λ_usage L_soft_marginal_usage
        + λ_inv (1 - cosine(z,z_sampled))
        + λ_var L_variance
        + λ_cov L_offdiag_covariance
        + λ_aux mean(CE(speed), CE(turn), CE(road_transition), CE(grid_transition))
```

consistency/variance/covariance 在量化前、同 `traj_id/event index` latent 上计算。speed/turn/transition 标签按 8-point anchor event 构造；速度三阈值和转向两阈值只从该 city train population 分位数冻结并写入 resolved config。

## 6. 初始化与训练

Stage1 每个 city/seed/cell 从同 seed 随机初始化，不加载旧 checkpoint；全部模块可训练，不 early stop。optimizer `AdamW(lr=1e-4, weight_decay=1e-4)`，global gradient clip 1.0，CUDA BF16 autocast。linear warmup 10 epochs（起始 ratio .01），之后 cosine 到 ratio .10；若总 epoch≤10，则整个 run 为线性 warmup。resolved config 记录 steps/epoch 与实际 epoch learning rate。

Stage2 为本地 Qwen3-1.7B；LoRA `r=16, alpha=32, dropout=.05`，target modules `q/k/v/o_proj, gate/up/down_proj`，并训练全部新协议 token embedding。optimizer `adamw_torch`、lr `5e-4`、weight decay `1e-4`、warmup ratio `.05`、linear scheduler、grad clip 1.0、BF16、gradient checkpointing。每个 cell 相同 batch 和 optimizer step 预算。

Stage3 当前冻结为 greedy (`do_sample=false`)、`max_new_tokens=256`；generation seed 仍记录。只解析新 continuation，不从 prompt 误取 `<road_n>`；malformed/truncated/duplicate/out-of-range 均保留失败记录。

## 7. G 2×2

| G | SID | sampling type | Road |
|---|---|---|---|
| G00 | oracle Stage1 | oracle Stage1 | oracle/prompt |
| G10 | generated | oracle Stage1 | oracle/prompt |
| G01 | oracle Stage1 | generated | oracle/prompt |
| G11 | generated | generated | oracle/prompt |

四组必须是完全相同 ID population。越小越好指标 `f`：

```text
phi_sid = .5[(f10-f00)+(f11-f01)]
phi_sampling = .5[(f01-f00)+(f11-f10)]
interaction = f11-f10-f01+f00
phi_sid+phi_sampling = f11-f00
```

越大越好先转 regret。只对同轨迹配对量做 bootstrap，失败在主分析中按预注册 penalty 计入，并另报 successful-only sensitivity。

## 8. 指标、gate 与统计

Stage1：reconstruction/percent/dxy/RQ loss；逐层 active fraction、perplexity fraction、max share；prefix conditional entropy；sampling layer agreement/AMI 与 tuple agreement；native speed/turn/Road/Grid transition；collision。有效性要求 loss finite、各层 active≥.10、perplexity fraction≥.05、max share≤.50，oracle decode GPS error 不得比同 city/seed E0/Q0 高 25%。E0 ratio 在跨 cell 汇总后回填；之前状态必须是 pending，不得伪造 pass。

Stage2/3：teacher-forced NLL/perplexity、sampling/逐 RQ layer accuracy、semantic sequence exact、first-error histogram；free generation validity、exact block、invalid reason、decode coverage。

Stage4 保持 HTP 方向：GPS travel/segment/radius JSD 越小越好；Road density cosine、top-100 F1 越大越好，point-to-road JSD 越小；Grid density cosine/top-100 F1 越大；另报 valid-generation/decode population coverage。

统计：trajectory paired bootstrap 95% CI；seed mean/std/CI；city interaction/方向异质性。选择要求：通过 validity；GPS/Road/Grid 无显著劣化；至少一个域显著改善；跨 seed/city Pareto 非支配。仍并列时依次按改善域数、最差 standardized regret、总失败率。

## 9. 方法变更表

| 方法组成 | Baseline E0/Q0 | 本轮变量 | 是否变化 |
|---|---|---|---|
| 样本物理单元 | 完整轨迹 | 与 baseline 相同 | 否 |
| 视图构造与覆盖 | GPS；完整 observation 支持 | E 改变 GPS/Road/Grid active mask | 是，主变量 |
| 输入单位 | MBR-normalized GPS；seconds | Road ID/Grid ID 可加入 | 是，主变量 |
| encoder/downsampling | HTP 3×stride-2 | 共享 encoder；等参数 adapter/null fusion | adapter 输入变化 |
| token 支持 | `ceil(T/8)` events | 与 baseline 相同 | 否 |
| quantizer/codebook | 4-layer RQ `[32,64,128,256]` | Q1–Q3 只改 loss | loss 变化，协议不变 |
| decoder/时间锚点 | oracle route；逐 observation percent+dxy | 与 baseline 相同 | 否 |
| 初始化/冻结 | 同 seed 从头训练；全部可训练 | 与 baseline 相同 | 否 |
| reconstruction/VQ | percent+dxy MSE+RQ | 与 baseline 相同 | 否 |
| 新增 loss | 无 | Q1 usage；Q2 inv/var/cov；Q3 aux | 是，Q 主变量 |
| 训练数据/超参 | 同 city/split/seed/budget | 与 baseline 相同 | 否 |
| Stage2 | Qwen3-1.7B/同 LoRA | 只有 answer SID 改变 | 监督内容变化 |
| Stage3/G | oracle route，生成 SID/sampling | 完整 2×2 counterfactual | 是，归因变量 |
| 评估 | 同 ID、Road、Grid 与统计实现 | paired contrasts/Shapley | 归因分析增加 |

## 10. 首次结果前的方法纠偏

在尚无任何正式 Stage4 attempt、未观察任何 cell 结果时，代码审计发现并冻结以下纠偏：

1. E0 是 GPS-only，不是空模态，因此不能作为标准 2×2 factorial 的 `f00`。七个非空 E 组合只报告可识别的条件二阶交互：

```text
GPS×Road | Grid = E6 - E4 - E5 + E2
GPS×Grid | Road = E6 - E3 - E5 + E1
Road×Grid | GPS = E6 - E3 - E4 + E0
```

旧实现中退化为 `E3-E1`/`E4-E2` 的字段不得出现在任何结果报告。

2. Stage4 Grid mapping 必须直接复用该 city aligned `metadata.json` 的 MBR、`meters_per_lon/lat`、`grid_size_m=100`、`n_grid_x/y`，与 Stage1 `gps_to_grid_ids` 完全相同；禁止另算近似 geodesic grid。
3. Grid/Road top-100 pattern F1 只取各分布正频 support 内的 top-k，使用对称集合 F1 `2|A∩B|/(|A|+|B|)`；禁止把零频 bin 加入 top-k。
4. 生成 GPS 任一点超出注册 city MBR 时，该轨迹记作 invalid，并在主分析保留预注册 penalty；禁止裁剪或静默丢点。

纠偏只修正可识别性与评估定义，不改变 E/Q cell、训练数据、seed、预算、权重或选择 gate。adaptive/selection 定向测试 23 passed、全套 166 passed 后才能形成新的 clean formal revision。

## 11. 失败、完成与证据边界

formal run 必须来自 clean 40-char Git revision。每个 run 保存 command、resolved config、数据/子集 hash、Stage1/2/3/4 artifacts、逐阶段 complete marker、顶层 immutable manifest、失败 traceback。已失败 attempt 不覆盖，另建 attempt 目录。

`report_results` 在统计前强制验证完整 Cartesian coverage、manifest identity、同一 clean revision、artifact 位于本 attempt 内，以及 marker size/SHA256。controlled 报告必须覆盖两城×三 seed×十 cell。confirmation 必须通过 `--cells` 显式给出 baseline `e0_q0` 与 controlled 选出的候选；缺少 baseline、重复/未知 cell 或省略选择时拒绝排队和汇总，禁止默认重跑全部十 cell。

十个核心 cell 之外，配置注册完整 7×4 E×Q 交叉组合，但非核心组合只能通过显式 `--cells` 运行。核心 controlled 后，`select_candidates --phase core` 从同 city 内三 seed 的 G11 regret 配对差值计算 95% t CI：任一城市的 GPS/Road/Grid/validity 指标显著劣化则拒绝；至少一个城市、一个 GPS/Road/Grid 指标显著改善，且所有 city/seed Stage1 gate 通过，方可进入 Pareto。

选择器分别冻结最佳 single（E0/E1/E2）、pair（E3/E4/E5）和 Q（E6/Q0..Q3）。若最佳 Q 不是 Q0，则生成 `best_single×best_Q` 与 `best_pair×best_Q` adaptive cell；不得人工替换。adaptive controlled 完成后用 `--phase final` 重新审计，confirmation 清单必须以 `e0_q0` 开头并只包含最终合格非支配候选。城市方向相反时保留 heterogeneity，不能用两城合并成六个独立重复来掩盖。

Stage1 gate 在汇总时用同 city/seed G00 `population_mean_penalized_error` 相对 E0/Q0 的比值回填 oracle ratio；codebook active/perplexity/max-share 与 loss finite 条件仍按第 8 节，不允许继续使用 `pending_e0_oracle_ratio` 参加候选选择。

任何没有 Stage2 adapter、generation、四路 decode、Stage4 JSON 的 cell 都是 incomplete；不得用 Stage1 排名或现有独立 tokenizer pilot 补齐。最终优于 baseline 的声称只允许来自 confirmation。smoke、controlled 与 confirmation 结果分别提交，负结果与失败不得删除。
