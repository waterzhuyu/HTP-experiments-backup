# GPS/Road/Grid 分阶段归因：Stage2 资源恢复协议

## 1. 文档状态与边界

- 实验 ID：`gps_road_grid_staged_attribution_v1_20260826`
- 记录日期：2026-08-27（Asia/Shanghai）
- 文档状态：`prepared / decision_pending`
- 实验类型：受控消融运行失败后的预结果（pre-result）执行协议修正候选
- 正式结果资格：`false`
- 当前结论：24 GiB GPU 上，原锁定的 Stage2 `microbatch=16, accumulation=1` 不可行；`microbatch=8, accumulation=2` 已通过早期容量和极端长度容量探针，但尚未完成任一正式全程训练。
- 边界：本文不授权启动正式队列，不改变当前正式配置，不把内存探针或候选子集审计计入消融结果。正式重启必须先明确选择 A/B/C，并用独立配置提交冻结选择。

本文只讨论 GPS、Road、Grid；POI 持续排除。Stage1 表示学习、tokenizer、Stage3 生成、G 归因和 Stage4 指标定义均不在本次资源修正中改变。

## 2. 代码、数据与产物权威性

### 2.1 Git worktree

| 用途 | 路径 | 分支 | 实现 revision | 状态 |
|---|---|---|---|---|
| 原正式训练源 | `/workspace/guozuyu/HTP-staged-attribution` | `exp/staged-attribution-gps-road-grid` | `730c2cc5b51003b31e5de49561bee39ca8979a75` | clean，保持冻结 |
| 校正重评与统计 | `/workspace/guozuyu/HTP-staged-attribution-eval-correction` | `fix/staged-attribution-eval-correction` | `e7ec3348e2f5db34b200355fd3c5f57bce99e360` | clean，176 tests passed |
| Stage2 资源控制候选 | `/workspace/guozuyu/HTP-staged-attribution-stage2-resource-controls` | `feat/staged-attribution-stage2-resource-controls` | `0ef07aab1121394b6c2657f553c22fce6a1519ff`（本文所审计实现） | 178 tests passed；本文另行提交 |
| 统一执行分支 | `/workspace/guozuyu/HTP-staged-attribution-integrated` | `exp/staged-attribution-gps-road-grid-integrated` | `ba8cf2e3f80edbff10d075affaa8fe6c7592ef8b` | clean，206 tests passed |

资源控制候选分支的提交：

1. `4357868`：显式记录 Stage2 训练集合、microbatch、梯度累积和有效 batch；默认值保持原行为。
2. `4e66048`：冻结同一城市跨 cell、跨训练 seed 共用的 Stage2 训练 ID 清单，并校验源集合和清单哈希。
3. `6f39ef6`：审计候选子集在 GPS、Road、Grid 和 HTP lens type 上的代表性。
4. `0ef07aa`：续跑时重新核对 marker 中 artifact 的大小和 SHA-256，拒绝被替换或损坏的阶段产物。

候选分支尚未合并到正式训练分支；正式配置 `stage_1/configs/staged_attribution/experiment.json` 仍为全量 Stage2、`batch=16`、`accumulation=1` 的原锁定口径。

统一执行分支通过 merge commit `e4ddc9848257a9f98030a2b02b8b1bf5bb14e498` 同时包含资源控制父提交 `8743e8ed38a608e6dd3539fa2fa8bc4166a5f36f` 和评估修正父提交 `e7ec3348e2f5db34b200355fd3c5f57bce99e360`；`ba8cf2e` 新增统一契约测试。该测试要求：默认正式 Stage2 仍为 full/16×1、core job 数仍为 60、adaptive `E3×Q2` 可解析、训练 DAG 包含 `stage2_plan.json`、corrected reevaluation Stage4 必须同时产生四组 failure-retaining contribution manifest。

源分支与统一分支的正式 `experiment.json` 逐字节 SHA-256 均为 `201b0e89425f1c159d5a779eb381a2fe9922614909bf98bcf2ee87014fc071dc`，两者在该文件上无 Git diff。因此，创建统一分支本身没有暗中选择 A/B/C。

### 2.2 数据切分

| 城市 | train 轨迹 | test 轨迹 | train ID 权威文件 | train ID 文件 SHA-256 |
|---|---:|---:|---|---|
| Porto | 537,993 | 59,778 | `/workspace/guozuyu/HTP-main/reproduction/runs/multiview_vq/porto/data/train_traj_ids.npy` | `3c833f86cef5ab822d7b67fd8b127e8fa63d46a7ee16876632add0bdd7dabe47` |
| Chengdu | 542,592 | 60,288 | `/workspace/guozuyu/HTP-staged-attribution-artifacts/aligned/chengdu/train_traj_ids.npy` | `fa665ad3172973ffe9c31c4cd65cc577fd9e0d5a4373e0afc49f479018fc38e4` |

样本物理单元是一条完整轨迹。train/test 使用冻结的 aligned split，按 `traj_id` 严格不相交；GPS、Road、Grid 是同一轨迹、同一观测点索引下的对齐视图。候选 Stage2 子集只从 train ID 集合抽取，不接触 test、Stage3 generation subset 或固定 Stage4 重评子集。

## 3. 原始失败事实

首批四个 Porto/seed 42/Q0 cell（E0、E1、E2、E3）的 `attempt_001` 均完成 Stage1、token 导出和 Stage2 数据构造，随后在 Stage2 完成 3 个 optimizer step 后，于第 4 步之前发生同型 CUDA OOM。共同口径：

- 模型：Qwen3-1.7B + LoRA，训练人口 537,993 条；
- epochs：2；预计 67,250 optimizer steps/cell；
- microbatch：16；gradient accumulation：1；有效 batch：16；
- 失败时一次申请约 5.76 GiB；GPU 总显存约 23.56 GiB；空闲约 445 MiB；
- PyTorch allocated 约 17.15 GiB，reserved-but-unallocated 约 5.64 GiB；
- 四次失败均已终止且不可变，没有 watcher、scheduler 或子进程继续运行。

终止审计：

- `controlled_first_batch_terminal_audit.json`
- SHA-256：`a229fd35fd7eab1c298d9838efc6dd80b4b51b442eeac7265262f779bec4850d`

旧 artifact 根保留原始失败证据；新的可写根为 `/home/guozuyu/HTP-staged-attribution-artifacts/staged_attribution/gps_road_grid_staged_attribution_v1_20260826`，其中 `attempt_001` 仅以符号链接指向原始不可变目录。没有创建或启动 `attempt_002`。

## 4. 内存归因与容量探针

机器记录：`provenance/stage2_memory_recovery_audit.json`，SHA-256 `f4d8aab0d5961e34663001015d1cabbbaff6cd88f2bc5a779b6a76bab1787d7a`。

| 探针 | 唯一变化 | 结果 | 归因 |
|---|---|---|---|
| allocator-only | 保持 16×1，仅设 `expandable_segments:True` | 第 17 步前 OOM | 碎片不是充分解释；存在真实容量不足 |
| microbatch | 改为 8×2，保持有效 batch=16；全量 537,993 条；20 optimizer steps | 20/20 完成；peak allocated 14.739 GiB；peak reserved 22.840 GiB | 8×2 解除早期稳定 OOM |
| worst-length | 10,000 个最长字符候选中精确 tokenize 后取最长 16 条；822–876 input tokens；8×2；1 optimizer step | 完成；peak allocated 17.931 GiB；peak reserved 22.803 GiB | 8×2 可容纳已检查的高长度 batch |

这些探针只证明早期和已检查极端长度容量，不证明完整 8,000 或 67,250 步全程不会出现 OOM，也不证明训练指标等价。

## 5. 待选择的执行口径

### Option A：全量训练，改为 8×2

- Stage2 train：每城市全量 train；
- epochs：2；
- microbatch：8；gradient accumulation：2；有效 batch：16；
- 每 cell 约 67,250 optimizer steps（Porto 基准）；
- 估算 Stage2 约 44.59 h/cell，四卡跑完 60 个 core run 约 27.87 天；不含 Stage1、adaptive、confirmation 和失败重试；
- 科学优点：只改变显存执行方式，训练人口保持原锁定口径；
- 代价：完成周期接近一个月以上。

### Option B（推荐）：冻结每城市 64,000 条 train ID，改为 8×2

- Stage2 train：Porto、Chengdu 各自冻结一个 64,000-ID 清单；同一城市所有 E×Q cell、三个训练 seed、adaptive 和 confirmation 共用同一清单；
- 抽样：`sha256_rank_fixed_city_v1`，排序键为 `SHA256("20260826:stage2_train:{city}:{traj_id}")`，取前 64,000 后按 `traj_id` 升序保存；
- subset seed：20260826；该 seed 与模型训练 seed 分离；
- epochs：2；microbatch：8；gradient accumulation：2；有效 batch：16；
- 每 cell 8,000 optimizer steps；估算 Stage2 约 5.30 h/cell，四卡跑完 60 个 core run 约 3.32 天；加 Stage1 后预计 core 约 4–5 天；
- 科学代价：相对原预注册改变 Stage2 训练人口，必须称为“预结果协议修正”，不可与原全量口径混称；
- 控制：所有 cell/seed 共享精确 ID，避免把样本差异归因给 E 或 Q；Stage1 仍使用全量训练集，Stage2 test、Stage3 和 Stage4 口径不变。

### Option C：更大显存 GPU，保持原 16×1 和全量人口

- 科学优点：完全保持原训练口径；
- 当前阻塞：3090s 仅有四张约 24 GiB GPU，没有可用的 >24 GiB 设备；
- 在资源出现前不可执行。

## 6. Option B 候选冻结清单（非正式结果）

根目录：`/home/guozuyu/HTP-staged-attribution-artifacts/staged_attribution/gps_road_grid_staged_attribution_v1_20260826/provenance/candidate_option_b_stage2_subsets`

候选配置：`candidate_option_b_experiment.json`，SHA-256 `908d07c0423e5f732d69dcc52e27e752d4d0f9f3f1c3f596f5eeb7854f74b155`。该文件带有 `candidate_only=true`，不会被正式队列自动读取。

索引 `index.json` SHA-256：`8673fd0992d5f41af0ba0d810db6770edc8e4f1624d6a76cb984480fa5524bbd`。

| 城市 | 源人口 | 冻结人口 | 所选 ID 集合 SHA-256 | 清单文件 SHA-256 |
|---|---:|---:|---|---|
| Porto | 537,993 | 64,000 | `a83b142ec63dfe8cbad1b178da16b744224bf32e91a3fec3a2e2457c4041715a` | `026720480bfed13f9f98aa4f34611a3a0ce06b98af03e712ca60ef3f13153942` |
| Chengdu | 542,592 | 64,000 | `003a6a0be6814ed9b4cdf527eab4f4d3e389d079cbae3cb9a274f82c9757d61b` | `ba7d822c6ec0dbfe46f88d5653b127017f62918b92c81090993a2e8251acf8e4` |

清单写入是不可变的：同一路径已有相同内容时幂等返回；内容不同则拒绝覆盖。运行时重新验证城市、请求样本数、subset seed、源 ID 人口、源 ID hash、源文件 hash、所选 ID 唯一性/排序/hash、集合包含关系和 SHA 排名复现结果。

## 7. Option B 代表性审计

审计直接使用 aligned GPS/Grid/Road point arrays；它不使用 test 数据，不训练模型。预先采用以下工程接受阈值判断候选集合是否存在明显结构偏差：

- 轨迹点数 KS ≤ 0.01；
- 轨迹点数绝对标准化均值差 ≤ 0.02；
- 8 个 HTP lens-type stratum 的最大绝对比例差 ≤ 0.005；
- GPS 两轴绝对标准化均值差均 ≤ 0.02；
- Grid/Road Jensen–Shannon divergence ≤ 0.01 bit；
- Grid/Road total variation distance ≤ 0.05。

这些阈值是本次资源恢复的探索性工程阈值，不是最终论文显著性阈值。

| 指标 | Porto | Chengdu | 阈值 | 状态 |
|---|---:|---:|---:|---|
| sampling fraction | 0.11896 | 0.11795 | 描述性 | — |
| length KS | 0.00293 | 0.00351 | ≤0.01 | pass |
| length SMD | −0.00061 | −0.00601 | |·|≤0.02 | pass |
| lens-type max proportion delta | 0.00171 | 0.00283 | ≤0.005 | pass |
| GPS lon SMD | −0.00404 | −0.00509 | |·|≤0.02 | pass |
| GPS lat SMD | −0.00194 | −0.00006 | |·|≤0.02 | pass |
| Grid JS (bit) | 0.00052 | 0.00268 | ≤0.01 | pass |
| Grid TV | 0.01537 | 0.03050 | ≤0.05 | pass |
| Road JS (bit) | 0.00156 | 0.00426 | ≤0.01 | pass |
| Road TV | 0.02182 | 0.04015 | ≤0.05 | pass |

长尾 support 覆盖率不作为 pass/fail 隐藏：Porto 的 Grid/Road support 覆盖为 95.81%/86.67%，Chengdu 为 77.23%/76.54%。64,000 条轨迹保留了整体频率分布，但必然缺失一部分极低频 Grid/Road ID；最终结论必须明确限定为“在该冻结 Stage2 训练预算下”的相对归因，不可宣称等价于全量训练。

审计文件：

- Porto：`porto_representativeness_audit.json`，SHA-256 `5e16f463dc38bee971e077a52b3d300e9589050b5670ff088f7dbba5a8141428`
- Chengdu：`chengdu_representativeness_audit.json`，SHA-256 `f63af8ef6c92f9e24c68969d8e999b3084ab5ef77ee352e5bd29067e4db8950f`

## 8. 方法变更表

| 方法组成 | 原锁定 baseline | Option B 候选 | 是否变化 |
|---|---|---|---|
| 样本物理单元 | 一条完整轨迹 | 与 baseline 相同 | 否 |
| 视图构造与覆盖范围 | 同一点索引对齐的 GPS/Road/Grid；排除 POI | 与 baseline 相同 | 否 |
| 输入特征与单位 | GPS 为 MBR 归一化经纬度；Road/Grid 为整数 ID；时间为秒级轨迹时间序列 | 与 baseline 相同 | 否 |
| encoder/downsampling | Attribution RQ-VAE；既定三层事件/downsampling 口径 | 与 baseline 相同 | 否 |
| token 时间/空间支持 | 既定变长 token 序列及边界处理 | 与 baseline 相同 | 否 |
| quantizer/codebook | 4 层 RQ，词表 `[32,64,128,256]` | 与 baseline 相同 | 否 |
| decoder 与输出时间锚点 | 既定 Stage1 decoder 与 Stage3/4 GPS 重构语义 | 与 baseline 相同 | 否 |
| 初始化与冻结模块 | 各 cell/seed 从同 seed 初始化；Stage2 Qwen3-1.7B + LoRA | 与 baseline 相同 | 否 |
| reconstruction/VQ loss | 既定 E/Q cell 权重 | 与 baseline 相同 | 否 |
| 新增 loss 与权重 | 无 | 无 | 否 |
| Stage1 训练数据与超参数 | 全量 train，20 epochs（controlled） | 与 baseline 相同 | 否 |
| Stage2 训练人口 | 每城市全量 train | 每城市固定 64,000 IDs | **是** |
| Stage2 microbatch/累积 | 16×1，有效 batch 16 | 8×2，有效 batch 16 | **是** |
| Stage2 epochs/optimizer step | 2 epochs；Porto 约 67,250 steps | 2 epochs；8,000 steps | **是（人口变化导致）** |
| Stage2 optimizer/scheduler/precision | 既定 `run_hf_stage2` 口径 | 与 baseline 相同 | 否 |
| 评估 split、对齐与指标 | 固定 test、固定 1024-ID 分层重评、失败保留统计 | 与 baseline 相同 | 否 |

所有与 Stage2 训练人口和显存执行方式同时发生的变化均已列出，因此 Option B 的结果不能单独归因为 tokenizer、view 或 Q loss；E/Q cell 之间仍可做受控相对归因，因为它们共享完全相同的修正口径。

## 9. 正式执行门与复现命令

### 9.1 启动前必须同时满足

1. 用户明确选择 A、B 或 C；
2. 在资源控制分支上提交单独的正式配置 commit；
3. 若选 B，正式配置同时登记两城市冻结清单路径，controlled/adaptive/confirmation 使用同一清单；
4. `git status --short` 为空，记录最终 revision；
5. `PYTHONPATH="$work:$work/stage_1/src" ... -m pytest stage_1/tests/staged_attribution -q` 全部通过；
6. 新 attempt 路径不存在，旧失败 attempt 保持不可变；
7. 四卡满足既定安全阈值，且没有同一 run key 的活动进程；
8. dry-run 核对 60 个 core job 的 city/seed/cell 顺序和目标 artifact 根；
9. 将正式配置、命令、revision、清单 SHA、数据 split、seed 和输出根同步到 Notion。

### 9.2 候选清单生成命令

```bash
cd /workspace/guozuyu/HTP-staged-attribution-stage2-resource-controls
PYTHONPATH="$PWD:$PWD/stage_1/src" \
  /workspace/guozuyu/HTP-staged-attribution/.venv/bin/python \
  -m staged_attribution.stage2_subset_manifest \
  --config /tmp/candidate_option_b_experiment.json \
  --budget controlled \
  --output-dir /home/guozuyu/HTP-staged-attribution-artifacts/staged_attribution/gps_road_grid_staged_attribution_v1_20260826/provenance/candidate_option_b_stage2_subsets
```

### 9.3 测试命令与结果

```bash
cd /workspace/guozuyu/HTP-staged-attribution-stage2-resource-controls
PYTHONPATH="$PWD:$PWD/stage_1/src" \
  /workspace/guozuyu/HTP-staged-attribution/.venv/bin/python \
  -m pytest stage_1/tests/staged_attribution -q
```

结果：`178 passed, 21 warnings in 10.64s`。warnings 均为现有 PyTorch Transformer nested-tensor 提示。

统一执行分支的相同完整测试命令结果：`206 passed, 21 warnings in 11.37s`；worktree clean。正式启动后，source training 和 corrected reevaluation/statistics 应统一从该分支的唯一冻结 revision 执行，避免训练与后处理分支发生未记录漂移。

## 10. 事实、解释与决策分离

### 观测事实

- 原 16×1 正式首批四个 cell 同型 OOM；allocator-only 仍 OOM。
- 8×2 完成 20 个全量人口早期 optimizer steps，并完成一个 822–876 token 的极端长度 optimizer step。
- 候选 64k 集合的长度、GPS 均值、Grid/Road 频率分布偏差低于本文工程阈值。
- 3090s 当前四卡空闲，但没有 >24 GiB GPU。

### 解释

- 原失败主要是单 microbatch 的真实显存容量问题，而非单纯 allocator 碎片。
- 8×2 是当前 24 GiB 卡上最小且已验证的显存修正；它保持有效 batch，但浮点累积顺序与原 16×1 不完全相同。
- 64k 固定集合适合在有限周期内完成 E/Q 相对归因，但不能证明与全量 Stage2 的绝对指标等价。

### Gate 决策

- 当前 gate：`HOLD`。
- 原因：A/B/C 是会改变运行周期或训练人口的实质选择，尚未得到授权。
- 下一动作：收到选择后，提交唯一正式配置，做 dry-run/provenance audit，然后才创建 `attempt_002` 并启动队列。
