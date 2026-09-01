# GPS/Road/Grid 分阶段归因：实现与校正重评冻结记录

## 1. 实验身份、状态与研究问题

- 实验 ID：`gps_road_grid_staged_attribution_v1_20260826`
- 记录日期：2026-09-01（Asia/Shanghai）
- 类型：正式 controlled 消融运行中的实现验证与 pre-result 校正重评人口冻结
- 状态：`implementation_verified / reevaluation_subsets_frozen / scientific_result_pending`
- 研究范围：只包含 GPS、Road、Grid；POI 排除
- 正式运行 worktree：`/workspace/guozuyu/HTP-staged-attribution-integrated`
- 正式运行分支：`exp/staged-attribution-gps-road-grid-integrated`
- 正式运行 revision：`6ac87be094c9c1ce4783f10291a339f04623449b`
- 正式运行 config SHA-256：`39a33c9bef9cc06f1bc2cdbbf1c6010ea76dcb455a7855b330905868a1c7604a`
- 校正实现 worktree：`/workspace/guozuyu/HTP-staged-attribution-compact-adapter`
- 校正实现分支：`fix/staged-attribution-compact-adapter`
- 固定子集生成 revision：`a2991e14d509a96830d21c5cfa8f2ae9d4fa1967`
- 校正重评队列互斥 revision：`5d7cbc7efeb2142472733356694b48c32ba8db18`
- 校正重评来源/启动身份门禁实现 revision：`081b97bcdc8d8953c0e4d32587688a879945597d`
- 校正 config SHA-256：`b5d82f55040f47a627d2793cf8af0f9dde651c19ad04a8dda016245ffcc946bb`

研究问题分为两个层次：

1. 科学问题：在同一 Stage2 Option B 训练人口、同一 test population 与同一 G 2×2 口径下，
   GPS、Road、Grid 的 encoder 可见性和 Q loss 阶梯如何改变轨迹生成质量；
2. 本记录的工程问题：能否在任何结果汇总前，固定两城市×三 seed、与 cell 无关的 1,024 条
   corrected reevaluation population，并把源数据、算法、配置和生成 revision 封存在不可变清单中。

本次工程通过条件为：六份清单覆盖精确 city×seed Cartesian product；每份精确 1,024 个排序且
唯一的 test `traj_id`；选择不依赖 cell；八个 HTP lens stratum 按 Hamilton 法分配；相同输入重放
逐字节一致；正式重评加载器全部接受；生成器来自 clean 且锁定的 Git/config revision；完整测试
集通过；同一 reevaluation root 同时只能有一个批处理，且父进程异常退出时运行中的子进程继续
持锁。任一条件失败则不得启动 corrected reevaluation。

本文没有 controlled 或 confirmation Stage4 结果，因此不形成 GPS/Road/Grid 优劣结论。

## 2. 数据、样本单元、切分与泄漏控制

样本物理单元是一条完整轨迹，不切窗口。轨迹含 20–200 个 observation。Porto 与 Chengdu
分别使用冻结的 aligned train/test split；train 与 test `traj_id` 零交集。GPS、Road、Grid、
timestamp、percent 和 dxy 都以 `traj_id + point index` 对齐，不按未校验行号 join。

| 城市 | train population | test population | aligned test root |
|---|---:|---:|---|
| Porto | 537,993 | 59,778 | `/workspace/guozuyu/HTP-main/reproduction/runs/multiview_vq/porto/data` |
| Chengdu | 542,592 | 60,288 | `/workspace/guozuyu/HTP-staged-attribution-artifacts/aligned/chengdu` |

校正重评选择只读取 `test_traj_ids.npy` 和 `test_offsets.npy`。offset 必须一维、整数、从 0
开始、严格递增，且长度为 `len(traj_ids)+1`；ID 必须一维、整数、非空、唯一。源 ID 的文件
顺序不要求升序，因为该顺序与 offset 行严格对齐。清单同时记录：

- 两个源文件的绝对路径、字节数和 SHA-256；
- 源 ID 原始顺序 SHA-256；
- 源 ID 排序后集合 SHA-256；
- population、最短和最长 observation 数。

Stage2 训练集合与本 test 子集完全分离。Option B 从每城市 train population 固定选 64,000 条，
而 corrected reevaluation 只从 test population 选 1,024 条。模型训练 seed、Stage2 subset seed
和 reevaluation seed 的职责分离；cell 不能参与选择，因此 E/Q 对比共享精确人口。

## 3. 输入、视图、表示、token 与解码

- GPS 输入为按城市 MBR 归一化的 `(lon,lat)`；时间为 Unix seconds；
- Grid 是 local-equirectangular 100 m cell ID；
- Road 是每个 GPS observation 对应的 map-matched road ID；
- 三个 adapter 始终实例化且可训练：GPS `Linear(2,64)`，Road
  `Embedding(128)+Linear`，Grid `Embedding(64)`；各自 LayerNorm；
- 禁用视图由 learned null vector 替换；active 表示求和后除以 `sqrt(active_count)`；
- encoder 三次 stride-2，输出事件数为 `ceil(T/8)`；事件 `j` 锚定
  `[8j,min(8j+8,T))`，但卷积感受野不是严格互斥八点块；
- residual quantizer 为四层 RQ，固定词表 `[32,64,128,256]`，完整事件 token 为
  `<a_i><b_j><c_k><d_l>`；
- sampling type 是三次下采样长度奇偶产生的三位类型 `<t_0>...<t_7>`；
- Stage1 decoder 输出逐 observation standardized percent increment 与 dxy；反标准化后 percent
  绝对值累积并 clip 到 `[0,.9999999]`，沿 oracle route 插值，再加 dxy 得到 GPS；
- Stage3 使用 Qwen3-1.7B + LoRA 生成 SID/sampling；greedy，`max_new_tokens=256`；
- Stage4 在同一 ID population 上执行 G00/G10/G01/G11 四路 decode，并对失败保留 penalty。

sampling view 对原 observation 以 `.65` Bernoulli keep mask 删除点，强制首尾；GPS 按原 time
index 线性插值恢复，Road/Grid 最近邻恢复。native 与 sampled view 的 target、轨迹边界和 latent
event 数不变；consistency 在同 `traj_id/event index` 的量化前 latent 上计算。

## 4. 损失、初始化与训练配置

Stage1 总损失保持：

```text
L_base = MSE(percent_hat, percent) + MSE(dxy_hat, dxy) + L_RQ
L_total = L_base
        + λ_usage L_soft_marginal_usage
        + λ_inv (1 - cosine(z,z_sampled))
        + λ_var L_variance
        + λ_cov L_offdiag_covariance
        + λ_aux mean(CE(speed), CE(turn), CE(road_transition), CE(grid_transition))
```

Q0 全部附加权重为 0；Q1 usage `.05`；Q2 再加入 invariance `.05`、variance `.05`、
covariance `.005`；Q3 再加入 auxiliary `.10`。每个 city/seed/cell 从同 seed 随机初始化，
不加载旧 checkpoint，全部模块可训练。controlled Stage1 为全量 train/test、20 epochs、batch
512，AdamW `lr=1e-4, weight_decay=1e-4`，gradient clip 1.0，CUDA BF16；10 epoch linear
warmup 后 cosine。

正式选择的 Stage2 Option B 为：

- Porto/Chengdu 各固定 64,000 条 train trajectory；
- 抽样算法 `sha256_rank_fixed_city_v1`；subset seed `20260826`；
- 2 epochs、microbatch 8、gradient accumulation 2、有效 batch 16、8,000 optimizer steps；
- Qwen3-1.7B LoRA `r=16, alpha=32, dropout=.05`；target 为
  `q/k/v/o_proj, gate/up/down_proj`，并训练新协议 token embedding；
- AdamW `lr=5e-4, weight_decay=1e-4`，warmup ratio `.05`，linear scheduler，clip 1.0，
  BF16 与 gradient checkpointing。

Option B 冻结证据：

| 证据 | SHA-256 |
|---|---|
| Stage2 subset index | `8673fd0992d5f41af0ba0d810db6770edc8e4f1624d6a76cb984480fa5524bbd` |
| Porto manifest | `026720480bfed13f9f98aa4f34611a3a0ce06b98af03e712ca60ef3f13153942` |
| Chengdu manifest | `ba7d822c6ec0dbfe46f88d5653b127017f62918b92c81090993a2e8251acf8e4` |
| Porto selected train IDs | `a83b142ec63dfe8cbad1b178da16b744224bf32e91a3fec3a2e2457c4041715a` |
| Chengdu selected train IDs | `003a6a0be6814ed9b4cdf527eab4f4d3e389d079cbae3cb9a274f82c9757d61b` |

Option B 是首次正式结果前、因 24 GiB Stage2 16×1 OOM 而冻结的资源协议修正。所有 cell/seed
共享该修正，因此仍可做 E/Q 相对归因；最终结论必须限定为该 64k/城 Stage2 训练预算，不能
宣称等价于全量 Stage2。

## 5. 固定分层 corrected reevaluation 方法

对每条 test trajectory，以 `length = diff(test_offsets)` 计算三位 lens type：

```text
b0 = T mod 2
T1 = ceil(T/2)
b1 = T1 mod 2
T2 = ceil(T1/2)
b2 = T2 mod 2
stratum = b0 b1 b2
```

八个注册 stratum 为 `000` 到 `111`，缺任一 stratum 即失败。对城市总体各 stratum 人口
`N_s` 和目标 `n=1024`，先计算 `n*N_s/N`，取 floor，再按最大小数余数、stratum lexical
顺序分配剩余额度（Hamilton apportionment）。每个 stratum 内对 trajectory 使用：

```text
SHA256(f"{seed}:{stratum}:{traj_id}")
```

升序排名，按 quota 取前若干；并列时按 `traj_id`。最终 ID 全局升序保存。算法 ID 为
`lens3_hamilton_sha256_v1`。相同 city/seed 的所有 cell 使用同一清单；模型输出、metric 或
cell ID 均不参与选择。

每份清单还封存生成 config artifact 和生成器 Git worktree/revision/dirty 状态。正式命令要求
clean worktree，并显式核对完整 revision 与 config SHA。已有同路径且逐字节相同时幂等返回；
已有不同内容、非普通文件、残留临时文件或并发内容不同时失败关闭，绝不覆盖。

## 6. 正式冻结产物与哈希

根目录：

`/home/guozuyu/HTP-staged-attribution-artifacts/staged_attribution/gps_road_grid_staged_attribution_v1_20260826/provenance/fixed_stratified_reevaluation_subsets_a2991e1`

index SHA-256：`58a130de8f7672ad0de175050b976af5db5a8f7aa2b8e1432cae32c3e771e849`

| 城市 | seed | selected ID SHA-256 | manifest SHA-256 |
|---|---:|---|---|
| Porto | 42 | `804299e36a2ea81f186210beb5a4ef722600ff6f554b77e0ddc43dfb226f0ec7` | `35b2c8a28fd9a763473e86069e731712fcb3173e4f72afe548211195b13fbc0b` |
| Porto | 314159 | `2357a5d1356a8ff3c52c01401e392618d6bb65a5e5c40e46b5ae32d512be8b2c` | `c9aa5559c0bb278a558a3bc5ef23385d80ae8a1f9669295b45621a73fa756f49` |
| Porto | 20260819 | `4415a172c9f898d0689aa7fa8de73a49f740de2324aa2e63f8aeb40e75a4052c` | `85c420fbc96f6b0bee976f82b1aa70eed166cc065505bf679a56cfe0c50bc012` |
| Chengdu | 42 | `28838eb8d80ff47abc6880d516b443e548dd5b6832856e1295ff13d736fd9281` | `4de14f0cc4641e7d347368b872127871e520f964d3ea6a3c755eb998d1b2f638` |
| Chengdu | 314159 | `b7658d7f3b776f6d517f77edadf79a4586383efdc91ba560d6caab149a83d66b` | `a110e52374290af9bf8afd96ca39e1c0d5ebf76a857586eeadc7db907af22cc6` |
| Chengdu | 20260819 | `6bc4473df9870618da14c410b599c01bdcabd7ec59750478875b95cfc9c69f10` | `bc2c90752ead616443c1e7b2f45d31ad3465341eb582c234ec1e1fcdb4e53974` |

每份 selected population 均为 1,024；`selection_depends_on_cell=false`。幂等重放后 index SHA
不变，`load_stratified_manifest` 对 6/6 清单成功，且每份精确读回 1,024 条。

## 7. 真实数据兼容修正与失败记录

首次真实数据验证曾失败，原因是早期实现错误要求 aligned `test_traj_ids.npy` 自身升序。只读
审计确认 Porto 和 Chengdu 源 ID 分别为 59,778 与 60,288 个，均为 `int64`、唯一；offset
从 0 开始并严格递增，但 ID 的文件顺序不是升序。该顺序与每行 offset/trajectory 一一对应，
属于合法权威顺序。

修正没有重排源文件，也没有改变 ID 与 length 的配对。生成器改为接受任意唯一源顺序；算法
仍对 `(traj_id,length)` 对顺序无关；清单分别记录原始顺序 hash 与排序后集合 hash。新增测试将
同一 `(ID,length)` 人口整体逆序，证明三个 seed 的 selected ID 完全相同，同时原始顺序 hash
改变而总体集合 hash 不变。

这是输入契约修正，不是科学变量。没有删除或覆盖首次失败的临时验证证据，也没有触碰正式
attempt。

## 8. 方法变更表

| 方法组成 | 正式 Option B baseline | 本次冻结/校正 | 是否变化 |
|---|---|---|---|
| 样本物理单元 | 完整轨迹 | 与 baseline 相同 | 否 |
| 视图构造与覆盖范围 | pointwise GPS/Road/Grid；sampling view `.65` keep 后恢复 | 与 baseline 相同 | 否 |
| 输入特征与单位 | MBR-normalized GPS、Road/Grid ID、Unix seconds | 与 baseline 相同 | 否 |
| encoder/downsampling | 三次 stride-2，`ceil(T/8)` events | 与 baseline 相同 | 否 |
| token 时间/空间支持 | event anchor `[8j,min(8j+8,T))` | 与 baseline 相同 | 否 |
| quantizer/codebook | 四层 RQ `[32,64,128,256]` | 与 baseline 相同 | 否 |
| decoder 与输出时间锚点 | oracle route + percent/dxy，逐 observation GPS | 与 baseline 相同 | 否 |
| 初始化与冻结模块 | Stage1 从头训练；Stage2 Qwen3-1.7B LoRA | 与 baseline 相同 | 否 |
| reconstruction/VQ loss | percent+dxy MSE + RQ | 与 baseline 相同 | 否 |
| 新增 loss 与权重 | Q0–Q3 预注册权重 | 与 baseline 相同 | 否 |
| 训练数据与超参数 | Option B 64k/城、8×2、2 epochs | 与 baseline 相同 | 否 |
| controlled Stage3 population | 既定 1,024 test IDs | 显式冻结为 lens-stratified city×seed manifests | 只补齐可复现证据 |
| 源 ID 顺序契约 | aligned 行顺序为权威 | 允许任意唯一顺序并封存双 hash | 输入验证修正 |
| 评估对齐与指标 | 同 ID G00/G10/G01/G11、paired penalty-retaining metrics | 与 baseline 相同 | 否 |
| provenance | 纯函数存在，无正式清单生成入口 | clean revision/config gate + immutable 6-manifest index | 是（非科学变量） |

## 9. 自动化验证与复现命令

相邻验证：

```bash
cd /workspace/guozuyu/HTP-staged-attribution-compact-adapter
uv run --frozen pytest -q \
  stage_1/tests/staged_attribution/test_freeze_stratified_subsets.py \
  stage_1/tests/staged_attribution/test_stratified_subset.py \
  stage_1/tests/staged_attribution/test_run_stratified_reevaluation_batch.py \
  stage_1/tests/staged_attribution/test_reevaluation_evidence.py
```

结果：`17 passed in 3.16s`。

队列锁专项验证：

```bash
uv run --frozen pytest -q \
  stage_1/tests/staged_attribution/test_run_stratified_reevaluation_batch.py
```

结果：`4 passed in 0.67s`，覆盖并发拒绝、锁描述符传给子进程、父引用关闭后锁仍存活、最后
一个子进程退出后释放，以及完成队列幂等重放。

完整回归：

```bash
cd /workspace/guozuyu/HTP-staged-attribution-compact-adapter
uv run --frozen pytest -q
```

校正重评来源/启动身份门禁加固后的最新结果：`287 passed, 23 warnings in 19.60s`。warnings
为既有 PyTorch Transformer nested-tensor 提示。

正式冻结命令：

```bash
cd /workspace/guozuyu/HTP-staged-attribution-compact-adapter
uv run --frozen python -m staged_attribution.freeze_stratified_subsets \
  --config stage_1/configs/staged_attribution/experiment.json \
  --output-dir /home/guozuyu/HTP-staged-attribution-artifacts/staged_attribution/gps_road_grid_staged_attribution_v1_20260826/provenance/fixed_stratified_reevaluation_subsets_a2991e1 \
  --expected-revision a2991e14d509a96830d21c5cfa8f2ae9d4fa1967 \
  --expected-config-sha256 b5d82f55040f47a627d2793cf8af0f9dde651c19ad04a8dda016245ffcc946bb \
  --require-clean
```

该命令已执行两次；第二次幂等返回，没有替换现有内容。

## 10. 正式队列边界、结果边界与下一步

本次提交和 provenance 冻结没有启动、暂停、终止或重复派发任何正式训练。正式 watcher 继续
锁定 `6ac87be...`/`39a33c9b...`，候选 revision 未热替换到运行中的 `run_cell`。旧 attempt
保持不可变；六份清单位于新的 commit 命名目录。

2026-09-01 15:24 CST 的只读正式快照：watcher uv/Python（PID 3811733/3811770）、锁
`pid=3811770` 和四个精确 `run_cell` 均存活；正式 branch/HEAD/config/clean 口径未漂移。
Porto seed42 `e6_q2/attempt_001` 已完成 Stage1 并推进到 Stage2 727/8000；`e6_q3` 仍在
Stage1。Porto seed314159 `e0_q0`、`e1_q0` 分别为 Stage2 5439/8000、4812/8000；四份
当前日志均未发现 Traceback 或 CUDA OOM。四卡 free/utilization 分别为
2453/100%、8665/15%、1772/100%、1022 MiB/100%，没有 GPU 同时满足 free>=16000 MiB 且
utilization<=20%，因此未启动任何新队列。

同一时点对完整 60-job contract 执行 watcher 自带 `--validate-only`：
`contract_valid=true`、`resource_option=B`、`completed_count=8`、`pending_count=52`、
`safe_gpus=[]`。全 attempt 审计另确认 4 个旧 revision `730c2cc...` 的 seed42 E0–E3
`attempt_001` 为历史 OOM failure；相同 run key 的正式 `6ac87be.../attempt_002` 已完成 Stage4，
没有覆盖或删除旧失败。当前 16 个 attempt 对应 12 个唯一 run key：8 个正式完成、4 个正式
运行、4 个历史失败。

只有完整 60 个 core source attempt 通过 immutable marker/manifest 审计后，才能执行校正重评。
早期仅检查 source/subset 存在性的命令模板自 `081b97b...` 起已经失效；新入口强制要求完整
evaluator revision、config SHA、completed-source audit 路径与 audit SHA，不能省略或回退。
第 13 节给出新的两阶段审计/启动协议。

批处理 preflight 会在缺任一 source attempt、任一 city×seed 清单或任一身份/哈希证据时拒绝执行；持久 flock
保证同一 reevaluation root 只有一条队列。锁描述符通过 `pass_fds` 由所有运行子进程继承，
因此父进程异常结束后，第二条队列仍会在最后一个子进程退出前被拒绝。GPU 不安全时只等待，
不抢占其他用户进程。之后才依次执行 core 汇总、
population-aware 候选选择、adaptive cell、final selection、full confirmation 和最终报告。

Notion 同步状态为 `pending_explicit_destination_authorization`。此前外部写入因目标归属未验证而
被安全审查拒绝；本文作为仓库内版本化权威记录保留完整事实，不绕过门禁。最终结论必须来自
confirmation，且 Stage1 gate 全通过、GPS/Road/Grid/validity 无显著劣化、至少一个注册域显著
改善；否则明确报告不支持或负结果。

## 11. 已完成 source 的全链路证据审计与人口差异

为避免等 60 个 core 全部完成后才发现源证据不可复评，在不暂停或重复派发正式队列的前提下，
新增 `staged_attribution.audit_completed_sources`。实现提交为
`de3935db22e2c8e59ebebcab42bbb18c7522fa5e`，随后由真实正式产物发现“原始 Stage4 人口必须等于
未来固定分层人口”这一过强假设；语义修正提交为
`5aff73140ed1f66f475c3b123cd91c82dae04866`。修正后的门禁分别验证：

- source manifest 的实验身份、clean/full revision 与完成状态；
- Stage1 model、test tokens、完整 LLM test parquet、Stage2 adapter manifest 和 adapter tree 的
  immutable marker、大小与 SHA256；
- aligned test ID/offset、token `lens_type` 公式和完整 LLM test population 一致；
- fixed city×seed manifest 的算法、大小、排序唯一性、配额和 traj-ID SHA256；
- 原始 Stage4 的 G00/G10/G01/G11 使用完全相同且无重复的 1,024 条人口；
- 原始 Stage4 人口大小等于注册 corrected reevaluation 大小，但不要求其 ID 与未来 fixed subset
  相同；二者不同会被显式记录为 `corrected_stage3_stage4_reevaluation_required=true`；
- 精确 completed count、候选 config SHA、审计生成器 revision 与 clean worktree；
- 输出采用 exclusive temporary file、`fsync` 和 hard-link，已有不同内容时拒绝替换。

聚焦测试为 `7 passed in 2.91s`，该阶段仓库完整回归为 `275 passed, 23 warnings in 18.49s`。
正式审计命令如下：

```bash
cd /workspace/guozuyu/HTP-staged-attribution-compact-adapter
uv run --frozen python -m staged_attribution.audit_completed_sources \
  --config stage_1/configs/staged_attribution/experiment.json \
  --source-root /home/guozuyu/HTP-staged-attribution-artifacts/staged_attribution/gps_road_grid_staged_attribution_v1_20260826 \
  --subset-root /home/guozuyu/HTP-staged-attribution-artifacts/staged_attribution/gps_road_grid_staged_attribution_v1_20260826/provenance/fixed_stratified_reevaluation_subsets_a2991e1 \
  --output /home/guozuyu/HTP-staged-attribution-artifacts/staged_attribution/gps_road_grid_staged_attribution_v1_20260826/provenance/completed_source_audit_5aff731/index.json \
  --expected-source-revision 6ac87be094c9c1ce4783f10291a339f04623449b \
  --expected-completed-count 8 \
  --expected-generator-revision 5aff73140ed1f66f475c3b123cd91c82dae04866 \
  --expected-config-sha256 b5d82f55040f47a627d2793cf8af0f9dde651c19ad04a8dda016245ffcc946bb \
  --require-clean
```

不可变输出 SHA256 为
`6905e0f99b0f193bd4a4146ded59d99d1fc1dc61393a29e86635bb21e4a2bf6d`；完全相同命令第二次执行后
SHA256 不变。审计覆盖当前 8 个正式完成 source（Porto/seed42 的 E0Q0–E5Q0、E6Q0、E6Q1），
8/8 均为 source revision `6ac87be...`，每个 adapter tree 6 个文件，完整测试人口 59,778，
token/LLM/aligned ID 哈希均为
`5e15c827a074a611a9a7bcddd87e85fd8a3754314462b6576e8845e92b0b33aa`。

8/8 原始 Stage4 的四个 G 条件内部人口完全一致，原始人口哈希均为
`02cb240090d83160845eb2f90583d6e5bf881bced2f72ba2d9410c8478e2b705`；固定分层人口哈希为
`804299e36a2ea81f186210beb5a4ef722600ff6f554b77e0ddc43dfb226f0ec7`，因此 8/8 均明确标记必须
执行 corrected Stage3→4 reevaluation。这不构成 GPS/Road/Grid 科学结论，也不表示原始 Stage4
损坏；它证明原始 1,024 条人口不能冒充预注册分层人口，后续跨 cell/city/seed 结论必须来自
固定清单上的纠正版复评。审计同时列出尚缺 52 个 completed source，未据此提前启动复评队列。

第一次正式执行在输出前因上述人口哈希不同而拒绝，未留下 provenance 文件；该失败触发了
`5aff731...` 的语义修正并新增“同人口大小但不同 ID 应记录而非拒绝”以及“人口大小漂移必须
拒绝”的回归测试。整个过程没有修改任何正式 attempt、watcher、GPU 进程或历史产物。

## 12. 下游分析、候选冻结与确认报告的不可覆盖证据协议

在审计 core 汇总、复评与确认流水线时发现，`report_results`、`report_stratified_results`、
`select_candidates` 和 `report_confirmation_results` 虽然先写 `.tmp`，但随后使用 `replace`
静默覆盖同名目标。这会使
同一路径上的旧 `analysis.json`、`analysis.md` 或 frozen selection 被后来一次运行改写，不能满足
实验结论的不可变证据要求。提交
`af9d03e2ffc6321a7c2751e2daebc99ee4f0598f`（`fix(eval): preserve immutable analysis evidence`）
新增共享 `immutable_artifacts` 协议，并将三个入口全部接入。

后续完整写入面审计发现 core evidence index 的 `report_results.write_report` 仍有同类问题；提交
`e39f5ebebc5537af1f85bd29d02ec7483cf183a2`（`fix(eval): preserve immutable core report`）将第四个
顶层证据入口接入同一协议。其余 `replace` 位于受 attempt 非空/完成门禁保护的阶段产物、运行中
状态清单，或已有“内容不同拒绝”的冻结清单，不作为可反复覆盖的顶层结论输出。

四个入口共用的协议具体为：

- 对一组相关产物先逐个预检；任一已有目标内容不同、不是普通文件或为符号链接时，在创建其他
  同组目标前拒绝；
- 完全相同内容视为幂等重放；若同组一个目标完全相同而另一个缺失，只补齐缺失目标；
- 新目标先写 exclusive `.tmp`，`flush` + `fsync` 后以 hard-link 原子发布，并 `fsync` 父目录；
- 陈旧 `.tmp` 明确拒绝，不猜测或删除；并发发布若内容不同明确拒绝；
- core/corrected/confirmation analysis 的 JSON/Markdown 分别作为相关证据集整体预检；candidate
  selection 使用同一不可变单文件协议；
- 未改变 bootstrap、候选 eligibility、adaptive cell、confirmation gate、GPS/Road/Grid 指标、人口或
  任何训练超参数，属于 provenance/落盘语义加固，不是科学变量。

聚焦验证命令：

```bash
cd /workspace/guozuyu/HTP-staged-attribution-compact-adapter
uv run --frozen pytest -q \
  stage_1/tests/staged_attribution/test_immutable_artifacts.py \
  stage_1/tests/staged_attribution/test_report_stratified_results.py \
  stage_1/tests/staged_attribution/test_report_confirmation_results.py \
  stage_1/tests/staged_attribution/test_report_results.py
```

结果为 `18 passed in 7.91s`。覆盖正常首次发布、完全相同重放、缺失同伴补齐、任一同伴内容冲突时
全组预检拒绝且尚未创建的同伴保持不存在、陈旧临时文件、符号链接，以及三个实际调用入口的
不同内容拒绝与旧字节保持不变。core report 的追加聚焦命令
`uv run --frozen pytest -q stage_1/tests/staged_attribution/test_report_results.py` 为
`7 passed in 3.00s`。最终完整命令 `uv run --frozen pytest -q` 结果为
`281 passed, 23 warnings in 18.80s`；warnings 仍为既有 Transformer nested-tensor 提示。

本提交后候选 worktree 为 clean。它没有创建真实 core/corrected analysis、selection 或 confirmation
产物，因为正式 core source 仍未齐备；也没有修改正式 watcher、`run_cell`、attempt 或 GPU
派发。该加固只保证未来结论一旦写入命名目录就不能被同路径不同内容静默替换，不能替代尚待
执行的固定人口复评与 full confirmation。

## 13. core/adaptive 来源审计与校正重评启动身份门禁

提交 `081b97bcdc8d8953c0e4d32587688a879945597d`（`fix(eval): bind corrected launch
evidence`）封闭了 completed source audit 与 corrected reevaluation 之间的身份空隙。该变更不修改
GPS/Road/Grid 输入、模型、loss、训练人口、固定 test population、bootstrap 或选择规则，属于
provenance 与启动语义加固，不是科学变量。

### 13.1 completed-source audit 的 phase population 契约

`audit_completed_sources` 新增 `--cells`。不传时仍严格生成 10-cell、2-city、3-seed 的 60-job
core contract；传入显式 cell 集时，生成对应 adaptive phase 的精确 Cartesian contract。审计新增：

- `requested_cell_ids`：规范化后的精确 cell 集；
- `requested_job_count`：该 phase 的精确 job 数；
- `requested_run_key_sha256`：排序 run-key 集合的换行规范 SHA-256；
- `all_requested_sources_completed`：该 phase 是否全部完成；
- `formal_core_contract`：是否为默认 60-job core；adaptive audit 不再冒充 formal core。

每一行继续封存 source attempt/revision、immutable marker 与 adapter tree、fixed subset
path/SHA、aligned/token/LLM population audit 和原始 Stage4 population audit。core 与 adaptive 必须
分别生成新的不可变 audit artifact；不得用 partial audit、另一 cell 集或另一 phase 的 audit 启动。

### 13.2 batch 与单次 runner 的 fail-closed 身份链

batch 在创建 reevaluation root、锁或 attempt 前同时验证：

1. 实际加载模块所在 worktree clean，且 HEAD 等于显式 40 位 evaluator revision；
2. config 实际 SHA 等于显式 SHA，audit 内 config path/SHA 同时匹配；
3. audit 是普通非符号链接文件，实际 SHA 等于显式 audit SHA；
4. audit schema/kind/experiment/budget/source root、生成器 clean revision、fixed subset index
   path/SHA 全部匹配；
5. requested cell、job count、run-key hash、missing=0、completed count 与当前 queue 完全一致；
6. audit 行无重复、无缺失，并逐 job 匹配 cell/city/seed/budget、source attempt、source revision、
   fixed subset path/SHA。

batch 将 evaluator revision、config SHA、逐 job source revision、audit path/SHA 全部传给单次 runner。
单次 runner 在创建 `run_dir` 前独立重验 config、audit 文件及 audit 中该 run-key 的唯一行，并把
完整身份写入 `reevaluation_manifest.json`；pipeline 再把 expected source revision 传入 immutable
source evidence resolver。即使绕过 batch 直接调用，也不能替换 source attempt 或 subset。

已有 completed reevaluation 仅在 manifest 的 experiment/cell/city/seed/budget、config SHA、source
attempt/revision、subset、audit path/SHA 和 evaluator Git identity 全部匹配时复用。身份不匹配的旧
attempt 保持不可变并跳过；新运行使用下一编号，例如 `attempt_002`，不覆盖旧证据。

### 13.3 验证证据

聚焦命令覆盖四个入口：

```bash
cd /workspace/guozuyu/HTP-staged-attribution-compact-adapter
PYTHONPATH=stage_1/src uv run --frozen pytest -q \
  stage_1/tests/staged_attribution/test_audit_completed_sources.py \
  stage_1/tests/staged_attribution/test_run_stratified_reevaluation_batch.py \
  stage_1/tests/staged_attribution/test_run_stratified_reevaluation.py \
  stage_1/tests/staged_attribution/test_stratified_reevaluation_pipeline.py
```

结果：`19 passed in 4.01s`。负向测试证明错误 evaluator revision、config SHA、audit SHA、cell
population、source attempt substitution 都在创建目标 attempt 前失败且零子进程；另验证来源
revision 确实传到 resolver，以及身份不同的旧 completed attempt 产生新的 `attempt_002`。

完整 staged-attribution 集：`268 passed, 23 warnings in 18.27s`；完整 `stage_1/tests`：
`287 passed, 23 warnings in 19.60s`。config SHA 仍为
`b5d82f55040f47a627d2793cf8af0f9dde651c19ad04a8dda016245ffcc946bb`。

### 13.4 正式 core 启动模板（尚未执行）

以下模板只能在 60/60 source 完成后使用。`<EVALUATION_REVISION>` 必须替换为当时 clean 候选
worktree 的完整 HEAD；`<SOURCE_AUDIT_PATH>` 是由同一 revision 生成的新不可变 60-job audit；
`<SOURCE_AUDIT_SHA256>` 是该文件的实际 SHA。任何占位符、旧 partial audit 或当前 8-source audit
都会被拒绝。

```bash
cd /workspace/guozuyu/HTP-staged-attribution-compact-adapter
uv run --frozen python -m staged_attribution.audit_completed_sources \
  --config stage_1/configs/staged_attribution/experiment.json \
  --source-root /home/guozuyu/HTP-staged-attribution-artifacts/staged_attribution/gps_road_grid_staged_attribution_v1_20260826 \
  --subset-root /home/guozuyu/HTP-staged-attribution-artifacts/staged_attribution/gps_road_grid_staged_attribution_v1_20260826/provenance/fixed_stratified_reevaluation_subsets_a2991e1 \
  --output <SOURCE_AUDIT_PATH> \
  --expected-source-revision 6ac87be094c9c1ce4783f10291a339f04623449b \
  --expected-completed-count 60 \
  --expected-generator-revision <EVALUATION_REVISION> \
  --expected-config-sha256 b5d82f55040f47a627d2793cf8af0f9dde651c19ad04a8dda016245ffcc946bb \
  --require-clean

uv run --frozen python -m staged_attribution.run_stratified_reevaluation_batch \
  --config stage_1/configs/staged_attribution/experiment.json \
  --source-root /home/guozuyu/HTP-staged-attribution-artifacts/staged_attribution/gps_road_grid_staged_attribution_v1_20260826 \
  --reevaluation-root /home/guozuyu/HTP-staged-attribution-artifacts/staged_attribution/gps_road_grid_staged_attribution_v1_20260826_corrected_stratified_<REVISION_SHORT> \
  --subset-dir /home/guozuyu/HTP-staged-attribution-artifacts/staged_attribution/gps_road_grid_staged_attribution_v1_20260826/provenance/fixed_stratified_reevaluation_subsets_a2991e1 \
  --source-audit <SOURCE_AUDIT_PATH> \
  --expected-revision <EVALUATION_REVISION> \
  --expected-config-sha256 b5d82f55040f47a627d2793cf8af0f9dde651c19ad04a8dda016245ffcc946bb \
  --expected-source-audit-sha256 <SOURCE_AUDIT_SHA256> \
  --maximum-parallel 4 \
  --wait-for-gpu \
  --poll-seconds 60 \
  --minimum-free-mib 16000 \
  --maximum-utilization 20
```

adaptive phase 使用相同两阶段协议，但 audit 与 batch 都必须传同一显式 `--cells <CELL_A>
<CELL_B> ...`，`--expected-completed-count` 必须等于 `len(cells) × 2 cities × 3 seeds`。选定
cell 与其 source revision 只能来自已冻结的 core selection/adaptive plan；不得在观察结果后另换
cell 集。当前未生成可启动的 60/60 full audit、未启动 corrected/adaptive queue，也未修改正式
watcher 或任何 attempt；第 13.5 节仅记录 fail-closed 的 8/60 partial audit。

### 13.5 新 schema 的真实 partial audit（不可用于启动）

在正式 watcher `--validate-only` 返回 `contract_valid=true`、`completed_count=8`、
`pending_count=52`、`safe_gpus=[]` 后，以 clean revision
`bc15a72b5e3635a88b78886dec6cae7e4a4dfdea` 对真实正式产物执行新 schema 审计。输出为：

`/home/guozuyu/HTP-staged-attribution-artifacts/staged_attribution/gps_road_grid_staged_attribution_v1_20260826/provenance/completed_source_audit_bc15a72_partial_8/index.json`

文件 SHA-256：`f58422c678dc53e4b9da00ef3d49868ad6583d95c63086ef8ba03c4780539c70`。
完全相同命令第二次执行后 SHA 不变，候选 worktree 仍 clean。审计记录：

- `requested_cell_ids` 为完整 10 个 formal core cells；
- `requested_job_count=60`；
- `requested_run_key_sha256=196d96871d40c1c05835aa82a933e3c5d050d85ae0080d3bd97cff8cfb72ed68`；
- `formal_core_contract=true`、`formal_core_job_count=60`；
- `audited_completed_source_count=8`、`missing_completed_source_count=52`；
- `all_requested_sources_completed=false`、`all_formal_core_sources_completed=false`；
- generator revision 为 `bc15a72...`，8 行 source revision 均为正式 `6ac87be...`。

该 artifact 的用途是验证真实源数据上的新 population/identity schema，并保留当前阶段证据。它
明确为 partial，batch 会因 `all_requested_sources_completed=false`、missing=52 和 completed count
不等于 60 而拒绝，不能用于启动 corrected reevaluation。60/60 完成后必须在当时 clean HEAD 上
生成新的不可变 full audit；不得修改、删除、移动或覆盖本 partial audit。
