# GPS/Road/Grid 分阶段归因：Stage2 compact-adapter 存储协议

## 1. 状态与适用边界

- 实验 ID：`gps_road_grid_staged_attribution_v1_20260826`
- 记录日期：2026-08-31（Asia/Shanghai）
- 修复 worktree：`/workspace/guozuyu/HTP-staged-attribution-compact-adapter`
- 修复分支：`fix/staged-attribution-compact-adapter`
- 基线 revision：`6ac87be094c9c1ce4783f10291a339f04623449b`
- 首次候选提交：`1870926cc1abf170d54ce930e1ca455fe4b61251`
- 精确 extension-row 提交：`aa0cbc580eb5b33880d0577665fa0d09459b8578`
- 状态：`verified / not_activated_on_formal_queue`

本协议只改变每个 Stage2 PEFT adapter 的落盘表示，不改变 GPS/Road/Grid 数据、Stage1、
Stage2 训练人口、模型初始化、LoRA、可训练 token delta、loss、optimizer、scheduler、
microbatch、梯度累积、Stage3 生成参数或 Stage4 评估。POI 持续排除。

正式 Option B 队列仍锁定在基线 revision；本文不授权暂停、重启或切换正在运行的 watcher，
也不修改任何既有 attempt。激活必须形成单独的 Git revision，并在实验记录中区分旧的完整
adapter 与后续 compact adapter。

## 2. 问题与精确归因

Qwen3-1.7B + PEFT 在保存当前 Stage2 adapter 时，同时写入 LoRA、两份可训练 token delta，
以及两份完整的 resized base embedding。真实 Porto 与 Chengdu adapter 中，以下两份 base
embedding 的 shape、dtype 和每个字节均相同：

1. `base_model.model.lm_head.token_adapter.base_layer.weight`
2. `base_model.model.model.embed_tokens.token_adapter.base_layer.weight`

前 `base_model.config.vocab_size` 行来自冻结基础模型；只有 tokenizer 新增行需要额外保存。
LoRA 和两份 `trainable_tokens_delta` 承载训练更新，必须原样保留。

### 2.1 被否证的 seed-only 方案

首个候选提交尝试仅记录训练 seed，并在 Stage3 重新执行随机 embedding 初始化。Porto smoke
adapter 得到 logits 全等，但 Chengdu 历史完整 adapter 的加载后 base-embedding SHA 不一致，
生产门禁正确拒绝继续。这证明“seed 相同”不足以覆盖不同历史代码路径中的 RNG 消耗顺序。

该失败只发生在 `/tmp` 副本，没有修改正式或历史产物，也没有激活到正式队列。最终协议不再
依赖 seed 重建新增行，而是保存其精确字节。`training_seed` 仅作为 provenance，不参与恢复。

## 3. 最终存储格式与失败关闭门禁

`compact_peft_adapter` 只在满足以下全部条件时省略两份完整 base embedding：

1. adapter model 与 `adapter_config.json` 同时存在；
2. adapter 配置中的基础模型路径与调用方锁定路径一致；
3. 基础模型 config 提供有效的 `vocab_size`；
4. 恰好发现 input/output 两份 `token_adapter.base_layer.weight`；
5. 两份张量 shape、dtype 和每个 4096-row chunk 完全一致；
6. compact adapter 的张量清单只减少这两个名字；
7. 将一份 `[base_vocab_size:]` 新增行精确保存为
   `base_embedding_extension.safetensors`；
8. 原始/compact model、extension、adapter config、基础模型 config、软件版本、训练 seed、
   被省略完整张量 SHA-256 和字节数全部写入 `compact_adapter_manifest.json`。

发布顺序为先发布 manifest，再发布 extension，最后原子替换 model 文件。若中途崩溃，文件
缺失或 SHA 不一致，后续加载会失败；不会把不完整 compact 文件静默当作 legacy adapter。

Stage3 先按既有 seed 加载基础模型、resize embedding 和 PEFT delta，再把 extension 精确回填
到 input/output `token_adapter.base_layer.weight` 的新增行。随后对两份完整张量逐字节计算
SHA-256，并与 manifest 比较；inventory、shape、dtype 或 digest 任一不一致即拒绝生成。
最后重新设置既有 generation seed，使模型加载期间的 RNG 消耗不改变生成口径。无 manifest
的旧 adapter 保持原加载路径。

## 4. 自动化测试

定向测试覆盖：

- 只移除两份完整 base embedding；
- LoRA 与两份 token delta 原样保留；
- extension 只含新增行且被计入总 compact bytes；
- 拒绝不相同的 input/output base embedding；
- 拒绝 compact model 或 extension 篡改；
- legacy adapter 为 no-op；
- 加载后精确回填并验证完整张量 digest；
- Stage3 加载前后重置 generation seed；
- 现有 pipeline、collator、metrics 与生成解析回归。

定向命令：

```bash
cd /workspace/guozuyu/HTP-staged-attribution-compact-adapter
PYTHONPATH=stage_1:stage_1/src uv run --frozen pytest -q \
  stage_1/tests/staged_attribution/test_adapter_storage.py \
  stage_1/tests/staged_attribution/test_generation_parser.py \
  stage_1/tests/staged_attribution/test_pipeline.py \
  stage_1/tests/staged_attribution/test_stage2_collator.py \
  stage_1/tests/staged_attribution/test_stage2_metrics.py
```

结果：`34 passed in 6.59s`。

完整回归命令：

```bash
cd /workspace/guozuyu/HTP-staged-attribution-compact-adapter
PYTHONPATH=stage_1:stage_1/src uv run --frozen pytest -q stage_1/tests
```

结果：在加入评估进度证据后为 `246 passed, 21 warnings in 19.35s`。21 个 warning 均为既有 PyTorch
Transformer nested-tensor 提示。

### 4.1 Stage2 全量评估进度证据

功能提交 `804dd003ede42e3eaa7dacc660e597bcbd97ba28` 不改变 Stage2 数据、模型、
forward、loss、batch size 或指标聚合，只增加原子更新的
`stage2/evaluation_progress.json`。该文件记录 schema/status、总/完成 batch、总/完成样本、
监督 token、耗时、吞吐、ETA、时间戳和失败类型；默认每 100 batches 更新一次。正常结束时
状态为 `completed`，异常时保存最后一个已完成 batch 并标记 `failed`，随后原异常继续抛出。

最终进度文件已加入 Stage2 DAG 的 required artifacts，由 `stage2/complete.json` 记录 size 与
SHA-256。因此它既可在运行中提供可观测性，也会在成功运行后成为不可变实验记录。新增定向
测试覆盖 completed、failed 和 DAG 哈希契约；与既有 `run_cell` 测试合计
`10 passed in 2.05s`。

## 5. 双城市真实 adapter 等价性探针

全部探针均为 CPU-only，只读取权威源并操作独立 `/tmp` 副本。源 adapter 的 SHA-256 在
探针前后相同；没有移动、删除或覆盖任何历史或正式产物。

### 5.1 Porto

- 源：staged-attribution smoke Porto / seed 42 / E0-Q0 / `attempt_004`
- 源 model SHA-256：
  `671b4d0b705745bb6173e652d1bdb3fd2dead858a04ca4ada4dff4d43abc2c03`
- 探针副本：`/tmp/staged_compact_verified_72ec6zbx/adapter`
- 原始：1,583,305,008 bytes
- compact model：250,482,712 bytes
- exact extension：44,081,272 bytes
- compact 总计：294,563,984 bytes
- 节省：1,288,741,024 bytes（81.395626%）
- 完整 base tensor：BF16 `[162698, 2048]`
- 完整 base tensor SHA-256：
  `04121496dda6e94ba68b81329e4ed9b5f7116e04ec33df064ae6940efd55c4f0`
- logits：shape `[1, 17, 162698]`，逐元素全等，最大绝对差 `0.0`

### 5.2 Chengdu

- 源：HTP reproduction Chengdu / full / seed 42 / `saved_lora_model`
- 源 model SHA-256：
  `ad7169cc77ab693cc4be8b4a916631fb5e0883aa3e3dafc7385fc64d71884342`
- 探针副本：`/tmp/staged_compact_verified_u9kh3qlu/adapter`
- 原始：2,295,173,424 bytes
- compact model：725,061,656 bytes
- exact extension：162,726,008 bytes
- compact 总计：887,787,664 bytes
- 节省：1,407,385,760 bytes（61.319365%）
- 完整 base tensor：BF16 `[191664, 2048]`
- 完整 base tensor SHA-256：
  `67dd3788ddc5fc469099e408a9ff5ddbfcdacfbe7ae5951e2698cb89fa211f1d`
- logits：shape `[1, 17, 191664]`，逐元素全等，最大绝对差 `0.0`

两个探针均使用同一含 road/protocol token 的 prompt，input IDs 完全相同；compact 加载均
通过生产路径的 extension 文件 SHA、回填后完整张量 SHA 和 logits 全等验证。

若 60 个 core run 分为 30 个 Porto 与 30 个 Chengdu，且各城市 adapter 形状与上述对应，
仅 adapter 文件部分预计少写 80,883,803,520 bytes（约 75.33 GiB）。该数字不含其它阶段
产物，不作为总实验体积估计。

## 6. 激活门与回滚边界

正式激活前必须同时满足：

1. follow-up 修复提交完成，worktree clean；
2. 完整回归和双城市真实 adapter 全等探针持续通过；
3. 正式配置 SHA-256 保持
   `39a33c9bef9cc06f1bc2cdbbf1c6010ea76dcb455a7855b330905868a1c7604a`；
4. 明确记录正式队列 revision 的切换时点；
5. 不修改四个旧 Porto `attempt_001`，不修改已创建或完成的任何正式 attempt；
6. 不在活跃 `run_cell` 内热替换代码；如要切换 watcher，必须先得到授权并在安全边界执行；
7. 每个 compact attempt 的 Stage2 marker 必须记录 `adapter_storage` manifest；
8. 每个新 attempt 必须原子更新 `evaluation_progress.json`，并在 Stage2 marker 中锁定其终态；
9. Stage3 开始前必须通过 model/config/extension/version/full-tensor 全部校验。

回滚只允许通过新的 Git revision 恢复完整 adapter 保存；不删除、不覆盖已经产生的 compact 或
legacy artifact。两种格式都保留各自 provenance，并由 manifest 是否存在显式区分。

## 7. 正式激活容量门禁与调度边界

2026-08-31 16:15 UTC 的只读快照中，正式 worktree 仍锁定在 `6ac87be...`，正式配置
SHA-256 仍为 `39a33c9b...c7604a`，工作树 clean，4 个 `run_cell` 存活，正式 attempt 仍只有
既定的 4 个 `attempt_002`，失败标记为 0。E0-Q0、E1-Q0、E2-Q0 已完成 8,000/8,000
optimizer steps 并写出 legacy adapter；E3-Q0 为 7,655/8,000。

三个已写 adapter 的 Porto attempt 当前分别占 2,097,798,178、2,097,895,042 和
2,096,909,201 bytes，其中每个 `adapter_model.safetensors` 为 1,583,305,008 bytes。
home filesystem 精确可用 142,705,381,376 bytes。以真实 Porto attempt 为基准，并用双城市
真实 adapter 的大小差估算 Chengdu，若首批 4 个 Porto 保持 legacy，剩余 26 Porto +
30 Chengdu 也全部使用 legacy，仅 controlled core 预计还需 138,782,967,736 bytes。这会把
余量压到约 2–3 GB，且尚未计入 Stage3/4、corrected reevaluation、adaptive 和 confirmation，
因此不能证明完整实验可落盘。

若首批 4 个 attempt 保持不可变、只对剩余 56 个启用本文协议，预计后续 controlled core
写入约 63,054,128,312 bytes，相比 legacy 少写 75,728,839,424 bytes（约 70.52 GiB）。
这是当前可以证明完整后处理仍有容量余量的方案。

修复 revision 的 watcher `--validate-only` 已通过：`contract_valid=true`、resource option B、
config SHA 匹配、当前 completed 0 / pending 60、GPU `safe_gpus=[]`；该检查不取正式锁，也没有
启动任务。正式进程写出 adapter 后仍对完整 Porto test population 59,778 条执行
teacher-forced evaluation（batch size 4，共 14,945 batches），因此 `8000/8000` 不是
`run_cell` terminal 证据。

推荐切换步骤如下，未获授权前不得执行：

1. 仅暂停按命令行和锁识别出的 watcher Python 父进程，不向 4 个 `run_cell` 发信号；
2. 等待 4 个 attempt 自然 terminal，逐项审计 completion/failure marker、revision、config、
   adapter SHA、Stage3/4 与 run size；
3. 将正式分支 fast-forward 到已验证修复 revision，保持配置 SHA 不变并重跑完整回归；
4. 用正式 branch/name/head 再执行一次 `--validate-only`；
5. 释放旧 watcher 后，以新 expected HEAD 启动唯一 watcher，保留所有旧 attempt；
6. 若授权前 watcher 已派发新一批，则新 attempt 同样不可变，并在下一安全边界重新计算容量。

## 8. 2026-09-01 正式队列与容量重基线

2026-09-01 10:56 CST 的只读审计确认：正式 revision、branch、配置 SHA-256 与 clean
状态均未漂移；controlled core 已有 8 个 Stage4 complete、4 个 active、48 个尚未创建，失败
标记为 0。已完成且必须保持不可变的是 Porto/seed 42 的 `e0_q0`–`e3_q0/attempt_002`
以及 `e4_q0`、`e5_q0`、`e6_q0`、`e6_q1/attempt_001`。当前 active 且同样不可变的是
Porto/seed 42 的 `e6_q2`、`e6_q3/attempt_001` 与 Porto/seed 314159 的 `e0_q0`、
`e1_q0/attempt_001`；四者仍由正式 watcher 父进程持有，并处于 Stage1，尚无阶段 terminal
marker。

8 个完成 Porto attempt 的精确总大小为 17,024,332,454 bytes，均值为
2,128,041,556.75 bytes。扣除 legacy Porto adapter 的 1,583,305,008 bytes 后，观测到的
平均非 adapter 体积为 544,736,548.75 bytes。用该非 adapter 均值以及双城市真实 adapter
探针的字节数估算单个最终 run：

- Porto legacy：2,128,041,556.75 bytes；
- Porto compact：839,300,532.75 bytes；
- Chengdu legacy：2,839,909,972.75 bytes；
- Chengdu compact：1,432,524,212.75 bytes。

同一快照附近 home filesystem 的精确可用空间为 132,323,553,280 bytes。若当前 4 个
Porto active 继续按 legacy 自然完成，随后 18 个尚未创建 Porto 与 30 个尚未创建 Chengdu
全部使用 compact，预计从当前状态还需约 66,595,280,367 bytes，预计余量约
65,728,272,913 bytes。若当前 4 个及其余 48 个全部继续 legacy，预计还需约
132,014,191,599 bytes，只余约 309,361,681 bytes。后者无法为 corrected reevaluation、
adaptive、confirmation、临时文件和体积波动提供安全余量。

上述数字是容量门禁估算而不是科学结果：它假设已完成 Porto run 的平均非 adapter 体积可近似
外推到后续 run，并以真实 Chengdu adapter 与 Porto adapter 的大小差修正城市差异。每次出现
新的 terminal attempt 或新派发批次后，必须重新读取 `du -sb` 与 `df -B1`，不得复用静态
余量。

当前激活候选为 `d60c0b8c3026a603de76846f71ac99370b52d878`。它包含 exact extension
compact adapter、双城市全等探针和原子 `stage2/evaluation_progress.json`；完整
`stage_1/tests` 为 246 passed。该 revision 的 `--validate-only` 已在 8 completed / 52
non-completed 的实时状态上通过，报告 `contract_valid=true`、Option B/config SHA 匹配和
`safe_gpus=[]`，没有取锁或创建 attempt。

新的安全切换边界如下，未获用户明确授权前仍不得执行：

1. 只控制由命令行、锁和 watcher 日志共同确认的唯一 watcher 父进程停止新派发，不向任何
   active `run_cell` 发信号；
2. 等待当前 4 个 attempt 自然 terminal，并再次审计 completion/failure marker、run size、
   实时磁盘余量与剩余城市/cell 计数；
3. 保留全部 12 个既有 attempt，不移动、不删除、不覆盖；
4. 将正式 worktree fast-forward 到上述候选 revision，保持正式配置逐字节不变，重跑完整回归
   与 `--validate-only`；
5. 确认旧 watcher 已不再持锁后，以新 expected HEAD 启动唯一 watcher；
6. 若授权或切换前又出现新派发，立即把新 attempt 纳入不可变集合并从步骤 2 重新计算容量。
