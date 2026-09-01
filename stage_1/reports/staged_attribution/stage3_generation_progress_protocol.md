# GPS/Road/Grid 分阶段归因：Stage3 生成原子进度协议

## 1. 实验身份、状态与研究问题

- 实验 ID：`gps_road_grid_staged_attribution_v1_20260826`
- 记录日期：2026-09-01（Asia/Shanghai）
- 候选 worktree：`/workspace/guozuyu/HTP-staged-attribution-compact-adapter`
- 候选分支：`fix/staged-attribution-compact-adapter`
- 正式基线 revision：`6ac87be094c9c1ce4783f10291a339f04623449b`
- 功能 revision：`1fe012ebd56c4f458f3c3f378100996668a6a7f6`
- 状态：`verified / not_activated_on_formal_queue`
- 类型：运行证据与故障可观测性协议；不是科学消融变量
- POI：不使用；研究上下文只包含 GPS、Road 与 Grid

研究问题是：Stage3 对完整生成 population 执行确定性自回归生成时，能否在不改变输入人口、
模型、prompt、解码、随机数状态或最终输出的前提下，持续写出可审计的批次级进度，并在异常时
保留最后一个完整批次的证据。

预先定义的验收条件：

1. 模型加载前即创建状态为 `loading_model` 的进度文件；
2. 模型就绪后进入 `running`，默认每 10 个完整 batch 以及最后一个 batch 原子更新；
3. 只累计已完成解析的完整 batch，不把进行中的 batch 计入；
4. 正常结束时，batch、population、completed/failed 计数必须与最终 audit 完全一致；
5. 异常时写出 `failed`、异常类型/消息和最后一个完整 batch，然后原异常继续抛出；
6. 最终 `generations.jsonl` 仍只在全人口验证通过后原子发布，不形成 partial checkpoint；
7. terminal progress 文件必须成为 Stage3 DAG required artifact，并由 stage marker 锁定大小与 SHA-256；
8. controlled core 与 corrected stratified reevaluation 使用同一记录协议；
9. 完整测试集通过，且正式队列在未激活前不受影响。

失败条件为任一计数越界、空 batch、非法输出状态、终止时人口未完成、progress 与 audit 不一致、
异常被吞掉、partial generations 被发布，或 DAG 未锁定 terminal progress。

## 2. 数据、样本单元与切分

本变更不构造新数据，也不改变任何实验人口。正式 Option B 仍使用：

- Porto 与 Chengdu 各固定 64,000 条 Stage2 训练轨迹；
- subset seed `20260826`；
- generation population 来自各 run 的
  `stage2_data/subset_manifest.json["generation"]["traj_ids"]`；
- controlled core 从 `llm_test.parquet` 中精确筛出该 ID 集；
- corrected stratified reevaluation 从已冻结的 stratified selection 精确筛出 ID 集；
- 输入样本的物理单元仍是一条完整轨迹；
- 轨迹 ID 在进入生成前升序排序，重复 ID 失败关闭。

轨迹、车辆、道路、时间和多视图的切分与泄漏防护均与正式基线相同。本协议只观察既有生成
循环，不重新抽样、不删除失败输出，也不改变 train/test 边界。

## 3. 输入、表示、token、模型与生成口径

以下科学方法全部与正式基线相同：

- Stage3 输入为 `desc` 文本，包含冻结的轨迹上下文与 oracle route 描述；
- prompt 使用既有 system/user/assistant 模板和空 `<think>` block；
- tokenizer、协议 token、road vocabulary 与 adapter 路径不变；
- Qwen3-1.7B 基础模型、PEFT adapter、BF16 CUDA 加载口径不变；
- compact adapter 存在时仍先完成 exact extension-row 恢复与完整 tensor digest 校验；
- 加载后按既有逻辑重新设置 generation seed；
- `do_sample=False`，`max_new_tokens=256`，默认 batch size 16；
- continuation 解析、completed/failed 判定和失败原因分类不变；
- Stage3 不执行 GPS 解码；GPS 重建仍由后续 decode stage 完成；
- Stage4 的 ADE/FDE、route 与 grid 评估口径不变。

进度记录不参与 forward、logits、token sampling、parse decision 或 Stage4 指标计算。

## 4. 原子进度文件

默认路径为每个 attempt 的 `stage3/generation_progress.json`。schema version 为 1，固定
`stage="stage3_generation"`，状态机为：

```text
loading_model -> running -> completed
loading_model -> failed
running       -> failed
```

每次写入先生成同目录临时文件，再用原子 replace 发布。字段为：

| 字段 | 精确定义 |
|---|---|
| `started_at` / `updated_at` | UTC ISO-8601 时间 |
| `batch_size` | 本次生成调用的 batch size |
| `batches_total` | `ceil(population_total / batch_size)` |
| `batches_completed` | 已完整生成、解析并累计的 batch 数 |
| `population_total` | 冻结 generation population 大小 |
| `population_completed` | 已完整处理的轨迹数 |
| `outputs_completed` | 已解析为有效协议序列的轨迹数 |
| `outputs_failed` | 保留为显式失败记录的轨迹数 |
| `fraction_completed` | `population_completed / population_total` |
| `elapsed_seconds` | 单调时钟计算的累计耗时 |
| `records_per_second` | `population_completed / elapsed_seconds` |
| `eta_seconds` | 按当前吞吐估计的剩余秒数；无样本时为 null |
| `last_traj_id` | 最近一个完整 batch 的末尾轨迹 ID |
| `audit` | 仅 terminal completed 时写入最终人口 audit |
| `error` | 仅 failed 时写入异常类型与消息 |

`update()` 拒绝空 batch、人口越界以及非 `completed|failed` 状态。`complete()` 要求
batch 数、population 数和 audit 的 completed/failed 全部一致。任何失败都会保留此前最后一次
完整更新，不会把半个 batch 宣称为完成。

## 5. DAG、产物与恢复语义

controlled core 的 Stage3 required artifacts 现在是：

1. `generations.jsonl`
2. `generation_audit.json`
3. `generation_progress.json`

corrected stratified reevaluation 使用同一三件套。只有三者全部存在且 Stage3 正常返回后，
`stage3/complete.json` 才会记录每个文件的大小与 SHA-256。

`generation_progress.json` 是观察证据，不是 resume checkpoint。Stage3 中断后不得从
`population_completed` 继续拼接输出；恢复必须创建新的 attempt 或按既有不可变运行规则重新
执行完整 Stage3。这样避免把不同模型状态、revision 或 RNG 上下文的片段拼成一个人口。

## 6. 方法变更表

| 方法组成 | Baseline | 本次协议 | 是否变化 |
|---|---|---|---|
| 样本物理单元 | 一条完整轨迹 | 与 baseline 相同 | 否 |
| 视图构造与覆盖范围 | Stage1/Stage2 既定 GPS/Road/Grid 视图 | 与 baseline 相同 | 否 |
| 输入特征与单位 | 既定 GPS/Road/Grid 与 Stage3 `desc` | 与 baseline 相同 | 否 |
| encoder/downsampling | 既定 Stage1 encoder | 与 baseline 相同 | 否 |
| token 时间/空间支持 | 既定 RQ token 协议 | 与 baseline 相同 | 否 |
| quantizer/codebook | 既定 residual quantizer/codebook | 与 baseline 相同 | 否 |
| decoder 与输出时间锚点 | 既定 Stage3 token 生成和后续 GPS decode | 与 baseline 相同 | 否 |
| 初始化与冻结模块 | 既定基础模型、PEFT 与 seed | 与 baseline 相同 | 否 |
| reconstruction/VQ loss | 既定 Stage1 loss | 与 baseline 相同 | 否 |
| 新增 loss 与权重 | 无 | 无 | 否 |
| 训练数据与超参数 | Option B、64k/城、microbatch 8、grad accumulation 2 | 与 baseline 相同 | 否 |
| 评估对齐与指标 | 既定全人口与 Stage4 口径 | 与 baseline 相同 | 否 |
| 运行证据 | Stage3 仅 terminal audit/marker | 新增原子批次进度与 terminal DAG 哈希 | 是（非科学变量） |

因此，该 revision 不能被当作 GPS/Road/Grid 效果变化的解释变量；它只改善长阶段的可观测性
和失败归因。

## 7. 自动化验证

测试覆盖：

- 初始 `loading_model` 与 `running` 状态；
- completed/failed 输出分别计数；
- 最后一个完整 batch 与 `last_traj_id`；
- synthetic 中途异常后保留 partial progress evidence；
- 提前 complete、计数不一致和非法状态失败关闭；
- controlled core 返回并锁定 progress artifact；
- corrected reevaluation 返回并锁定 progress artifact；
- compact/legacy adapter 加载回归；
- 既有生成解析、人口审计和 DAG marker 回归。

完整回归命令：

```bash
cd /workspace/guozuyu/HTP-staged-attribution-compact-adapter
uv run --frozen pytest -q
```

结果：`255 passed, 23 warnings in 15.97s`。warnings 均为既有 PyTorch Transformer
nested-tensor 提示。另执行 Python compileall 与 `git diff --check`，均通过。验证后候选
worktree clean。

## 8. 正式队列边界与当前证据

正式 watcher 仍锁定在 `6ac87be...` 和配置 SHA-256
`39a33c9bef9cc06f1bc2cdbbf1c6010ea76dcb455a7855b330905868a1c7604a`；本文 revision
未激活，当前正式 attempts 因而不会写入新增 progress 文件，不能把缺少该文件误判为故障。

2026-09-01 13:58 CST 的只读快照确认：

- watcher uv/python 与四个正式 run_cell 均存活；
- 正式 worktree branch、HEAD、config SHA 与 clean 状态正确；
- Porto seed42 `e6_q2`、`e6_q3/attempt_001` 仍在 Stage1；
- Porto seed314159 `e0_q0/attempt_001` 约为 Stage2 3414/8000；
- Porto seed314159 `e1_q0/attempt_001` 约为 Stage2 2937/8000；
- 四张 GPU 均未满足 free >= 16000 MiB 且 utilization <= 20%；
- 没有训练 Traceback 或 CUDA OOM 证据。

不得热替换运行中 run_cell 的代码，不得由本协议暂停 watcher、发送信号、切换正式 revision、
重启或重复派发。正式激活只能在用户明确授权的安全边界进行，并继续保持所有既有 attempt
不可变。

## 9. 记录同步状态与下一步

仓库中的本报告、功能 revision、测试结果和未来 terminal progress/marker 是权威可复现证据。
Notion 页面的同内容同步曾被外部写入安全审查拒绝，状态为
`pending_explicit_destination_authorization`；未绕过该门禁，也未缩减或改写敏感实验事实。

下一步：

1. 继续只读监督正式 Option B 队列；
2. 获得明确授权后，在所有活跃 attempt 自然结束的安全边界评估是否激活候选 revision；
3. 对首次产生的正式 `generation_progress.json` 验证状态机、计数与 stage marker SHA；
4. 完整 core、corrected reevaluation、adaptive 与 confirmation 结束后，汇总双城市多 seed
   GPS/Road/Grid 分阶段归因结论；
5. 获得明确 Notion 目标授权后同步同一份方法、revision、命令、路径、失败与下一步记录。

