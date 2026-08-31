# GPS/Road/Grid 分阶段归因：Stage2 compact-adapter 存储协议

## 1. 状态与适用边界

- 实验 ID：`gps_road_grid_staged_attribution_v1_20260826`
- 记录日期：2026-08-31（Asia/Shanghai）
- 修复 worktree：`/workspace/guozuyu/HTP-staged-attribution-compact-adapter`
- 修复分支：`fix/staged-attribution-compact-adapter`
- 基线 revision：`6ac87be094c9c1ce4783f10291a339f04623449b`
- 状态：`verified / not_activated_on_formal_queue`

本协议只改变每个 Stage2 PEFT adapter 的落盘表示，不改变 GPS/Road/Grid 数据、Stage1、
Stage2 训练人口、模型初始化、LoRA、可训练 token delta、loss、optimizer、scheduler、
microbatch、梯度累积、Stage3 生成参数或 Stage4 评估。POI 持续排除。

正式 Option B 队列仍锁定在基线 revision；本文不授权暂停、重启或切换正在运行的 watcher，
也不修改任何既有 attempt。激活必须形成单独的 Git revision，并在实验记录中区分旧的完整
adapter 与后续 compact adapter。

## 2. 问题与精确归因

Qwen3-1.7B + PEFT 在保存当前 Stage2 adapter 时，同时写入 LoRA、两份可训练 token delta，
以及两份完整的 resized base embedding。真实 Porto smoke adapter 中共有 396 个张量；以下
两份 base embedding 形状均为 `[162698, 2048]`、dtype 均为 BF16，且逐字节完全相同：

1. `base_model.model.lm_head.token_adapter.base_layer.weight`
2. `base_model.model.model.embed_tokens.token_adapter.base_layer.weight`

它们是冻结基础模型在固定 tokenizer 扩展和固定训练 seed 下执行
`resize_token_embeddings(..., mean_resizing=True)` 的确定性产物，不是训练所得参数。LoRA 和
两份 `trainable_tokens_delta` 才承载训练更新，必须原样保留。

## 3. 存储格式与失败关闭门禁

`compact_peft_adapter` 只在满足以下全部条件时省略上述两份张量：

1. adapter model 与 `adapter_config.json` 同时存在；
2. adapter 配置中的基础模型路径与调用方锁定路径一致；
3. 恰好发现 input/output 两份 `token_adapter.base_layer.weight`；
4. 两份张量 shape、dtype 和每个 4096-row chunk 完全一致；
5. compact 文件的张量清单只减少这两个名字；
6. 原始/compact model、adapter config、基础模型 config、软件版本、训练 seed、被省略张量
   SHA-256 和字节数全部写入 `compact_adapter_manifest.json`。

发布顺序为先发布 manifest、再原子替换 model 文件。若中途崩溃，manifest/model SHA 不一致，
后续加载会失败；不会把缺少 base embedding 的文件静默当作 legacy adapter。

Stage3 加载 compact adapter 时先用记录的训练 seed 重建 resized embedding，再加载 PEFT delta。
加载完成后，对模型中的两份重建张量逐字节计算 SHA-256，并与 manifest 比较；shape、dtype、
张量清单或 digest 任一不一致即拒绝生成。随后重新设置独立的 Stage3 generation seed。因此，
重建 seed 不会改变生成随机性口径。无 manifest 的旧 adapter 继续使用原加载路径。

## 4. 自动化测试

定向测试命令：

```bash
cd /workspace/guozuyu/HTP-staged-attribution-compact-adapter
PYTHONPATH=stage_1:stage_1/src uv run --frozen pytest -q \
  stage_1/tests/staged_attribution/test_adapter_storage.py \
  stage_1/tests/staged_attribution/test_generation_parser.py \
  stage_1/tests/staged_attribution/test_pipeline.py \
  stage_1/tests/staged_attribution/test_reevaluation_evidence.py \
  stage_1/tests/staged_attribution/test_stage2_collator.py \
  stage_1/tests/staged_attribution/test_stage2_metrics.py
```

结果：`36 passed in 16.55s`。

完整 Stage1 回归命令：

```bash
cd /workspace/guozuyu/HTP-staged-attribution-compact-adapter
PYTHONPATH=stage_1:stage_1/src uv run --frozen pytest -q stage_1/tests
```

结果：`242 passed, 21 warnings in 12.16s`。21 个 warning 均为既有 PyTorch Transformer
nested-tensor 提示。

测试覆盖：只移除两份 base embedding、保留 LoRA/delta、拒绝不相同的 base embedding、
拒绝 compact model 篡改、legacy fallback、重建 seed 与 generation seed 顺序、加载后张量
digest 复核、Stage2 manifest 传播，以及现有 pipeline/生成/评估回归。

## 5. 真实 adapter 等价性探针

权威只读源：

`/workspace/guozuyu/HTP-staged-attribution-artifacts/staged_attribution/gps_road_grid_staged_attribution_v1_20260826/smoke/porto/seed_42/e0_q0/attempt_004/stage2/adapter`

- 源 `adapter_model.safetensors` SHA-256（探针前后相同）：
  `671b4d0b705745bb6173e652d1bdb3fd2dead858a04ca4ada4dff4d43abc2c03`
- 探针副本：`/tmp/staged_compact_verified_bp9r9wzd/adapter`
- 原始大小：1,583,305,008 bytes
- compact 大小：250,482,712 bytes
- 节省：1,332,822,296 bytes（84.179756%）
- 被省略张量共同 SHA-256：
  `04121496dda6e94ba68b81329e4ed9b5f7116e04ec33df064ae6940efd55c4f0`
- 重建 seed：42

CPU-only 探针分别用完整 adapter 和 compact 副本加载同一基础模型与同一含 road/protocol token
的 prompt。输入 IDs 完全相同；输出 logits shape 为 `[1, 17, 162698]`；逐元素完全相同，
最大绝对差为 `0.0`。compact 加载同时通过生产代码的加载后张量 digest 复核。原始 smoke
adapter 未被移动、删除或改写。

若 60 个 core run 的 adapter 形状均相同，仅 adapter 文件部分预计可少写
79,969,337,760 bytes（约 74.48 GiB）；这不含其它阶段产物，因此不是总实验体积估算。

## 6. 激活门与回滚边界

正式激活前必须同时满足：

1. 修复分支形成 focused commit，worktree clean；
2. 完整回归和真实 adapter 全等探针持续通过；
3. 正式配置 SHA-256 保持
   `39a33c9bef9cc06f1bc2cdbbf1c6010ea76dcb455a7855b330905868a1c7604a`；
4. 明确记录正式队列 revision 的切换时点；
5. 不修改四个旧 Porto `attempt_001`，不修改已创建或完成的任何正式 attempt；
6. 不在活跃 `run_cell` 内热替换代码；如要切换 watcher，必须先得到授权并在安全边界执行；
7. 每个 compact attempt 的 Stage2 marker 必须记录 `adapter_storage` manifest；
8. Stage3 开始前必须通过 model/config/version/reconstructed-tensor 全部校验。

回滚只允许通过新的 Git revision 恢复完整 adapter 保存；不删除、不覆盖已经产生的 compact 或
legacy artifact。两种格式都保留各自 provenance，并由 manifest 是否存在显式区分。
