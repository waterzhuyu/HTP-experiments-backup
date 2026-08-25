# Staged Attribution Smoke Audit

- 实验 ID：`gps_road_grid_staged_attribution_v1_20260826`
- 状态：running，不能用于科学结论
- 城市/seed：Porto / 42
- 目标：十个核心 cell 各自完整到 Stage4；当前尚未满足

## 已验证证据

截至 revision `472db57` 后的真实 CPU 检查，E0/Q0 使用实际 Porto aligned/processed/road network 数据完成：

1. Stage1：32 train / 16 test，1 epoch，batch 8；生成可重载 `stage1_model.pt`（48,557,673 bytes）、resolved config 和 Stage1 JSON；
2. tokens：32 train / 16 test 全人口一一导出，产生 `train_tokens.jsonl`、`test_tokens.jsonl` 和 SHA256 audit；
3. Stage2 data：复刻原 HTP prompt，产生 32/16 parquet 与固定 8-ID generation manifest。

临时验证路径：`/tmp/staged_attr_real_stage1_e0`。这是可丢弃的 pre-commit runtime check，不登记为完成 run；正式 smoke 使用 artifact root、clean revision 与 immutable run manifest 重跑。

全量单元测试：最新 pre-formal revision 为 158 passed；唯一 warnings 为 PyTorch nested Transformer 优化未启用，不改变数值。

## 首次结果前的评估审计

在没有任何正式 Stage4 attempt 的前提下发现并修复：

- E0/GPS-only 被误当成空模态导致不可识别的交互字段；改为三个给定第三模态的条件交互；
- Stage4 使用近似重划网格而非 Stage1 aligned grid；改为复用 metadata 和同一 mapping；
- top-100 F1 把零频 bin 加入集合，极小 population 时可能虚高；改为正频 top-k 对称 F1；
- 越界生成 GPS 被静默丢点/裁剪；改为 invalid 并保留 penalty。

这些是看结果前的方法纠偏，不改变 cell/seed/预算；详见预注册第 10 节。

## 首次真实失败与修复

第一次 E0/Q0 Stage1 CPU smoke 在 decoder route context 失败：

```text
RuntimeError: The size of tensor a (59) must match the size of tensor b (52)
```

原因：RoadContext geometry 按全路网最大 59 points padding，而 batch 内所选 route 最大 geometry 仅 52；`get_padding_mask(geometry_lengths)` 使用 52，geometry tensor 未同步 slice。

修复：collator 先取 batch route 的 `selected_lengths`，再将 selected geometry slice 到 `max(selected_lengths)`；新增全局 padded support 大于 batch support 的回归夹具。修复后真实 Stage1、token export、Stage2 data 均完成。

## GPU blocker（预期等待，不是完成）

检查时四张 24 GB GPU 占用约为：

| GPU | used MiB | utilization |
|---:|---:|---:|
| 0 | 22294 | 100% |
| 1 | 20676 | 96% |
| 2 | 20818 | 100% |
| 3 | 20976 | 64% |

安全阈值为 free≥16,000 MiB 且 utilization≤20%。当前无 GPU 合格；未抢占或终止其他用户进程。Stage2 真实 Qwen 2-step、Stage3 和十 cell 完整 smoke 仍为 pending。

## 当前不可声称内容

- 不能声称任一 E/Q cell 更好；
- 不能声称完整 smoke 已通过；
- 不能把前三阶段接口检查记作 full-pipeline cell；
- 不能用 CPU fixture 或旧 independent-tokenizer 结果替代 Stage2/3/4。

## 下一动作

1. 等待安全 GPU；先跑 E0/Q0 完整 smoke，验证 PEFT trainable token、teacher-forced eval、generation、G00/G10/G01/G11 decode、Stage4；
2. 修复并提交任何真实接口失败；
3. 从新的 clean revision 运行十个核心 smoke cell；
4. 冻结 smoke audit 和 controlled preregistration revision 后，排队两城×三 seed×十 cell controlled；
5. 依据预注册规则选择 interaction/confirmation，不能提前看 test 结果改权重。
