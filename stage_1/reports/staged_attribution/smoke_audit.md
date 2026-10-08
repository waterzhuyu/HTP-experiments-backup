# Staged Attribution Smoke Audit

- 实验 ID：`gps_road_grid_staged_attribution_v1_20260826`
- 状态：**completed（仅接口验证，不用于科学结论）**
- 城市/seed：Porto / 42
- 最终统一 revision：`433b3421b4a5b568365a66fb4b19853b203070d3`
- 完成时间：2026-08-27 11:45（Asia/Shanghai）

## 1. 覆盖范围与固定预算

本轮只使用 GPS、pointwise Road 和 100 m Grid，不使用 POI。十个核心 cell 为 E0–E6/Q0 与 E6/Q1–Q3；每个 cell 都实际完成 Stage1、token export、Stage2 data、Stage2 Qwen/LoRA、Stage3 generation、G00/G10/G01/G11、decode 和 Stage4 JSON。

Smoke 固定为 32 条训练轨迹、16 条测试轨迹、Stage1 1 epoch/batch 8、Stage2 最多 2 optimizer steps、固定 8 条生成轨迹。所有成功 attempt 来自 clean revision，失败 attempt 均保持 immutable，没有覆盖或删除。

## 2. 最终成功证据

artifact root：

```text
/workspace/guozuyu/HTP-staged-attribution-artifacts/staged_attribution/
gps_road_grid_staged_attribution_v1_20260826
```

最终同 revision completed attempts：

| Cell | Attempt | 完整终点 |
|---|---|---|
| E0/Q0 | `smoke/porto/seed_42/e0_q0/attempt_004` | Stage4 |
| E1/Q0 | `smoke/porto/seed_42/e1_q0/attempt_002` | Stage4 |
| E2/Q0 | `smoke/porto/seed_42/e2_q0/attempt_002` | Stage4 |
| E3/Q0 | `smoke/porto/seed_42/e3_q0/attempt_002` | Stage4 |
| E4/Q0 | `smoke/porto/seed_42/e4_q0/attempt_002` | Stage4 |
| E5/Q0 | `smoke/porto/seed_42/e5_q0/attempt_001` | Stage4 |
| E6/Q0 | `smoke/porto/seed_42/e6_q0/attempt_001` | Stage4 |
| E6/Q1 | `smoke/porto/seed_42/e6_q1/attempt_001` | Stage4 |
| E6/Q2 | `smoke/porto/seed_42/e6_q2/attempt_001` | Stage4 |
| E6/Q3 | `smoke/porto/seed_42/e6_q3/attempt_001` | Stage4 |

结果审计命令：

```sh
uv run python -m staged_attribution.report_results \
  --config stage_1/configs/staged_attribution/experiment.json \
  --experiment-root /workspace/guozuyu/HTP-staged-attribution-artifacts/staged_attribution/gps_road_grid_staged_attribution_v1_20260826 \
  --budget smoke \
  --output-dir /workspace/guozuyu/HTP-staged-attribution-artifacts/staged_attribution/gps_road_grid_staged_attribution_v1_20260826/analysis/smoke_433b342 \
  --n-resamples 200 --bootstrap-seed 20260826
```

审计器验证了完整 Cartesian coverage、manifest identity、同一 clean revision、artifact 路径约束、size/SHA256、Stage1/Stage4 JSON，以及四个 G condition 的完全相同 trajectory population。输出为：

- `analysis/smoke_433b342/analysis.json`
- `analysis/smoke_433b342/analysis.md`
- 10 completed runs、160 Stage4 metric rows、20 G Shapley rows、40 trajectory-bootstrap G rows、28 E/Q contrast rows、9 paired E/Q rows。

## 3. Smoke 观察与边界

- 十个 cell 的 G00 validity 均为 1.0，证明 oracle SID/sampling、route-conditioned decode 与 Stage4 接口可贯通。
- 十个 cell 的 G11 validity 均为 0.0。2-step Qwen smoke 不足以学会协议，因此该结果只说明 invalid population 能被保留和惩罚，不能用于比较 E/Q 方法。
- 1-epoch Stage1 gate 有 8/10 通过；E0/Q0 与 E3/Q0 因 active fraction、perplexity fraction 或 max share 未过阈值而失败。按预注册，smoke gate 不用于提前筛 cell，controlled 仍完整运行全部十个核心 cell。
- 所有 smoke contrast 和 G attribution 均不得写入最终科学结论；最终优于 baseline 的声称只允许来自 confirmation。

## 4. 真实失败、归因与修复

### 4.0 Pre-formal CPU runtime：route geometry 支持长度不一致

- 失败：decoder route context 中 geometry tensor 使用全路网 padding 59，而 batch 内 mask support 为 52，触发 tensor size mismatch。
- 归因：collator 选择 batch routes 后没有把 geometry 同步 slice 到 `max(selected_lengths)`。
- 修复：按 batch route support 截断 geometry，并增加全局 padding 大于 batch support 的回归夹具。该检查发生在 formal attempt 之前，临时路径不记为完成 run。

### 4.1 E0/Q0 attempt_001：模型与 batch 设备不一致

- revision：`518c7d5fb3881f81f9f56caf6b564782b2723696`
- 失败：batch 在 `cuda:0`，Stage1 模型仍在 CPU，`Linear` 报 device mismatch。
- 归因：训练入口只迁移 batch，没有保证 model 移到传入 device。
- 修复：`23bcd4f0a3e6050f2c208b6dc3385efb967b074e`，`train_epochs` 入口统一迁移模型，并增加回归测试。

### 4.2 E0/Q0 attempt_002：锁定 Transformers API 不兼容

- revision：`23bcd4f0a3e6050f2c208b6dc3385efb967b074e`
- 已完成：Stage1、tokens、Stage2 data。
- 失败：Transformers 5.13.0 的 `TrainingArguments` 已移除 `overwrite_output_dir`。
- 修复：`912e8edcd7b938c43b6f1b23c54955ee52da6ec0`，集中构造锁定版本参数并增加真实 API 构造测试。

### 4.3 E1–E4/Q0 首次并行 attempt：跨卡 DataParallel OOM

- revision：`912e8edcd7b938c43b6f1b23c54955ee52da6ec0`
- 失败：四个进程都能看见四张 GPU，Transformers 为每个进程自动启用 DataParallel；四份 Qwen 同时跨四卡复制，导致 OOM。
- 归因：`--device cuda:N` 只约束 Stage1 tensor，不会限制 Trainer 的可见设备。
- 修复：`433b3421b4a5b568365a66fb4b19853b203070d3`。launcher 为每个子进程设置 `CUDA_VISIBLE_DEVICES=<physical GPU>`，子进程统一使用逻辑 `cuda:0`，同时把可见物理卡写入 resolved config。
- 真实验证：隔离后 E1/E2/E3 同时运行时各自只占一张卡约 5.3 GiB；十个 cell 全部完成，无 OOM。

## 5. 测试与环境

最终 revision：

```text
168 passed, 21 warnings
```

warnings 为 Transformer nested-tensor 优化提示，不改变数值或执行路径。正式依赖由 `pyproject.toml` 与 `uv.lock` 锁定；Stage2 使用本地 Qwen3-1.7B、相同 LoRA 目标层和固定 smoke token budget。

## 6. 下一动作

1. 从本 smoke 记录后的 clean revision 排队两城市 × 三 seeds × 十核心 cells 的 controlled full pipeline；
2. 审计 60 个 run 的同 revision、population、gate、E/Q contrast 与 G 2×2；
3. 按冻结 Pareto 规则选择 best single、best pair 和 best Q，运行两个 adaptive interaction cells；
4. 最后对 baseline 与最终候选运行 confirmation，才能形成 GPS/Road/Grid 轨迹上下文结论。
