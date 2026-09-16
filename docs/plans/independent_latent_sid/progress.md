# 独立 latent → 分层 SID：执行记录

- 日期：2026-09-17（Asia/Shanghai）。
- [Notion实验记录](https://app.notion.com/p/3ddc24a261be816886fae93e48896efc)。已在Research 9.10–9.17区段添加进度链接；研究依据为9.3–9.10分析。
- 权威工作区：`3090s:/workspace/guozuyu/HTP-independent-latent-sid`。
- 实现提交：`b5e7dec07f2b4c362e6ed0e96f16b00123e25a56`。
- fork：`52b6666e6a65eeee7fbe7ecc7cae8b84548f7ea3`；独立branch `exp/independent-latent-sid`。旧容量消融不变。
- 环境：`uv sync --frozen`，Python3.12.3；依赖锁未修改。
- 完整测试：`uv run pytest -q`，307 passed / 49 warnings，29.45s。11项新测试；第一次9通过2失败（辅助梯度仍走继承共享latent），提交前已修复。warnings为继承Transformer nested-tensor提示。
- 配置：`stage_1/configs/independent_latent_sid/pilot.json`，SHA256 `796f63b2472ab9e5aec5a6a0efa49879598e50376626230ff6857f5bfbecd193`。
- preflight manifest SHA256：`a8857cd044076731d133841b1f4ba45fe7e44e7198c2242c939239a25bbaad28`。
- smoke命令：`CUDA_VISIBLE_DEVICES=2 uv run python -m independent_latent_sid.run --config stage_1/configs/independent_latent_sid/pilot.json --preflight /workspace/guozuyu/HTP-hierarchical-rqvae-artifacts/hierarchical_rqvae_gps_grid_road_v1_20260907/preflight --phase smoke --cells LF HV --seed 42`；同命令加 `--validate-only` 通过。
- smoke产物：`/workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/smoke/porto/42/{LF,HV}/attempt_001/`。
- LF smoke已完成9阶段，229.70s；tokenizer4step、LLM2step，G00 8/8解码成功，G10/G01/G11各0/8解码成功（生成不合法，按失败保留）。短程smoke不评判科学效果。HV smoke已完成9阶段，221.41s；G00 8/8解码成功，其他三G各0/8，所有失败保留。两组均为短程LLM2step，不能用于科学判断。
- 当前科学指标/相对改善/跨seed统计：N/A。单seed探索未完成，不能声称建立了视图分工或提高轨迹质量。
- 完整方法和差异表见spec.md；Python快照仅备份，运行以远端Git代码为准。

- 启动验收新增2项测试通过（0.28s）：拒绝产物SHA变化、缺失或config不匹配的smoke。正式启动器在启动时重新核验全部产物，不使用仅状态标签的验收。

## 单 seed pilot 已启动

启动提交 `70b82db195f600e3271030ddb5e6c228e4915f31`；GPU2，PID2602619。已确认HV进入Stage1，随后LF/H1/B2串行。四组24GPU小时/40GiB预算，seed42，8192 train / 1024 validation。完整命令与smoke SHA验收证据见pilot_launch.json。日志 `/workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/pilot.log`；科学结果尚未产生。

## HV 语言模型阶段完成（2026-09-17 01:55 快照）

相同提交、配置和数据；LLM 1024/1024 step、2 epoch 完成，训练耗时3188.0391s。原始指标归档 `hv_stage2_metrics.json`，远端来源为上述产物根目录下 `pilot/porto/42/HV/attempt_001/stage2/stage2_metrics.json`。

验证1024条轨迹、5996个SID事件：teacher-forced SID NLL=0.568885；四层准确率依次0.946965/0.790193/0.626418/0.735490，sampling准确率0.720703，整序列精确率18/1024=0.017578。完整四码元组准确率尚未测量，不能用逐层准确率替代。

第一层准确率94.6965%仅略高于验证集最大码占比94.6798%，因此高准确率本身不支持语义分工成立。此占比来自先前全验证集latent诊断，比较只用于揭示多数码基线，不是独立评估对照。待完整产物验收后计算训练unigram NLL基线、完整SID元组准确率和四G质量。

进程2602623已进入stage3；01:55:48快照生成17/1024条，进度文件记录outputs_failed=0。该计数不等于最终可解码或轨迹质量合格率。LF/H1/B2尚未启动，暂无组间结论。
