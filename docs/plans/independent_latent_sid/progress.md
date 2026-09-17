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

## HV生成与stage4完成，诊断仍在运行（2026-09-17 02:57阶段快照）

相同训练revision、配置、seed42和1024条验证集。stage3耗时3534.006s，1023条解析成功，1条截断（traj_id=58954，缺少完整HTP结束标记），失败按方案保留。四G解码完成：G00 1024/1024，其余三G均1023/1024；后三者失败原因均为generation_truncated。

全体轨迹平均惩罚误差（无量纲，四项km统计量abs-log1p差的均值，失败惩罚1）：G00=0.0182425980，G10=0.0222109983，G01=0.0193492120，G11=0.0217968975。该指标不是ADE或相对误差百分比。四条件沿用相同oracle路线，低误差不能独立证明SID保留了运动信息。完整对照、GPS对齐误差、冻结probe、tuple准确率及惩罚10敏感性尚待汇总，不据此声明分层方法有收益。

精简汇总与原始来源SHA见hv_stage4_summary.json；权威原始逐轨迹数据保留远端stage4/stage4_metrics.json。当前HV已进入diagnostics（冻结特征与probe），整个run尚未验收完成；LF/H1/B2尚未产生结果。

## HV完整验收与SID元组补测

`analysis/results_snapshot_002.json` 已由已提交分析器验收HV：九阶段、训练步数、85项产物及SHA通过；LF处于stage1，H1/B2尚未启动。因此整体状态仍为incomplete，不生成组间比较。对应本地归档results_snapshot_002.json。

HV的24项冻结probe均训练10epoch、1860step，seed42、AdamW lr=.001/wd=.0001、batch_events256，按有效点逆轨迹长度加权；输入为单层码向量或累计前缀及点内offset，目标和单位继承spec。前缀1→4：GPS MSE 0.030492→0.028385，Grid准确率0.007675→0.012349，Road准确率0.009518→0.017931；各深度总体单调改善。当前未观察到后续前缀使先前任务退化，不能把公式上的信息覆盖风险当作已发生的事实。绝对准确率需对照LF及常量基线，不等于信息量的直接测量。原始归档hv_probe_metrics.json。

补测命令（分析revision `9f67a346f77cef999dbaeb8c4ca8a6cbf0742d26`）：`CUDA_VISIBLE_DEVICES=3 uv run python -m hierarchical_rqvae.audit_tuple_accuracy --root /workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917 --cells HV --output /workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/analysis/tuple_hv_001`。冻结已保存adapter，无重新训练；1024轨迹/5996目标事件，teacher-forced eval batch4。恢复SID NLL=0.569082514，与原0.568885164绝对差0.000197351<.005，分母一致；恢复评估小幅数值差异不覆盖原指标。

完整四码同时正确：teacher-forced 2217/5996=36.97465%；free-running 1685/5996=28.10207%。自由生成按目标事件序号对齐，缺失事件及失败轨迹计错，额外45事件单列；不是时间或语义对齐。自由生成四层准确率94.179/76.518/48.949/46.848%，对应teacher-forced94.713/79.053/62.475/73.566%。第三、四层下降更明显，但不能仅凭条件相关性宣称误差传播的因果机制。原始归档hv_tuple_metrics.json含逐轨迹数据、源manifest SHA及完整revision。输出指标计算耗时87.779s（不含全部模型加载）。后续仍需LF/H1/B2对照及相同诊断后形成最终建议。
## 最终状态（2026-09-17）

四组HV/LF/H1/B2及全部SID元组补评均完成；`results_final_001.json`已验收九阶段、每组85项产物和固定训练预算。此前区段保留为历史快照，不代表当前状态。

HV未通过探索门槛：对LF的G11相对改善−2.787%，95%CI[−14.058%,+4.359%]；前两层分布集中，四组后两层均未通过码本健康门槛。LF/HV相对H1的G11分别恶化5.883%/8.834%，H1对B2仅改善.305%且CI跨零。单seed、oracle路线，不作训练稳定性或跨域声明。

完整方法、结果、门槛判断和优先级见`final_analysis.md`。建议先单独验证HV重构ST路由，再独立验证码本初始化/训练机制；不自动追加训练，不直接扩规模。
