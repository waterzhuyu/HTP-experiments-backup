# HV-AllRecon 启动记录

状态：running。核查时刻：2026-09-17T03:42:08.869726+00:00（北京时间11:42）。GPU 2，Python PID 3641128，uv launcher PID 3641123。

代码 revision：60f140aa92799c36b21dccec87bef5223480bba4。配置 SHA256：d94c57c886d37883d29696612f1cd383471d98343b3b2355ed71731557ccc1a0。基线 manifest SHA256：c840e067328926ed6fe11290fa6bd3135f3a3f1878b27ceec5d5efb744f6a095。

正式训练已执行 120/1280 步、epoch 2/20。当前 batch loss=2.137845、reconstruction_loss=1.811115、quant_loss=0.076309；仅为过程遥测，不是最终验证指标。继承日志 auxiliary_enabled=false 指旧式辅助目标，新模型 Grid/Road/GPS 辅助loss仍按.1启用。

19项专项测试及完整仓库317项测试通过。smoke五阶段completed，耗时13.43秒；32 train/16 validation、1epoch、4step，所有16条G00恢复成功，24个probe各1epoch，运行产物SHA由正式启动器复核。smoke不作为科学结果。

正式首批固定validation批次的只读诊断：gps_only与all_views reconstruction_loss分别为1.921738267、1.921738267。前向相同；梯度L2如下（clip前）：

|路径|Grid encoder|Road encoder|GPS encoder|Codebook|
|---|---|---|---|---|
|gps_only|0.000000|0.000000|2.419854|0.000000|
|all_views|1.136610|1.462697|2.419854|0.000000|

运行命令（cwd=/workspace/guozuyu/HTP-hv-allrecon）：

```sh
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python -m independent_latent_sid.recon_run --config /workspace/guozuyu/HTP-hv-allrecon/stage_1/configs/independent_latent_sid/hv_allrecon.json --preflight /workspace/guozuyu/HTP-hierarchical-rqvae-artifacts/hierarchical_rqvae_gps_grid_road_v1_20260907/preflight --phase pilot --cells HVAllRecon --seed 42
```

通过已提交的recon_launch --phase pilot --gpu 2启动，独占receipt防止重复；通过smoke SHA/版本检查、数据/模型hash核查后进入正式训练。

产物根：/workspace/guozuyu/HTP-independent-latent-sid-artifacts/hv_allrecon_20260917。
正式run：pilot/porto/42/HVAllRecon/attempt_001；日志pilot.log；启动凭据pilot_launch.json。run内保存run_manifest.json、stage1/training_progress.json与checkpoints、oracle/g00_metrics.json与逐轨迹GPS、diagnostics码本/probe/latent_rank、initial/final_route_audit.json。

本轮自动完成tokenizer→tokens→G00→冻结probe/码本/秩/采样/梯度诊断，然后停止。LLM阶段未启动，G10/G01/G11及LLM NLL为N/A。结束后依据spec.md门槛对比原HV；当前没有收益结论。完整方法与强制变更表见spec.md及配置JSON。

失败/异常：无实验失败；初次SSH受本地网络沙箱限制，授权网络执行后连接成功，无VPN故障；远端缺rg，文件定位使用grep/find。环境依赖未更改，未动GPU0/1的其他任务。
