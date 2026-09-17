## HV-AllRecon 完成结果（2026-09-17）

状态：completed_tokenizer_only；1280步/20epoch及全部5阶段完成，耗时694.22秒（约11.6分钟）。验证人口1024轨迹，训练seed42。64项本次产物与85项原HV产物的bytes/SHA256全部核验通过；24个冻结probe均10epoch/1860step。没有运行LLM、重跑或新增训练。

**结论：重构梯度到达三个视图后，前层码集中明显缓解，G00位置ADE改善；但主误差改善不确定，第2–4层仍未通过码本健康门槛，暂不进入LLM阶段。** 这是单seed探索证据，不能宣称端到端生成收益，也不能区分梯度可达性与总梯度尺度增加的作用。

### 原HV与HV-AllRecon结果

|指标|原HV|HV-AllRecon|判断|
|---|---|---|---|
|G00全人口主误差↓|0.018242598|0.017997285|改善1.345%，95%区间[-2.536%,+5.081%]，跨零|
|G00 ADE(m)↓|196.32|180.24|减少16.08m，差值95%区间[-23.18,-8.86]m|
|G00 FDE(m)↓|163.88|166.93|增加3.04m，差值95%区间[-5.20,+11.94]m|
|有效轨迹|1024/1024|1024/1024|通过有效率门槛|
|第1层最大码占比↓|94.68%|37.31%|集中显著缓解（描述性）|
|第2层最大码占比↓|65.41%|36.52%|集中缓解|
|完整tuple熵(nats)|3.524|5.367|经验事件码分布更丰富，不等于已证明语义更好|
|tuple碰撞概率↓|6.610%|0.668%|经验事件对碰撞下降|

G00使用真实SID、sampling与route。主误差为travel distance、mean segment distance、radius、mean point-to-road distance四项km统计的abs(log1p(pred)−log1p(real))平均，无量纲；全1024条含失败罚1。本轮两组G00均无失败，罚10敏感性结果相同。ADE/FDE按100个观测索引进度锚点，非真实时间对齐。配对bootstrap2000次、seed271828、普通percentile95区间，仅反映验证轨迹不确定性，不包含训练随机性，也未做多重比较校正。

### 逐层健康门槛

|层 / K|原HV active / perplexity / max share|AllRecon active / perplexity / max share|AllRecon判定|
|---|---|---|---|
|1 / 32|29 / 1.378 / 94.68%|4 / 3.815 / 37.31%|通过|
|2 / 64|27 / 2.942 / 65.41%|6 / 3.801 / 36.52%|active/K=9.375% <10%，未通过|
|3 / 128|6 / 3.988 / 30.40%|6 / 4.706 / 31.79%|active及perplexity比例未通过|
|4 / 256|7 / 4.631 / 35.34%|7 / 5.374 / 31.99%|active及perplexity比例未通过|

门槛保持预设：每层active/K≥.10、perplexity/K≥.05、max share≤.50。前两层active数量减少但perplexity增加，说明稀有码数量与有效使用容量并非同一指标；不能只看active个数判断改善或恶化。G00不恶化超过10%的点估计门槛通过，但不是等效证明。

### 辅助证据与解释边界

Grid s1 probe准确率0.768%→1.123%，CE7.292→7.231；Road s2准确率1.149%→1.611%，但CE7.200→7.430变差，指标不一致，不能宣称前缀信息全面改善。GPS s4 probe归一化MSE 0.028385→0.018015。probe配置、有效点逆轨迹长度加权与原HV一致；无常量参照或跨seed区间，不将绝对值当作信息量。

AllRecon最终验证latent的中心化奇异值熵有效秩：Grid8.064、Road8.351、GPS24.487（5996事件）。本报告不直接与旧诊断的秩作比值，需先统一统计定义/样本口径。初始化和最终checkpoint的固定批次梯度诊断已保存，未执行optimizer，仅检验路由；它不代表整个训练期间的梯度统计。

**机制判断：** 结果支持“重构梯度隔离是前层码集中问题的一个相关机制”的解释，但还不能认定唯一原因。梯度路径变化也增加整体梯度尺度；尚未做GPS-only增益对照。码本后层低利用仍存在；当前健康门槛未过，不启动LLM，也不自动加初始化/正则。G10/G01/G11和LLM NLL均为未执行。

### 复现与记录

完整架构、样本/视图对齐、输入单位、损失、训练参数与强制变更表沿用本页HV-AllRecon方案及spec.md，本次只是只读结果分析，无训练方法变更。训练revision：60f140aa92799c36b21dccec87bef5223480bba4；原HV：70b82db195f600e3271030ddb5e6c228e4915f31；分析revision：a927df185a0bd102dc7715c621764b19f5ec17ce。

分析命令（cwd=/workspace/guozuyu/HTP-hv-allrecon）：`uv run python docs/plans/hv_allrecon/analyze_completion.py`。脚本验证完成状态、全部登记产物hash、1024唯一且配对一致的轨迹ID、24个probe训练预算及无失败条件，再计算区间。环境沿用uv.lock，无依赖修改；未修改模型代码。

权威产物根：`/workspace/guozuyu/HTP-independent-latent-sid-artifacts/hv_allrecon_20260917`。正式run：`pilot/porto/42/HVAllRecon/attempt_001`；比较JSON：`analysis/completion_comparison.json`；详细报告：`/workspace/guozuyu/HTP-hv-allrecon/docs/plans/hv_allrecon/completion.md`。本地同步于`experiments/hv_allrecon/`。原始启动记录保留，当前状态由本次完成记录更新。

后续动作：停止本轮进度检查。是否执行梯度尺度控制或独立码本初始化实验，另行决定；本轮未启动后续训练。
