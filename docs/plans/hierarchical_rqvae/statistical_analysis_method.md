# Pilot 统计分析实施补充（2026-09-08）

训练源固定 280f83b；分析独立 worktree `/workspace/guozuyu/HTP-hierarchical-rqvae-analysis`。主终点、配对单位、2000次 bootstrap、seed271828、失败惩罚1及敏感性10、门槛沿用原spec。四G先逐轨迹配对再计算SID/sampling Shapley，二者之和必须等于G11−G00；另报interaction。

已有 stage4 的 JSD 分箱由每个模型真实/生成样本共同确定，不适合把不同支持的模型直接作差。因此三域次要分析另设仅使用冻结8192条训练轨迹的公共支持：0到1含端点的21个等距分位点，去重，加浮点数最大有限值正负端点接住underflow/overflow；常数训练分布使用c±max(1e−10,|c|×1e−10)避免将所有新值也当常数。四个直方图指标分别构造支持，所有cell及bootstrap保持不变。保存support.json及SHA；原stage4指标另行保留。

次要统计实现细节在已检查B2/H1主终点后敲定，明确标为实施后的次要分析，不能称完全预注册。未根据验证集分布选择分箱或追求显著性。

按trajectory多项式权重重采样，在每次draw重新聚合真实/生成直方图、Grid/Road折叠连续重复ID计数，并重新计算JSD、cosine、top100 F1。总体regret=failure_fraction+valid_fraction×successful_regret，cosine/F1转为1−score；成功指标与失败率单独报告。失败轨迹保留在原1024总分母。

所有8个三域指标×5个对照（H1−B2、B1−B0、B2−B1、H2−B2、H3−B2）组成40检验Holm family。使用近似双侧null-centered bootstrap p=(1+count(|draw−estimate|≥|estimate|))/(2000+1)，不能将P(draw<0)称p值。未完成对照p按1暂占位，全部完成后重算。主比较与三域次要family分开解释。

G00 GPS门槛以1024条完整oracle population的matched-progress ADE均值落实；原spec未逐项指定GPS error，此处显式说明，而非默默替换。G11 ADE/FDE/DTW是成功条件下指标，必须连同失败数量和主要全population终点解释。单seed bootstrap仅代表给定训练模型上的轨迹不确定性，不能替代独立训练seed。
