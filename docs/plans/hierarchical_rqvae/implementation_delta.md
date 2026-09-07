# 正式实验前的实现补充（2026-09-08）

本记录说明相对首版 smoke revision `5e2dc7df4ca7cb6d88fff15a3b17f0d1a12b80bd` 的补充。新增代码尚须在 authoritative worktree 提交、完成真实数据诊断审计后才启用 pilot。

## 训练方法保持项

六 cell 的样本、视图、prefix routing、RQ/codebook、straight-through、权重、head、encoder、主 decoder、初始化、epochs、batch和LR方案保持 spec 不变。新增 optional callback 在资源充足时不改变梯度或更新步数。新的 `decode_quantized` 只用于诊断；四层完整前缀已通过与正式decoder数值等价测试。

## 评估补充

- teacher-forced CE 在 FP32 计算，逐 token 累积；报告每层 NLL 和 SID-only NLL，剔除 sampling 与分隔符。它们补充原有 overall NLL/accuracy，不替换最终 GPS 生成主要终点。
- 前缀与单层 embedding 均使用实际离散 codebook lookup，冻结后分别拟合 4×3×2=24 个 head。结构为 `(latent_dim+8)→64→target_dim`、SiLU，seed42，AdamW lr1e-3/weight_decay1e-4，正式10epoch、256event/batch；smoke1epoch。原始权重和event顺序在同目标的各表示之间一致。
- probe 分类用 raw CE，回归用normalized GPS MSE；按逆轨迹长度对有效点加权，每batch按权重和归一化。probe CE未除log词表大小，与tokenizer辅助损失区别明确；同view对照使用相同口径。记录checkpoint、每epoch进度、steps和验证loss/accuracy。不得把probe当作独立语义标签真值。
- 2×/3×恢复视图保留首尾，GPS线性插值，Grid/Road最近时间邻居；距离相同取早者。诊断时间从原始record读float64，修复 inherited Unix float32精度问题；训练模型未使用time特征，因此该修复不改变本轮训练输入的GPS/Grid/Road。
- 前缀截断以真实累计量化和解码（缺失后层向量为零）；层交换在相同event数、route长度整除5分桶内，按hash排序循环选donor，单层1..4和L3+L4分别交换；没有donor标记N/A。交换结果与完全相同eligible population的native结果比较。均属于诊断，不声称因果解耦。
- GPS误差在100个均匀observation-progress锚点插值，输出ADE、FDE、DTW累计距离和点数误差；单位米。使用reference平均纬度的local-equirectangular尺度；不是时钟时间ADE。invalid保留，不用successful-only均值代替主population指标。

## 运行与证据

新增训练step/评估batch/生成batch的预算检查；生成中断保留partial输出；训练超限保留可训练权重和optimizer。记录每阶段耗时、GPU allocated/reserved峰值；消费上游产物前核对hash，包括adapter目录。pilot前要求至少23GiB可用显存。

还要对冻结cohort进行原始/processed/normalized GPS、Grid、pointwise Road、timestamp逐点一致性审计，并显式报告train/validation车辆、道路、日期重叠。只允许held-out-trip解释；不能声称这些维度全部独立。

新的诊断smoke会读取已完成、hash验证通过的原smoke模型；通过只读链接在独立审计目录输出结果和新的执行SHA，不重写原attempt、不重复训练同一模型。首版smoke的overall指标与原SHA保持不变。
