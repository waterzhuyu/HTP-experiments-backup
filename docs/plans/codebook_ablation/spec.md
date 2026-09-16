# 码字数量 × 分层前缀监督消融（2026-09-16）

状态：实现验证中；尚无本轮科学结果。来源：Notion Research 最新有内容周报 Research 9.3–9.10；9.10–9.17 尚空白。
研究问题：原 H1 未优于 B2，是否可由码本容量或前层容量分配解释？不能预先承诺“排除”；未显著不等于等效，容量增大但实际利用仍低也不能声称排除了有效容量不足。

## 设计与预先判断

每种设置都做 B2/H1 配对，seed 42、314159、20260907（30 个完整 run）。两组均输入 GPS/Grid/Road、三 head 权重各 0.1；B2 读取 4/4/4，H1 读取 1/2/4。固定四层、每 event 四 SID token。

| 设置 | K1/K2/K3/K4 | 名义 bits/event | 目的 |
|---|---|---|---|
| base | 32/64/128/256 | 26 | 当前实现的复跑基准 |
| quarter | 8/16/32/64 | 18 | 更小容量的敏感性 |
| scale4 | 128/256/512/1024 | 34 | 整体扩容 |
| front_matched | 256/128/64/32 | 26 | 相同总码字数480、总码率26，容量前移 |
| front_large | 4096/8192/128/256 | 40 | 前两层大容量压力测试，覆盖训练观测类别数量量级 |

front_large 不意味着码字必须一一对应类别。改变 K 会改变码字参数量、LLM 新增 token 行数和名义码率，均是容量干预的伴随变化，必须记录。相同 K 内比较 H1−B2，再比较该差值跨 K 的交互；不能拿大码本 H1 对小码本 B2。

所有设置均执行完整 tokenizer→LLM→四 G→诊断，不用旧 pilot 的码本利用率门槛筛掉不利设置。资源或数值失败完整记录，不以成功子集代替30组计划。总预算96 GPU小时、100GiB产物，最多两 GPU各一个进程；每容量每seed配对上限12小时。不因达到预算声称完成，扩展需求另记。

主要指标为 G11 population_mean_penalized_error，四项 km 统计量 abs(log1p(generated)−log1p(real))平均，无效固定1，敏感性惩罚10。报告各seed H1/B2比值及相对改善、seed均值/标准差/范围；2000次配对轨迹 bootstrap（seed271828）用于每seed轨迹不确定性，另做seed与轨迹的两层配对重采样，注明仅3seed限制。五个容量对比及四个相对base交互同时报告，Holm校正，不挑最佳设置。

若任一容量主误差改善≥5%、跨seed同向且区间支持改善，认为容量可能调节方法效果，不能排除。若所有容量均无实用收益，且各容量改善的多重比较保守上界均低于5%，可限定说“在已测范围和预算内未见≥5%收益”；只有交互的同时区间落在预设±5个百分点内，才称对已测容量的实用效应稳健。区间过宽则证据不足。健康利用不足、前缀probe未改善时，区分名义容量与有效容量，不以NLL下降替代端到端效果。必要的等效判断不由不显著代替。

## 可复现方法

数据、切分沿用冻结 preflight：Porto，官方train中的8192 train / 1024 validation，20–200观测点整条轨迹；同GPS/time哈希去重，固定split seed20260907。官方封存test不读取。车辆/道路/日期跨split有重叠，只支持同城held-out-trip，不声称跨车/跨道路泛化。GPS经纬度按继承metadata中的固定Porto边界归一化（经度−8.69至−8.55，纬度41.13至41.19），该MBR并非本轮8192条训练轨迹拟合；dxy和percent_dist的均值/标准差由本轮训练集重新估计，Grid为100m ID，Road为逐点map-matched ID；按traj_id/point_index对齐，无新采样增强。padding按length mask。输入GPS B×T×2，Grid/Road B×T；时间不进入融合特征，仅诊断对齐。

ViewFusion：GPS Linear2→64、Road冻结128维Node2Vec→Linear64、Grid embedding64，各自LayerNorm，同点求和除sqrt(3)；共享HTP Encoder hidden64、ch_multi[1,2,2,4]、每级2残差块、dropout0.1、三次stride2。Encoder输出256维经Linear→64 latent；事件数ceil(T/8)，不是独立8点窗口，存在上下文感受野。辅助head为[64维累计前缀+8维offset onehot]→64→SiLU→每点目标，预测块内所有有效观测，末块mask。

RQ采用四层欧式最近邻，依次量化残差，码字向量求和；straight-through为z+stopgrad(sum−z)。损失为原percent/dxy重构 + 四前缀RQ均值（每层MSE(stopgrad(z),s_l)+0.25MSE(z,stopgrad(s_l))) + 0.1Grid CE/log(train observed Grid) + 0.1Road CE/log(train observed Road) + 0.1GPS坐标MSE；辅助损失按每轨迹有效点平均再跨轨迹平均。无usage、sampling、视图独占正则。精确原重构公式见继承stage_1/src/staged_attribution/model.py和上轮spec；执行前归档对应源码与解析配置。

初始化：所有模块从头。先按原K与相同seed构建原模型，再在隔离RNG中替换量化器，保证Encoder/Decoder/heads等非码本参数跨K逐tensor相同。各层保留前min(K,baseK)个原码字，新增码字以seed+10000+layer初始化，uniform幅度固定为该层原1/baseK；不随新K缩小。base设置与旧初始化完全一致。不加载旧训练checkpoint；这是受控初始化容量消融，不能泛化为所有初始化策略。Node2Vec保留原来源与冻结hash，缺少独立生成manifest的限制沿用。

Tokenizer AdamW lr1e-4、wd1e-4、betas(.9,.999)、eps1e-8、clip1、20epoch、batch128、1280steps；10epoch warmup起始ratio.01后cosine到.10；BF16、量化和CE FP32，num_workers0；固定seed和data order，cudnn deterministic true/benchmark false，无early stop。最终epoch用于比较。相同seed的B2/H1初始state一致，仅prefix routing不同。

Decoder为继承route-conditioned HTP：64→256后解码road percent+dxy；真实route几何和z-score恢复GPS；sampling三bit决定恢复点数，保持原观测进度语义，不声称真实时钟重建。正式解码逐轨迹执行；不采用已知有批量敏感性的旧干预结果。

Tokenizer冻结后各run独立Qwen3-1.7B LoRA：2epoch、1024steps、microbatch2×accum8，AdamW lr5e-4/wd1e-4/clip1，linear warmup.05，BF16+gradient checkpointing；r16/alpha32/dropout.05，q/k/v/o/gate/up/down；base冻结，新token行可训练。四SID串行a/b/c/d加sampling，assistant-only labels，最大2048，无额外模态标签。容量必须贯穿序列化、边界校验、词表、SID分层NLL和adapter恢复；保存rq_protocol及config hash，错配拒绝。生成greedy、batch1、max_new_tokens256，不重试无效样本；prompt与decoder真实route相同。

G00真实SID/真实sampling，G10生成SID/真实sampling，G01真实SID/生成sampling，G11均生成。相同validation原始population保留失败。报告主误差、有效率、G00→G11增量及Shapley，100个observation-progress锚点ADE/FDE（米），GPS/Road/Grid分布统计（共同train分箱，次要）、SID分层NLL/准确率、完整tuple、unigram校正、码本used/perplexity/max share、tuple熵/碰撞。冻结probe每单层与累计前缀预测三视图：MLP同辅助head、AdamW lr.001、10epoch、batch256events、seed42；不把低CE解释为视图独占。

## 方法变更表（每个K内B2 vs H1；跨K）

| 项目 | 同K B2→H1 | 跨K |
|---|---|---|
| 样本单元/视图/特征单位 | 相同 | 相同 |
| Encoder/downsampling/token支撑 | 相同 | 相同 |
| RQ/码本 | 相同 | 仅各层K及伴随名义码率、参数量 |
| Decoder/时间锚点 | 相同 | 相同 |
| 初始化/冻结 | 相同seed参数一致 | 非码本相同，共有码字嵌套，幅度固定 |
| 重构/VQ loss | 相同 | 相同 |
| 辅助loss/权重 | 权重相同；位置4/4/4→1/2/4 | 相同 |
| 训练数据/超参数 | 相同 | 相同；新增LLM词表行数随K变化 |
| 评估对齐/指标 | 相同 | 相同 |

## 路径、版本、命令与状态

工作区 /workspace/guozuyu/HTP-codebook-ablation；继承归档revision f0074bc52f712aa66560bf5f6f8870e47a20d748，原训练280f83b3e4fda51db95911203368788255fceda3；本轮实现revision须在提交后写入每run。
配置 stage_1/configs/codebook_ablation/{setting}.json。preflight沿用 /workspace/guozuyu/HTP-hierarchical-rqvae-artifacts/hierarchical_rqvae_gps_grid_road_v1_20260907/preflight，config冻结其manifest SHA，执行前校验原始数据/模型SHA。
命令：CUDA_VISIBLE_DEVICES=<gpu> uv run python -m codebook_ablation.run --config stage_1/configs/codebook_ablation/<setting>.json --preflight <above> --phase smoke或pilot --cells B2 H1 --seed <seed>。
产物 /workspace/guozuyu/HTP-codebook-ablation-artifacts/codebook_ablation_20260916_<setting>/<phase>/porto/<seed>/<cell>/attempt_NNN/，保存run_manifest、配置、所有stage哈希、checkpoint、adapter、metrics、资源耗时与失败。smoke不是科学结果。提交前测试，禁止dirty运行，不覆盖旧attempt。
已完成环境：uv sync --frozen，Python3.12.3；继承测试282 passed/32 warnings。新实现测试、smoke、科学指标当前N/A（待执行）。
