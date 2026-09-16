# 四组结果分析与完成验收

状态：pending experiments；分析入口已就绪。此文不声明实验完成。

## 完整结果需要的证据

1. HV/LF/H1/B2，seed42，冻结8192/1024 cohort相同；每组tokenizer1280step/20epoch、LLM1024step，九阶段completed；逐产物SHA、bytes、run cell/config/preflight/revision一致。不能用smoke替代。
2. G00/G10/G01/G11每组完整1024轨迹，无重复、不丢失败；主误差保留invalid=1，另做10敏感性。确认G00重构和G11有效率后解释主误差；不能以success-only ADE/JSD代替全人口主指标。
3. 所有24个冻结probe（四层×单层/前缀×三视图）均10epoch，比较指定前缀信息及前缀增加后的信息保留。不把probe可提取信息视为独占、因果或真实运动语义。
4. NLL按四层报告，与同一train unigram(alpha=.5)在validation上的交叉熵比较；逐层denominator必须等于冻结validation event数。报告码本max share/perplexity/active、tuple熵及碰撞，避免仅凭NLL下结论。
5. 补充teacher-forced完整四层tuple准确率及free-running按event序号对齐的tuple准确率，所有缺失/失败视为错、额外event单列；恢复adapter与原SID NLL差≤.005、denominator一致。现成入口 `hierarchical_rqvae.audit_tuple_accuracy` 可直接用于本次固定32/64/128/256协议，无tokenizer工厂依赖，待run completed后运行，不对运行中的adapter评估。
6. 在结果报告中逐条区分观测、机制解释、候选改进与验证优先级。至少结合HV−LF（同模块、目标及路由变化）、LF−H1（独立encoder联合参数量变化）、HV−H1（总体方法）、H1−B2（复跑旧比较）。只对已测范围作结论；若负面结果成立，保留并解释，不修改原run或挑有利种子。

## 统计口径

配对单元为1024条验证轨迹；2000次bootstrap，seed271828，各组共用同一抽样权重。每G输出candidate−baseline均值差和相对改善1−candidate/baseline；primary为HV−LF的G11。另报告4个预定G11对比的Bonferroni同时区间（描述性，不替代跨训练seed推断）。没有基于未显著的等效判断。GPS ADE仅在两组共同成功人口配对、明确剔除数；全人口失败已进入主指标。Shapley SID/sampling分量和交互项按同一轨迹计算，再bootstrap。

bootstrap及当前门槛来自spec.md；本轮仅1个训练seed，置信区间不能说明训练随机性。生成route始终oracle；无跨道路/车辆/城市泛化主张。独立JSD分箱可能不同，因此不从当前success-only JSD作主因果比较；需要分布结论时再使用共同训练支持和全人口统计。

## 已完成工程与验证

分析worktree `/workspace/guozuyu/HTP-independent-latent-sid-analysis`，训练worktree未改。分析代码提交 `223ee45`，4项测试通过（0.12s），涵盖缺失/重复轨迹、失败惩罚、已知改善方向、Shapley分解和拒绝不完整训练预算。原冻结latent分析提交 `869a8ef`、报告提交 `90ff6ec`。

```bash
uv run python -m independent_latent_sid.analyze_results --root /workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917 --config stage_1/configs/independent_latent_sid/pilot.json --output /workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/analysis/results_snapshot_001.json
```

首份快照 status=incomplete、verified_cells=[]；HV正在stage2，LF/H1/B2未开始。快照只描述阶段完整性，不能以该文件代替活进程检查。PID2602619及子进程2602623已现场确认存活，日志从98继续增长；不重启、不暂停、不干预原实验。

补充tuple评估的待执行命令（输出目录每次唯一）：

```bash
CUDA_VISIBLE_DEVICES=3 uv run python -m hierarchical_rqvae.audit_tuple_accuracy --root /workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917 --cells HV LF H1 B2 --output /workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/analysis/tuple_accuracy_001
```

分析新增训练、初始化、优化器、loss、encoder、量化器、decoder变化均不适用；完全读取已提交训练run的冻结产物。输入、sample unit、mask、token支持和GPS恢复定义继承spec.md。新的统计输出不是新训练实验。结果JSON保存分析revision、命令、source manifest路径及SHA；路径和config/preflight身份可追溯。
