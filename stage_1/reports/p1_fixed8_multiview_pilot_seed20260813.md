# P1 原始固定 8 点 Token 多视图 Pilot 报告

> 日期：2026-08-13
> 状态：单随机种子对照 pilot；当前实现未通过 P1
> 环境：`3090s:/workspace/guozuyu/HTP`
> 代码提交：`f9f98e983e9a64be914bc45f4ed4d21c00dca01d`
> 结果提交：`0c9ea89bf073f925939189b132888483331276ab`
> 记录规范：[实验记录方法规范](experiment_recording_standard.md)

## 目标与协议

检验原始 HTP 的“约每 8 个输入点产生一个 token”结构能否通过多视图训练获得采样鲁棒性。该实验不改为固定时间窗，也不改 RQ-VAE 层数或词表大小。

- 数据：成都、Porto；每城 8,192 条长度至少 48 点的训练轨迹，1,024 条冻结确认轨迹；
- 视图：native、2×、3×；仅删除原始点，不插值；
- 初始化：各城市原始 HTP 100 epoch、seed 42 checkpoint；
- 训练：seed 20260813，15 epochs，batch size 32，学习率 `1e-5`；
- 目标：三视图各自执行原 HTP 重构与量化，在共享原始点锚点处加入量化前 embedding cosine 一致性，`λ=1.0`；
- 解码器与四层 RQ 词表保持原样，词表大小依次为 `[32, 64, 128, 256]`。

### 方法变更表

| 方法组成 | 原始 HTP baseline | 本次固定 8 点多视图 | 是否变化 |
|---|---|---|---|
| 样本物理单元 | 一条变长轨迹 | 同一条变长轨迹 | 否 |
| 视图构造 | baseline 单视图训练；扰动只用于测试 | 训练中加入 native/2×/3× 纯删点视图 | 是 |
| 输入与道路条件 | 原 HTP batch loader | 与 baseline 相同 | 否 |
| encoder/downsampling | 原 HTP，三次 2× 下采样 | 与 baseline 相同 | 否 |
| token 支持 | 每约 8 个当前视图输入点一个 token | 与 baseline 相同；因此不同视图仍覆盖不同物理时长 | 否 |
| RQ/codebook | 四层 `[32,64,128,256]` | 与 baseline 相同 | 否 |
| decoder/重构目标 | road-conditioned `percent_dist+dxy` | 三视图分别使用同一原始 decoder/目标定义 | 否 |
| 初始化 | 100-epoch、seed-42 checkpoint | 从该 checkpoint 继续微调，所有模型参数可训练 | 是 |
| 新增损失 | 无 | 完全共享原始点锚点的量化前 cosine，λ=1 | 是 |
| 评估 | 冻结确认集上的扰动 | 同一确认集、同一评估脚本与 baseline checkpoint 对照 | 否 |

### 视图、token 对齐与损失细节

- 一个训练 record 必须至少 48 个 GPS 点；native 保留索引 `0,1,2,...`，2×保留 `0,2,4,...`，3×保留 `0,3,6,...`；不强制追加末点、不做插值；`gps_seq/time_seq/dxy/percent_dist` 同步删点，road `path` 保持不变；
- 三次 2× encoder 下采样使第 `j` 个 token 的代表原始点定义为 `view_indices[min(8*j, len(view_indices)-1)]`。native 与降采样视图只有在该代表原始点索引完全相同时才构成一致性正样本；未对齐 token 不计算 consistency；
- 这只是离散锚点相等，不代表 token 覆盖的邻域或物理时长相同。例如相同数量的当前视图输入点在 2×/3×下覆盖约 2/3 倍原生时长；
- 通过 forward hook 捕获 `rq_pre_quant_linear` 输出，按每条轨迹有效 token 数拆分；一致性在量化前 embedding 上计算，不直接监督 code id，也不使用负样本；
- 三个视图分别由完整原 HTP 模型重构并计算 `recon_loss` 与 `quant_loss`。总损失为：

```text
L_rec  = mean(recon_native, recon_2x, recon_3x)
L_quant = mean(quant_native, quant_2x, quant_3x)
L_cons = 0.5 * [mean_aligned(1-cos(pre_native,pre_2x))
                + mean_aligned(1-cos(pre_native,pre_3x))]
L_total = L_rec + L_quant + 1.0*L_cons
```

- AdamW，learning rate `1e-5`，weight decay `1e-4`，batch size 32，15 epochs，gradient clip 1.0；seed 20260813；训练中未冻结模块。

### Decoder 与评估口径

- 本实验不新定义“8 点解码为 9 个 GPS 点”。它完整沿用 HTP road-conditioned decoder，以量化 token、道路序列/道路几何、`dxy` 与 `road_percent` 为条件，优化原 HTP 的 `percent_dist+dxy` reconstruction；
- 本次 agreement 实验没有把 decoder 输出另行反变换、拼接成 GPS 并报告 GPS 误差；因此此处的主要结论只针对 tokenizer code，而不是解码 GPS 的采样一致性；
- 确认集为每城 1,024 条、seed 314159，并排除 seed-42 的 256 条探索轨迹；baseline 是微调前的同一 100-epoch checkpoint；
- 每个扰动使用原评估脚本的 token 对齐，分别报告四层逐层 exact agreement 和四个 code 全相等的 tuple agreement；同时报告第一层 token 直方图 JS 选择性 margin、速度/停留/道路 probe 以及逐层 codebook 使用；
- 训练记录来自 `processed_train_traj.pkl` 中长度至少 48 点的轨迹，以 seed 20260813 无放回抽取 8,192 条。

### 运行命令

```bash
env VIRTUAL_ENV=/workspace/guozuyu/HTP-main/.venv \
uv run --active python stage_1/src/p1_fixed8_multiview_pilot.py \
  --city <chengdu|porto> \
  --checkpoint <对应城市100epoch-seed42-checkpoint> \
  --output-dir stage_1/runs/p1_fixed8_multiview \
  --train-examples 8192 --epochs 15 --batch-size 32 \
  --learning-rate 1e-5 --consistency-weight 1.0 \
  --seed 20260813 --device <cuda:0|cuda:1>
```

实际城市参数：

| 城市 | 初始化 checkpoint | device | 指标 JSON |
|---|---|---|---|
| 成都 | `/workspace/guozuyu/HTP-main/reproduction/runs/chengdu/repro_full_100epoch_seed42_20260717/rqvae/rqvae.pt` | cuda:0 | `stage_1/runs/p1_fixed8_multiview/chengdu_fixed8_multiview_seed20260813.json` |
| Porto | `/workspace/guozuyu/HTP-main/reproduction/runs/porto/repro_full_100epoch_seed42_20260716/rqvae/rqvae.pt` | cuda:1 | `stage_1/runs/p1_fixed8_multiview/porto_fixed8_multiview_seed20260813.json` |

两份 JSON 中解析后的共同配置为：`train_examples=8192`、`epochs=15`、`batch_size=32`、`learning_rate=0.00001`、`consistency_weight=1.0`、`seed=20260813`。

## Agreement 结果

### 第一层精确 agreement

| 城市 | 模型 | 2× | 3× | 相位变化 | 20% 缺失 | 10 m 噪声 |
|---|---|---:|---:|---:|---:|---:|
| 成都 | 原始 HTP | .177 | .078 | .154 | .374 | .923 |
| 成都 | 固定 8 点多视图 | **.878** | **.833** | **.859** | **.863** | **.973** |
| Porto | 原始 HTP | .122 | .086 | .108 | .205 | .848 |
| Porto | 固定 8 点多视图 | **.913** | **.878** | **.915** | **.913** | **.975** |

### 四层 RQ tuple 精确 agreement

| 城市 | 模型 | 2× | 3× | 相位变化 | 20% 缺失 | 10 m 噪声 |
|---|---|---:|---:|---:|---:|---:|
| 成都 | 原始 HTP | .000 | .000 | .000 | .003 | .229 |
| 成都 | 固定 8 点多视图 | .002 | .001 | .002 | .016 | .458 |
| Porto | 原始 HTP | .000 | .000 | .000 | .004 | .066 |
| Porto | 固定 8 点多视图 | .001 | .000 | .002 | .012 | .289 |

提升主要集中在第一层。以 2× 为例，成都四层 agreement 依次为 `.878/.287/.057/.011`，Porto 为 `.913/.237/.048/.007`；完整 tuple 仍几乎不一致。

## 防坍缩与语义检查

| 城市 | L1 使用码数 | L1 最大码占比 | 速度 macro-F1：原始→多视图 | 停留 macro-F1：原始→多视图 | 选择性一致性 margin |
|---|---:|---:|---:|---:|---:|
| 成都 | 13/32 | .537 | .498 → .253 | .785 → .599 | +.145 |
| Porto | 12/32 | **.894** | .448 → .252 | .662 → .631 | +.019 |

原始 HTP 在完整测试集上的第一层使用 32/32 个码，最大码占比仅为成都 `.050`、Porto `.068`。尽管评估样本范围不同，多视图模型特别是 Porto 的第一层压缩仍非常明显。Porto 的正样本与同路困难负样本 JS 相似度都很高，选择性 margin 未达到暂定 `.05` 门槛。

## 判断

1. **原始固定 8 点结构并非天然无法提高 agreement。** 加入多视图目标后，第一层 2×/3× agreement 可达到 `.83–.91`，说明训练协议本身是原始 HTP 低一致性的一个重要原因。
2. **当前固定 8 点多视图实现未通过 P1。** 高 agreement 主要由第一层码本压缩获得，完整 RQ tuple 仍不稳定，第一层速度与停留可分性明显下降。
3. **agreement 不能单独作为通过标准。** 必须同时检查逐层与 tuple 一致性、码本利用、行为 probe，以及正样本相对真实行为困难负样本的 margin。
4. **固定时间窗仍更适合作为主线。** 固定 8 点的 native、2×、3× token 覆盖不同物理时长；即使输出码接近，也不能保证它们表达相同时间尺度的运动事件。
5. **Gate 1 继续关闭。** 下一版需要选择性一致性、防坍缩约束和 P2 行为辅助目标。

准确表述是：**多视图训练能够把原始固定 8 点 HTP tokenizer 的粗层跨采样 agreement 提高到 `.83–.91`，但当前纯正样本 cosine 主要通过粗层码本压缩获得该收益，尚未得到采样鲁棒且行为可辨识的运动原语。**

## 可复现产物

- 远端代码：`stage_1/src/p1_fixed8_multiview_pilot.py`
- 远端测试：`stage_1/tests/test_p1_fixed8_multiview_pilot.py`；Stage 1 回归共 13 tests passed
- 远端完整报告：`stage_1/reports/p1_fixed8_multiview_pilot_seed20260813.md`
- 远端指标：`stage_1/runs/p1_fixed8_multiview/{chengdu,porto}_fixed8_multiview_seed20260813.json`
- checkpoint 保留在远端运行目录，未提交 Git
