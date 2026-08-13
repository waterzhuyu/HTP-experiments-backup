# P1 固定 8 点 Token 多视图训练 Pilot

> 日期：2026-08-13  
> 状态：单随机种子 pilot；当前变体未通过  
> 环境：`3090s:/workspace/guozuyu/HTP`  
> 实验代码提交：`f9f98e983e9a64be914bc45f4ed4d21c00dca01d`

## 研究问题

检验原始 HTP 的“每约 8 个采样点产生一个 token”结构在加入多视图训练后，能否提高跨采样 agreement。该实验用于区分两种解释：

1. 固定点数结构本身无法获得跨采样稳定性；
2. 原始 HTP 低 agreement 主要因为没有接受跨采样多视图训练。

本实验不改成固定时间窗口，也不使用窗口级池化。

## 方法与协议

- 初始化：成都、Porto 各自 100-epoch、seed-42 的原始 HTP RQ-VAE checkpoint；
- 模型：保留原始 HTP encoder、三次 2× 序列下采样、道路条件 decoder 和四层 RQ；
- 多视图：从同一轨迹纯删点构造 native、2×、3×，不插值；
- 原始重构：三个视图各自计算 HTP 的 `percent_dist + dxy` reconstruction 和 quantization loss；
- 一致性：将各视图的量化前 embedding 映射回原始点索引，只在完全相同的固定 8 点锚点处计算 cosine loss；
- 总损失：三个视图平均 reconstruction + 平均 quantization + `λ=1.0` 的 2×/3× cosine consistency；
- 训练：每城 8,192 条长度至少 48 点的训练轨迹，15 epochs，batch size 32，learning rate `1e-5`；
- seed：`20260813`；
- 确认集：每城 1,024 条、seed `314159`，排除之前 256 条探索轨迹；
- 比较对象：微调前的同一 100-epoch checkpoint。

运行命令：

```bash
env VIRTUAL_ENV=/workspace/guozuyu/HTP-main/.venv \
uv run --active python stage_1/src/p1_fixed8_multiview_pilot.py \
  --city <chengdu|porto> \
  --checkpoint <city-100epoch-rqvae.pt> \
  --output-dir stage_1/runs/p1_fixed8_multiview \
  --train-examples 8192 --epochs 15 --batch-size 32 \
  --learning-rate 1e-5 --consistency-weight 1.0 \
  --seed 20260813 --device <cuda:0|cuda:1>
```

## Agreement 结果

### 第一层 code agreement

| 城市 | 模型 | 2× | 3× | 相位变化 | 20% 缺失 | 10 m 噪声 |
|---|---|---:|---:|---:|---:|---:|
| 成都 | 原始 HTP | .177 | .078 | .154 | .374 | .923 |
| 成都 | 固定 8 点多视图 | **.878** | **.833** | **.859** | **.863** | **.973** |
| Porto | 原始 HTP | .122 | .086 | .108 | .205 | .848 |
| Porto | 固定 8 点多视图 | **.913** | **.878** | **.915** | **.913** | **.975** |

固定 8 点结构在接受多视图训练后，两城 2×/3× 第一层 agreement 均超过预注册的 `.70/.60` 门槛。因此，原始 HTP 的低 agreement 不能仅归因于固定点数结构；训练协议是重要因素。

### 四层 tuple agreement

| 城市 | 模型 | 2× | 3× | 相位变化 | 20% 缺失 | 10 m 噪声 |
|---|---|---:|---:|---:|---:|---:|
| 成都 | 原始 HTP | .0000 | .0000 | .0000 | .0035 | .2289 |
| 成都 | 固定 8 点多视图 | .0020 | .0005 | .0016 | .0164 | .4578 |
| Porto | 原始 HTP | .0000 | .0000 | .0000 | .0038 | .0663 |
| Porto | 固定 8 点多视图 | .0009 | .0003 | .0018 | .0122 | .2890 |

完整 tuple 在降采样下仍接近零。提升主要集中在第一层粗 code，第二至第四层的 exact agreement 仍低。

## 选择性一致性

使用第一层 token 直方图的 Jensen–Shannon similarity：

| 城市 | 模型 | 同轨迹 2× | 同道路困难负样本 | Margin |
|---|---|---:|---:|---:|
| 成都 | 原始 HTP | .312 | .341 | -.029 |
| 成都 | 固定 8 点多视图 | .921 | .776 | **+.145** |
| Porto | 原始 HTP | .220 | .239 | -.019 |
| Porto | 固定 8 点多视图 | .941 | .921 | **+.019** |

成都通过暂定 `+.05` margin；Porto 虽由负转正，但未达到 `+.05`。Porto 的正样本和困难负样本都高度相似，提示表示区分度不足。

## 行为信息与码本健康度

### 第一层行为 probe

| 城市 | 指标 | 原始 HTP | 固定 8 点多视图 |
|---|---|---:|---:|
| 成都 | 速度 Macro-F1 | .498 | **.253** |
| 成都 | 停留 Macro-F1 | .785 | **.599** |
| 成都 | top-32 road Macro-F1 | .176 | .055 |
| Porto | 速度 Macro-F1 | .448 | **.252** |
| Porto | 停留 Macro-F1 | .662 | **.631** |
| Porto | top-32 road Macro-F1 | .047 | .016 |

道路 probe 下降，但同时速度和停留信息也下降，因此不能把 road probe 降低单独解释为成功去身份化。

### 第一层码本

| 城市 | 模型/评估范围 | 使用码数 | 归一化 perplexity | 最大 code 占比 |
|---|---|---:|---:|---:|
| 成都 | 原始 HTP / 完整 test | 32/32 | .947 | .050 |
| 成都 | 多视图 / 1,024 确认集 | 13/32 | **.103** | **.537** |
| Porto | 原始 HTP / 完整 test | 32/32 | .965 | .068 |
| Porto | 多视图 / 1,024 确认集 | 12/32 | **.050** | **.894** |

原始和微调模型的码本统计评估范围不同，数值不用于精确同分布比较；但多视图模型在确认集上的集中程度已足以判定粗层严重压缩，Porto 接近单 code 主导。

第二至第四层仍使用较多 code，部分 layer-2/tuple 行为 probe 比原始模型更高，但无法抵消第一层作为 coarse primitive 的坍缩和完整 tuple 不稳定。

## 训练状态

| 城市 | Epoch 15 reconstruction | quantization | consistency |
|---|---:|---:|---:|
| 成都 | .818 | .118 | .027 |
| Porto | .670 | .091 | .033 |

训练过程数值稳定，没有出现显存错误或 loss 发散。上述 reconstruction 是三个训练视图的平均训练损失，不能与原始 HTP 的 test reconstruction 直接比较。

## 结论

1. **固定 8 点结构可以通过多视图训练获得高 coarse-code agreement。** 因此先前 `.177/.122` 的结果不是固定点数结构的不可避免结果。
2. **当前方法仍未通过 P1。** 第一层 agreement 的大幅提升伴随明显码本压缩和行为信息下降，尤其 Porto 最大 code 占比达到 `.894`。
3. **固定时间窗仍有物理解释优势。** 本实验提高了同中心 token 的一致性，但 native/2×/3× token 仍分别覆盖不同时间范围；高 agreement 不代表它们描述了同一尺度的运动原语。
4. **不能只用 layer-1 agreement 判定成功。** 必须联合检查完整 tuple、选择性 margin、行为 probe、码本 perplexity 和最大 code 占比。

准确表述是：

> 多视图训练能够把原始固定 8 点 HTP tokenizer 的粗层跨采样 agreement 提高到 `.83–.91`，但当前纯正样本 cosine 主要通过粗层码本压缩获得该收益，尚未得到采样鲁棒且行为可辨识的运动原语。

## 下一步

- 加入不同速度/停留状态的困难负样本，改用选择性对比目标；
- 对每层加入 variance/covariance 或 code-usage 防坍缩约束；
- 分层设置一致性权重：粗层保持采样稳定，细层保留重构残差；
- 与固定时间窗模型在相同 codebook、decoder 和训练数据下做严格受控消融；
- 至少三个随机种子后再决定是否保留固定 8 点路线。

## 产物

- 代码：`stage_1/src/p1_fixed8_multiview_pilot.py`
- 测试：`stage_1/tests/test_p1_fixed8_multiview_pilot.py`
- 结果：`stage_1/runs/p1_fixed8_multiview/{chengdu,porto}_fixed8_multiview_seed20260813.json`
- Checkpoint：同目录 `.pt` 文件，不提交 Git；
- 测试：Stage 1 全套 `13 passed`。
