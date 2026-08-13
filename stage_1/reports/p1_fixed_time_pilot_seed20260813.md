# P1 固定时间窗 Pilot 报告

> 日期：2026-08-13
> 状态：单随机种子 pilot；不能替代多种子确认实验
> 环境：`3090s:/workspace/guozuyu/HTP`
> 实现与报告提交：`0ecbe11 feat(stage1): add fixed-time P1 pilot`
> 记录规范：[实验记录方法规范](experiment_recording_standard.md)

## 研究问题与方法变更

本 pilot 同时包含两层比较，必须分开解释：

1. 原始 HTP B6 与 P0-time：不仅把可变物理时长的固定点数 token 改成 120 秒窗口，也同时更换了 encoder、窗口池化方式、RQ 层数和 decoder 目标。因此该比较验证的是“窗口级方案整体是否可行”，**不是只改变时间支持的严格单变量消融**。
2. P0-time 与 P1-time：两者数据、模型、量化器和 decoder 完全相同，只改变 cosine consistency 权重；这是对一致性损失的受控比较。

| 方法组成 | 原始 HTP B6 | P0/P1-time | 是否变化 |
|---|---|---|---|
| 样本物理单元 | 变长轨迹，经三次 2× 下采样后每约 8 个输入点产生一个 token | 一个 120 秒窗口产生一个 embedding/RQ tuple | 是 |
| 多视图 | 原始评估时纯删点；不同视图 token 覆盖不同物理时长 | `[start,start+120s]` 相同边界内纯删点并强制保留末点 | 是 |
| 输入 | HTP 原始归一化 GPS、道路与运动特征 | `(局部 x km, 局部 y km, 窗内相对时间)` | 是 |
| encoder | 原始 HTP 序列 encoder，三次 2× 下采样 | 两层 Transformer + attention pooling，每窗一个 64 维向量 | 是 |
| quantizer | 四层 RQ，词表 `[32,64,128,256]` | 两层 Residual VQ，每层 32 码、64 维 | 是 |
| decoder | 原始 road-conditioned decoder | `64→128→18` MLP，输出 9 个固定时间锚点的局部 XY | 是 |
| 初始化 | 100-epoch HTP checkpoint | 从头随机初始化 | 是 |
| reconstruction/VQ | 原 HTP `percent_dist+dxy` 与四层 RQ | 9 点局部坐标 MSE 与两层 VQ loss | 是 |
| consistency | 无 | P0 为 0；P1 在量化前窗口 embedding 上计算 native–2×/3× cosine | P0→P1 时是 |
| 训练数据 | 论文 baseline 数据流程 | 每城 8,192 个训练窗 | 是 |
| 评估 | token 序列位置对齐 | 同一窗口天然一一对应 | 是 |

## 可复现方法

### 数据与固定时间窗构造

- 从 `traj_data/<city>/traj.parquet` 读取 `traj_id/time/geometry`，分别按 `train_index.npy` 和 `test_index.npy` 构造训练与测试；训练索引用 seed `20260813` 打乱，测试用 `20260814`；
- 从每条轨迹首个时间戳开始，以 120 秒为步长产生不重叠候选窗，窗口使用闭区间 `[start,start+120]`；末端不足完整 120 秒的窗口丢弃；
- native 窗至少 7 点，且首末观测实际覆盖至少 `0.8×120=96` 秒；3×视图至少 3 点；
- 经纬度以 native 窗首点为局部原点，用局部等距近似转为米，再除以 1,000 得到 km；每点输入为 `(x_km,y_km,(t-start)/120)`；
- native 使用窗内全部点；2×/3× 分别取位置 `0,2,4,...` 与 `0,3,6,...`。若最后一个 native 点未被选中，额外追加该末点；不插值输入点；
- 三个视图共享同一物理起止边界，但点数不同。变长序列按 batch padding，并用真实长度产生 Transformer padding mask。

### 目标、token 与解码语义

- 在归一化时间 `0,1/8,...,1` 上设置 9 个锚点；用 native 观测在时间轴上分别线性插值局部 `x/y`，形成 18 维 reconstruction target；
- `3→64→64` point MLP 后进入 2 层、4 头 Transformer，hidden size 64、FFN 256、dropout 0.1；learned query attention pooling 把整窗变成一个向量，再经 `LayerNorm+Linear` 得到 64 维量化前 embedding；
- 每个 120 秒窗口只产生一个两层 RQ tuple，不产生长度为 9 的 token 序列；两层 codebook 均为 `32×64`；第二层量化第一层残差，量化向量相加后用 straight-through 送入 decoder；
- decoder 为 `64→128→18` MLP，18 维 reshape 后代表 9 个时间锚点的局部 XY；本 pilot 在局部 km 坐标中计算重构 MSE，**没有执行反投影到经纬度、道路约束或相邻窗口 GPS 拼接**。

### 损失与优化

对三个视图分别前向，并计算：

```text
L_rec  = mean_v MSE(decoder(q_v), target_native_9_points)
L_vq   = mean_v mean_layer [MSE(code,residual.detach)
                             + 0.25*MSE(residual,code.detach)]
L_cons = 0.5 * [(1-cos(pre_native,pre_2x))
                + (1-cos(pre_native,pre_3x))]
L_total = L_rec + L_vq + lambda * L_cons
```

- P0-time：`lambda=0`；P1-time：`lambda=1.0` 或 `0.1`；一致性发生在量化前的单窗口 embedding 上，不使用负样本；
- AdamW，learning rate `3e-4`，weight decay `1e-4`，batch size 128，15 epochs，gradient clip 1.0；seed 20260813；模型从头训练。

### 评估口径

- 每城 2,048 个独立 test 窗；一个 native 窗与其 2×/3×视图直接一一对应；
- layer-1 agreement 比较第一个 RQ code，tuple agreement 要求两个 RQ code 都完全相同；
- retrieval 用量化前/量化后 cosine 矩阵在 2,048 个 native 候选中检索同一窗口；
- speed label 由 native 窗总距离/实际覆盖时间按 `[1,5,15] m/s` 分四档；stop label 为相邻点速度低于 `1 m/s` 的比例是否至少 0.5；probe 仅用 native 第一层 code 作类别到训练集多数标签映射；
- codebook 报告 used codes、归一化 perplexity 与最大码占比；重构 MSE 单位为局部 `km²` 坐标元素均值。

### 运行配置

正式运行对成都与 Porto 分别执行下列配置，`variant=p0_time` 时脚本强制 `lambda=0`：

```bash
uv run python stage_1/src/p1_fixed_time_pilot.py \
  --city <chengdu|porto> --variant <p0_time|p1_time> \
  --output-dir <对应运行目录> --window-seconds 120 \
  --train-windows 8192 --test-windows 2048 \
  --epochs 15 --batch-size 128 --learning-rate 3e-4 \
  --consistency-weight <0|0.1|1.0> --seed 20260813 \
  --device <cuda:0|cuda:1>
```

实际解析配置矩阵：

| 城市 | variant | λ | device | output directory |
|---|---|---:|---|---|
| 成都 | p0_time | 0 | cuda:0 | `stage_1/runs/p1_fixed_time_pilot` |
| 成都 | p1_time | 1.0 | cuda:0 | `stage_1/runs/p1_fixed_time_pilot` |
| 成都 | p1_time | 0.1 | cuda:0 | `stage_1/runs/p1_fixed_time_pilot_lambda0p1` |
| Porto | p0_time | 0 | cuda:1 | `stage_1/runs/p1_fixed_time_pilot` |
| Porto | p1_time | 1.0 | cuda:1 | `stage_1/runs/p1_fixed_time_pilot` |
| Porto | p1_time | 0.1 | cuda:1 | `stage_1/runs/p1_fixed_time_pilot_lambda0p1` |

六个 JSON 中解析后的共同配置均为：`window_seconds=120`、`train_windows=8192`、`test_windows=2048`、`epochs=15`、`batch_size=128`、`learning_rate=0.0003`、`seed=20260813`。

## 主要结果

| 城市 | 版本 | 2×/3× layer-1 agreement | 2× tuple | 2×/3×检索 Top-1 | 重构 MSE | 速度 F1/多数类 | L1 使用码数/最大占比 |
|---|---|---:|---:|---:|---:|---:|---:|
| 成都 | P0-time | .971 / .951 | .864 | .836 / .623 | .0119 | .315 / .202 | 10 / .189 |
| 成都 | P1 λ=1 | .980 / .973 | .933 | .787 / .537 | .0201 | .202 / .202 | 6 / .640 |
| 成都 | P1 λ=.1 | .982 / .975 | .919 | .802 / .563 | .0180 | .202 / .202 | 5 / .627 |
| Porto | P0-time | .985 / .957 | .891 | .583 / .215 | .0165 | .336 / .189 | 9 / .404 |
| Porto | P1 λ=1 | .984 / .958 | .950 | .572 / .248 | .0354 | .337 / .189 | 4 / .499 |
| Porto | P1 λ=.1 | .990 / .972 | .935 | .609 / .233 | .0257 | .319 / .189 | 5 / .793 |

所有模型量化前 2×/3× cosine 均超过 .997。P0/P1-time 方案保证所有视图具有相同物理边界，消除了原 HTP 中“8 个当前视图点对应不同物理时间”的结构性错位；但由于该 pilot 同时更换了 encoder、quantizer 和 decoder，当前结果不能把相对 B6 的全部提升单独归因于固定时间支持。

## 判断

1. **P0-time 整体方案通过稳定性 pilot：** P0-time 在两城的 2× layer-1 agreement 均超过预注册 .70 门槛，并远高于原 HTP 的成都 .177、Porto .122；这不是只改变窗口定义的单变量因果结论。
2. **纯 cosine P1 未通过：** 一致性损失只带来很小的 agreement 增益，却恶化重构与码本利用；成都速度 probe 退回多数类，Porto `λ=.1` 的最大 token 占比达到 .793。
3. **不能把高一致性直接解释为语义原语：** 当前停留 probe 未超过多数类，速度语义仍弱，且只有一个 seed。
4. **Gate 1 继续关闭。** 当前结果支持保留固定时间窗，但不支持采用无防坍缩约束的纯 cosine 一致性目标。

作为控制实验，原始固定 8 点 HTP 也完成了同类多视图训练。其第一层 2×/3× agreement 升至成都 `.878/.833`、Porto `.913/.878`，但完整 tuple 的 2× agreement 仍只有 `.002/.001`，且 Porto 第一层最大码占比达到 `.894`。因此固定 8 点结构也能学到粗层一致性，但当前收益主要伴随码本压缩，不能否定固定物理时间窗的必要性。详见：[原始固定 8 点 Token 多视图 Pilot 报告](stage_1_p1_fixed8_multiview_pilot_report.md)。

## 下一步

- 把 P1 改成选择性一致性：正样本为同物理窗不同采样，加入不同速度/停留状态的困难负样本；
- 增加 variance/covariance 或 code-usage 正则，避免表征和码本坍缩；
- 进入 P2：加入速度、停留比例、持续时间辅助预测；
- 比较 60/120/180 秒窗口，并补固定道路距离窗消融；
- 完成至少三个随机种子后再进行确认集与 Gate 1 决策。

## 可复现产物

- `stage_1/src/p1_fixed_time_pilot.py`
- `stage_1/tests/test_p1_fixed_time_pilot.py`（4 tests passed）
- `stage_1/runs/p1_fixed_time_pilot/`
- `stage_1/runs/p1_fixed_time_pilot_lambda0p1/`
