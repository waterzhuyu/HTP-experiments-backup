# 实验记录方法规范

> 适用范围：本仓库所有 baseline、pilot、ablation、confirmation、training 与 evaluation run。
> 核心原则：**实验名称或概念标签不能代替实验方法。** “固定 8 点 token”“固定时间窗口”“多视图训练”“加入一致性损失”等只能作为简称，必须同时给出下面的可复现细节。

## 1. 每次实验必须记录的内容

### 1.1 实验身份与研究问题

- 实验 ID、日期、状态：planned / running / completed / failed；
- 要检验的具体假设；
- baseline、实验变量、保持不变的控制变量；
- 预先定义的通过标准和失败条件；
- 本次是探索集、pilot、ablation 还是冻结确认集。

### 1.2 数据来源、样本单元与切分

- 数据集、城市、文件或版本、原始字段；
- 一个训练/评估样本在物理上代表什么：整条轨迹、一个时间窗、一个距离窗或一个点序列；
- 轨迹筛选条件、窗口边界条件、最少点数、异常点处理；
- train/validation/test 或 exploration/confirmation 的构造方法；
- 轨迹、车辆、道路、时间以及多视图之间如何防止泄漏；
- 样本数、随机抽样顺序和随机种子。

### 1.3 输入坐标、时间与归一化

- 输入每个点包含哪些量，例如 `(lon, lat, timestamp)` 或 `(x_km, y_km, relative_time)`；
- 坐标投影、局部原点、距离单位、时间单位；
- normalization、padding、mask 与变长序列处理；
- 模型实际收到的 tensor shape 和有效长度。

### 1.4 多视图的精确构造

对 native、2×、3×、相位、缺失、噪声等每种视图分别记录：

- 从哪些原始点开始；
- 保留点的索引规则或伪代码，例如 `indices[0::2]`；
- 是否强制保留首点/末点；
- 是删点、插值还是重新采样；若插值，写明时间网格与方法；
- 各视图覆盖的物理起止时间是否相同；
- 输入点数、采样间隔与物理持续时间如何变化；
- 不同视图中哪些位置被认为是同一运动事件，以及对齐映射的具体公式。

不得只写“构造 native/2×/3× 多视图”。

### 1.5 表示、token 与解码的精确定义

- encoder 架构、层数、每层 stride/downsampling、hidden dimension；
- 一个 embedding/token 对应多少输入点、多少秒或多少米；边界 token 如何处理；
- quantizer 类型、层数、每层 codebook size、距离函数、straight-through 方式；
- 报告的是第一层 code、某一层 code，还是完整 RQ tuple；
- decoder 的输入、输出维数与目标；
- token 如何被 decode 成坐标/GPS：输出点的时间锚点、点数、坐标反变换和轨迹拼接方法；
- 若某次实验不执行 GPS 解码，必须明确写 `未执行解码`，不能让读者推断。

### 1.6 初始化、可训练参数与训练配置

- 从头训练还是加载 checkpoint；checkpoint 的来源、epoch、seed 和路径；
- 哪些模块冻结、哪些模块可训练；
- optimizer、learning rate、weight decay、batch size、epochs/steps；
- gradient clipping、scheduler、early stopping、precision 与 device；
- 所有随机种子及 deterministic 设置。

### 1.7 损失函数

必须给出总损失的公式或等价伪代码，并逐项解释：

```text
L_total = w_rec L_rec + w_vq L_vq + w_cons L_cons + ...
```

同时记录：

- 每个子损失在哪个 tensor 上计算；
- native/2×/3× 是求和还是取平均；
- 一致性发生在量化前、量化后、code 分布还是 decoder 输出；
- 正样本与困难负样本如何选；
- 对齐点如何获得；
- 每个权重的确切数值。

不得只写“加入 cosine 一致性损失”。

### 1.8 评估口径

- 使用哪个冻结 split、多少样本、seed；
- agreement 的比较对象、对齐规则、分母与边界处理；
- 分别报告逐层 agreement 和完整 tuple agreement，不能用第一层代替完整 RQ；
- 重构、检索、probe、码本利用、选择性 margin 等指标的定义；
- probe 的训练/测试数据、特征、标签、baseline；
- codebook used codes、perplexity、最大码占比等防坍缩指标；
- 阈值来自预注册、探索性暂定还是事后分析。

### 1.9 可复现信息

- authoritative Git revision；
- 完整运行命令，不省略与方法有关的默认参数；
- config 文件与实际解析后的配置；
- 数据 split、seed、日志、checkpoint、JSON 指标和报告路径；
- 测试命令与结果；
- 失败、异常、重跑原因以及与首次运行的差异。

### 1.10 结果与结论

- 报告原始数值、baseline 差值和跨 seed 统计；
- 把“观测事实”“解释”和“Gate 决策”分开；
- 记录不支持假设的结果、坍缩、泄漏和其他失败；
- 给出下一步，但不得用下一步掩盖本次失败。

## 2. 方法变更表是强制项

每份实验报告都必须包含下面的逐项差异表。没有变化的项目也要写“与 baseline 相同”，不能省略。

| 方法组成 | Baseline | 本次实验 | 是否变化 |
|---|---|---|---|
| 样本物理单元 |  |  |  |
| 视图构造与覆盖范围 |  |  |  |
| 输入特征与单位 |  |  |  |
| encoder/downsampling |  |  |  |
| token 时间/空间支持 |  |  |  |
| quantizer/codebook |  |  |  |
| decoder 与输出时间锚点 |  |  |  |
| 初始化与冻结模块 |  |  |  |
| reconstruction/VQ loss |  |  |  |
| 新增 loss 与权重 |  |  |  |
| 训练数据与超参数 |  |  |  |
| 评估对齐与指标 |  |  |  |

## 3. “固定 8 点”与“固定时间窗”的最低解释要求

### 固定 8 点 token

必须说明它究竟是：输入先被切成互不重叠的 8 点块，还是 encoder 通过多层 stride 使输出长度约为输入长度的 `1/8`。同时记录：

- native/2×/3× 下每个 token 实际覆盖的原始点范围与物理时长；
- 不同长度 token 序列如何建立一一比较位置；
- 边界 token 如何映射；
- RQ 每一层 code 与 tuple 如何评估；
- decoder 输出多少点、这些点位于哪些时间位置。

### 固定时间窗

必须说明：

- 窗口长度、步长、闭区间/开区间和覆盖不足时的处理；
- native 视图如何取点，2×/3× 如何删点或插值；
- 是否共享完全相同的物理起止时间；
- 一个窗口产生一个 token 还是一个 token 序列；
- decoder 的目标时间锚点、输出点数和插值方法；
- 与固定 8 点 baseline 相比，除了时间支持外是否同时更换了 encoder、quantizer、decoder 或训练目标。

因此，“从固定 8 点 token 改到固定时间窗口”不是完整的方法描述。若同时更换模型结构，必须明确把“时间支持变化”和“架构变化”拆开记录，避免把结果错误归因于单一因素。

## 4. 记录位置

每次正式实验至少同步保存三份信息：

1. 代码仓库中的 run-level JSON/config/log，保存机器可读的完整配置与指标；
2. 代码仓库中的 Markdown 报告，保存详细方法、差异表、结果与判断；
3. Notion 实验进度页，保存同样的方法摘要、Git revision、命令、路径、失败和下一步，并链接到完整报告。

若三处内容冲突，以 authoritative remote workspace 中对应 Git revision 的代码和解析后配置为准，并立即修正文档。
