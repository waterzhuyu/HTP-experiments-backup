# GPS / Road / Grid 对齐数据审计

实验 ID：`staged_attribution_data_alignment_v1_20260826`
状态：completed
正式运行 revision：`f9cd1d4b32239a219966faa5b841d7be0c13d089`

## 方法

本审计只覆盖 GPS、100 m Grid 和 pointwise Road，不使用 POI。GPS、Grid、Road、time、dxy 和 percent target 共享 observation index；route path 是 decoder/prompt 的轨迹级条件，允许与 observation 数量不同。

数组先按官方 train/test index 从 `traj.parquet` 构造，再以 `traj_id` 与 `processed_{train,test}_traj.pkl` join。审计逐条检查 population、offset、长度、有限 GPS、Grid/Road ID 范围和 split 泄漏。Grid 使用城市 MBR 中纬度的 local-equirectangular 近似，固定 100 m；GPS 使用 HTP MBR 归一化。

## 结果

| City | Train trajectories | Test trajectories | Train points | Test points | Split overlap | Audit |
|---|---:|---:|---:|---:|---:|---|
| Porto | 537,993 | 59,778 | 23,066,053 | 2,570,491 | 0 | pass |
| Chengdu | 542,592 | 60,288 | 31,237,070 | 3,475,397 | 0 | pass |

Chengdu 的网格为 278×255，共 70,890 个 Grid ID；Road 数为 39,503。Porto 沿用已有已验证 artifact。两城市所有 point fields 与 processed target 长度一致，route path 长度差异未被误判为 pointwise misalignment。

机器可读记录 `data_preparation_audit.json` 保存两城市 metadata hash、20 个数组 SHA256、source/split hash、路径、失败和 gate。大数组不提交 Git：Porto 401M，Chengdu 539M。

## 命令

```sh
uv run python -m staged_attribution.prepare_data \
  --config stage_1/configs/staged_attribution/experiment.json \
  --city chengdu
uv run python -m staged_attribution.prepare_data \
  --config stage_1/configs/staged_attribution/experiment.json \
  --city chengdu --audit-only
uv run python -m staged_attribution.prepare_data \
  --config stage_1/configs/staged_attribution/experiment.json \
  --city porto --audit-only
```

## 失败记录

第一次 Chengdu prepare 基于 `a43f55d`，在写数组前失败：实现为了统计道路数量读取 Porto 特有的 `from_node_id` 列，而 Chengdu edge table 的列为 `road_name/bridge/...`。失败未生成部分数组。

修复 `f9cd1d4` 改为与列名无关的 CSV record count，并为两种表头增加回归测试。修复后完整 pytest 为 42 passed，随后从 clean revision 重新执行 prepare 和两城市 audit。

## 观察、解释与 Gate

- 观察：两城市官方 split 均无 `traj_id` overlap；所有逐点视图和 HTP reconstruction target 一一对齐。
- 解释：数据层满足比较 GPS/Road/Grid encoder 输入的必要条件；这不证明三种表达在语义上等价。
- 风险：当前锁文件沿用上游 `pandas==3.0.4`，uv 标记该版本已因 datetime segfault 报告被 yank；本轮 parquet/CSV/array 全量流程成功，但正式环境报告必须保留该警告。
- Gate：通过，允许进入 E-stage 模型实现。
