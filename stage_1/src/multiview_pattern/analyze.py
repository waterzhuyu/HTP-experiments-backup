from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .metrics import (
    bootstrap_metrics,
    contingency,
    distribution_similarity,
    event_condition_labels,
    global_permutation,
    grouped_permutation,
    json_ready,
    linear_probe,
    metrics_from_contingency,
    optimal_mapping,
    reindex_matrix,
)
from .model import ENCODER_FAMILIES, VIEWS


VIEW_PAIRS = (("gps", "grid"), ("gps", "road"), ("grid", "road"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze independent tokenizer correspondence and robustness.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 314159, 20260819])
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--probe-device", default="cuda:0")
    return parser.parse_args()


def run_path(run_dir: Path, family: str, seed: int, view: str) -> Path:
    return run_dir / family / f"seed_{seed}" / view


def load_representation(path: Path, split: str = "test") -> dict[str, np.ndarray]:
    name = "test_representation.npz" if split == "test" else "train_probe_representation.npz"
    with np.load(path / name) as source:
        return {key: source[key] for key in source.files}


def assert_aligned(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> None:
    if not np.array_equal(left["offsets"], right["offsets"]):
        raise ValueError("representation offsets are not aligned")
    if not np.array_equal(left["traj_ids"], right["traj_ids"]):
        raise ValueError("trajectory IDs are not aligned")


def compare(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    size: int,
    global_index: np.ndarray | None = None,
    conditional_index: np.ndarray | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    assert_aligned(left, right)
    matrix = contingency(left["codes"], right["codes"], size)
    result: dict[str, object] = {"actual": metrics_from_contingency(matrix)}
    if global_index is not None:
        result["global_shuffle"] = metrics_from_contingency(
            contingency(left["codes"], right["codes"][global_index], size)
        )
    if conditional_index is not None:
        result["conditional_shuffle"] = metrics_from_contingency(
            contingency(left["codes"], right["codes"][conditional_index], size)
        )
    if conditional_index is not None:
        actual = result["actual"]
        conditional = result["conditional_shuffle"]
        result["conditional_lift"] = {
            "adjusted_mutual_information": float(actual["adjusted_mutual_information"])
            - float(conditional["adjusted_mutual_information"]),
            "optimal_match_rate": float(actual["optimal_match_rate"])
            - float(conditional["optimal_match_rate"]),
            "right_given_left_entropy_reduction": float(conditional["normalized_h_right_given_left"])
            - float(actual["normalized_h_right_given_left"]),
            "left_given_right_entropy_reduction": float(conditional["normalized_h_left_given_right"])
            - float(actual["normalized_h_left_given_right"]),
        }
    return result, matrix


def validity(run_dir: Path, seeds: list[int]) -> tuple[list[dict[str, object]], bool]:
    rows: list[dict[str, object]] = []
    for family in ENCODER_FAMILIES:
        for seed in seeds:
            for view in VIEWS:
                metrics = json.loads((run_path(run_dir, family, seed, view) / "metrics.json").read_text())
                codebook = metrics["test"]["codebook"]
                reconstruction = metrics["test"]["losses"]["reconstruction"]
                passed = (
                    codebook["active_codes"] >= 32
                    and codebook["perplexity"] >= 16
                    and codebook["max_token_share"] <= 0.20
                    and math.isfinite(reconstruction)
                )
                rows.append(
                    {
                        "family": family,
                        "seed": seed,
                        "view": view,
                        **codebook,
                        "reconstruction": reconstruction,
                        "passed": passed,
                        "runtime_seconds": metrics["runtime_seconds"],
                    }
                )
    return rows, all(row["passed"] for row in rows)


def event_attributes(data_dir: Path, representation_offsets: np.ndarray) -> dict[str, np.ndarray]:
    metadata = json.loads((data_dir / "metadata.json").read_text())
    point_offsets = np.load(data_dir / "test_offsets.npy", mmap_mode="r")
    gps = np.load(data_dir / "test_gps.npy", mmap_mode="r")
    grid = np.load(data_dir / "test_grid.npy", mmap_mode="r")
    road = np.load(data_dir / "test_road.npy", mmap_mode="r")
    meters_x = (metadata["mbr"]["max_lon"] - metadata["mbr"]["min_lon"]) * metadata["grid"]["meters_per_lon"]
    meters_y = (metadata["mbr"]["max_lat"] - metadata["mbr"]["min_lat"]) * metadata["grid"]["meters_per_lat"]
    names = ("path_length_m", "displacement_m", "mean_abs_turn_rad", "grid_transition_rate", "road_transition_rate", "trajectory_points")
    values = {name: np.empty(int(representation_offsets[-1]), dtype=np.float32) for name in names}
    trajectory_for_event = np.empty(int(representation_offsets[-1]), dtype=np.int64)
    cursor = 0
    for trajectory in range(len(point_offsets) - 1):
        start, end = int(point_offsets[trajectory]), int(point_offsets[trajectory + 1])
        trajectory_gps = np.asarray(gps[start:end], dtype=np.float64)
        trajectory_grid = np.asarray(grid[start:end])
        trajectory_road = np.asarray(road[start:end])
        expected_events = int(representation_offsets[trajectory + 1] - representation_offsets[trajectory])
        for position in range(expected_events):
            chunk_start = position * 8
            chunk_end = min((position + 1) * 8, len(trajectory_gps))
            points = trajectory_gps[chunk_start:chunk_end].copy()
            points[:, 0] *= meters_x
            points[:, 1] *= meters_y
            steps = np.diff(points, axis=0)
            step_lengths = np.linalg.norm(steps, axis=1)
            path_length = float(step_lengths.sum())
            displacement = float(np.linalg.norm(points[-1] - points[0]))
            if len(steps) >= 2:
                angles = np.arctan2(steps[:, 1], steps[:, 0])
                turns = np.abs(np.angle(np.exp(1j * np.diff(angles))))
                mean_turn = float(turns.mean())
            else:
                mean_turn = 0.0
            chunk_grid = trajectory_grid[chunk_start:chunk_end]
            chunk_road = trajectory_road[chunk_start:chunk_end]
            denominator = max(len(chunk_grid) - 1, 1)
            values["path_length_m"][cursor] = path_length
            values["displacement_m"][cursor] = displacement
            values["mean_abs_turn_rad"][cursor] = mean_turn
            values["grid_transition_rate"][cursor] = float(np.sum(np.diff(chunk_grid) != 0) / denominator)
            values["road_transition_rate"][cursor] = float(np.sum(np.diff(chunk_road) != 0) / denominator)
            values["trajectory_points"][cursor] = len(trajectory_gps)
            trajectory_for_event[cursor] = trajectory
            cursor += 1
    values["trajectory_index"] = trajectory_for_event
    return values


def semantic_audit(codes: np.ndarray, attributes: dict[str, np.ndarray], top_k: int = 10) -> list[dict[str, object]]:
    counts = np.bincount(codes.astype(np.int64), minlength=256)
    rows: list[dict[str, object]] = []
    for code in np.argsort(counts)[::-1][:top_k]:
        mask = codes == code
        trajectories = attributes["trajectory_index"][mask]
        row: dict[str, object] = {
            "code": int(code),
            "count": int(mask.sum()),
            "representative_trajectory_indices": [int(value) for value in np.unique(trajectories)[:5]],
        }
        for name, values in attributes.items():
            if name == "trajectory_index":
                continue
            row[f"{name}_mean"] = float(values[mask].mean())
            row[f"{name}_median"] = float(np.median(values[mask]))
        rows.append(row)
    return rows


def aggregate_verdict(
    valid: bool,
    same_view: list[dict[str, object]],
    cross_view: list[dict[str, object]],
    block_rows: list[dict[str, object]],
) -> dict[str, object]:
    gate0_groups: dict[str, list[float]] = {}
    for row in same_view:
        key = f"{row['family']}:{row['view']}"
        gate0_groups.setdefault(key, []).append(float(row["metrics"]["actual"]["adjusted_mutual_information"]))
    gate0_summary = {key: {"mean_ami": float(np.mean(values)), "min_ami": float(np.min(values))} for key, values in gate0_groups.items()}
    gate0_pass = valid and all(value["mean_ami"] >= 0.50 for value in gate0_summary.values())

    h1_groups: dict[str, list[float]] = {}
    pair_groups: dict[str, list[dict[str, object]]] = {}
    for row in cross_view:
        key = f"{row['family']}:{row['left']}-{row['right']}"
        h1_groups.setdefault(key, []).append(float(row["metrics"]["conditional_lift"]["adjusted_mutual_information"]))
        pair_groups.setdefault(key, []).append(row)
    h1_summary = {key: {"mean_ami_lift": float(np.mean(values)), "min_ami_lift": float(np.min(values))} for key, values in h1_groups.items()}
    h1_pass = gate0_pass and all(value["mean_ami_lift"] >= 0.05 for value in h1_summary.values())

    non_equivalent = []
    for key, rows in pair_groups.items():
        match = np.mean([float(row["metrics"]["actual"]["optimal_match_rate"]) for row in rows])
        forward = np.mean([float(row["metrics"]["actual"]["normalized_h_right_given_left"]) for row in rows])
        reverse = np.mean([float(row["metrics"]["actual"]["normalized_h_left_given_right"]) for row in rows])
        if match < 0.80 and max(forward, reverse) >= 0.30:
            non_equivalent.append({"group": key, "mean_match": float(match), "forward_entropy": float(forward), "reverse_entropy": float(reverse)})
    h2_pass = h1_pass and bool(non_equivalent)

    seed_stability = {}
    for key, rows in pair_groups.items():
        ami = [float(row["metrics"]["actual"]["adjusted_mutual_information"]) for row in rows]
        direction = [
            np.sign(
                float(row["metrics"]["actual"]["normalized_h_right_given_left"])
                - float(row["metrics"]["actual"]["normalized_h_left_given_right"])
            )
            for row in rows
        ]
        majority = max(direction.count(-1), direction.count(0), direction.count(1)) / len(direction)
        seed_stability[key] = {"ami_std": float(np.std(ami)), "direction_agreement": float(majority)}
    seed_pass = all(value["ami_std"] <= 0.05 and value["direction_agreement"] >= 2 / 3 for value in seed_stability.values())

    family_vectors = {}
    for family in ENCODER_FAMILIES:
        vector = []
        for left, right in VIEW_PAIRS:
            rows = pair_groups[f"{family}:{left}-{right}"]
            vector.extend(
                [
                    float(np.mean([row["metrics"]["actual"]["normalized_h_right_given_left"] for row in rows])),
                    float(np.mean([row["metrics"]["actual"]["normalized_h_left_given_right"] for row in rows])),
                ]
            )
        family_vectors[family] = vector
    encoder_rank = float(spearmanr(family_vectors["transformer"], family_vectors["gru"]).statistic)
    within_block = [row["cosine_similarity"] for row in block_rows if row["comparison"] == "within_family_seed"]
    across_block = [row["cosine_similarity"] for row in block_rows if row["comparison"] == "across_encoder"]
    block_pass = float(np.mean(within_block)) >= 0.80 and float(np.mean(across_block)) >= 0.60
    h3_pass = h2_pass and seed_pass and encoder_rank >= 0.70 and block_pass
    return {
        "validity_pass": valid,
        "gate0_same_view_stability_pass": gate0_pass,
        "h1_cross_view_association_pass": h1_pass,
        "h2_non_equivalence_pass": h2_pass,
        "h3_encoder_robustness_pass": h3_pass,
        "gate0_summary": gate0_summary,
        "h1_summary": h1_summary,
        "h2_non_equivalent_groups": non_equivalent,
        "seed_stability": seed_stability,
        "encoder_directional_entropy_spearman": encoder_rank,
        "within_family_block_cosine_mean": float(np.mean(within_block)),
        "across_encoder_block_cosine_mean": float(np.mean(across_block)),
        "decision": (
            "continue_to_shared_private_design"
            if h3_pass
            else "do_not_advance_to_shared_private_design"
        ),
    }


def markdown_report(result: dict[str, object], metadata: dict[str, object]) -> str:
    verdict = result["verdict"]
    validity_rows = result["validity"]
    cross_rows = result["cross_view"]
    lines = [
        "# Porto GPS / Grid / Road 独立 Tokenizer 跨视图 Pattern 实验",
        "",
        f"> **Gate 决策：`{verdict['decision']}`。** Gate 0={verdict['gate0_same_view_stability_pass']}，H1={verdict['h1_cross_view_association_pass']}，H2={verdict['h2_non_equivalence_pass']}，H3={verdict['h3_encoder_robustness_pass']}。POI 未纳入本实验。",
        "",
        "## 方法冻结",
        "",
        "- 数据：Porto 官方 train/test 索引；537,993/59,778 条轨迹。GPS、100 m Grid、逐点 map-matched `opath` Road 共享轨迹边界与点索引。",
        "- 样本单元：三层 stride-2 后的一个 latent event，输出长度 `ceil(T/8)`；末尾不足 8 点保留。Transformer event 具有全轨迹注意力上下文，GRU event 具有双向递归上下文，因此“8 点”仅是下采样锚定而非严格局部感受野。",
        "- 独立性：每个 view × encoder × seed 拥有独立 input adapter、encoder、K=256 spherical VQ codebook 和 decoder；无共享参数、无共享码本、无跨视图 loss。",
        "- Encoder：3×(Transformer 或双向 GRU + stride-2 Conv1d)，hidden=64，latent=64。Road 输入使用冻结的 128 维 Node2Vec，再投影到 64 维。",
        "- Decoder：latent 重复 8 次；GPS 重构归一化经纬度，Grid 重构网格中心，Road 重构 L2-normalized Node2Vec。未执行 GPS 轨迹拼接式解码。",
        "- 损失：`L = L_recon + L_codebook + 0.25 L_commit + 0.05 L_soft_usage`。AdamW 2e-4、weight decay 1e-4、batch 256、3 epochs、BF16、clip=1.0。",
        "- Seeds：42、314159、20260819。Encoder families：Transformer、GRU。",
        "- 条件打乱：在 2 km origin 区域、8方向 OD、轨迹长度 bin、latent 相对位置 quartile 内打乱 event 配对；bootstrap 以完整轨迹为重采样单位。",
        "",
        "## 方法变更表",
        "",
        "| 方法组成 | 共享 VQ 先前基线 | 本实验 | 是否变化 |",
        "|---|---|---|---|",
        "| 样本物理单元 | 同轨迹的 ceil(T/8) latent 位置 | 相同 | 否 |",
        "| 视图构造与覆盖范围 | GPS/Grid/Road 逐点对齐 | 相同；POI 跳过 | POI 不适用 |",
        "| 输入特征与单位 | normalized GPS / grid ID / road Node2Vec | 相同 | 否 |",
        "| encoder/downsampling | 三个 Transformer encoder | 独立 Transformer 与 GRU 控制 | 是 |",
        "| token 时间/空间支持 | ceil(T/8) | 相同 | 否 |",
        "| quantizer/codebook | 三视图共享 K=256 | 每个 tokenizer 私有 K=256 | 是（主变量） |",
        "| decoder 与输出时间锚点 | view-specific，repeat ×8 | 相同 | 否 |",
        "| 初始化与冻结模块 | road embedding 冻结 | 相同；每 run 从头初始化 | 否 |",
        "| reconstruction/VQ loss | 三视图损失求和 | 单视图独立优化 | 是（独立训练必需） |",
        "| 新增 loss 与权重 | alignment 可选 | 无 alignment | 是 |",
        "| 训练数据与超参数 | full train, 3 epochs, seed 42 | full train, 3 seeds | 是 |",
        "| 评估对齐与指标 | exact/NMI/shuffle | 匹配率、双向条件熵、VI、AMI、global/conditional shuffle、probe、bootstrap | 是 |",
        "",
        "## 码本有效性",
        "",
        "| Encoder | Seed | View | Active | Perplexity | Max share | Recon | Pass |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in validity_rows:
        lines.append(
            f"| {row['family']} | {row['seed']} | {row['view']} | {row['active_codes']} | {row['perplexity']:.3f} | {row['max_token_share']:.4f} | {row['reconstruction']:.6f} | {row['passed']} |"
        )
    lines.extend(
        [
            "",
            "## 跨视图结果（同 seed）",
            "",
            "| Encoder | Seed | Pair | AMI | Cond-shuffle AMI | Lift | Match | H(R given L) | H(L given R) |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in cross_rows:
        actual = row["metrics"]["actual"]
        shuffled = row["metrics"]["conditional_shuffle"]
        lift = row["metrics"]["conditional_lift"]
        lines.append(
            f"| {row['family']} | {row['seed']} | {row['left']}-{row['right']} | {actual['adjusted_mutual_information']:.4f} | {shuffled['adjusted_mutual_information']:.4f} | {lift['adjusted_mutual_information']:.4f} | {actual['optimal_match_rate']:.4f} | {actual['normalized_h_right_given_left']:.4f} | {actual['normalized_h_left_given_right']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Gate 判定",
            "",
            f"- Gate 0（同视图跨 seed AMI 均值 ≥0.50 且无坍塌）：**{verdict['gate0_same_view_stability_pass']}**。",
            f"- H1（三对在两种 encoder 下的 AMI 相对条件打乱提升均值 ≥0.05）：**{verdict['h1_cross_view_association_pass']}**。",
            f"- H2（至少一对不是近似一一置换：match<0.80 且方向条件熵≥0.30）：**{verdict['h2_non_equivalence_pass']}**。",
            f"- H3（AMI seed std≤0.05、方向一致、跨 encoder 熵排序 Spearman≥0.70、块结构相似）：**{verdict['h3_encoder_robustness_pass']}**。",
            f"- Transformer/GRU 双向条件熵排序 Spearman：{verdict['encoder_directional_entropy_spearman']:.4f}。",
            f"- 同 family 跨 seed 共现块 cosine：{verdict['within_family_block_cosine_mean']:.4f}；跨 encoder：{verdict['across_encoder_block_cosine_mean']:.4f}。",
            "",
            "## 解释边界",
            "",
            "GPS、Grid、Road 是同源派生视图，本实验验证的是同一移动事件的跨表达稳定性，不是统计独立模态的一致性。共现与 probe 只证明可预测关系；语义审计使用几何、转向、grid/road transition 和轨迹长度，未使用 POI。Road reconstruction 是冻结 Node2Vec MSE，不等价于 road ID 分类准确率。",
            "",
            "## 可复现信息",
            "",
            f"- Git revision：`{metadata['git_revision']}`；branch：`exp/porto-independent-gps-road-grid`。",
            f"- 完整 resolved config、checkpoint、test/probe representations：`{metadata['run_dir']}`。",
            f"- 机器可读汇总、共现矩阵与报告：`{metadata['output_dir']}`。",
            f"- Bootstrap replicates：{metadata['bootstrap_replicates']}；probe 使用冻结 latent 的线性分类器，5 epochs，最多 120k train/test events。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(args.seeds)
    reference = load_representation(run_path(args.run_dir, "transformer", seeds[0], "gps"))
    event_meta = event_condition_labels(args.data_dir, reference["offsets"])
    global_index, global_meta = global_permutation(event_meta["trajectory_ids"], 20260819)
    conditional_index, conditional_meta = grouped_permutation(
        event_meta["condition"], event_meta["trajectory_ids"], 20260820
    )
    validity_rows, validity_pass = validity(args.run_dir, seeds)
    matrices: dict[str, np.ndarray] = {}
    same_view: list[dict[str, object]] = []
    for family in ENCODER_FAMILIES:
        for view in VIEWS:
            for seed_index, left_seed in enumerate(seeds):
                for right_seed in seeds[seed_index + 1 :]:
                    left = load_representation(run_path(args.run_dir, family, left_seed, view))
                    right = load_representation(run_path(args.run_dir, family, right_seed, view))
                    row_metrics, matrix = compare(left, right, args.codebook_size)
                    key = f"same_{family}_{view}_{left_seed}_{right_seed}"
                    matrices[key] = matrix.astype(np.int64)
                    same_view.append(
                        {
                            "family": family,
                            "view": view,
                            "left_seed": left_seed,
                            "right_seed": right_seed,
                            "metrics": row_metrics,
                        }
                    )

    cross_view: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    for family in ENCODER_FAMILIES:
        for seed in seeds:
            for left_view, right_view in VIEW_PAIRS:
                left_path = run_path(args.run_dir, family, seed, left_view)
                right_path = run_path(args.run_dir, family, seed, right_view)
                left = load_representation(left_path)
                right = load_representation(right_path)
                row_metrics, matrix = compare(
                    left,
                    right,
                    args.codebook_size,
                    global_index,
                    conditional_index,
                )
                row_metrics["bootstrap"] = bootstrap_metrics(
                    left["codes"],
                    right["codes"],
                    left["offsets"],
                    args.codebook_size,
                    seed=seed + 1000,
                    replicates=args.bootstrap_replicates,
                )
                key = f"cross_{family}_{seed}_{left_view}_{right_view}"
                matrices[key] = matrix.astype(np.int64)
                cross_view.append(
                    {
                        "family": family,
                        "seed": seed,
                        "left": left_view,
                        "right": right_view,
                        "metrics": row_metrics,
                    }
                )
                left_train = load_representation(left_path, "probe")
                right_train = load_representation(right_path, "probe")
                assert_aligned(left_train, right_train)
                for source_name, source_train, source_test, target_name, target_train, target_test in (
                    (left_view, left_train, left, right_view, right_train, right),
                    (right_view, right_train, right, left_view, left_train, left),
                ):
                    probe_rows.append(
                        {
                            "family": family,
                            "seed": seed,
                            "source": source_name,
                            "target": target_name,
                            **linear_probe(
                                source_train["latents"],
                                target_train["codes"],
                                source_test["latents"],
                                target_test["codes"],
                                args.codebook_size,
                                args.probe_device,
                                seed + 2000,
                            ),
                        }
                    )

    block_rows: list[dict[str, object]] = []
    for family in ENCODER_FAMILIES:
        reference_seed = seeds[0]
        for left_view, right_view in VIEW_PAIRS:
            left_reference = load_representation(run_path(args.run_dir, family, reference_seed, left_view))
            right_reference = load_representation(run_path(args.run_dir, family, reference_seed, right_view))
            reference_matrix = contingency(left_reference["codes"], right_reference["codes"], args.codebook_size)
            for seed in seeds[1:]:
                left_current = load_representation(run_path(args.run_dir, family, seed, left_view))
                right_current = load_representation(run_path(args.run_dir, family, seed, right_view))
                left_mapping = optimal_mapping(left_current["codes"], left_reference["codes"], args.codebook_size)
                right_mapping = optimal_mapping(right_current["codes"], right_reference["codes"], args.codebook_size)
                current_matrix = contingency(left_current["codes"], right_current["codes"], args.codebook_size)
                aligned = reindex_matrix(current_matrix, left_mapping, right_mapping)
                block_rows.append(
                    {
                        "comparison": "within_family_seed",
                        "family": family,
                        "seed": seed,
                        "pair": f"{left_view}-{right_view}",
                        **distribution_similarity(reference_matrix, aligned),
                    }
                )
    for seed in seeds:
        for left_view, right_view in VIEW_PAIRS:
            left_reference = load_representation(run_path(args.run_dir, "transformer", seed, left_view))
            right_reference = load_representation(run_path(args.run_dir, "transformer", seed, right_view))
            left_current = load_representation(run_path(args.run_dir, "gru", seed, left_view))
            right_current = load_representation(run_path(args.run_dir, "gru", seed, right_view))
            left_mapping = optimal_mapping(left_current["codes"], left_reference["codes"], args.codebook_size)
            right_mapping = optimal_mapping(right_current["codes"], right_reference["codes"], args.codebook_size)
            reference_matrix = contingency(left_reference["codes"], right_reference["codes"], args.codebook_size)
            current_matrix = contingency(left_current["codes"], right_current["codes"], args.codebook_size)
            aligned = reindex_matrix(current_matrix, left_mapping, right_mapping)
            block_rows.append(
                {
                    "comparison": "across_encoder",
                    "seed": seed,
                    "pair": f"{left_view}-{right_view}",
                    **distribution_similarity(reference_matrix, aligned),
                }
            )

    attributes = event_attributes(args.data_dir, reference["offsets"])
    semantic_rows = []
    for family in ENCODER_FAMILIES:
        for view in VIEWS:
            representation = load_representation(run_path(args.run_dir, family, seeds[0], view))
            semantic_rows.append(
                {
                    "family": family,
                    "seed": seeds[0],
                    "view": view,
                    "top_codes": semantic_audit(representation["codes"], attributes),
                }
            )

    verdict = aggregate_verdict(validity_pass, same_view, cross_view, block_rows)
    first_config = json.loads(
        (run_path(args.run_dir, "transformer", seeds[0], "gps") / "resolved_config.json").read_text()
    )
    metadata = {
        "git_revision": first_config["git_revision"],
        "run_dir": str(args.run_dir),
        "output_dir": str(args.output_dir),
        "data_dir": str(args.data_dir),
        "bootstrap_replicates": args.bootstrap_replicates,
        "global_shuffle": global_meta,
        "conditional_shuffle": conditional_meta,
        "seeds": seeds,
        "codebook_size": args.codebook_size,
    }
    result = {
        "metadata": metadata,
        "validity": validity_rows,
        "same_view": same_view,
        "cross_view": cross_view,
        "continuous_latent_probes": probe_rows,
        "block_stability": block_rows,
        "semantic_audit": semantic_rows,
        "verdict": verdict,
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(json_ready(result), indent=2))
    np.savez_compressed(args.output_dir / "cooccurrence_matrices.npz", **matrices)
    (args.output_dir / "REPORT_PORTO_INDEPENDENT_TOKENIZERS.md").write_text(markdown_report(result, metadata))
    print(json.dumps(json_ready(verdict), indent=2), flush=True)


if __name__ == "__main__":
    main()
