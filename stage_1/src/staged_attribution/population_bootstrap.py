from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy import sparse

from .stage4_metrics import (
    MetricPopulationError,
    _grid_context,
    _grid_ids,
    _point_road_distances,
    _radius,
    _segments,
)


HISTOGRAM_METRICS = (
    "travel_distance_jsd",
    "segment_distance_jsd",
    "radius_jsd",
    "point_to_road_jsd",
)
DENSITY_METRICS = (
    "grid_density_cosine",
    "grid_pattern_f1",
    "road_density_cosine",
    "road_pattern_f1",
)
DOMAIN_METRICS = HISTOGRAM_METRICS + DENSITY_METRICS
BOOTSTRAP_METHOD = "trajectory_paired_population_aware_fixed_support_v1"
SELECTION_BOOTSTRAP_METHOD = (
    "trajectory_paired_population_aware_fixed_support_stratified_seed_v1"
)
POPULATION_REGRET_METHOD = "failure_weighted_population_regret_v1"
CONTRIBUTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PopulationMetricContributions:
    traj_ids: tuple[int, ...]
    valid: np.ndarray
    histograms: Mapping[str, tuple[np.ndarray, np.ndarray]]
    densities: Mapping[str, tuple[sparse.csr_matrix, sparse.csr_matrix]]
    supports: Mapping[str, Sequence[float] | str]
    top_k: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _fixed_edges(first: Sequence[float], second: Sequence[float]) -> np.ndarray | None:
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if not len(first_values) or not len(second_values):
        raise MetricPopulationError("fixed histogram support requires paired values")
    minimum = min(float(first_values.min()), float(second_values.min()))
    maximum = max(float(first_values.max()), float(second_values.max()))
    if math.isclose(minimum, maximum):
        return None
    bins = min(50, max(5, int(math.log2(len(first_values) + len(second_values))) + 1))
    return np.linspace(minimum - 1e-10, maximum + 1e-10, bins + 1)


def _histogram_rows(values: Sequence[np.ndarray], edges: np.ndarray | None) -> np.ndarray:
    if edges is None:
        return np.asarray([[len(value)] for value in values], dtype=np.float64)
    return np.asarray([np.histogram(value, bins=edges)[0] for value in values], dtype=np.float64)


def _collapsed_counts(
    sequences: Sequence[Sequence[int]],
    valid: np.ndarray,
    width: int,
) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row, sequence in enumerate(sequences):
        if not valid[row]:
            continue
        counts: dict[int, int] = {}
        previous = None
        for raw in sequence:
            value = int(raw)
            if value < 0 or value >= width:
                raise MetricPopulationError(f"density ID {value} outside [0,{width})")
            if previous is None or value != previous:
                counts[value] = counts.get(value, 0) + 1
            previous = value
        for column, count in counts.items():
            rows.append(row)
            columns.append(column)
            values.append(float(count))
    return sparse.csr_matrix((values, (rows, columns)), shape=(len(sequences), width))


def build_metric_contributions(
    real_gps: Mapping[int, np.ndarray],
    real_roads: Mapping[int, Sequence[int]],
    generated_records: Sequence[Mapping[str, object]],
    road_geometries: Mapping[int, np.ndarray],
    city: str,
    *,
    top_k: int = 100,
    mbr: Mapping[str, float] | None = None,
    grid: Mapping[str, float | int] | None = None,
) -> PopulationMetricContributions:
    if set(real_gps) != set(real_roads):
        raise MetricPopulationError("real GPS/Road population mismatch")
    generated_by_id: dict[int, Mapping[str, object]] = {}
    for record in generated_records:
        traj_id = int(record["traj_id"])
        if traj_id in generated_by_id:
            raise MetricPopulationError(f"duplicate generated traj_id={traj_id}")
        generated_by_id[traj_id] = record
    ids = tuple(sorted(real_gps))
    if set(generated_by_id) != set(ids):
        raise MetricPopulationError("generated/real population mismatch")
    bounds, grid_spec = _grid_context(city, mbr, grid, 100.0)
    num_grids = int(grid_spec["n_grid_x"]) * int(grid_spec["n_grid_y"])
    num_roads = max(road_geometries, default=-1) + 1
    if num_roads <= 0:
        raise MetricPopulationError("road geometry population must be non-empty")

    valid = np.zeros(len(ids), dtype=bool)
    histogram_values: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        metric: ([], []) for metric in HISTOGRAM_METRICS
    }
    real_road_sequences: list[Sequence[int]] = []
    generated_road_sequences: list[Sequence[int]] = []
    real_grid_sequences: list[Sequence[int]] = []
    generated_grid_sequences: list[Sequence[int]] = []
    for row, traj_id in enumerate(ids):
        record = generated_by_id[traj_id]
        real_road = [int(value) for value in real_roads[traj_id]]
        real_road_sequences.append(real_road)
        if record.get("status") != "reconstructed":
            generated_road_sequences.append([])
            real_grid_sequences.append([])
            generated_grid_sequences.append([])
            for first, second in histogram_values.values():
                first.append(np.asarray([], dtype=np.float64))
                second.append(np.asarray([], dtype=np.float64))
            continue
        try:
            real = np.asarray(real_gps[traj_id], dtype=np.float64)
            generated = np.asarray(record["gps_traj"], dtype=np.float64)
            generated_road = [int(value) for value in record["road_traj"]]
            real_segments = _segments(real)
            generated_segments = _segments(generated)
            real_point = np.asarray(
                _point_road_distances(real, real_road, road_geometries), dtype=np.float64
            )
            generated_point = np.asarray(
                _point_road_distances(generated, generated_road, road_geometries), dtype=np.float64
            )
            real_grid = _grid_ids(real, city, mbr=bounds, grid=grid_spec)
            generated_grid = _grid_ids(generated, city, mbr=bounds, grid=grid_spec)
        except Exception:
            generated_road_sequences.append([])
            real_grid_sequences.append([])
            generated_grid_sequences.append([])
            for first, second in histogram_values.values():
                first.append(np.asarray([], dtype=np.float64))
                second.append(np.asarray([], dtype=np.float64))
            continue
        valid[row] = True
        generated_road_sequences.append(generated_road)
        real_grid_sequences.append(real_grid)
        generated_grid_sequences.append(generated_grid)
        values = {
            "travel_distance_jsd": (
                np.asarray([real_segments.sum()]), np.asarray([generated_segments.sum()])
            ),
            "segment_distance_jsd": (real_segments, generated_segments),
            "radius_jsd": (np.asarray([_radius(real)]), np.asarray([_radius(generated)])),
            "point_to_road_jsd": (real_point, generated_point),
        }
        for metric, (first, second) in values.items():
            histogram_values[metric][0].append(first)
            histogram_values[metric][1].append(second)

    histograms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    supports: dict[str, Sequence[float] | str] = {}
    for metric, (first_rows, second_rows) in histogram_values.items():
        if not valid.any():
            first_matrix = second_matrix = np.zeros((len(ids), 1), dtype=np.float64)
            supports[metric] = "no_success"
        else:
            first_full = np.concatenate([value for value in first_rows if len(value)])
            second_full = np.concatenate([value for value in second_rows if len(value)])
            edges = _fixed_edges(first_full, second_full)
            first_matrix = _histogram_rows(first_rows, edges)
            second_matrix = _histogram_rows(second_rows, edges)
            supports[metric] = "constant" if edges is None else edges.tolist()
        histograms[metric] = (first_matrix, second_matrix)

    real_road = _collapsed_counts(real_road_sequences, valid, num_roads)
    generated_road = _collapsed_counts(generated_road_sequences, valid, num_roads)
    real_grid = _collapsed_counts(real_grid_sequences, valid, num_grids)
    generated_grid = _collapsed_counts(generated_grid_sequences, valid, num_grids)
    return PopulationMetricContributions(
        traj_ids=ids,
        valid=valid,
        histograms=histograms,
        densities={"road": (real_road, generated_road), "grid": (real_grid, generated_grid)},
        supports=supports,
        top_k=int(top_k),
    )


def _jsd_rows(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    epsilon = 1e-10
    first_probability = (first + epsilon) / (first.sum(axis=1, keepdims=True) + epsilon * first.shape[1])
    second_probability = (second + epsilon) / (second.sum(axis=1, keepdims=True) + epsilon * second.shape[1])
    midpoint = 0.5 * (first_probability + second_probability)
    first_term = np.where(first_probability > 0, first_probability * np.log2(first_probability / midpoint), 0)
    second_term = np.where(second_probability > 0, second_probability * np.log2(second_probability / midpoint), 0)
    values = 0.5 * (first_term.sum(axis=1) + second_term.sum(axis=1))
    return np.clip(values, 0.0, 1.0)


def _cosine_rows(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ij,ij->i", first, second)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    result = np.zeros(len(first), dtype=np.float64)
    nonzero = denominator > 0
    result[nonzero] = numerator[nonzero] / denominator[nonzero]
    both_zero = (~nonzero) & np.all(first == second, axis=1)
    result[both_zero] = 1.0
    return np.clip(result, 0.0, 1.0)


def _top_k_set(row: np.ndarray, top_k: int) -> set[int]:
    positive = np.flatnonzero(row > 0)
    if len(positive) <= top_k:
        return set(positive.tolist())
    selected = positive[np.argpartition(row[positive], -top_k)[-top_k:]]
    return set(selected.tolist())


def _pattern_rows(first: np.ndarray, second: np.ndarray, top_k: int) -> np.ndarray:
    result = np.zeros(len(first), dtype=np.float64)
    for index, (left, right) in enumerate(zip(first, second, strict=True)):
        first_top = _top_k_set(left, top_k)
        second_top = _top_k_set(right, top_k)
        if not first_top and not second_top:
            result[index] = 1.0
        elif first_top and second_top:
            result[index] = 2 * len(first_top & second_top) / (len(first_top) + len(second_top))
    return result


def _validate_weights(weights: np.ndarray, population: int) -> np.ndarray:
    values = np.asarray(weights)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != population:
        raise ValueError("bootstrap weights must have shape [replicates,population]")
    if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
        raise ValueError("bootstrap weights must be non-negative integers")
    if np.any(values.sum(axis=1) != population):
        raise ValueError("each bootstrap replicate must contain exactly N draws")
    return values.astype(np.float64, copy=False)


def population_regrets(
    contributions: PopulationMetricContributions,
    weights: np.ndarray | None = None,
    *,
    chunk_size: int = 64,
) -> dict[str, np.ndarray]:
    population = len(contributions.traj_ids)
    if population == 0:
        raise ValueError("population must be non-empty")
    raw = np.ones((1, population), dtype=np.int64) if weights is None else weights
    weight_matrix = _validate_weights(raw, population)
    outputs = {metric: [] for metric in DOMAIN_METRICS + ("validity_rate",)}
    valid = contributions.valid.astype(np.float64)
    for start in range(0, len(weight_matrix), chunk_size):
        current = weight_matrix[start : start + chunk_size]
        valid_weights = current * valid[None, :]
        valid_count = valid_weights.sum(axis=1)
        validity = valid_count / population
        failure = 1.0 - validity
        outputs["validity_rate"].append(failure)
        for metric, (first_rows, second_rows) in contributions.histograms.items():
            if isinstance(contributions.supports[metric], str):
                success_regret = np.zeros(len(current), dtype=np.float64)
            else:
                success_regret = _jsd_rows(valid_weights @ first_rows, valid_weights @ second_rows)
            success_regret[valid_count == 0] = 1.0
            outputs[metric].append(np.clip(failure + validity * success_regret, 0.0, 1.0))
        for domain, (first_sparse, second_sparse) in contributions.densities.items():
            first = np.asarray(valid_weights @ first_sparse)
            second = np.asarray(valid_weights @ second_sparse)
            cosine_regret = 1.0 - _cosine_rows(first, second)
            pattern_regret = 1.0 - _pattern_rows(first, second, contributions.top_k)
            cosine_regret[valid_count == 0] = 1.0
            pattern_regret[valid_count == 0] = 1.0
            outputs[f"{domain}_density_cosine"].append(
                np.clip(failure + validity * cosine_regret, 0.0, 1.0)
            )
            outputs[f"{domain}_pattern_f1"].append(
                np.clip(failure + validity * pattern_regret, 0.0, 1.0)
            )
    return {metric: np.concatenate(chunks) for metric, chunks in outputs.items()}


def save_metric_contributions(
    contributions: PopulationMetricContributions,
    output_dir: str | Path,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arrays_path = output / "histograms.npz"
    temporary_arrays = output / "histograms.tmp.npz"
    arrays: dict[str, np.ndarray] = {
        "traj_ids": np.asarray(contributions.traj_ids, dtype=np.int64),
        "valid": contributions.valid.astype(np.uint8),
    }
    for metric, (first, second) in contributions.histograms.items():
        arrays[f"{metric}__real"] = np.asarray(first, dtype=np.float64)
        arrays[f"{metric}__generated"] = np.asarray(second, dtype=np.float64)
    np.savez_compressed(temporary_arrays, **arrays)
    temporary_arrays.replace(arrays_path)
    density_artifacts: dict[str, dict[str, object]] = {}
    for domain, (first, second) in contributions.densities.items():
        real_path = output / f"{domain}_real.npz"
        generated_path = output / f"{domain}_generated.npz"
        temporary_real = output / f"{domain}_real.tmp.npz"
        temporary_generated = output / f"{domain}_generated.tmp.npz"
        sparse.save_npz(temporary_real, first, compressed=True)
        sparse.save_npz(temporary_generated, second, compressed=True)
        temporary_real.replace(real_path)
        temporary_generated.replace(generated_path)
        density_artifacts[domain] = {
            "real": _artifact(real_path),
            "generated": _artifact(generated_path),
        }
    manifest = {
        "schema_version": CONTRIBUTION_SCHEMA_VERSION,
        "method": BOOTSTRAP_METHOD,
        "population": len(contributions.traj_ids),
        "traj_id_sha256": hashlib.sha256(
            "\n".join(map(str, contributions.traj_ids)).encode()
        ).hexdigest(),
        "top_k": contributions.top_k,
        "supports": contributions.supports,
        "histograms": _artifact(arrays_path),
        "densities": density_artifacts,
    }
    manifest_path = output / "contributions_manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _verified_artifact(root: Path, info: Mapping[str, object]) -> Path:
    path = Path(str(info["path"])).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"contribution artifact escapes output directory: {path}")
    if not path.is_file() or path.stat().st_size != int(info["size_bytes"]):
        raise ValueError(f"contribution artifact missing or size mismatch: {path}")
    if _sha256(path) != str(info["sha256"]):
        raise ValueError(f"contribution artifact SHA256 mismatch: {path}")
    return path


def load_metric_contributions(manifest_path: str | Path) -> PopulationMetricContributions:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CONTRIBUTION_SCHEMA_VERSION:
        raise ValueError("unsupported contribution schema version")
    if manifest.get("method") != BOOTSTRAP_METHOD:
        raise ValueError("unexpected contribution method")
    root = path.parent.resolve()
    arrays_path = _verified_artifact(root, manifest["histograms"])
    with np.load(arrays_path, allow_pickle=False) as archive:
        traj_ids = tuple(int(value) for value in archive["traj_ids"].tolist())
        valid = archive["valid"].astype(bool)
        histograms = {
            metric: (
                np.asarray(archive[f"{metric}__real"], dtype=np.float64),
                np.asarray(archive[f"{metric}__generated"], dtype=np.float64),
            )
            for metric in HISTOGRAM_METRICS
        }
    expected_hash = hashlib.sha256("\n".join(map(str, traj_ids)).encode()).hexdigest()
    if len(traj_ids) != int(manifest["population"]) or expected_hash != manifest["traj_id_sha256"]:
        raise ValueError("contribution population identity mismatch")
    if len(valid) != len(traj_ids) or any(
        first.shape[0] != len(traj_ids) or second.shape != first.shape
        for first, second in histograms.values()
    ):
        raise ValueError("histogram contribution shape mismatch")
    densities = {}
    for domain in ("road", "grid"):
        info = manifest["densities"][domain]
        first = sparse.load_npz(_verified_artifact(root, info["real"])).tocsr()
        second = sparse.load_npz(_verified_artifact(root, info["generated"])).tocsr()
        if first.shape[0] != len(traj_ids) or second.shape != first.shape:
            raise ValueError("density contribution shape mismatch")
        densities[domain] = (first, second)
    return PopulationMetricContributions(
        traj_ids=traj_ids,
        valid=valid,
        histograms=histograms,
        densities=densities,
        supports=manifest["supports"],
        top_k=int(manifest["top_k"]),
    )


def _interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = 1.0 - confidence
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def paired_metric_effects(
    baseline: PopulationMetricContributions,
    candidate: PopulationMetricContributions,
    *,
    n_resamples: int,
    seed: int,
    confidence: float = 0.95,
    chunk_size: int = 64,
) -> list[dict[str, object]]:
    if baseline.traj_ids != candidate.traj_ids:
        raise ValueError("paired populations differ")
    if n_resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    population = len(baseline.traj_ids)
    generator = np.random.default_rng(seed)
    weights = generator.multinomial(population, np.full(population, 1 / population), size=n_resamples)
    baseline_values = population_regrets(baseline, weights, chunk_size=chunk_size)
    candidate_values = population_regrets(candidate, weights, chunk_size=chunk_size)
    baseline_point = population_regrets(baseline)
    candidate_point = population_regrets(candidate)
    result = []
    for metric in DOMAIN_METRICS + ("validity_rate",):
        differences = candidate_values[metric] - baseline_values[metric]
        low, high = _interval(differences, confidence)
        result.append(
            {
                "metric": metric,
                "method": BOOTSTRAP_METHOD,
                "n_pairs": population,
                "estimate": float(candidate_point[metric][0] - baseline_point[metric][0]),
                "ci_low": low,
                "ci_high": high,
                "p_improvement": float(np.mean(differences < 0)),
                "p_degradation": float(np.mean(differences > 0)),
                "seed": int(seed),
                "n_resamples": int(n_resamples),
            }
        )
    return result


def paired_metric_effects_stratified_seeds(
    baseline_by_seed: Mapping[int, PopulationMetricContributions],
    candidate_by_seed: Mapping[int, PopulationMetricContributions],
    *,
    n_resamples: int,
    seed: int,
    confidence: float = 0.95,
    chunk_size: int = 64,
) -> list[dict[str, object]]:
    seed_ids = sorted(int(value) for value in baseline_by_seed)
    if not seed_ids or set(seed_ids) != {int(value) for value in candidate_by_seed}:
        raise ValueError("baseline and candidate seed strata differ")
    if n_resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    bootstrap_differences = {
        metric: np.zeros(n_resamples, dtype=np.float64)
        for metric in DOMAIN_METRICS + ("validity_rate",)
    }
    point_differences = {metric: 0.0 for metric in bootstrap_differences}
    populations: dict[int, int] = {}
    for current_seed in seed_ids:
        baseline = baseline_by_seed[current_seed]
        candidate = candidate_by_seed[current_seed]
        if baseline.traj_ids != candidate.traj_ids:
            raise ValueError(f"paired populations differ for seed {current_seed}")
        population = len(baseline.traj_ids)
        populations[current_seed] = population
        generator = np.random.default_rng(np.random.SeedSequence([int(seed), current_seed]))
        weights = generator.multinomial(
            population, np.full(population, 1 / population), size=n_resamples
        )
        baseline_values = population_regrets(baseline, weights, chunk_size=chunk_size)
        candidate_values = population_regrets(candidate, weights, chunk_size=chunk_size)
        baseline_point = population_regrets(baseline)
        candidate_point = population_regrets(candidate)
        for metric in bootstrap_differences:
            bootstrap_differences[metric] += (
                candidate_values[metric] - baseline_values[metric]
            ) / len(seed_ids)
            point_differences[metric] += float(
                candidate_point[metric][0] - baseline_point[metric][0]
            ) / len(seed_ids)
    result = []
    for metric, differences in bootstrap_differences.items():
        low, high = _interval(differences, confidence)
        result.append(
            {
                "metric": metric,
                "method": SELECTION_BOOTSTRAP_METHOD,
                "seed_strata": seed_ids,
                "seed_populations": populations,
                "n_pairs": sum(populations.values()),
                "estimate": point_differences[metric],
                "ci_low": low,
                "ci_high": high,
                "p_improvement": float(np.mean(differences < 0)),
                "p_degradation": float(np.mean(differences > 0)),
                "seed": int(seed),
                "n_resamples": int(n_resamples),
            }
        )
    return result


def g_metric_attribution(
    variants: Mapping[str, PopulationMetricContributions],
    *,
    n_resamples: int,
    seed: int,
    confidence: float = 0.95,
    chunk_size: int = 64,
) -> list[dict[str, object]]:
    required = ("g00", "g10", "g01", "g11")
    if set(variants) != set(required):
        raise ValueError(f"G contributions must be exactly {required}")
    populations = {variants[name].traj_ids for name in required}
    if len(populations) != 1:
        raise ValueError("G contribution populations differ")
    population = len(variants["g00"].traj_ids)
    if n_resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    generator = np.random.default_rng(seed)
    weights = generator.multinomial(
        population, np.full(population, 1 / population), size=n_resamples
    )
    sampled = {
        name: population_regrets(variants[name], weights, chunk_size=chunk_size)
        for name in required
    }
    points = {name: population_regrets(variants[name]) for name in required}
    result = []
    for metric in DOMAIN_METRICS + ("validity_rate",):
        sampled_components = {
            "sid": 0.5 * (
                (sampled["g10"][metric] - sampled["g00"][metric])
                + (sampled["g11"][metric] - sampled["g01"][metric])
            ),
            "sampling": 0.5 * (
                (sampled["g01"][metric] - sampled["g00"][metric])
                + (sampled["g11"][metric] - sampled["g10"][metric])
            ),
            "interaction": (
                sampled["g11"][metric] - sampled["g10"][metric]
                - sampled["g01"][metric] + sampled["g00"][metric]
            ),
            "total": sampled["g11"][metric] - sampled["g00"][metric],
        }
        point_components = {
            "sid": 0.5 * (
                (points["g10"][metric][0] - points["g00"][metric][0])
                + (points["g11"][metric][0] - points["g01"][metric][0])
            ),
            "sampling": 0.5 * (
                (points["g01"][metric][0] - points["g00"][metric][0])
                + (points["g11"][metric][0] - points["g10"][metric][0])
            ),
            "interaction": (
                points["g11"][metric][0] - points["g10"][metric][0]
                - points["g01"][metric][0] + points["g00"][metric][0]
            ),
            "total": points["g11"][metric][0] - points["g00"][metric][0],
        }
        for component, values in sampled_components.items():
            low, high = _interval(values, confidence)
            result.append(
                {
                    "metric": metric,
                    "component": component,
                    "method": BOOTSTRAP_METHOD,
                    "n_pairs": population,
                    "estimate": float(point_components[component]),
                    "ci_low": low,
                    "ci_high": high,
                    "p_improvement": float(np.mean(values < 0)),
                    "p_degradation": float(np.mean(values > 0)),
                    "seed": int(seed),
                    "n_resamples": int(n_resamples),
                }
            )
    return result
