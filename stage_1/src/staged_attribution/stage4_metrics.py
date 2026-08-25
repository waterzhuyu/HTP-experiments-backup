from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import entropy
from shapely.geometry import LineString, Point


CITY_BOUNDS = {
    "porto": {"min_lat": 41.13, "max_lat": 41.19, "min_lon": -8.69, "max_lon": -8.55},
    "chengdu": {"min_lat": 30.56, "max_lat": 30.79, "min_lon": 103.92, "max_lon": 104.21},
}


class MetricPopulationError(ValueError):
    """Raised when Stage4 inputs do not describe the exact same population."""


def geodistance(first: Sequence[float], second: Sequence[float]) -> float:
    lon1, lat1 = map(math.radians, map(float, first))
    lon2, lat2 = map(math.radians, map(float, second))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * math.asin(math.sqrt(min(1.0, value))) * 6371.0


def _segments(trajectory: np.ndarray) -> np.ndarray:
    return np.asarray(
        [geodistance(first, second) for first, second in zip(trajectory[:-1], trajectory[1:])],
        dtype=np.float64,
    )


def _radius(trajectory: np.ndarray) -> float:
    center = trajectory.mean(axis=0)
    return float(np.mean([geodistance(center, point) for point in trajectory]))


def _grid_ids(trajectory: np.ndarray, city: str, grid_size: int = 100) -> list[int]:
    if city not in CITY_BOUNDS:
        raise ValueError(f"unsupported city {city!r}")
    bounds = CITY_BOUNDS[city]
    lat_num = max(
        1,
        int(
            geodistance(
                (bounds["min_lon"], bounds["min_lat"]),
                (bounds["min_lon"], bounds["max_lat"]),
            )
            * 1000
            // grid_size
        ),
    )
    lon_num = max(
        1,
        int(
            geodistance(
                (bounds["min_lon"], bounds["min_lat"]),
                (bounds["max_lon"], bounds["min_lat"]),
            )
            * 1000
            // grid_size
        ),
    )
    lat_size = (bounds["max_lat"] - bounds["min_lat"]) / lat_num
    lon_size = (bounds["max_lon"] - bounds["min_lon"]) / lon_num
    result: list[int] = []
    for lon, lat in trajectory:
        if not (
            bounds["min_lon"] < lon < bounds["max_lon"]
            and bounds["min_lat"] < lat < bounds["max_lat"]
        ):
            continue
        column = min(lon_num - 1, int((lon - bounds["min_lon"]) / lon_size))
        row = min(lat_num - 1, int((lat - bounds["min_lat"]) / lat_size))
        value = row * lon_num + column
        if not result or result[-1] != value:
            result.append(value)
    return result


def _point_road_distances(
    trajectory: np.ndarray,
    road_traj: Sequence[int],
    road_geometries: Mapping[int, np.ndarray],
) -> list[float]:
    if len(trajectory) != len(road_traj):
        raise ValueError("GPS and pointwise road trajectories must align")
    result: list[float] = []
    for point, road in zip(trajectory, road_traj):
        if int(road) not in road_geometries:
            raise ValueError(f"missing road geometry {road}")
        line = LineString(np.asarray(road_geometries[int(road)], dtype=np.float64))
        source = Point(float(point[0]), float(point[1]))
        nearest = line.interpolate(line.project(source))
        result.append(geodistance(point, (nearest.x, nearest.y)))
    return result


def trajectory_sufficient_statistics(
    gps_traj: np.ndarray,
    road_traj: Sequence[int],
    road_geometries: Mapping[int, np.ndarray],
    city: str,
) -> dict[str, float | int]:
    trajectory = np.asarray(gps_traj, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[0] < 2 or trajectory.shape[1] != 2:
        raise ValueError("trajectory must have shape [points>=2,2]")
    if not np.isfinite(trajectory).all():
        raise ValueError("trajectory contains non-finite coordinates")
    segments = _segments(trajectory)
    point_road = _point_road_distances(trajectory, road_traj, road_geometries)
    return {
        "points": len(trajectory),
        "travel_distance_km": float(segments.sum()),
        "mean_segment_distance_km": float(segments.mean()),
        "radius_km": _radius(trajectory),
        "mean_point_to_road_km": float(np.mean(point_road)),
        "unique_road_count": len(set(int(value) for value in road_traj)),
        "unique_grid_count": len(set(_grid_ids(trajectory, city))),
    }


def _jsd(first: Sequence[float], second: Sequence[float]) -> float:
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if not len(first_values) or not len(second_values):
        return float("nan")
    minimum = min(float(first_values.min()), float(second_values.min()))
    maximum = max(float(first_values.max()), float(second_values.max()))
    if math.isclose(minimum, maximum):
        return 0.0
    bins_count = min(50, max(5, int(math.log2(len(first_values) + len(second_values))) + 1))
    bins = np.linspace(minimum - 1e-10, maximum + 1e-10, bins_count + 1)
    first_hist, _ = np.histogram(first_values, bins=bins)
    second_hist, _ = np.histogram(second_values, bins=bins)
    first_probability = (first_hist + 1e-10) / (first_hist.sum() + 1e-10 * len(first_hist))
    second_probability = (second_hist + 1e-10) / (second_hist.sum() + 1e-10 * len(second_hist))
    midpoint = 0.5 * (first_probability + second_probability)
    return float(
        0.5 * entropy(first_probability, midpoint, base=2)
        + 0.5 * entropy(second_probability, midpoint, base=2)
    )


def _density(sequences: Mapping[int, Sequence[int]], size: int) -> np.ndarray:
    density = np.zeros(size, dtype=np.float64)
    for sequence in sequences.values():
        collapsed = [
            int(value)
            for index, value in enumerate(sequence)
            if index == 0 or int(value) != int(sequence[index - 1])
        ]
        for value in collapsed:
            if value < 0 or value >= size:
                raise ValueError(f"density ID {value} outside [0,{size})")
            density[value] += 1
    return density


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0:
        return 1.0 if np.array_equal(first, second) else 0.0
    return float(np.dot(first, second) / denominator)


def _pattern(first: np.ndarray, second: np.ndarray, top_k: int) -> float:
    k = min(top_k, len(first))
    first_top = set(np.argsort(first)[::-1][:k].tolist())
    second_top = set(np.argsort(second)[::-1][:k].tolist())
    intersection = len(first_top & second_top)
    precision = intersection / k
    recall = intersection / len(first_top | second_top)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def htp_aggregate_metrics(
    real_gps: Mapping[int, np.ndarray],
    generated_gps: Mapping[int, np.ndarray],
    real_roads: Mapping[int, Sequence[int]],
    generated_roads: Mapping[int, Sequence[int]],
    road_geometries: Mapping[int, np.ndarray],
    city: str,
    top_k: int = 100,
) -> dict[str, float]:
    populations = [set(values) for values in (real_gps, generated_gps, real_roads, generated_roads)]
    if any(values != populations[0] for values in populations[1:]):
        raise MetricPopulationError("aggregate metric population mismatch")
    real_segments = {key: _segments(np.asarray(value)) for key, value in real_gps.items()}
    generated_segments = {key: _segments(np.asarray(value)) for key, value in generated_gps.items()}
    real_travel = [float(value.sum()) for value in real_segments.values()]
    generated_travel = [float(value.sum()) for value in generated_segments.values()]
    real_radius = [_radius(np.asarray(value)) for value in real_gps.values()]
    generated_radius = [_radius(np.asarray(value)) for value in generated_gps.values()]
    real_point = [
        distance
        for key in real_gps
        for distance in _point_road_distances(np.asarray(real_gps[key]), real_roads[key], road_geometries)
    ]
    generated_point = [
        distance
        for key in generated_gps
        for distance in _point_road_distances(
            np.asarray(generated_gps[key]), generated_roads[key], road_geometries
        )
    ]
    real_grids = {key: _grid_ids(np.asarray(value), city) for key, value in real_gps.items()}
    generated_grids = {
        key: _grid_ids(np.asarray(value), city) for key, value in generated_gps.items()
    }
    max_grid = max(
        [value for sequence in list(real_grids.values()) + list(generated_grids.values()) for value in sequence],
        default=0,
    )
    max_road = max(road_geometries, default=0)
    real_grid_density = _density(real_grids, max_grid + 1)
    generated_grid_density = _density(generated_grids, max_grid + 1)
    real_road_density = _density(real_roads, max_road + 1)
    generated_road_density = _density(generated_roads, max_road + 1)
    return {
        "travel_distance_jsd": _jsd(real_travel, generated_travel),
        "segment_distance_jsd": _jsd(
            np.concatenate(list(real_segments.values())),
            np.concatenate(list(generated_segments.values())),
        ),
        "radius_jsd": _jsd(real_radius, generated_radius),
        "grid_density_cosine": _cosine(real_grid_density, generated_grid_density),
        "grid_pattern_f1": _pattern(real_grid_density, generated_grid_density, top_k),
        "road_density_cosine": _cosine(real_road_density, generated_road_density),
        "road_pattern_f1": _pattern(real_road_density, generated_road_density, top_k),
        "point_to_road_jsd": _jsd(real_point, generated_point),
    }


def evaluate_population(
    real_gps: Mapping[int, np.ndarray],
    real_roads: Mapping[int, Sequence[int]],
    generated_records: Sequence[Mapping[str, object]],
    road_geometries: Mapping[int, np.ndarray],
    city: str,
    invalid_penalty: float = 1.0,
) -> dict[str, object]:
    if set(real_gps) != set(real_roads):
        raise MetricPopulationError("real GPS/Road population mismatch")
    generated_by_id: dict[int, Mapping[str, object]] = {}
    for record in generated_records:
        traj_id = int(record["traj_id"])
        if traj_id in generated_by_id:
            raise MetricPopulationError(f"duplicate generated traj_id={traj_id}")
        generated_by_id[traj_id] = record
    if set(generated_by_id) != set(real_gps):
        raise MetricPopulationError("generated/real population mismatch")
    per_trajectory: list[dict[str, object]] = []
    generated_gps: dict[int, np.ndarray] = {}
    generated_roads: dict[int, Sequence[int]] = {}
    failures: Counter[str] = Counter()
    for traj_id in sorted(real_gps):
        record = generated_by_id[traj_id]
        if record.get("status") != "reconstructed":
            reason = str(record.get("failure", {}).get("reason", "unknown"))
            failures[reason] += 1
            per_trajectory.append(
                {"traj_id": traj_id, "valid": False, "failure_reason": reason, "penalized_error": invalid_penalty}
            )
            continue
        try:
            generated = np.asarray(record["gps_traj"], dtype=np.float64)
            road = [int(value) for value in record["road_traj"]]
            real_stats = trajectory_sufficient_statistics(
                np.asarray(real_gps[traj_id]), real_roads[traj_id], road_geometries, city
            )
            generated_stats = trajectory_sufficient_statistics(generated, road, road_geometries, city)
            error = sum(
                abs(math.log1p(float(generated_stats[name])) - math.log1p(float(real_stats[name])))
                for name in ("travel_distance_km", "mean_segment_distance_km", "radius_km", "mean_point_to_road_km")
            ) / 4
            generated_gps[traj_id] = generated
            generated_roads[traj_id] = road
            per_trajectory.append(
                {"traj_id": traj_id, "valid": True, "penalized_error": error, "real": real_stats, "generated": generated_stats}
            )
        except Exception as error:
            failures["metric_error"] += 1
            per_trajectory.append(
                {"traj_id": traj_id, "valid": False, "failure_reason": "metric_error", "message": str(error), "penalized_error": invalid_penalty}
            )
    successful_ids = set(generated_gps)
    successful_metrics = None
    if successful_ids:
        successful_metrics = htp_aggregate_metrics(
            {key: real_gps[key] for key in successful_ids},
            generated_gps,
            {key: real_roads[key] for key in successful_ids},
            generated_roads,
            road_geometries,
            city,
        )
    successful = len(successful_ids)
    return {
        "population": len(real_gps),
        "successful": successful,
        "failed": len(real_gps) - successful,
        "validity_rate": successful / len(real_gps),
        "failure_reasons": dict(failures),
        "population_mean_penalized_error": float(np.mean([row["penalized_error"] for row in per_trajectory])),
        "successful_only": {"population": successful, "metrics": successful_metrics},
        "per_trajectory": per_trajectory,
    }
