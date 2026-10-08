"""Gate 0 audit for Stage 1 motion-primitive validation."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
EARTH_RADIUS_M = 6_371_008.8


def quantiles(values: list[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {f"p{int(q * 100):02d}": None for q in QUANTILES}
    return {f"p{int(q * 100):02d}": float(np.quantile(array, q)) for q in QUANTILES}


def haversine_m(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.empty(0, dtype=np.float64)
    lon = np.deg2rad(points[:, 0].astype(np.float64))
    lat = np.deg2rad(points[:, 1].astype(np.float64))
    dlon = np.diff(lon)
    dlat = np.diff(lat)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def sample_values(target: list[float], values: np.ndarray, limit: int, rng: np.random.Generator) -> None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values) or len(target) >= limit:
        return
    remaining = limit - len(target)
    if len(values) > remaining:
        values = values[rng.choice(len(values), remaining, replace=False)]
    target.extend(values.tolist())


def audit_city(data_root: Path, city: str, sample_limit: int, seed: int) -> dict:
    path = data_root / city / "traj.parquet"
    frame = pd.read_parquet(path, columns=["traj_id", "user_id", "time", "geometry", "cpath", "opath"])
    rng = np.random.default_rng(seed)

    point_counts: list[int] = []
    durations: list[float] = []
    dt_sample: list[float] = []
    gap_sample: list[float] = []
    speed_sample: list[float] = []
    accel_sample: list[float] = []
    road_counts: Counter[int] = Counter()
    cadence_counts: Counter[int] = Counter()
    matched_point_count = 0
    total_point_count = 0
    aligned_opath_trajectories = 0
    nonempty_cpath_trajectories = 0
    positive_time_trajectories = 0
    three_view_eligible = 0
    valid_kinematic_trajectories = 0

    for row in frame.itertuples(index=False):
        times = np.asarray(row.time, dtype=np.int64)
        raw_points = np.asarray(row.geometry, dtype=object)
        if len(raw_points):
            points = np.vstack(raw_points).astype(np.float64)
        else:
            points = np.empty((0, 2), dtype=np.float64)
        opath = np.asarray(row.opath, dtype=np.int64)
        cpath = np.asarray(row.cpath, dtype=np.int64)
        n = min(len(times), len(points))
        times = times[:n]
        points = points[:n]
        point_counts.append(n)
        total_point_count += n

        if len(cpath):
            nonempty_cpath_trajectories += 1
            road_counts.update(int(value) for value in np.unique(cpath) if value >= 0)
        if len(opath) == n:
            aligned_opath_trajectories += 1
            matched_point_count += int(np.sum(opath >= 0))

        if n < 2:
            continue
        dt = np.diff(times).astype(np.float64)
        positive = dt > 0
        if not np.all(positive):
            continue
        positive_time_trajectories += 1
        duration = float(times[-1] - times[0])
        durations.append(duration)
        rounded_dt = np.rint(dt).astype(np.int64)
        cadence_counts.update(rounded_dt.tolist())
        sample_values(dt_sample, dt, sample_limit, rng)

        local_mode = int(Counter(rounded_dt.tolist()).most_common(1)[0][0])
        gaps = np.maximum(0.0, dt / max(local_mode, 1) - 1.0)
        sample_values(gap_sample, gaps[gaps > 0], sample_limit, rng)

        if n >= 48 and duration >= 24 * local_mode:
            three_view_eligible += 1

        distances = haversine_m(points)
        if len(distances) != len(dt):
            continue
        speeds = distances / dt
        sample_values(speed_sample, speeds, sample_limit, rng)
        if len(speeds) >= 2:
            midpoint_dt = (dt[:-1] + dt[1:]) / 2
            acceleration = np.diff(speeds) / midpoint_dt
            sample_values(accel_sample, acceleration, sample_limit, rng)
        valid_kinematic_trajectories += 1

    modal_dt, modal_dt_count = cadence_counts.most_common(1)[0]
    users_per_trajectory = frame.groupby("user_id", sort=False).size().to_numpy()
    road_trajectory_counts = np.asarray(list(road_counts.values()), dtype=np.int64)
    dt_array = np.asarray(dt_sample, dtype=np.float64)
    speed_array = np.asarray(speed_sample, dtype=np.float64)
    accel_array = np.asarray(accel_sample, dtype=np.float64)
    trajectories = len(frame)

    three_view_ratio = three_view_eligible / max(trajectories, 1)
    if positive_time_trajectories < 0.99 * trajectories:
        gate = "blocked"
        reason = "More than 1% of trajectories lack strictly increasing timestamps."
    elif modal_dt <= 5 and three_view_ratio >= 0.2:
        gate = "pass"
        reason = "Sub-5-second reference data and at least three downsampling views are available."
    elif modal_dt <= 15 and three_view_ratio >= 0.2:
        gate = "degraded_pass"
        reason = "Three views are available, but native cadence only supports coarse motion states."
    else:
        gate = "blocked"
        reason = "Insufficient cadence or eligible trajectory coverage for three views."

    return {
        "city": city,
        "source": str(path),
        "gate_0": {"decision": gate, "reason": reason},
        "counts": {
            "trajectories": trajectories,
            "unique_trajectory_ids": int(frame["traj_id"].nunique()),
            "unique_users": int(frame["user_id"].nunique()),
            "unique_roads": len(road_counts),
            "points": total_point_count,
        },
        "field_completeness": {
            "strictly_increasing_time_ratio": positive_time_trajectories / max(trajectories, 1),
            "opath_length_matches_points_ratio": aligned_opath_trajectories / max(trajectories, 1),
            "matched_point_ratio": matched_point_count / max(total_point_count, 1),
            "nonempty_cpath_ratio": nonempty_cpath_trajectories / max(trajectories, 1),
            "kinematics_computable_ratio": valid_kinematic_trajectories / max(trajectories, 1),
        },
        "sampling": {
            "modal_interval_seconds": modal_dt,
            "modal_interval_share": modal_dt_count / max(sum(cadence_counts.values()), 1),
            "interval_seconds_quantiles": quantiles(dt_array),
            "positive_missing_intervals_quantiles": quantiles(gap_sample),
            "three_view_eligible_ratio": three_view_ratio,
            "supported_views_seconds": [modal_dt, modal_dt * 2, modal_dt * 3],
        },
        "trajectory": {
            "point_count_quantiles": quantiles(point_counts),
            "duration_seconds_quantiles": quantiles(durations),
            "trajectories_per_user_quantiles": quantiles(users_per_trajectory),
            "trajectories_per_road_quantiles": quantiles(road_trajectory_counts),
        },
        "kinematics": {
            "speed_mps_quantiles": quantiles(speed_array),
            "acceleration_mps2_quantiles": quantiles(accel_array),
            "speed_over_55_mps_ratio": float(np.mean(speed_array > 55)) if len(speed_array) else None,
            "abs_acceleration_over_6_mps2_ratio": float(np.mean(np.abs(accel_array) > 6)) if len(accel_array) else None,
            "sampled_speed_count": len(speed_array),
            "sampled_acceleration_count": len(accel_array),
        },
        "limitations": [
            "GPS-derived speed and acceleration are approximate rather than external ground truth.",
            "Ten-to-fifteen-second native cadence cannot resolve brief stops or sharp acceleration reliably.",
        ],
    }


def markdown_report(report: dict) -> str:
    lines = ["# Stage 1 Gate 0 Data Audit", ""]
    for city in report["cities"]:
        q = city["sampling"]["interval_seconds_quantiles"]
        length_q = city["trajectory"]["point_count_quantiles"]
        duration_q = city["trajectory"]["duration_seconds_quantiles"]
        fields = city["field_completeness"]
        lines.extend([
            f"## {city['city'].title()}", "",
            f"- Decision: **{city['gate_0']['decision']}** — {city['gate_0']['reason']}",
            f"- Trajectories/users/roads: {city['counts']['trajectories']:,} / {city['counts']['unique_users']:,} / {city['counts']['unique_roads']:,}",
            f"- Native interval: {city['sampling']['modal_interval_seconds']} s ({city['sampling']['modal_interval_share']:.2%} of intervals); p01–p99 {q['p01']}–{q['p99']} s",
            f"- Supported views: {city['sampling']['supported_views_seconds']} s; eligible trajectory ratio {city['sampling']['three_view_eligible_ratio']:.2%}",
            f"- Median points/duration: {length_q['p50']} points / {duration_q['p50']} s",
            f"- Timestamp/path/map-match completeness: {fields['strictly_increasing_time_ratio']:.2%} / {fields['opath_length_matches_points_ratio']:.2%} / {fields['matched_point_ratio']:.2%}",
            f"- Speed >55 m/s: {city['kinematics']['speed_over_55_mps_ratio']:.3%}; |acceleration| >6 m/s²: {city['kinematics']['abs_acceleration_over_6_mps2_ratio']:.3%}",
            "",
        ])
    lines.extend([
        "## Gate 0 Conclusion", "",
        report["conclusion"], "",
        "The Stage 1 claim must initially be restricted to coarse states (stop/slow/cruise/fast and persistent turn/stop-go regimes). Short acceleration bursts and brief signal stops are outside the supported resolution.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("traj_data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cities", nargs="+", default=["chengdu", "porto"])
    parser.add_argument("--sample-limit", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cities = [audit_city(args.data_root, city, args.sample_limit, args.seed) for city in args.cities]
    decisions = {city["gate_0"]["decision"] for city in cities}
    if "blocked" in decisions:
        conclusion = "Gate 0 is blocked for at least one city; do not proceed to semantic claims."
    elif decisions == {"pass"}:
        conclusion = "Gate 0 passes for fine-grained motion validation in both cities."
    else:
        conclusion = "Gate 0 passes with reduced scope: both datasets support multi-view tests, but only for coarse motion primitives."
    report = {"schema_version": 1, "seed": args.seed, "cities": cities, "conclusion": conclusion}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gate_0_data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "data_audit.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "conclusion": conclusion}, indent=2))


if __name__ == "__main__":
    main()
