from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


CORE_CELLS = (
    "e0_q0", "e1_q0", "e2_q0", "e3_q0", "e4_q0", "e5_q0", "e6_q0",
    "e6_q1", "e6_q2", "e6_q3",
)


def _core(values: Mapping[str, float]) -> dict[str, float]:
    missing = [cell for cell in CORE_CELLS if cell not in values]
    if missing:
        raise ValueError(f"missing core cells: {missing}")
    parsed = {cell: float(values[cell]) for cell in CORE_CELLS}
    if not all(math.isfinite(value) for value in parsed.values()):
        raise ValueError("core cell metrics must be finite")
    return parsed


def e_contrasts(values: Mapping[str, float]) -> dict[str, float]:
    parsed = _core(values)
    baseline = parsed["e0_q0"]
    result = {
        f"e{index}_q0_minus_e0_q0": parsed[f"e{index}_q0"] - baseline
        for index in range(1, 7)
    }
    result.update(
        {
            "gps_road_interaction": parsed["e3_q0"] - parsed["e1_q0"] - baseline + baseline,
            "gps_grid_interaction": parsed["e4_q0"] - parsed["e2_q0"] - baseline + baseline,
            "road_grid_interaction": parsed["e5_q0"] - parsed["e1_q0"] - parsed["e2_q0"] + baseline,
            "all_vs_best_pair": parsed["e6_q0"] - min(
                parsed["e3_q0"], parsed["e4_q0"], parsed["e5_q0"]
            ),
        }
    )
    return result


def q_incremental_contrasts(values: Mapping[str, float]) -> dict[str, float]:
    parsed = _core(values)
    return {
        "q1_minus_q0": parsed["e6_q1"] - parsed["e6_q0"],
        "q2_minus_q1": parsed["e6_q2"] - parsed["e6_q1"],
        "q3_minus_q2": parsed["e6_q3"] - parsed["e6_q2"],
        "q3_minus_q0": parsed["e6_q3"] - parsed["e6_q0"],
    }


def pareto_front(
    candidates: Mapping[str, Mapping[str, float]],
    minimize: Sequence[str],
    maximize: Sequence[str],
) -> list[str]:
    if not candidates:
        return []
    metrics = tuple(minimize) + tuple(maximize)
    result: list[str] = []
    for name, values in candidates.items():
        if any(metric not in values or not math.isfinite(float(values[metric])) for metric in metrics):
            raise ValueError(f"candidate {name} missing a finite Pareto metric")
        dominated = False
        for other_name, other in candidates.items():
            if other_name == name:
                continue
            no_worse = all(float(other[metric]) <= float(values[metric]) for metric in minimize) and all(
                float(other[metric]) >= float(values[metric]) for metric in maximize
            )
            strictly_better = any(float(other[metric]) < float(values[metric]) for metric in minimize) or any(
                float(other[metric]) > float(values[metric]) for metric in maximize
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(name)
    return sorted(result)


def select_interaction_cells(
    interactions: Mapping[str, Mapping[str, float]], threshold: float
) -> list[str]:
    if threshold < 0:
        raise ValueError("interaction threshold must be non-negative")
    selected = []
    for name, result in interactions.items():
        estimate = float(result["estimate"])
        low = float(result["ci_low"])
        high = float(result["ci_high"])
        significant = low > 0 or high < 0
        if abs(estimate) >= threshold and significant:
            selected.append(name)
    return sorted(selected)


def city_heterogeneity(city_values: Mapping[str, float]) -> dict[str, object]:
    if len(city_values) < 2:
        raise ValueError("city heterogeneity requires at least two cities")
    parsed = {name: float(value) for name, value in city_values.items()}
    if not all(math.isfinite(value) for value in parsed.values()):
        raise ValueError("city effects must be finite")
    values = list(parsed.values())
    return {
        "city_values": parsed,
        "range": max(values) - min(values),
        "sign_heterogeneous": min(values) < 0 < max(values),
    }
