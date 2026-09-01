from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


CORE_CELLS = (
    "e0_q0", "e1_q0", "e2_q0", "e3_q0", "e4_q0", "e5_q0", "e6_q0",
    "e6_q1", "e6_q2", "e6_q3",
)

E_LINEAR_CONTRASTS = {
    **{
        f"e{index}_q0_minus_e0_q0": {
            f"e{index}_q0": 1.0,
            "e0_q0": -1.0,
        }
        for index in range(1, 7)
    },
    "gps_road_given_grid": {
        "e6_q0": 1.0,
        "e4_q0": -1.0,
        "e5_q0": -1.0,
        "e2_q0": 1.0,
    },
    "gps_grid_given_road": {
        "e6_q0": 1.0,
        "e3_q0": -1.0,
        "e5_q0": -1.0,
        "e1_q0": 1.0,
    },
    "road_grid_given_gps": {
        "e6_q0": 1.0,
        "e3_q0": -1.0,
        "e4_q0": -1.0,
        "e0_q0": 1.0,
    },
}
Q_LINEAR_CONTRASTS = {
    "q1_minus_q0": {"e6_q1": 1.0, "e6_q0": -1.0},
    "q2_minus_q1": {"e6_q2": 1.0, "e6_q1": -1.0},
    "q3_minus_q2": {"e6_q3": 1.0, "e6_q2": -1.0},
    "q3_minus_q0": {"e6_q3": 1.0, "e6_q0": -1.0},
}
E_BEST_PAIR_CONTRAST = "all_minus_best_pair_for_minimize"
E_BEST_PAIR_EXPRESSION = "e6_q0 - min(e3_q0, e4_q0, e5_q0)"


def staged_contrast_definitions() -> list[dict[str, object]]:
    """Return the registered, serializable E/Q estimands in stable order."""
    definitions = [
        {
            "family": "E",
            "contrast": name,
            "kind": "linear",
            "role": "primary",
            "coefficients": dict(coefficients),
        }
        for name, coefficients in E_LINEAR_CONTRASTS.items()
    ]
    definitions.append(
        {
            "family": "E",
            "contrast": E_BEST_PAIR_CONTRAST,
            "kind": "nonlinear",
            "role": "descriptive_sensitivity",
            "expression": E_BEST_PAIR_EXPRESSION,
        }
    )
    definitions.extend(
        {
            "family": "Q",
            "contrast": name,
            "kind": "linear",
            "role": "primary",
            "coefficients": dict(coefficients),
        }
        for name, coefficients in Q_LINEAR_CONTRASTS.items()
    )
    return definitions


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
    result = {
        name: sum(parsed[cell] * coefficient for cell, coefficient in coefficients.items())
        for name, coefficients in E_LINEAR_CONTRASTS.items()
    }
    result[E_BEST_PAIR_CONTRAST] = parsed["e6_q0"] - min(
        parsed["e3_q0"], parsed["e4_q0"], parsed["e5_q0"]
    )
    return result


def q_incremental_contrasts(values: Mapping[str, float]) -> dict[str, float]:
    parsed = _core(values)
    return {
        name: sum(parsed[cell] * coefficient for cell, coefficient in coefficients.items())
        for name, coefficients in Q_LINEAR_CONTRASTS.items()
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
