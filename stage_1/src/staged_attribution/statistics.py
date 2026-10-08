from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import t


def paired_bootstrap(
    baseline: Mapping[int, float],
    candidate: Mapping[int, float],
    n_resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    if set(baseline) != set(candidate):
        raise ValueError("paired population mismatch")
    if not baseline:
        raise ValueError("paired bootstrap requires observations")
    if n_resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    ids = sorted(baseline)
    differences = np.asarray([float(candidate[key]) - float(baseline[key]) for key in ids])
    if not np.isfinite(differences).all():
        raise ValueError("paired values must be finite")
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, len(ids), size=(n_resamples, len(ids)))
    estimates = differences[samples].mean(axis=1)
    alpha = 1 - confidence
    estimate = float(differences.mean())
    return {
        "n_pairs": len(ids),
        "estimate": estimate,
        "ci_low": float(np.quantile(estimates, alpha / 2)),
        "ci_high": float(np.quantile(estimates, 1 - alpha / 2)),
        "p_improvement": float(np.mean(estimates < 0)),
        "p_degradation": float(np.mean(estimates > 0)),
        "seed": int(seed),
        "n_resamples": int(n_resamples),
    }


def g_shapley(values: Mapping[str, float]) -> dict[str, float]:
    required = {"g00", "g10", "g01", "g11"}
    if set(values) != required:
        raise ValueError(f"G values must be exactly {sorted(required)}")
    numeric = {name: float(value) for name, value in values.items()}
    if not all(math.isfinite(value) for value in numeric.values()):
        raise ValueError("G values must be finite")
    sid = 0.5 * ((numeric["g10"] - numeric["g00"]) + (numeric["g11"] - numeric["g01"]))
    sampling = 0.5 * (
        (numeric["g01"] - numeric["g00"]) + (numeric["g11"] - numeric["g10"])
    )
    interaction = numeric["g11"] - numeric["g10"] - numeric["g01"] + numeric["g00"]
    return {
        "sid": sid,
        "sampling": sampling,
        "interaction": interaction,
        "total": numeric["g11"] - numeric["g00"],
    }


def aggregate_seeds(values: Sequence[float], confidence: float = 0.95) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("seed values must be a non-empty finite sequence")
    mean = float(array.mean())
    if len(array) == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "sem": 0.0, "ci_low": mean, "ci_high": mean}
    standard = float(array.std(ddof=1))
    sem = standard / math.sqrt(len(array))
    critical = float(t.ppf(0.5 + confidence / 2, df=len(array) - 1))
    return {
        "n": len(array),
        "mean": mean,
        "std": standard,
        "sem": sem,
        "ci_low": mean - critical * sem,
        "ci_high": mean + critical * sem,
    }
