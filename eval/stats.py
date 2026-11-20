"""Small, dependency-free statistics used by every evaluation script."""

import numpy as np


def bootstrap_ci(values, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Mean with a percentile bootstrap CI: resample questions with replacement, recompute the mean."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(x))}


def paired_bootstrap_diff(a, b, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> dict:
    """CI for mean(a - b) when a and b are scores on the SAME questions. Resampling questions jointly keeps the
    pairing, which is much tighter than comparing two independent CIs (question difficulty cancels out)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    ci = bootstrap_ci(d, n_boot, seed, alpha)
    return {**ci, "significant": not (ci["lo"] <= 0 <= ci["hi"])}


def cohen_kappa(a, b) -> float:
    """Agreement between two binary labelers corrected for chance: (p_o - p_e) / (1 - p_e)."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    p_o = float((a == b).mean())
    p_e = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return 1.0 if p_e == 1 else (p_o - p_e) / (1 - p_e)


def percentiles(seconds) -> dict:
    ms = np.asarray(seconds, dtype=float) * 1000
    return {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)), "n": int(len(ms))}
