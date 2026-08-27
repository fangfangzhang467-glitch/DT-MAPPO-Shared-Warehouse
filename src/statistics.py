from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(values: Iterable[float], confidence: float = 0.95) -> dict:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    if x.size > 1:
        sem = stats.sem(x)
        h = float(stats.t.ppf((1 + confidence) / 2.0, x.size - 1) * sem)
    else:
        h = 0.0
    return {"n": int(x.size), "mean": mean, "std": std, "ci_low": mean - h, "ci_high": mean + h}


def cohen_dz(a: Iterable[float], b: Iterable[float]) -> float:
    d = np.asarray(list(a), dtype=float) - np.asarray(list(b), dtype=float)
    if d.size < 2 or np.std(d, ddof=1) < 1e-12:
        return 0.0
    return float(np.mean(d) / np.std(d, ddof=1))


def paired_test(a: Iterable[float], b: Iterable[float], alpha: float = 0.05) -> dict:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return {"test": "insufficient", "statistic": np.nan, "p_value": np.nan, "effect_size": np.nan}
    diff = x - y
    normal_p = stats.shapiro(diff).pvalue if 3 <= len(diff) <= 5000 else 0.0
    if normal_p >= alpha:
        stat, p = stats.ttest_rel(x, y)
        test = "paired_t"
    else:
        if np.allclose(diff, 0.0):
            stat, p = 0.0, 1.0
        else:
            stat, p = stats.wilcoxon(x, y, zero_method="wilcox")
        test = "wilcoxon"
    return {"test": test, "statistic": float(stat), "p_value": float(p), "effect_size": cohen_dz(x, y), "normality_p": float(normal_p)}


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    out = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    idx = np.where(finite)[0]
    if not len(idx):
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, original in enumerate(order):
        adjusted = min(1.0, (m - rank) * p[original])
        running = max(running, adjusted)
        out[original] = running
    return out


def aggregate_seed_results(df: pd.DataFrame, group_cols=("method", "scenario"), confidence: float = 0.95) -> pd.DataFrame:
    metrics = [c for c in df.columns if c not in set(group_cols) | {"seed", "episode"} and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for keys, grp in df.groupby(list(group_cols)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for metric in metrics:
            s = mean_ci(grp[metric], confidence)
            rows.append({**base, "metric": metric, **s})
    return pd.DataFrame(rows)
