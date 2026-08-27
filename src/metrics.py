from __future__ import annotations

from typing import Dict, Iterable
import numpy as np


def jain_fairness(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    if a.size == 0 or np.allclose(a, 0.0):
        return 1.0
    return float((a.sum() ** 2) / (a.size * np.square(a).sum() + 1e-12))


def normalized_imbalance(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    if a.size == 0 or np.mean(a) <= 1e-12:
        return 0.0
    return float(np.std(a) / (np.mean(a) + 1e-12))


def congestion_stability(congestion_history: np.ndarray) -> float:
    c = np.asarray(congestion_history, dtype=float)
    if c.ndim < 2 or c.shape[0] < 2:
        return 1.0
    delta = np.diff(c, axis=0)
    oscillation = float(np.mean(np.square(delta)))
    return float(1.0 / (1.0 + oscillation))


def summarize_episode(records: list[dict]) -> Dict[str, float]:
    if not records:
        return {}
    retrieval = np.asarray([r.get("retrieval_time", 0.0) for r in records], dtype=float)
    distance = np.asarray([r.get("travel_distance", 0.0) for r in records], dtype=float)
    throughput = np.asarray([r.get("throughput", 0.0) for r in records], dtype=float)
    fairness = np.asarray([r.get("fairness", 1.0) for r in records], dtype=float)
    reward = np.asarray([r.get("reward", 0.0) for r in records], dtype=float)
    congestion = np.stack([np.asarray(r.get("congestion", [0.0]), dtype=float) for r in records])
    return {
        "average_retrieval_time": float(retrieval.mean()),
        "travel_distance": float(distance.mean()),
        "throughput": float(throughput.mean()),
        "congestion_mean": float(congestion.mean()),
        "congestion_variance": float(congestion.var()),
        "congestion_stability": congestion_stability(congestion),
        "fairness": float(fairness.mean()),
        "cumulative_reward": float(reward.sum()),
    }


def composite_index(metrics: Dict[str, float]) -> float:
    # Higher is better. Transform costs to bounded benefits rather than mixing raw units.
    efficiency = 1.0 / (1.0 + max(metrics.get("average_retrieval_time", 0.0), 0.0))
    stability = metrics.get("congestion_stability", 0.0)
    fairness = metrics.get("fairness", 0.0)
    throughput = metrics.get("throughput", 0.0)
    throughput_term = throughput / (throughput + 10.0) if throughput > 0 else 0.0
    return float(0.35 * efficiency + 0.25 * stability + 0.25 * fairness + 0.15 * throughput_term)
