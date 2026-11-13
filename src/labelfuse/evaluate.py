"""Evaluation of aggregated labels against gold labels.

Accuracy gives an item full credit when its single top class is the gold class and 1/t when the
gold class is one of t classes tied at the top, which is the expected score of breaking ties
uniformly at random. Intervals come from a percentile bootstrap over items. Each method is also
compared with a baseline (majority vote by default) on the same items: a paired bootstrap interval
for the difference in accuracy and an exact McNemar test on the items only one of the two gets
right. The posteriors are checked with a reliability table, the expected calibration error and
the Brier score.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from labelfuse.data import LabelSet
from labelfuse.result import TIE_TOLERANCE, AggregationResult, resolve_labels


def gold_indices(labels: LabelSet) -> np.ndarray:
    """Class index of each item's gold label: -1 if it is not a class, -2 if there is no gold."""
    if labels.gold is None:
        raise ValueError("the label set has no gold labels")
    class_of = {name: index for index, name in enumerate(labels.classes)}
    return np.array(
        [-2 if value is None else class_of.get(value, -1) for value in labels.gold], dtype=np.int64
    )


def tie_credit(posterior: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Per-item score: 1/t when the gold class is among the t top classes, otherwise 0."""
    posterior = np.asarray(posterior, dtype=float)
    at_top = posterior >= posterior.max(axis=1, keepdims=True) - TIE_TOLERANCE
    rows = np.arange(posterior.shape[0])
    hit = np.where(gold >= 0, at_top[rows, np.maximum(gold, 0)], False)
    return hit / at_top.sum(axis=1)


def bootstrap_mean_interval(
    values: np.ndarray, *, n_boot: int = 2000, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values``, resampling the entries.

    Resampling n entries with replacement only changes how often each distinct value occurs, and
    those counts are multinomial, so the resamples are drawn as counts instead of index arrays.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("no values to resample")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    distinct, counts = np.unique(values, return_counts=True)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(values.size, counts / values.size, size=n_boot)
    means = draws @ distinct / values.size
    low, high = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(low), float(high)


def mcnemar_exact(only_first: int, only_second: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts.

    Under the null hypothesis each discordant item is equally likely to favour either method, so
    the smaller count is Binomial(n, 1/2); the p-value is twice its lower tail, capped at 1.
    """
    if only_first < 0 or only_second < 0:
        raise ValueError("counts must not be negative")
    n = only_first + only_second
    if n == 0:
        return 1.0
    term, tail = 1, 0
    for i in range(min(only_first, only_second) + 1):
        tail += term
        term = term * (n - i) // (i + 1)
    return min(1.0, 2 * tail / 2**n)


@dataclass(frozen=True)
class ReliabilityBin:
    """Items whose top probability falls in ``[lower, upper)`` (the last bin includes 1)."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


def reliability_table(
    confidence: np.ndarray, correct: np.ndarray, *, n_bins: int = 10
) -> tuple[ReliabilityBin, ...]:
    """Group items by top probability and compare it with the share that is correct.

    Bins are equal-width on [0, 1]; empty bins are left out.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    index = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)
    table = []
    for b in range(n_bins):
        inside = index == b
        if inside.any():
            table.append(
                ReliabilityBin(
                    lower=b / n_bins,
                    upper=(b + 1) / n_bins,
                    count=int(inside.sum()),
                    mean_confidence=float(confidence[inside].mean()),
                    accuracy=float(correct[inside].mean()),
                )
            )
    return tuple(table)


def expected_calibration_error(table: tuple[ReliabilityBin, ...]) -> float:
    """Count-weighted mean gap between accuracy and mean top probability over the bins."""
    total = sum(row.count for row in table)
    return sum(row.count * abs(row.accuracy - row.mean_confidence) for row in table) / total


def brier_score(posterior: np.ndarray, gold: np.ndarray) -> float:
    """Mean squared distance between the posterior and the one-hot gold class (0 best, 2 worst)."""
    posterior = np.asarray(posterior, dtype=float)
    target = np.zeros_like(posterior)
    known = gold >= 0
    target[np.flatnonzero(known), gold[known]] = 1.0
    return float(((posterior - target) ** 2).sum(axis=1).mean())


@dataclass(frozen=True)
class MethodEvaluation:
    """One method against gold, and against the baseline on the same items.

    ``only_method`` and ``only_baseline`` count the items that exactly one of the two gets right
    once ties are broken at random with the seed; ``mcnemar_p`` is the exact test on them.
    """

    method: str
    n_items: int
    accuracy: float
    interval: tuple[float, float]
    n_tied: int
    difference: float
    difference_interval: tuple[float, float]
    only_method: int
    only_baseline: int
    mcnemar_p: float
    ece: float
    brier: float
    reliability: tuple[ReliabilityBin, ...]


def evaluate(
    labels: LabelSet,
    results: Mapping[str, AggregationResult],
    *,
    baseline: str = "majority",
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    n_bins: int = 10,
) -> list[MethodEvaluation]:
    """Score every result on the items that have a gold label."""
    if baseline not in results:
        raise ValueError(f"the baseline {baseline!r} is not among the results")
    gold = gold_indices(labels)
    known = gold != -2
    if not known.any():
        raise ValueError("no item has a gold label")
    gold = gold[known]
    truth = np.array([value for value in labels.gold or () if value is not None], dtype=object)

    credits, hard = {}, {}
    for name, result in results.items():
        if result.items != labels.items:
            raise ValueError(f"the result {name!r} was computed on other items")
        posterior = result.posterior[known]
        credits[name] = tie_credit(posterior, gold)
        names, _ = resolve_labels(posterior, result.classes, tie="random", seed=seed)
        hard[name] = np.array(names, dtype=object) == truth

    rows = []
    for name, result in results.items():
        posterior = result.posterior[known]
        score = credits[name]
        difference = score - credits[baseline]
        only_method = int((hard[name] & ~hard[baseline]).sum())
        only_baseline = int((~hard[name] & hard[baseline]).sum())
        table = reliability_table(posterior.max(axis=1), score, n_bins=n_bins)
        rows.append(
            MethodEvaluation(
                method=name,
                n_items=int(known.sum()),
                accuracy=float(score.mean()),
                interval=bootstrap_mean_interval(score, n_boot=n_boot, level=level, seed=seed),
                n_tied=int(result.tied[known].sum()),
                difference=float(difference.mean()),
                difference_interval=bootstrap_mean_interval(
                    difference, n_boot=n_boot, level=level, seed=seed
                ),
                only_method=only_method,
                only_baseline=only_baseline,
                mcnemar_p=mcnemar_exact(only_method, only_baseline),
                ece=expected_calibration_error(table),
                brier=brier_score(posterior, gold),
                reliability=table,
            )
        )
    return rows
