"""Audits of annotators and items, and a planner for further labels.

Everything here is computed from a label set and the posterior of an aggregation, so it works
with any aggregator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from labelfuse.data import LabelSet
from labelfuse.evaluate import gold_indices
from labelfuse.result import AggregationResult


def spammer_score(confusion: np.ndarray) -> float:
    """Spammer score of Raykar and Yu (2012) for one confusion matrix.

    An annotator whose answer does not depend on the true class has a confusion matrix with equal
    rows. The score is the squared distance of the matrix from its nearest matrix with equal rows,
    divided by its largest possible value:

        S = 1 / (K (K - 1)) * sum over pairs c < c' and classes k of (A[c, k] - A[c', k])^2

    It is 0 for such an annotator, 1 for a perfect one, and for two classes it equals
    (sensitivity + specificity - 1)^2. An annotator who reliably gives a wrong class also scores
    high: the score measures information, not accuracy.
    """
    matrix = np.asarray(confusion, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("confusion must be a square matrix with at least two classes")
    centred = matrix - matrix.mean(axis=0)
    return float((centred**2).sum() / (matrix.shape[0] - 1))


def wilson_interval(
    successes: float, n: float, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Wilson score interval for a proportion (95 % by default); (0, 1) when ``n`` is 0."""
    if n <= 0:
        return 0.0, 1.0
    share = successes / n
    centre = (share + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(share * (1 - share) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _soft_confusions(weights: np.ndarray, given: np.ndarray) -> np.ndarray:
    """Confusion matrices of one annotator from posterior rows and one-hot given labels.

    Both arguments have shape ``(batch, labels, classes)``: ``weights[b, o, k]`` is the posterior
    of class k for the item of label o. Row k of a matrix is the weighted distribution of the
    given labels, plus one pseudo-label spread like the annotator's overall label distribution,
    so that a row with little evidence stays close to "no information".
    """
    counts = np.einsum("bok,bol->bkl", weights, given)
    overall = given.mean(axis=1)[:, None, :]
    return (counts + overall) / (counts.sum(axis=2, keepdims=True) + 1.0)


def _spammer_scores(confusions: np.ndarray) -> np.ndarray:
    centred = confusions - confusions.mean(axis=1, keepdims=True)
    return (centred**2).sum(axis=(1, 2)) / (confusions.shape[1] - 1)


@dataclass(frozen=True, eq=False)
class AnnotatorReport:
    """What the labels and the aggregate say about one annotator.

    ``accuracy`` is the expected share of the annotator's labels that are right under the
    posterior, with a Wilson interval that treats the posterior as known. ``gold_accuracy`` is the
    share right on items with a gold label. ``bias`` is how much more often the annotator gives
    ``bias_class`` than the posterior expects on the same items. ``agreement`` is the share of
    labels equal to the aggregate label. ``flag`` is ``"random"``, ``"constant"`` or ``None``.
    """

    annotator: str
    n_labels: int
    confusion: np.ndarray
    accuracy: float
    accuracy_interval: tuple[float, float]
    n_gold: int
    gold_accuracy: float | None
    gold_interval: tuple[float, float] | None
    spammer_score: float
    spammer_interval: tuple[float, float]
    bias_class: str
    bias: float
    top_label: str
    top_label_share: float
    agreement: float
    flag: str | None


def audit_annotators(
    labels: LabelSet,
    result: AggregationResult,
    *,
    spam_threshold: float = 0.05,
    constant_share: float = 0.95,
    min_labels: int = 10,
    n_boot: int = 200,
    level: float = 0.95,
    seed: int = 0,
) -> list[AnnotatorReport]:
    """Audit every annotator against the posterior of ``result``.

    The spammer score is computed from a confusion matrix estimated against the posterior, with a
    percentile bootstrap interval from resampling the annotator's labels. An annotator with at
    least ``min_labels`` labels is flagged when the upper end of that interval is below
    ``spam_threshold``: ``"constant"`` when one class makes up at least ``constant_share`` of
    their labels, ``"random"`` otherwise.
    """
    if result.items != labels.items or result.classes != labels.classes:
        raise ValueError("the result was computed on another label set")
    n_classes = labels.n_classes
    if n_classes < 2:
        raise ValueError("the annotator audit needs at least two classes")
    rng = np.random.default_rng(seed)
    posterior = result.posterior
    gold = gold_indices(labels) if labels.gold is not None else None
    aggregate = np.array(
        [-1 if name is None else labels.classes.index(name) for name in result.labels]
    )
    order = np.argsort(labels.annotator_index, kind="stable")
    bounds = np.searchsorted(labels.annotator_index[order], np.arange(labels.n_annotators + 1))

    reports = []
    for j, name in enumerate(labels.annotators):
        rows = order[bounds[j] : bounds[j + 1]]
        items, given = labels.item_index[rows], labels.label_index[rows]
        n = int(rows.size)
        weights = posterior[items]
        one_hot = np.eye(n_classes)[given]
        confusion = _soft_confusions(weights[None], one_hot[None])[0]
        score = spammer_score(confusion)
        resampled = []
        block = max(1, min(n_boot, 2_000_000 // max(n * n_classes, 1)))
        for start in range(0, n_boot, block):
            draws = rng.integers(0, n, size=(min(block, n_boot - start), n))
            resampled.append(_spammer_scores(_soft_confusions(weights[draws], one_hot[draws])))
        low, high = np.quantile(np.concatenate(resampled), [(1 - level) / 2, (1 + level) / 2])

        right = float(weights[np.arange(n), given].sum())
        share = np.bincount(given, minlength=n_classes) / n
        excess = share - weights.mean(axis=0)
        top = int(share.argmax())
        decided = aggregate[items] >= 0
        agreement = (
            float((given[decided] == aggregate[items][decided]).mean()) if decided.any() else 0.0
        )
        n_gold, gold_accuracy, gold_interval = 0, None, None
        if gold is not None:
            known = gold[items] != -2
            n_gold = int(known.sum())
            if n_gold:
                hits = int((given[known] == gold[items][known]).sum())
                gold_accuracy, gold_interval = hits / n_gold, wilson_interval(hits, n_gold)
        flag = None
        if n >= min_labels and high < spam_threshold:
            flag = "constant" if share[top] >= constant_share else "random"
        reports.append(
            AnnotatorReport(
                annotator=name,
                n_labels=n,
                confusion=confusion,
                accuracy=right / n,
                accuracy_interval=wilson_interval(right, n),
                n_gold=n_gold,
                gold_accuracy=gold_accuracy,
                gold_interval=gold_interval,
                spammer_score=score,
                spammer_interval=(float(low), float(high)),
                bias_class=labels.classes[int(excess.argmax())],
                bias=float(excess.max()),
                top_label=labels.classes[top],
                top_label_share=float(share[top]),
                agreement=agreement,
                flag=flag,
            )
        )
    return reports


def posterior_entropy(posterior: np.ndarray) -> np.ndarray:
    """Entropy of each posterior row in bits."""
    posterior = np.asarray(posterior, dtype=float)
    safe = np.where(posterior > 0, posterior, 1.0)
    return -(posterior * np.log2(safe)).sum(axis=-1)


@dataclass(frozen=True)
class ItemReport:
    """One item: its label, how certain the aggregate is, and the votes behind it."""

    item: str
    label: str | None
    probability: float
    entropy: float
    margin: float
    n_labels: int
    votes: dict[str, int]
    uncertain: bool


def audit_items(
    labels: LabelSet, result: AggregationResult, *, certain_at: float = 0.9
) -> list[ItemReport]:
    """Posterior entropy (bits), top probability and margin of every item.

    An item is marked uncertain when its top probability is below ``certain_at``.
    """
    if result.items != labels.items:
        raise ValueError("the result was computed on another label set")
    ordered = np.sort(result.posterior, axis=1)
    margin = ordered[:, -1] - (ordered[:, -2] if ordered.shape[1] > 1 else 0.0)
    entropy = posterior_entropy(result.posterior)
    votes = labels.vote_counts()
    return [
        ItemReport(
            item=item,
            label=result.labels[i],
            probability=float(ordered[i, -1]),
            entropy=float(entropy[i]),
            margin=float(margin[i]),
            n_labels=int(votes[i].sum()),
            votes={
                name: int(count)
                for name, count in zip(labels.classes, votes[i], strict=True)
                if count
            },
            uncertain=bool(ordered[i, -1] < certain_at),
        )
        for i, item in enumerate(labels.items)
    ]


def average_confusion(labels: LabelSet, result: AggregationResult) -> np.ndarray:
    """Confusion matrix of the average annotator: all labels pooled against the posterior.

    Row k is the distribution of the labels given to items, weighted by each item's posterior of
    class k. A row without any weight is uniform.
    """
    n_classes = labels.n_classes
    weights = result.posterior[labels.item_index]
    counts = np.zeros((n_classes, n_classes))
    for label in range(n_classes):
        counts[:, label] = weights[labels.label_index == label].sum(axis=0)
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.full_like(counts, 1.0 / n_classes), where=totals > 0)


def expected_entropy_reduction(posterior: np.ndarray, confusion: np.ndarray) -> np.ndarray:
    """Expected drop in posterior entropy (bits) from one more label per item.

    The new label l comes from an annotator with the given confusion matrix A. It has probability
    P(l) = sum_k q[k] A[k, l] under the current posterior q, and seeing it turns the posterior
    into q[k] A[k, l] / P(l). The expected drop H(q) - sum_l P(l) H(q | l) is the mutual
    information between the true class and the new label, so it is never negative.
    """
    posterior = np.asarray(posterior, dtype=float)
    joint = posterior[:, :, None] * np.asarray(confusion, dtype=float)[None, :, :]
    predictive = joint.sum(axis=1)
    safe = np.where(predictive > 0, predictive, 1.0)
    updated = joint / safe[:, None, :]
    after = (predictive * posterior_entropy(np.moveaxis(updated, 1, 2))).sum(axis=1)
    return np.maximum(posterior_entropy(posterior) - after, 0.0)


@dataclass(frozen=True)
class PlanRow:
    """One item to send out again and what a further label is expected to buy."""

    item: str
    label: str | None
    probability: float
    entropy: float
    expected_reduction: float
    n_labels: int


def plan_relabelling(labels: LabelSet, result: AggregationResult, budget: int) -> list[PlanRow]:
    """Choose the ``budget`` items for which one more label is expected to help most.

    Items are ranked by :func:`expected_entropy_reduction` under the average annotator
    (:func:`average_confusion`); ties keep item order. Each item gets at most one new label, so
    at most ``n_items`` rows are returned.
    """
    if budget < 0:
        raise ValueError("budget must not be negative")
    if result.items != labels.items:
        raise ValueError("the result was computed on another label set")
    gain = expected_entropy_reduction(result.posterior, average_confusion(labels, result))
    entropy = posterior_entropy(result.posterior)
    per_item = labels.labels_per_item()
    chosen = np.argsort(-gain, kind="stable")[: min(budget, labels.n_items)]
    return [
        PlanRow(
            item=labels.items[i],
            label=result.labels[i],
            probability=float(result.posterior[i].max()),
            entropy=float(entropy[i]),
            expected_reduction=float(gain[i]),
            n_labels=int(per_item[i]),
        )
        for i in chosen
    ]
