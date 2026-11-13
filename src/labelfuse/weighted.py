"""Accuracy-weighted vote: each annotator's vote counts in proportion to an accuracy estimate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from labelfuse.data import LabelSet
from labelfuse.majority import vote_shares
from labelfuse.result import AggregationResult, TiePolicy, resolve_labels


@dataclass(frozen=True, eq=False)
class WeightedVoteResult(AggregationResult):
    """Weighted vote shares and the weight given to each annotator."""

    annotators: tuple[str, ...]
    weights: np.ndarray


def leave_one_out_agreement(labels: LabelSet, *, smoothing: float = 1.0) -> np.ndarray:
    """How often each annotator agrees with the majority of the other annotators.

    For every label, the other labels of the same item are tallied; the label scores 1 when it is
    their single most frequent class, 1/t when it is one of t tied classes and 0 otherwise. Labels
    on items nobody else labelled are skipped. The result is
    ``(score + smoothing) / (labels + n_classes * smoothing)``, which is ``1 / n_classes`` for an
    annotator without any comparable label.
    """
    if smoothing < 0:
        raise ValueError("smoothing must not be negative")
    n_classes = labels.n_classes
    rows = np.arange(labels.n_labels)
    others = labels.vote_counts()[labels.item_index].astype(float)
    others[rows, labels.label_index] -= 1
    comparable = others.sum(axis=1) > 0
    at_top = others == others.max(axis=1, keepdims=True)
    score = at_top[rows, labels.label_index] / at_top.sum(axis=1) * comparable
    agreed = np.bincount(labels.annotator_index, weights=score, minlength=labels.n_annotators)
    seen = np.bincount(labels.annotator_index, weights=comparable, minlength=labels.n_annotators)
    denominator = seen + n_classes * smoothing
    return np.divide(
        agreed + smoothing,
        denominator,
        out=np.full(labels.n_annotators, 1.0 / n_classes),
        where=denominator > 0,
    )


def weighted_vote(
    labels: LabelSet,
    *,
    weights: Mapping[str, float] | None = None,
    smoothing: float = 1.0,
    tie: TiePolicy = "abstain",
    seed: int = 0,
) -> WeightedVoteResult:
    """Aggregate by a vote in which annotator j's label counts with weight ``w[j]``.

    By default the weight is the annotator's leave-one-out agreement with the majority of the
    others (:func:`leave_one_out_agreement`). ``weights`` replaces it with known values, for
    example accuracies measured on gold items; annotators it does not name get weight 0. The
    posterior of an item is its share of the weighted votes, which is not a calibrated
    probability.
    """
    if weights is None:
        weight = leave_one_out_agreement(labels, smoothing=smoothing)
    else:
        weight = np.array([float(weights.get(name, 0.0)) for name in labels.annotators])
        if (weight < 0).any() or not np.isfinite(weight).all():
            raise ValueError("weights must be finite and not negative")
    cell = labels.item_index * labels.n_classes + labels.label_index
    scores = np.bincount(
        cell, weights=weight[labels.annotator_index], minlength=labels.n_items * labels.n_classes
    ).reshape(labels.n_items, labels.n_classes)
    posterior = vote_shares(scores)
    names, tied = resolve_labels(posterior, labels.classes, tie=tie, seed=seed)
    return WeightedVoteResult(
        method="weighted",
        items=labels.items,
        classes=labels.classes,
        posterior=posterior,
        labels=names,
        tied=tied,
        annotators=labels.annotators,
        weights=weight,
    )
