"""Majority vote: the share of votes per class, with an explicit policy for ties."""

from __future__ import annotations

import numpy as np

from labelfuse.data import LabelSet
from labelfuse.result import AggregationResult, TiePolicy, resolve_labels


def vote_shares(counts: np.ndarray) -> np.ndarray:
    """Normalise vote counts to shares per item; an item without votes gets equal shares."""
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    uniform = np.full_like(counts, 1.0 / counts.shape[1])
    return np.divide(counts, totals, out=uniform, where=totals > 0)


def majority_vote(
    labels: LabelSet, *, tie: TiePolicy = "abstain", seed: int = 0
) -> AggregationResult:
    """Aggregate by majority vote.

    The posterior of an item is its vote share per class. That is a description of the votes, not
    a calibrated probability. The label is the class with the most votes; ``tie`` decides what
    happens when several classes share the most votes (see :func:`labelfuse.resolve_labels`).
    """
    posterior = vote_shares(labels.vote_counts())
    names, tied = resolve_labels(posterior, labels.classes, tie=tie, seed=seed)
    return AggregationResult(
        method="majority",
        items=labels.items,
        classes=labels.classes,
        posterior=posterior,
        labels=names,
        tied=tied,
    )
