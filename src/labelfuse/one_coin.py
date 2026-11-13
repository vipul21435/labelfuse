"""The one-coin model: Dawid-Skene with a single accuracy per annotator.

Annotator j gives the true class with probability a[j], whatever that class is, and otherwise one
of the other K - 1 classes with equal probability. That is the Dawid-Skene model with every
confusion matrix constrained to a[j] on the diagonal and (1 - a[j]) / (K - 1) elsewhere, so it has
one parameter per annotator instead of K (K - 1). The E-step is the same; the M-step is

    a[j] = (expected number of correct labels of j + s) / (labels of j + K * s)

With s > 0 this is the posterior mode under a Beta(1 + s, 1 + (K - 1) s) prior on a[j], whose
mode is 1 / K.

The class prior is fixed to uniform by default. With symmetric errors a skewed prior can explain
the labels as well as the annotators do, and EM with a free prior can drift to "nearly every item
is of one class"; ``estimate_prior=True`` re-estimates it as Dawid-Skene does.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from labelfuse.data import LabelSet
from labelfuse.dawid_skene import (
    DEFAULT_MAX_ITER,
    DEFAULT_SMOOTHING,
    DEFAULT_TOL,
    DawidSkeneResult,
    _em_fields,
)
from labelfuse.result import TiePolicy


@dataclass(frozen=True, eq=False)
class OneCoinResult(DawidSkeneResult):
    """A Dawid-Skene result whose confusion matrices are symmetric, plus the accuracies."""

    accuracy: np.ndarray


def _accuracies(posterior: np.ndarray, labels: LabelSet, smoothing: float) -> np.ndarray:
    right = np.bincount(
        labels.annotator_index,
        weights=posterior[labels.item_index, labels.label_index],
        minlength=labels.n_annotators,
    )
    total = labels.labels_per_annotator() + labels.n_classes * smoothing
    return np.divide(
        right + smoothing,
        total,
        out=np.full(labels.n_annotators, 1.0 / labels.n_classes),
        where=total > 0,
    )


def _m_step(
    posterior: np.ndarray, labels: LabelSet, smoothing: float, *, estimate_prior: bool
) -> tuple[np.ndarray, np.ndarray]:
    n_classes = labels.n_classes
    accuracy = _accuracies(posterior, labels, smoothing)[:, None, None]
    diagonal = np.eye(n_classes)
    confusion = accuracy * diagonal + (1 - accuracy) / (n_classes - 1) * (1 - diagonal)
    if estimate_prior:
        counts = posterior.sum(axis=0) + smoothing
        class_prior = counts / (labels.n_items + n_classes * smoothing)
    else:
        class_prior = np.full(n_classes, 1.0 / n_classes)
    return confusion, class_prior


def _penalty(confusion: np.ndarray, class_prior: np.ndarray, smoothing: float) -> float:
    if smoothing == 0:
        return 0.0
    n_classes = confusion.shape[1]
    right = confusion[:, 0, 0]
    wrong = confusion[:, 0, 1] * (n_classes - 1)
    spread = np.log(right).sum() + (n_classes - 1) * np.log(wrong).sum()
    return float(smoothing * (spread + np.log(class_prior).sum()))


def one_coin(
    labels: LabelSet,
    *,
    smoothing: float = DEFAULT_SMOOTHING,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    n_restarts: int = 1,
    restart_noise: float = 0.5,
    seed: int = 0,
    tie: TiePolicy = "abstain",
    estimate_prior: bool = False,
) -> OneCoinResult:
    """Fit the one-coin model by EM.

    The arguments are those of :func:`labelfuse.dawid_skene`, plus ``estimate_prior``: the class
    prior is uniform unless it is true.
    """
    if labels.n_classes < 2:
        raise ValueError("the one-coin model needs at least two classes")
    fields = _em_fields(
        labels,
        method="one-coin",
        smoothing=smoothing,
        max_iter=max_iter,
        tol=tol,
        n_restarts=n_restarts,
        restart_noise=restart_noise,
        seed=seed,
        tie=tie,
        m_step=partial(_m_step, estimate_prior=estimate_prior),
        penalty=_penalty,
    )
    return OneCoinResult(**fields, accuracy=fields["confusion"][:, 0, 0].copy())
