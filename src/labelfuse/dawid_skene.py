"""The Dawid-Skene model, fitted by expectation-maximisation.

Each item i has a hidden true class z_i drawn from the class prior p. Annotator j, shown an item
of class k, answers l with probability pi[j, k, l]: one confusion matrix per annotator. Given the
true class, labels are independent. The likelihood of the observed labels is

    L = prod_i  sum_k  p[k] * prod_{(j, l) in labels of i}  pi[j, k, l]

EM alternates two steps, starting from the vote shares of each item as the first posterior:

    M-step   pi[j, k, l] = (sum_i T[i, k] * n[i, j, l] + s) / (sum_i T[i, k] * n[i, j, .] + K * s)
             p[k]        = (sum_i T[i, k] + s) / (I + K * s)
    E-step   T[i, k]  proportional to  p[k] * prod_{(j, l) in labels of i}  pi[j, k, l]

where n[i, j, l] counts how often annotator j gave label l to item i, K is the number of classes,
I the number of items and s the smoothing pseudo-count. With s = 0 this is maximum likelihood;
with s > 0 it is the posterior mode under a Dirichlet(1 + s) prior on every confusion row and on
the class prior, and the quantity that EM increases is the log-likelihood plus
s * (sum of log pi + sum of log p).

Reference: A. P. Dawid and A. M. Skene (1979), "Maximum Likelihood Estimation of Observer
Error-Rates Using the EM Algorithm", Applied Statistics 28(1), doi:10.2307/2346806.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from labelfuse.data import LabelSet
from labelfuse.majority import vote_shares
from labelfuse.result import AggregationResult, TiePolicy, resolve_labels

DEFAULT_SMOOTHING = 0.01
DEFAULT_MAX_ITER = 1000
DEFAULT_TOL = 1e-8


@dataclass(frozen=True, eq=False)
class DawidSkeneResult(AggregationResult):
    """Posterior and fitted parameters of the Dawid-Skene model.

    ``confusion[j, k, l]`` is the estimated probability that annotator ``annotators[j]`` answers
    ``classes[l]`` when the true class is ``classes[k]``. ``log_likelihood_trace`` and
    ``objective_trace`` have one entry per EM iteration of the selected start; the objective is
    the log-likelihood plus the smoothing term and never decreases. ``restart_objectives`` holds
    the final objective of every start and ``best_restart`` the index of the one returned.
    """

    annotators: tuple[str, ...]
    confusion: np.ndarray
    class_prior: np.ndarray
    log_likelihood: float
    log_likelihood_trace: tuple[float, ...]
    objective_trace: tuple[float, ...]
    n_iter: int
    converged: bool
    smoothing: float
    restart_objectives: tuple[float, ...]
    best_restart: int


@dataclass
class _Fit:
    posterior: np.ndarray
    confusion: np.ndarray
    class_prior: np.ndarray
    log_likelihoods: list[float]
    objectives: list[float]
    converged: bool


def _m_step(
    posterior: np.ndarray, labels: LabelSet, smoothing: float
) -> tuple[np.ndarray, np.ndarray]:
    """Confusion matrices and class prior from expected counts."""
    n_annotators, n_classes = labels.n_annotators, labels.n_classes
    cell = labels.annotator_index * n_classes + labels.label_index
    weights = posterior[labels.item_index]
    counts = np.empty((n_annotators, n_classes, n_classes))
    for k in range(n_classes):
        column = np.bincount(cell, weights=weights[:, k], minlength=n_annotators * n_classes)
        counts[:, k, :] = column.reshape(n_annotators, n_classes)
    counts += smoothing
    totals = counts.sum(axis=2, keepdims=True)
    # A row with no expected count (possible only when smoothing is 0) does not enter the
    # likelihood; it is set to uniform so that it stays a probability vector.
    confusion = np.divide(
        counts, totals, out=np.full_like(counts, 1.0 / n_classes), where=totals > 0
    )
    class_prior = (posterior.sum(axis=0) + smoothing) / (labels.n_items + n_classes * smoothing)
    return confusion, class_prior


def _e_step(
    confusion: np.ndarray, class_prior: np.ndarray, labels: LabelSet
) -> tuple[np.ndarray, float]:
    """Posterior over the true class of every item, and the log-likelihood of the labels."""
    n_items, n_classes = labels.n_items, labels.n_classes
    with np.errstate(divide="ignore"):
        log_confusion = np.log(confusion)
        log_prior = np.log(class_prior)
    terms = log_confusion[labels.annotator_index, :, labels.label_index]
    log_joint = np.empty((n_items, n_classes))
    for k in range(n_classes):
        log_joint[:, k] = np.bincount(labels.item_index, weights=terms[:, k], minlength=n_items)
    log_joint += log_prior
    top = log_joint.max(axis=1, keepdims=True)
    if not np.isfinite(top).all():
        raise FloatingPointError(
            "an item has probability zero under every class: zeros in the confusion matrices "
            "or the class prior rule out its labels"
        )
    log_evidence = top[:, 0] + np.log(np.exp(log_joint - top).sum(axis=1))
    posterior = np.exp(log_joint - log_evidence[:, None])
    posterior /= posterior.sum(axis=1, keepdims=True)
    return posterior, float(log_evidence.sum())


def _penalty(confusion: np.ndarray, class_prior: np.ndarray, smoothing: float) -> float:
    """Log of the Dirichlet(1 + smoothing) prior density, up to a constant."""
    if smoothing == 0:
        return 0.0
    return float(smoothing * (np.log(confusion).sum() + np.log(class_prior).sum()))


def _fit(start: np.ndarray, labels: LabelSet, smoothing: float, max_iter: int, tol: float) -> _Fit:
    posterior = start
    log_likelihoods: list[float] = []
    objectives: list[float] = []
    converged = False
    for _ in range(max_iter):
        confusion, class_prior = _m_step(posterior, labels, smoothing)
        updated, log_likelihood = _e_step(confusion, class_prior, labels)
        log_likelihoods.append(log_likelihood)
        objectives.append(log_likelihood + _penalty(confusion, class_prior, smoothing))
        change = float(np.abs(updated - posterior).max())
        posterior = updated
        if change < tol:
            converged = True
            break
    return _Fit(posterior, confusion, class_prior, log_likelihoods, objectives, converged)


def _check_parameters(
    labels: LabelSet, confusion: np.ndarray, class_prior: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    confusion = np.asarray(confusion, dtype=float)
    class_prior = np.asarray(class_prior, dtype=float)
    shape = (labels.n_annotators, labels.n_classes, labels.n_classes)
    if confusion.shape != shape:
        raise ValueError(f"confusion must have shape {shape}, got {confusion.shape}")
    if class_prior.shape != (labels.n_classes,):
        raise ValueError(f"class_prior must have length {labels.n_classes}")
    if (confusion < 0).any() or not np.allclose(confusion.sum(axis=2), 1.0):
        raise ValueError("every row of every confusion matrix must be a probability vector")
    if (class_prior < 0).any() or not np.isclose(class_prior.sum(), 1.0):
        raise ValueError("class_prior must be a probability vector")
    return confusion, class_prior


def dawid_skene_posterior(
    labels: LabelSet, confusion: np.ndarray, class_prior: np.ndarray
) -> np.ndarray:
    """Posterior over the true class of every item for given, fixed parameters.

    ``confusion`` has shape ``(n_annotators, n_classes, n_classes)`` with rows indexed by the true
    class; ``class_prior`` has length ``n_classes``. This is the E-step on its own.
    """
    confusion, class_prior = _check_parameters(labels, confusion, class_prior)
    posterior, _ = _e_step(confusion, class_prior, labels)
    return posterior


def dawid_skene(
    labels: LabelSet,
    *,
    smoothing: float = DEFAULT_SMOOTHING,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    n_restarts: int = 1,
    restart_noise: float = 0.5,
    seed: int = 0,
    tie: TiePolicy = "abstain",
) -> DawidSkeneResult:
    """Fit the Dawid-Skene model by EM and return the posterior of every item.

    ``smoothing`` is a pseudo-count added to every cell of every confusion matrix and to every
    class of the prior (0 gives plain maximum likelihood). One iteration is an M-step followed by
    an E-step; iteration stops when no posterior entry moved by more than ``tol`` or after
    ``max_iter`` iterations.

    The first start uses the vote shares as the initial posterior and is deterministic. With
    ``n_restarts > 1`` the further starts mix the vote shares with Dirichlet(1, ..., 1) noise of
    weight ``restart_noise`` (drawn with ``seed``); the start with the highest objective is
    returned, the first one on ties.
    """
    if smoothing < 0:
        raise ValueError("smoothing must not be negative")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if tol < 0:
        raise ValueError("tol must not be negative")
    if n_restarts < 1:
        raise ValueError("n_restarts must be at least 1")
    if not 0 < restart_noise <= 1:
        raise ValueError("restart_noise must be in (0, 1]")

    votes = vote_shares(labels.vote_counts())
    rng = np.random.default_rng(seed)
    best: _Fit | None = None
    best_index = 0
    finals: list[float] = []
    for index in range(n_restarts):
        if index == 0:
            start = votes
        else:
            noise = rng.dirichlet(np.ones(labels.n_classes), size=labels.n_items)
            start = (1 - restart_noise) * votes + restart_noise * noise
        fit = _fit(start, labels, smoothing, max_iter, tol)
        final = fit.objectives[-1]
        finals.append(final)
        if best is None or final > best.objectives[-1] + 1e-9 * max(1.0, abs(final)):
            best, best_index = fit, index
    assert best is not None

    names, tied = resolve_labels(best.posterior, labels.classes, tie=tie, seed=seed)
    return DawidSkeneResult(
        method="dawid-skene",
        items=labels.items,
        classes=labels.classes,
        posterior=best.posterior,
        labels=names,
        tied=tied,
        annotators=labels.annotators,
        confusion=best.confusion,
        class_prior=best.class_prior,
        log_likelihood=best.log_likelihoods[-1],
        log_likelihood_trace=tuple(best.log_likelihoods),
        objective_trace=tuple(best.objectives),
        n_iter=len(best.objectives),
        converged=best.converged,
        smoothing=float(smoothing),
        restart_objectives=tuple(finals),
        best_restart=best_index,
    )
