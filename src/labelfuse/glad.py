"""GLAD for binary labels: annotator ability times item easiness.

Annotator j labels item i correctly with probability sigmoid(alpha[j] * beta[i]), with
beta[i] = exp(b[i]) > 0. A large alpha is a reliable annotator, alpha = 0 a random one and a
negative alpha one who tends to give the opposite label; a small beta is a hard item, on which
every annotator is close to a coin flip. The model is from Whitehill, Wu, Bergsma, Movellan and
Ruvolo, "Whose Vote Should Count More: Optimal Integration of Labels from Labelers of Unknown
Expertise", NIPS 2009.

It is fitted by EM. The E-step computes the posterior of each item's class from the current
alpha, beta and class prior. The M-step has no closed form: it maximises

    sum over labels of  c * log sigmoid(x) + (1 - c) * log sigmoid(-x)
        - sum_j (alpha[j] - 1)^2 / 2  -  sum_i b[i]^2 / 2

where x = alpha[j] * exp(b[i]) and c is the posterior probability that the label is correct, with
L-BFGS-B from scipy. The two quadratic terms are Gaussian priors, alpha ~ N(1, 1) and
b ~ N(0, 1); they are choices of this implementation that keep the scale identified (alpha * beta
is unchanged by multiplying one and dividing the other) and favour annotators who are better than
chance. The class prior is uniform by default; ``estimate_prior=True`` re-estimates it in each
M-step with one pseudo-count per class, which can drift to "nearly every item is of one class".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from labelfuse.data import LabelSet
from labelfuse.majority import vote_shares
from labelfuse.result import AggregationResult, TiePolicy, resolve_labels

_BOUND = 8.0


@dataclass(frozen=True, eq=False)
class GladResult(AggregationResult):
    """Posterior and fitted parameters of GLAD.

    ``ability[j]`` is alpha of ``annotators[j]`` and ``difficulty[i]`` is 1 / beta of
    ``items[i]``. ``objective_trace`` holds the log-likelihood plus the log prior after each
    iteration.
    """

    annotators: tuple[str, ...]
    ability: np.ndarray
    difficulty: np.ndarray
    class_prior: np.ndarray
    log_likelihood: float
    objective_trace: tuple[float, ...]
    n_iter: int
    converged: bool


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)


def _negative_q(
    theta: np.ndarray, labels: LabelSet, correct: np.ndarray
) -> tuple[float, np.ndarray]:
    """Minus the M-step objective and its gradient with respect to (alpha, b)."""
    n_annotators = labels.n_annotators
    alpha, b = theta[:n_annotators], theta[n_annotators:]
    beta = np.exp(b)
    a, e = alpha[labels.annotator_index], beta[labels.item_index]
    x = a * e
    value = correct @ _log_sigmoid(x) + (1 - correct) @ _log_sigmoid(-x)
    value -= 0.5 * ((alpha - 1) ** 2).sum() + 0.5 * (b**2).sum()
    slope = correct - 1 / (1 + np.exp(-x))
    grad_alpha = np.bincount(labels.annotator_index, weights=slope * e, minlength=n_annotators)
    grad_b = np.bincount(labels.item_index, weights=slope * x, minlength=labels.n_items)
    gradient = np.concatenate([grad_alpha - (alpha - 1), grad_b - b])
    return -float(value), -gradient


def _e_step(
    alpha: np.ndarray, b: np.ndarray, class_prior: np.ndarray, labels: LabelSet
) -> tuple[np.ndarray, float]:
    x = alpha[labels.annotator_index] * np.exp(b[labels.item_index])
    right, wrong = _log_sigmoid(x), _log_sigmoid(-x)
    log_joint = np.empty((labels.n_items, 2))
    for k in (0, 1):
        terms = np.where(labels.label_index == k, right, wrong)
        log_joint[:, k] = np.bincount(labels.item_index, weights=terms, minlength=labels.n_items)
    log_joint += np.log(class_prior)
    top = log_joint.max(axis=1, keepdims=True)
    log_evidence = top[:, 0] + np.log(np.exp(log_joint - top).sum(axis=1))
    posterior = np.exp(log_joint - log_evidence[:, None])
    posterior /= posterior.sum(axis=1, keepdims=True)
    return posterior, float(log_evidence.sum())


def glad(
    labels: LabelSet,
    *,
    max_iter: int = 200,
    tol: float = 1e-5,
    tie: TiePolicy = "abstain",
    seed: int = 0,
    estimate_prior: bool = False,
) -> GladResult:
    """Fit GLAD by EM on a label set with exactly two classes.

    Starts from the vote shares, alpha = 1 and beta = 1. One iteration is an M-step followed by an
    E-step; iteration stops once every posterior entry moves by less than ``tol`` or after
    ``max_iter`` iterations. ``seed`` is only used for ``tie="random"``.
    """
    if labels.n_classes != 2:
        raise ValueError(f"GLAD needs exactly two classes, got {labels.n_classes}")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if tol < 0:
        raise ValueError("tol must not be negative")

    n_annotators, n_items = labels.n_annotators, labels.n_items
    posterior = vote_shares(labels.vote_counts())
    theta = np.concatenate([np.ones(n_annotators), np.zeros(n_items)])
    bounds = [(-_BOUND, _BOUND)] * n_annotators + [(-_BOUND / 2, _BOUND / 2)] * n_items
    objectives: list[float] = []
    converged = False
    for _ in range(max_iter):
        correct = posterior[labels.item_index, labels.label_index]
        fit = minimize(
            _negative_q, theta, args=(labels, correct), jac=True, method="L-BFGS-B", bounds=bounds
        )
        theta = fit.x
        alpha, b = theta[:n_annotators], theta[n_annotators:]
        if estimate_prior:
            class_prior = (posterior.sum(axis=0) + 1.0) / (n_items + 2.0)
        else:
            class_prior = np.array([0.5, 0.5])
        updated, log_likelihood = _e_step(alpha, b, class_prior, labels)
        log_prior = -0.5 * ((alpha - 1) ** 2).sum() - 0.5 * (b**2).sum()
        objectives.append(log_likelihood + float(log_prior + np.log(class_prior).sum()))
        change = float(np.abs(updated - posterior).max())
        posterior = updated
        if change < tol:
            converged = True
            break

    names, tied = resolve_labels(posterior, labels.classes, tie=tie, seed=seed)
    return GladResult(
        method="glad",
        items=labels.items,
        classes=labels.classes,
        posterior=posterior,
        labels=names,
        tied=tied,
        annotators=labels.annotators,
        ability=alpha,
        difficulty=np.exp(-b),
        class_prior=class_prior,
        log_likelihood=log_likelihood,
        objective_trace=tuple(objectives),
        n_iter=len(objectives),
        converged=converged,
    )
