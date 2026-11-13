"""One entry point for every aggregator, by name."""

from __future__ import annotations

from labelfuse.data import LabelSet
from labelfuse.dawid_skene import DEFAULT_MAX_ITER, DEFAULT_SMOOTHING, DEFAULT_TOL, dawid_skene
from labelfuse.glad import glad
from labelfuse.majority import majority_vote
from labelfuse.one_coin import one_coin
from labelfuse.result import AggregationResult, TiePolicy
from labelfuse.weighted import weighted_vote

METHODS = ("majority", "weighted", "one-coin", "dawid-skene", "glad")


def applicable_methods(labels: LabelSet) -> tuple[str, ...]:
    """The methods that can run on this label set: GLAD needs exactly two classes."""
    if labels.n_classes < 2:
        return ("majority", "weighted", "dawid-skene")
    return tuple(name for name in METHODS if name != "glad" or labels.n_classes == 2)


def aggregate(
    labels: LabelSet,
    method: str = "dawid-skene",
    *,
    tie: TiePolicy = "abstain",
    seed: int = 0,
    smoothing: float = DEFAULT_SMOOTHING,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    n_restarts: int = 1,
) -> AggregationResult:
    """Run the named aggregator.

    ``smoothing``, ``max_iter``, ``tol`` and ``n_restarts`` go to the EM of ``"dawid-skene"`` and
    ``"one-coin"``; the other methods use their own defaults.
    """
    if method == "majority":
        return majority_vote(labels, tie=tie, seed=seed)
    if method == "weighted":
        return weighted_vote(labels, tie=tie, seed=seed)
    if method == "glad":
        return glad(labels, tie=tie, seed=seed)
    if method in ("dawid-skene", "one-coin"):
        fit = dawid_skene if method == "dawid-skene" else one_coin
        return fit(
            labels,
            smoothing=smoothing,
            max_iter=max_iter,
            tol=tol,
            n_restarts=n_restarts,
            seed=seed,
            tie=tie,
        )
    raise ValueError(f"unknown method {method!r}; choose one of {', '.join(METHODS)}")
