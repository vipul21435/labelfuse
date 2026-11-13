"""Accuracy-weighted vote, the one-coin model, GLAD and the method registry."""

import numpy as np
import pytest

from labelfuse import (
    METHODS,
    LabelSet,
    aggregate,
    applicable_methods,
    dawid_skene_posterior,
    glad,
    leave_one_out_agreement,
    majority_vote,
    one_coin,
    simulate,
    symmetric_confusion,
    weighted_vote,
)
from labelfuse.glad import ALPHA_BOUND, LOG_BETA_BOUND, _negative_q

VOTES = [
    ("i1", "a", "x"),
    ("i1", "b", "x"),
    ("i1", "c", "y"),
    ("i2", "a", "y"),
    ("i2", "b", "y"),
    ("i2", "c", "x"),
    ("i3", "a", "x"),
    ("i3", "b", "y"),
    ("i4", "c", "y"),
]


def test_leave_one_out_agreement_by_hand():
    """a agrees with the others on i1 (b x, c y: tie, 1/2), i2 (tie, 1/2), i3 (b y: 0).

    b: i1 (a x, c y: 1/2), i2 (a y, c x: 1/2), i3 (a x: 0). c: i1 (a x, b x: 0), i2 (a y, b y: 0);
    i4 has no other label and is skipped. With one pseudo-count per class the agreements are
    (1 + 1) / (3 + 2), (1 + 1) / (3 + 2) and (0 + 1) / (2 + 2).
    """
    labels = LabelSet.from_records(VOTES)
    np.testing.assert_allclose(leave_one_out_agreement(labels), [0.4, 0.4, 0.25])
    np.testing.assert_allclose(leave_one_out_agreement(labels, smoothing=0), [1 / 3, 1 / 3, 0])
    with pytest.raises(ValueError, match="smoothing"):
        leave_one_out_agreement(labels, smoothing=-1)


def test_weighted_vote_by_hand():
    labels = LabelSet.from_records(VOTES)
    result = weighted_vote(labels)
    np.testing.assert_allclose(result.weights, [0.4, 0.4, 0.25])
    expected = [[0.8 / 1.05, 0.25 / 1.05], [0.25 / 1.05, 0.8 / 1.05], [0.5, 0.5], [0, 1]]
    np.testing.assert_allclose(result.posterior, expected)
    assert result.labels == ("x", "y", None, "y")
    assert result.method == "weighted"


def test_weighted_vote_with_given_weights():
    labels = LabelSet.from_records(VOTES)
    result = weighted_vote(labels, weights={"a": 0.2, "b": 0.9})
    np.testing.assert_allclose(result.weights, [0.2, 0.9, 0.0])
    assert result.labels == ("x", "y", "y", None)
    equal = weighted_vote(labels, weights=dict.fromkeys("abc", 1.0))
    np.testing.assert_allclose(equal.posterior, majority_vote(labels).posterior)
    with pytest.raises(ValueError, match="weights"):
        weighted_vote(labels, weights={"a": -1.0})


def test_weighted_vote_beats_majority_when_annotators_differ():
    confusion = np.stack([symmetric_confusion(a, 3) for a in (0.95, 0.9, 0.5, 0.4, 0.35)])
    labels = simulate(2000, confusion, seed=6)
    gold = np.array(labels.gold)
    weighted = np.mean(np.array(weighted_vote(labels, tie="first").labels) == gold)
    majority = np.mean(np.array(majority_vote(labels, tie="first").labels) == gold)
    assert weighted > majority + 0.02
    assert np.all(np.diff(weighted_vote(labels).weights) < 0)


def test_one_coin_recovers_planted_accuracies():
    planted = np.array([0.9, 0.8, 0.7, 0.6, 0.34])
    labels = simulate(4000, np.stack([symmetric_confusion(a, 3) for a in planted]), seed=2)
    result = one_coin(labels)
    assert result.converged
    assert result.method == "one-coin"
    # 4000 labels per annotator: a standard error of at most sqrt(0.25 / 4000) = 0.008
    np.testing.assert_allclose(result.accuracy, planted, atol=0.03)
    np.testing.assert_allclose(result.class_prior, 1 / 3)
    for accuracy, matrix in zip(result.accuracy, result.confusion, strict=True):
        np.testing.assert_allclose(matrix, symmetric_confusion(accuracy, 3))
    np.testing.assert_allclose(
        result.posterior, dawid_skene_posterior(labels, result.confusion, result.class_prior)
    )
    assert (np.diff(result.objective_trace) >= -1e-9).all()


def test_one_coin_first_step_by_hand():
    """Vote shares of VOTES: i1 (2/3, 1/3), i2 (1/3, 2/3), i3 (1/2, 1/2), i4 (0, 1).

    Expected correct labels: a 2/3 + 2/3 + 1/2, b 2/3 + 2/3 + 1/2, c 1/3 + 1/3 + 1.
    """
    result = one_coin(LabelSet.from_records(VOTES), smoothing=0, max_iter=1)
    np.testing.assert_allclose(result.accuracy, [11 / 18, 11 / 18, 5 / 9])
    a, c = 11 / 18, 5 / 9
    first = a * a * (1 - c) / (a * a * (1 - c) + (1 - a) * (1 - a) * c)
    np.testing.assert_allclose(result.posterior[0], [first, 1 - first])
    np.testing.assert_allclose(result.posterior[3], [1 - c, c])


def test_one_coin_can_estimate_the_prior_and_checks_arguments():
    confusion = np.stack([symmetric_confusion(a, 2) for a in (0.9, 0.85, 0.8)])
    labels = simulate(3000, confusion, class_prior=[0.8, 0.2], seed=4)
    result = one_coin(labels, estimate_prior=True, smoothing=0.5)
    np.testing.assert_allclose(result.class_prior, [0.8, 0.2], atol=0.03)
    assert (np.diff(result.objective_trace) >= -1e-9).all()
    single = LabelSet.from_records([("i", "a", "x")])
    with pytest.raises(ValueError, match="two classes"):
        one_coin(single)


def glad_data(seed: int = 0, n_items: int = 600) -> tuple[LabelSet, np.ndarray, np.ndarray]:
    """Labels drawn from the GLAD model itself."""
    rng = np.random.default_rng(seed)
    ability = np.array([3.0, 2.0, 1.5, 1.0, 0.5, 0.0, -1.5])
    easiness = np.exp(rng.normal(0, 0.7, size=n_items))
    truth = rng.integers(2, size=n_items)
    right = rng.random((n_items, 7)) < 1 / (1 + np.exp(-np.outer(easiness, ability)))
    answers = np.where(right, truth[:, None], 1 - truth[:, None])
    records = [(i, j, int(answers[i, j])) for i in range(n_items) for j in range(7)]
    gold = {i: int(truth[i]) for i in range(n_items)}
    return LabelSet.from_records(records, gold=gold), ability, easiness


def test_glad_recovers_ability_order_and_beats_majority():
    labels, ability, easiness = glad_data()
    result = glad(labels)
    assert result.converged
    assert result.method == "glad"
    np.testing.assert_allclose(result.posterior.sum(axis=1), 1.0)
    assert np.corrcoef(result.ability, ability)[0, 1] > 0.95
    assert result.ability[-1] < 0 < result.ability[0]
    assert np.corrcoef(np.log(result.difficulty), -np.log(easiness))[0, 1] > 0.3
    gold = np.array(labels.gold)
    model = np.mean(np.array(result.labels) == gold)
    votes = np.mean(np.array(majority_vote(labels, tie="first").labels) == gold)
    assert model > votes + 0.02
    assert (np.diff(result.objective_trace) >= -1e-5).all()


def test_glad_parameters_stay_within_the_documented_bounds():
    assert (ALPHA_BOUND, LOG_BETA_BOUND) == (8.0, 4.0)
    # two annotators who always agree on 400 items pull each other's ability upward
    records = [(i, a, i % 2) for i in range(400) for a in "ab"]
    result = glad(LabelSet.from_records(records), max_iter=20)
    assert (result.ability > 2).all()
    assert np.abs(result.ability).max() <= ALPHA_BOUND
    assert np.abs(np.log(result.difficulty)).max() <= LOG_BETA_BOUND


def test_glad_gradient_matches_finite_differences():
    labels, _, _ = glad_data(seed=1, n_items=60)
    rng = np.random.default_rng(0)
    theta = rng.normal(0, 0.5, size=labels.n_annotators + labels.n_items)
    correct = rng.random(labels.n_labels)
    _, gradient = _negative_q(theta, labels, correct)
    for index in (0, 3, 6, 7, 30, 66):
        step = np.zeros_like(theta)
        step[index] = 1e-6
        up, _ = _negative_q(theta + step, labels, correct)
        down, _ = _negative_q(theta - step, labels, correct)
        assert gradient[index] == pytest.approx((up - down) / 2e-6, rel=1e-5, abs=1e-6)


def test_glad_does_not_depend_on_class_names():
    labels, _, _ = glad_data(seed=2, n_items=80)
    rename = {"0": "yes", "1": "no"}
    renamed = LabelSet.from_records([(i, a, rename[label]) for i, a, label in labels.records()])
    assert renamed.classes == ("no", "yes")
    base, other = glad(labels, tol=1e-7), glad(renamed, tol=1e-7)
    assert base.converged
    assert other.converged
    np.testing.assert_allclose(other.posterior[:, ::-1], base.posterior, atol=1e-4)
    np.testing.assert_allclose(other.ability, base.ability, atol=1e-3)


def test_glad_checks_arguments_and_can_estimate_the_prior():
    labels, _, _ = glad_data(seed=3, n_items=60)
    with pytest.raises(ValueError, match="max_iter"):
        glad(labels, max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        glad(labels, tol=-1)
    three = LabelSet.from_records([("i", "a", "x"), ("i", "b", "y"), ("j", "a", "z")])
    with pytest.raises(ValueError, match="exactly two classes"):
        glad(three)
    short = glad(labels, max_iter=2, estimate_prior=True)
    assert short.n_iter == 2
    assert not short.converged
    assert short.class_prior.sum() == pytest.approx(1.0)
    assert not np.allclose(short.class_prior, 0.5)


def test_registry_runs_every_method_by_name():
    binary = simulate(60, np.stack([symmetric_confusion(a, 2) for a in (0.9, 0.8, 0.7)]), seed=1)
    assert applicable_methods(binary) == METHODS
    for name in METHODS:
        result = aggregate(binary, name)
        assert result.method == name
        assert result.posterior.shape == (60, 2)
    three = simulate(60, np.stack([symmetric_confusion(a, 3) for a in (0.9, 0.8, 0.7)]), seed=1)
    assert applicable_methods(three) == ("majority", "weighted", "one-coin", "dawid-skene")
    single = LabelSet.from_records([("i", "a", "x"), ("i", "b", "x")])
    assert applicable_methods(single) == ("majority", "weighted", "dawid-skene")
    with pytest.raises(ValueError, match="unknown method"):
        aggregate(binary, "median")
