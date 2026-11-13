import math
from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest

from labelfuse import (
    LabelSet,
    dawid_skene,
    dawid_skene_posterior,
    majority_vote,
    resolve_labels,
    simulate,
    symmetric_confusion,
)

TIGHT = {"tol": 1e-12, "max_iter": 5000}

HAND_WORKED = [
    ("i1", "A", "1"),
    ("i1", "B", "1"),
    ("i2", "A", "1"),
    ("i2", "B", "0"),
    ("i3", "A", "0"),
    ("i3", "B", "0"),
    ("i4", "A", "1"),
    ("i4", "B", "1"),
    ("i5", "A", "0"),
    ("i5", "B", "1"),
    ("i6", "A", "1"),
    ("i6", "B", "0"),
]


def test_one_em_step_worked_by_hand_for_two_annotators():
    """One M-step and one E-step from the vote shares, without smoothing, in exact fractions.

    Vote shares (class 0, class 1):
        i1 (0, 1)   i2 (1/2, 1/2)   i3 (1, 0)   i4 (0, 1)   i5 (1/2, 1/2)   i6 (1/2, 1/2)

    M-step. The shares sum to 5/2 and 7/2 over six items: prior (5/12, 7/12).
    A said 0 on i3, i5 and 1 on i1, i2, i4, i6:
        true 0: weight 1 + 1/2 = 3/2 on "0", 0 + 1/2 + 0 + 1/2 = 1 on "1"    -> (3/5, 2/5)
        true 1: weight 0 + 1/2 = 1/2 on "0", 1 + 1/2 + 1 + 1/2 = 3 on "1"    -> (1/7, 6/7)
    B said 0 on i2, i3, i6 and 1 on i1, i4, i5:
        true 0: weight 1/2 + 1 + 1/2 = 2 on "0", 0 + 0 + 1/2 = 1/2 on "1"    -> (4/5, 1/5)
        true 1: weight 1/2 + 0 + 1/2 = 1 on "0", 1 + 1 + 1/2 = 5/2 on "1"    -> (2/7, 5/7)

    E-step: prior times the two confusion entries, for class 0 and class 1.
        i1, i4 (A 1, B 1): 5/12 * 2/5 * 1/5 = 1/30,  7/12 * 6/7 * 5/7 = 5/14   sum 41/105
        i2, i6 (A 1, B 0): 5/12 * 2/5 * 4/5 = 2/15,  7/12 * 6/7 * 2/7 = 1/7    sum 29/105
        i3     (A 0, B 0): 5/12 * 3/5 * 4/5 = 1/5,   7/12 * 1/7 * 2/7 = 1/42   sum 47/210
        i5     (A 0, B 1): 5/12 * 3/5 * 1/5 = 1/20,  7/12 * 1/7 * 5/7 = 5/84   sum 23/210
    The four sums are the model's probabilities of the four answer patterns and add up to 1.
    """
    prior = [F(5, 12), F(7, 12)]
    confusion = [
        [[F(3, 5), F(2, 5)], [F(1, 7), F(6, 7)]],
        [[F(4, 5), F(1, 5)], [F(2, 7), F(5, 7)]],
    ]
    joint = {
        "i1": (F(1, 30), F(5, 14)),
        "i2": (F(2, 15), F(1, 7)),
        "i3": (F(1, 5), F(1, 42)),
        "i4": (F(1, 30), F(5, 14)),
        "i5": (F(1, 20), F(5, 84)),
        "i6": (F(2, 15), F(1, 7)),
    }
    evidence = {item: sum(pair) for item, pair in joint.items()}
    assert evidence["i1"] == F(41, 105)
    assert evidence["i2"] == F(29, 105)
    assert evidence["i3"] == F(47, 210)
    assert evidence["i5"] == F(23, 210)
    assert evidence["i1"] + evidence["i2"] + evidence["i3"] + evidence["i5"] == 1
    posterior = [[value / evidence[item] for value in joint[item]] for item in sorted(joint)]
    assert posterior[0] == [F(7, 82), F(75, 82)]
    assert posterior[1] == [F(14, 29), F(15, 29)]
    assert posterior[2] == [F(42, 47), F(5, 47)]
    assert posterior[4] == [F(21, 46), F(25, 46)]

    result = dawid_skene(LabelSet.from_records(HAND_WORKED), smoothing=0, max_iter=1)

    assert result.annotators == ("A", "B")
    assert result.classes == ("0", "1")
    np.testing.assert_allclose(result.class_prior, np.array(prior, dtype=float), rtol=1e-12)
    np.testing.assert_allclose(result.confusion, np.array(confusion, dtype=float), rtol=1e-12)
    np.testing.assert_allclose(result.posterior, np.array(posterior, dtype=float), rtol=1e-12)
    expected_log_likelihood = sum(math.log(value) for value in evidence.values())
    assert result.log_likelihood == pytest.approx(expected_log_likelihood, rel=1e-12)
    assert result.log_likelihood_trace == (result.log_likelihood,)
    assert result.labels == ("1", "1", "0", "1", "1", "1")
    assert result.n_iter == 1
    assert not result.converged


def test_one_smoothed_em_step_worked_by_hand():
    """The same step with a pseudo-count of 1 in every cell.

    Prior: (5/2 + 1, 7/2 + 1) / (6 + 2) = (7/16, 9/16).
    A, true 0: (3/2 + 1, 1 + 1) / (5/2 + 2) = (5/9, 4/9); true 1: (1/2 + 1, 3 + 1) / (7/2 + 2)
    = (3/11, 8/11). B, true 0: (2 + 1, 1/2 + 1) / (9/2) = (2/3, 1/3); true 1: (1 + 1, 5/2 + 1)
    / (11/2) = (4/11, 7/11).

    E-step, for class 0 and class 1:
        i1, i4 (A 1, B 1): 7/16 * 4/9 * 1/3 = 7/108,   9/16 * 8/11 * 7/11 = 63/242
        i2, i6 (A 1, B 0): 7/16 * 4/9 * 2/3 = 7/54,    9/16 * 8/11 * 4/11 = 18/121
        i3     (A 0, B 0): 7/16 * 5/9 * 2/3 = 35/216,  9/16 * 3/11 * 4/11 = 27/484
        i5     (A 0, B 1): 7/16 * 5/9 * 1/3 = 35/432,  9/16 * 3/11 * 7/11 = 189/1936
    """
    prior = [F(7, 16), F(9, 16)]
    confusion = [
        [[F(5, 9), F(4, 9)], [F(3, 11), F(8, 11)]],
        [[F(2, 3), F(1, 3)], [F(4, 11), F(7, 11)]],
    ]
    posterior = [
        [F(847, 4249), F(3402, 4249)],
        [F(847, 1819), F(972, 1819)],
        [F(4235, 5693), F(1458, 5693)],
        [F(847, 4249), F(3402, 4249)],
        [F(4235, 9338), F(5103, 9338)],
        [F(847, 1819), F(972, 1819)],
    ]
    assert F(7, 108) / (F(7, 108) + F(63, 242)) == posterior[0][0]
    assert F(7, 54) / (F(7, 54) + F(18, 121)) == posterior[1][0]
    assert F(35, 216) / (F(35, 216) + F(27, 484)) == posterior[2][0]
    assert F(35, 432) / (F(35, 432) + F(189, 1936)) == posterior[4][0]

    labels = LabelSet.from_records(HAND_WORKED)
    result = dawid_skene(labels, smoothing=1, max_iter=1)

    np.testing.assert_allclose(result.class_prior, np.array(prior, dtype=float), rtol=1e-12)
    np.testing.assert_allclose(result.confusion, np.array(confusion, dtype=float), rtol=1e-12)
    np.testing.assert_allclose(result.posterior, np.array(posterior, dtype=float), rtol=1e-12)
    penalty = sum(math.log(value) for matrix in confusion for row in matrix for value in row)
    penalty += sum(math.log(value) for value in prior)
    assert result.objective_trace[0] == pytest.approx(result.log_likelihood + penalty, rel=1e-12)
    # the items on which both annotators agree become less certain: 3402/4249 < 75/82 for i1
    sharp = dawid_skene(labels, smoothing=0, max_iter=1)
    agreed = [0, 2, 3]
    assert (result.confidence[agreed] < sharp.confidence[agreed]).all()


def exact_em(records, classes, smoothing, n_steps):
    """EM straight from the formulas, with plain loops and exact rational arithmetic."""
    items = sorted({item for item, _, _ in records})
    annotators = sorted({annotator for _, annotator, _ in records})
    n_classes = len(classes)
    shares = {}
    for item in items:
        given = [label for i, _, label in records if i == item]
        shares[item] = [F(given.count(name), len(given)) for name in classes]
    for _ in range(n_steps):
        prior = [
            (sum(shares[item][k] for item in items) + smoothing)
            / (len(items) + n_classes * smoothing)
            for k in range(n_classes)
        ]
        confusion = {}
        for annotator in annotators:
            for k in range(n_classes):
                weights = [
                    sum(shares[i][k] for i, a, label in records if a == annotator and label == name)
                    + smoothing
                    for name in classes
                ]
                total = sum(weights)
                confusion[annotator, k] = [
                    weight / total if total else F(1, n_classes) for weight in weights
                ]
        evidence = {}
        for item in items:
            joint = []
            for k in range(n_classes):
                value = prior[k]
                for i, annotator, label in records:
                    if i == item:
                        value *= confusion[annotator, k][classes.index(label)]
                joint.append(value)
            evidence[item] = sum(joint)
            shares[item] = [value / evidence[item] for value in joint]
    rows = [[confusion[annotator, k] for k in range(n_classes)] for annotator in annotators]
    return prior, rows, [shares[item] for item in items], [evidence[item] for item in items]


@pytest.mark.parametrize("smoothing", [F(0), F(1, 2)])
@pytest.mark.parametrize("n_steps", [1, 2, 3])
def test_em_steps_match_exact_rational_arithmetic(smoothing, n_steps):
    rng = np.random.default_rng(8)
    classes = ["x", "y", "z"]
    records = [
        (f"i{item}", f"a{annotator}", classes[rng.integers(3)])
        for item in range(6)
        for annotator in range(3)
        if rng.random() < 0.8
    ]
    prior, confusion, posterior, evidence = exact_em(records, classes, smoothing, n_steps)

    labels = LabelSet.from_records(records, classes=classes)
    result = dawid_skene(labels, smoothing=float(smoothing), max_iter=n_steps, tol=0)

    assert result.n_iter == n_steps
    np.testing.assert_allclose(result.class_prior, np.array(prior, dtype=float), rtol=1e-10)
    np.testing.assert_allclose(
        result.confusion, np.array(confusion, dtype=float), rtol=1e-10, atol=1e-300
    )
    np.testing.assert_allclose(
        result.posterior, np.array(posterior, dtype=float), rtol=1e-10, atol=1e-300
    )
    log_likelihood = sum(math.log(value) for value in evidence)
    assert result.log_likelihood == pytest.approx(log_likelihood, rel=1e-10)


def symmetric_three_annotator_data(unanimous: int, split: int) -> LabelSet:
    """Every answer pattern of three binary annotators: unanimous ones and 2-1 splits."""
    records = []
    for pattern in product("01", repeat=3):
        copies = unanimous if len(set(pattern)) == 1 else split
        for _ in range(copies):
            item = len(records)
            records += [(item, name, label) for name, label in zip("ABC", pattern, strict=True)]
    return LabelSet.from_records(records)


@pytest.mark.parametrize(("unanimous", "split"), [(7, 1), (3, 2), (20, 1)])
def test_identical_symmetric_annotators_give_majority_vote_and_a_closed_form(unanimous, split):
    """Data that is unchanged by swapping annotators or the two classes.

    EM started from the vote shares keeps that symmetry, so every annotator gets the same matrix
    [[a, 1 - a], [1 - a, a]] and the prior stays (1/2, 1/2). With u = a(1 - a) a unanimous pattern
    has probability (1 - 3u) / 2 and a split pattern u / 2, so the log-likelihood is
    2 * unanimous * log((1 - 3u) / 2) + 6 * split * log(u / 2), largest at
    u = split / (unanimous + 3 * split). At that point the model reproduces the pattern
    frequencies exactly, so it is the global maximum. The posterior of a split item is a for its
    majority class and that of a unanimous item is a^3 / (a^3 + (1 - a)^3): the label is the
    majority vote for every item.
    """
    labels = symmetric_three_annotator_data(unanimous, split)
    result = dawid_skene(labels, smoothing=0, **TIGHT)

    u = split / (unanimous + 3 * split)
    a = (1 + math.sqrt(1 - 4 * u)) / 2
    assert result.converged
    np.testing.assert_allclose(result.class_prior, [0.5, 0.5], atol=1e-9)
    for matrix in result.confusion:
        np.testing.assert_allclose(matrix, [[a, 1 - a], [1 - a, a]], atol=1e-8)
    most_votes = labels.vote_counts().max(axis=1)
    expected = np.where(most_votes == 3, a**3 / (a**3 + (1 - a) ** 3), a)
    np.testing.assert_allclose(result.confidence, expected, atol=1e-8)
    assert result.labels == majority_vote(labels).labels
    n_items = 2 * unanimous + 6 * split
    log_likelihood = 2 * unanimous * math.log((1 - 3 * u) / 2) + 6 * split * math.log(u / 2)
    assert labels.n_items == n_items
    assert result.log_likelihood == pytest.approx(log_likelihood, abs=1e-9)


def test_identical_symmetric_annotators_rank_classes_by_votes():
    """With one shared symmetric matrix and a uniform prior the posterior is a function of votes.

    Each vote for class k multiplies its posterior by a and every other class by (1 - a)/(K - 1),
    so the posterior is proportional to r^votes with r = a (K - 1) / (1 - a) > 1.
    """
    rng = np.random.default_rng(11)
    n_classes, accuracy = 4, 0.55
    records = [
        (item, annotator, int(rng.integers(n_classes)))
        for item in range(300)
        for annotator in range(7)
        if rng.random() < 0.7
    ]
    labels = LabelSet.from_records(records, classes=range(n_classes))
    confusion = np.tile(symmetric_confusion(accuracy, n_classes), (labels.n_annotators, 1, 1))
    posterior = dawid_skene_posterior(labels, confusion, np.full(n_classes, 1 / n_classes))

    votes = labels.vote_counts()
    expected = (accuracy * (n_classes - 1) / (1 - accuracy)) ** votes
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(posterior, expected, rtol=1e-10)
    by_model, model_ties = resolve_labels(posterior, labels.classes, tie="first")
    by_votes = majority_vote(labels, tie="first")
    assert by_model == by_votes.labels
    np.testing.assert_array_equal(model_ties, by_votes.tied)
    assert by_votes.tied.any()
    assert not by_votes.tied.all()


@pytest.fixture(scope="module")
def noisy() -> LabelSet:
    confusion = np.stack([symmetric_confusion(a, 3) for a in (0.9, 0.75, 0.6, 0.5, 0.4)])
    return simulate(150, confusion, labels_per_item=4, seed=5)


def test_result_shapes_and_bookkeeping(noisy):
    result = dawid_skene(noisy)
    assert result.method == "dawid-skene"
    assert result.items == noisy.items
    assert result.posterior.shape == (150, 3)
    assert result.confusion.shape == (5, 3, 3)
    np.testing.assert_allclose(result.posterior.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(result.confusion.sum(axis=2), 1.0, atol=1e-12)
    assert result.class_prior.sum() == pytest.approx(1.0)
    assert result.converged
    assert result.n_iter == len(result.log_likelihood_trace) == len(result.objective_trace)
    assert result.log_likelihood == result.log_likelihood_trace[-1]
    assert result.smoothing == 0.01
    assert result.restart_objectives == (result.objective_trace[-1],)
    assert result.best_restart == 0


@pytest.mark.parametrize("smoothing", [0.0, 0.01, 1.0])
def test_objective_never_decreases(noisy, smoothing):
    result = dawid_skene(noisy, smoothing=smoothing, tol=0, max_iter=150)
    trace = np.array(result.objective_trace)
    assert len(trace) == 150
    assert (np.diff(trace) >= -1e-9).all()
    assert trace[-1] > trace[0]
    if smoothing == 0:
        assert result.objective_trace == result.log_likelihood_trace
    else:
        assert (trace < np.array(result.log_likelihood_trace)).all()


def test_row_order_does_not_change_the_fit(noisy):
    records = list(noisy.records())
    order = np.random.default_rng(0).permutation(len(records))
    shuffled = LabelSet.from_records([records[index] for index in order])
    first, second = dawid_skene(noisy), dawid_skene(shuffled)
    np.testing.assert_array_equal(first.posterior, second.posterior)
    np.testing.assert_array_equal(first.confusion, second.confusion)
    assert first.labels == second.labels


def test_annotator_names_and_order_do_not_matter(noisy):
    rename = {"a1": "violet", "a2": "amber", "a3": "olive", "a4": "blue", "a5": "green"}
    renamed = LabelSet.from_records([(i, rename[a], label) for i, a, label in noisy.records()])
    assert renamed.annotators == ("amber", "blue", "green", "olive", "violet")
    base, other = dawid_skene(noisy, **TIGHT), dawid_skene(renamed, **TIGHT)
    np.testing.assert_allclose(other.posterior, base.posterior, rtol=0, atol=1e-8)
    np.testing.assert_allclose(other.class_prior, base.class_prior, rtol=0, atol=1e-8)
    for old, new in rename.items():
        np.testing.assert_allclose(
            other.confusion[renamed.annotators.index(new)],
            base.confusion[noisy.annotators.index(old)],
            rtol=0,
            atol=1e-8,
        )
    assert other.labels == base.labels
    assert other.log_likelihood == pytest.approx(base.log_likelihood, abs=1e-8)
    votes, renamed_votes = majority_vote(noisy), majority_vote(renamed)
    np.testing.assert_array_equal(votes.posterior, renamed_votes.posterior)
    assert votes.labels == renamed_votes.labels


def test_class_names_and_order_do_not_matter(noisy):
    rename = {"c0": "zebra", "c1": "ant", "c2": "mole"}
    renamed = LabelSet.from_records([(i, a, rename[label]) for i, a, label in noisy.records()])
    assert renamed.classes == ("ant", "mole", "zebra")
    columns = [renamed.classes.index(rename[name]) for name in noisy.classes]
    base, other = dawid_skene(noisy, **TIGHT), dawid_skene(renamed, **TIGHT)
    np.testing.assert_allclose(other.posterior[:, columns], base.posterior, rtol=0, atol=1e-8)
    np.testing.assert_allclose(other.class_prior[columns], base.class_prior, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        other.confusion[:, columns][:, :, columns], base.confusion, rtol=0, atol=1e-8
    )
    assert other.labels == tuple(rename.get(label) for label in base.labels)
    assert other.log_likelihood == pytest.approx(base.log_likelihood, abs=1e-8)
    votes, renamed_votes = majority_vote(noisy), majority_vote(renamed)
    np.testing.assert_array_equal(votes.posterior, renamed_votes.posterior[:, columns])
    assert renamed_votes.labels == tuple(rename.get(label) for label in votes.labels)


PLANTED = np.array(
    [
        [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]],
        [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]],
        [[0.6, 0.3, 0.1], [0.05, 0.9, 0.05], [0.3, 0.1, 0.6]],
        [[0.8, 0.1, 0.1], [0.3, 0.5, 0.2], [0.1, 0.1, 0.8]],
        [[1 / 3, 1 / 3, 1 / 3]] * 3,
    ]
)
PLANTED_PRIOR = np.array([0.5, 0.3, 0.2])


def accuracy(result, labels) -> float:
    return float(
        np.mean([got == want for got, want in zip(result.labels, labels.gold, strict=True)])
    )


def test_recovers_planted_confusion_matrices():
    labels = simulate(10000, PLANTED, class_prior=PLANTED_PRIOR, seed=1)
    result = dawid_skene(labels)
    assert result.converged
    # The rarest class has about 2000 items, so a cell estimated from known true classes would
    # have a standard error of at most sqrt(0.25 / 2000) = 0.011. The bound allows for the
    # true classes being inferred as well.
    assert np.abs(result.confusion - PLANTED).max() < 0.05
    assert np.abs(result.class_prior - PLANTED_PRIOR).max() < 0.03
    assert accuracy(result, labels) > accuracy(majority_vote(labels, tie="first"), labels) + 0.02


def test_recovery_improves_with_more_items():
    errors = []
    for n_items in (200, 20000):
        labels = simulate(n_items, PLANTED, class_prior=PLANTED_PRIOR, seed=2)
        result = dawid_skene(labels)
        errors.append(np.abs(result.confusion - PLANTED).max())
    assert errors[1] < errors[0] / 3
    assert errors[1] < 0.03


def test_restarts_report_every_start_and_return_the_best(noisy):
    single = dawid_skene(noisy)
    multi = dawid_skene(noisy, n_restarts=4, seed=3)
    assert len(multi.restart_objectives) == 4
    assert multi.restart_objectives[0] == single.objective_trace[-1]
    assert multi.objective_trace[-1] == multi.restart_objectives[multi.best_restart]
    assert multi.objective_trace[-1] >= max(multi.restart_objectives) - 1e-6
    again = dawid_skene(noisy, n_restarts=4, seed=3)
    assert again.restart_objectives == multi.restart_objectives
    np.testing.assert_array_equal(again.posterior, multi.posterior)


def test_maximum_likelihood_keeps_rows_without_data_as_probability_vectors():
    records = [
        (1, "a", "x"),
        (1, "b", "x"),
        (1, "c", "x"),
        (2, "a", "y"),
        (2, "b", "y"),
        (3, "a", "x"),
        (3, "b", "x"),
    ]
    result = dawid_skene(LabelSet.from_records(records), smoothing=0)
    assert np.isfinite(result.posterior).all()
    assert np.isfinite(result.log_likelihood)
    np.testing.assert_allclose(result.confusion.sum(axis=2), 1.0)
    # annotator c never labelled an item that could be of class y
    np.testing.assert_allclose(result.confusion[2, 1], [0.5, 0.5])
    np.testing.assert_allclose(result.posterior, [[1, 0], [0, 1], [1, 0]])
    assert result.labels == ("x", "y", "x")
    assert result.log_likelihood == pytest.approx(2 * math.log(2 / 3) + math.log(1 / 3))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"smoothing": -0.1}, "smoothing must not be negative"),
        ({"max_iter": 0}, "max_iter must be at least 1"),
        ({"tol": -1.0}, "tol must not be negative"),
        ({"n_restarts": 0}, "n_restarts must be at least 1"),
        ({"restart_noise": 0.0}, "restart_noise must be in"),
        ({"restart_noise": 1.5}, "restart_noise must be in"),
        ({"tie": "coin"}, "tie must be one of"),
    ],
)
def test_bad_arguments_are_errors(noisy, kwargs, message):
    with pytest.raises(ValueError, match=message):
        dawid_skene(noisy, **kwargs)


def test_posterior_for_given_parameters_checks_them():
    labels = LabelSet.from_records([("i", "a", "x"), ("j", "a", "y")])
    good = np.array([[[0.8, 0.2], [0.3, 0.7]]])
    posterior = dawid_skene_posterior(labels, good, [0.5, 0.5])
    np.testing.assert_allclose(posterior, [[0.8 / 1.1, 0.3 / 1.1], [0.2 / 0.9, 0.7 / 0.9]])
    with pytest.raises(ValueError, match="shape"):
        dawid_skene_posterior(labels, good[0], [0.5, 0.5])
    with pytest.raises(ValueError, match="length 2"):
        dawid_skene_posterior(labels, good, [1.0])
    with pytest.raises(ValueError, match="probability vector"):
        dawid_skene_posterior(labels, good * 2, [0.5, 0.5])
    with pytest.raises(ValueError, match="probability vector"):
        dawid_skene_posterior(labels, good, [0.7, 0.7])
    never_y = np.array([[[1.0, 0.0], [1.0, 0.0]]])
    with pytest.raises(FloatingPointError, match="probability zero"):
        dawid_skene_posterior(labels, never_y, [0.5, 0.5])
