import warnings

import numpy as np
import pytest
from scipy import stats

from labelfuse import (
    GoldOutsideClassesWarning,
    LabelSet,
    bootstrap_mean_interval,
    brier_score,
    dawid_skene,
    evaluate,
    expected_calibration_error,
    gold_outside_classes,
    majority_vote,
    mcnemar_exact,
    reliability_table,
    simulate,
    symmetric_confusion,
    tie_credit,
)
from labelfuse.evaluate import gold_indices


def test_tie_credit_splits_a_tie_and_ignores_unknown_gold():
    posterior = np.array([[0.7, 0.3, 0.0], [0.4, 0.4, 0.2], [1 / 3, 1 / 3, 1 / 3], [0.1, 0.2, 0.7]])
    gold = np.array([0, 1, 2, -1])
    np.testing.assert_allclose(tie_credit(posterior, gold), [1, 1 / 2, 1 / 3, 0])
    np.testing.assert_allclose(tie_credit(posterior, np.array([1, 2, 0, 2])), [0, 0, 1 / 3, 1])


def test_gold_indices():
    labels = LabelSet.from_records(
        [("i1", "a", "x"), ("i2", "a", "y"), ("i3", "a", "x")], gold={"i1": "y", "i3": "other"}
    )
    np.testing.assert_array_equal(gold_indices(labels), [1, -2, -1])
    with pytest.raises(ValueError, match="no gold labels"):
        gold_indices(LabelSet.from_records([("i1", "a", "x")]))


@pytest.mark.parametrize(
    ("b", "c"), [(0, 0), (1, 0), (3, 3), (15, 0), (5, 24), (40, 61), (300, 250)]
)
def test_mcnemar_matches_the_exact_binomial_test(b, c):
    expected = 1.0 if b + c == 0 else stats.binomtest(b, b + c, 0.5).pvalue
    assert mcnemar_exact(b, c) == pytest.approx(expected, rel=1e-10)
    assert mcnemar_exact(c, b) == pytest.approx(expected, rel=1e-10)


def test_mcnemar_by_hand():
    # 1 against 5: P(X <= 1) for Binomial(6, 1/2) is (1 + 6) / 64; doubled it is 14 / 64
    assert mcnemar_exact(1, 5) == pytest.approx(14 / 64)
    assert mcnemar_exact(0, 5) == pytest.approx(2 / 32)
    with pytest.raises(ValueError, match="negative"):
        mcnemar_exact(-1, 2)


def test_bootstrap_interval_of_binary_scores_matches_the_binomial():
    """Resampling n binary scores with k ones draws the number of ones from Binomial(n, k/n)."""
    values = np.array([1.0] * 180 + [0.0] * 20)
    low, high = bootstrap_mean_interval(values, n_boot=20000, seed=1)
    exact_low, exact_high = stats.binom.ppf([0.025, 0.975], 200, 0.9) / 200
    assert low == pytest.approx(exact_low, abs=0.006)
    assert high == pytest.approx(exact_high, abs=0.006)
    assert (low, high) == bootstrap_mean_interval(values, n_boot=20000, seed=1)


def test_bootstrap_interval_matches_plain_index_resampling():
    rng = np.random.default_rng(5)
    values = rng.choice([0.0, 1 / 3, 0.5, 1.0], size=400, p=[0.2, 0.1, 0.1, 0.6])
    low, high = bootstrap_mean_interval(values, n_boot=20000, seed=2)
    means = values[rng.integers(0, 400, size=(20000, 400))].mean(axis=1)
    plain_low, plain_high = np.quantile(means, [0.025, 0.975])
    assert low == pytest.approx(plain_low, abs=0.004)
    assert high == pytest.approx(plain_high, abs=0.004)
    assert bootstrap_mean_interval(np.ones(10)) == (1.0, 1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"values": np.array([])}, "no values"),
        ({"values": np.ones(3), "n_boot": 0}, "n_boot"),
        ({"values": np.ones(3), "level": 1.0}, "level"),
    ],
)
def test_bootstrap_rejects_bad_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        bootstrap_mean_interval(**kwargs)


def test_reliability_table_and_calibration_error_by_hand():
    confidence = np.array([0.55, 0.58, 0.62, 0.95, 0.95, 1.0])
    correct = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    table = reliability_table(confidence, correct, n_bins=10)
    assert [(row.lower, row.upper, row.count) for row in table] == [
        (0.5, 0.6, 2),
        (0.6, 0.7, 1),
        (0.9, 1.0, 3),
    ]
    assert table[0].mean_confidence == pytest.approx(0.565)
    assert table[0].accuracy == pytest.approx(0.5)
    assert table[2].mean_confidence == pytest.approx(2.9 / 3)
    assert table[2].accuracy == pytest.approx(2 / 3)
    gaps = 2 * 0.065 + 1 * 0.38 + 3 * (2.9 / 3 - 2 / 3)
    assert expected_calibration_error(table) == pytest.approx(gaps / 6)
    with pytest.raises(ValueError, match="n_bins"):
        reliability_table(confidence, correct, n_bins=0)


def test_brier_score_by_hand():
    posterior = np.array([[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]])
    # (0.2^2 + 0.2^2) + (0.5^2 + 0.5^2) + (0.1^2 + 0.9^2 with gold outside the classes)
    assert brier_score(posterior, np.array([0, 1, -1])) == pytest.approx((0.08 + 0.5 + 0.82) / 3)
    assert brier_score(np.eye(2), np.array([0, 1])) == 0.0
    assert brier_score(np.eye(2), np.array([1, 0])) == 2.0


@pytest.fixture(scope="module")
def noisy() -> LabelSet:
    confusion = np.stack([symmetric_confusion(a, 2) for a in (0.9, 0.8, 0.6, 0.55)])
    return simulate(400, confusion, seed=3)


def test_evaluate_reports_accuracy_difference_and_mcnemar(noisy):
    results = {"majority": majority_vote(noisy), "dawid-skene": dawid_skene(noisy)}
    by_name = {row.method: row for row in evaluate(noisy, results, n_boot=500, seed=4)}
    vote, model = by_name["majority"], by_name["dawid-skene"]

    gold = np.array(noisy.gold)
    hard = np.array(majority_vote(noisy, tie="first").labels) == gold
    tied = results["majority"].tied
    assert vote.n_items == 400
    assert vote.n_tied == int(tied.sum()) > 0
    assert vote.accuracy == pytest.approx((hard & ~tied).mean() + 0.5 * tied.mean())
    assert vote.difference == 0.0
    assert (vote.only_method, vote.only_baseline, vote.mcnemar_p) == (0, 0, 1.0)
    assert vote.interval[0] < vote.accuracy < vote.interval[1]

    correct = np.array(results["dawid-skene"].labels) == gold
    assert model.accuracy == pytest.approx(correct.mean())
    assert model.difference == pytest.approx(model.accuracy - vote.accuracy)
    assert model.difference_interval[0] < model.difference < model.difference_interval[1]
    assert model.accuracy > vote.accuracy
    assert model.only_method > model.only_baseline
    assert model.mcnemar_p == mcnemar_exact(model.only_method, model.only_baseline)
    assert model.brier < vote.brier
    assert sum(row.count for row in model.reliability) == 400
    assert 0 <= model.ece <= 1


def test_evaluate_uses_only_items_with_gold():
    records = [(i, a, "x" if i < 6 else "y") for i in range(10) for a in "abc"]
    gold = dict.fromkeys(range(4), "x") | {8: "x"}
    labels = LabelSet.from_records(records, gold=gold)
    (row,) = evaluate(labels, {"majority": majority_vote(labels)}, n_boot=50)
    assert row.n_items == 5
    assert row.accuracy == pytest.approx(4 / 5)


def test_evaluate_rejects_missing_gold_baseline_and_foreign_results(noisy):
    without = LabelSet.from_records(list(noisy.records()))
    with pytest.raises(ValueError, match="no gold labels"):
        evaluate(without, {"majority": majority_vote(without)})
    empty = LabelSet.from_records(list(noisy.records()), gold={})
    with pytest.raises(ValueError, match="no item has a gold label"):
        evaluate(empty, {"majority": majority_vote(empty)})
    with pytest.raises(ValueError, match="baseline"):
        evaluate(noisy, {"dawid-skene": dawid_skene(noisy)})
    other = simulate(30, np.stack([symmetric_confusion(0.8, 2)] * 3), seed=1)
    with pytest.raises(ValueError, match="other items"):
        evaluate(noisy, {"majority": majority_vote(other)})


def test_gold_outside_the_classes_is_scored_wrong_with_a_warning():
    records = [(i, a, "x") for i in range(4) for a in "ab"] + [(4, "a", "y"), (4, "b", "y")]
    gold = {0: "x", 1: "x", 2: "z", 3: "other", 4: "y"}
    labels = LabelSet.from_records(records, gold=gold)
    assert gold_outside_classes(labels) == (2, ("other", "z"))
    message = r"not among the classes \('x', 'y'\): 'other', 'z' on 2 of 5 items with gold"
    with pytest.warns(GoldOutsideClassesWarning, match=message):
        (row,) = evaluate(labels, {"majority": majority_vote(labels)}, n_boot=50)
    assert row.n_items == 5
    assert row.accuracy == pytest.approx(3 / 5)


def test_no_warning_when_every_gold_label_is_a_class(noisy):
    assert gold_outside_classes(noisy) == (0, ())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        evaluate(noisy, {"majority": majority_vote(noisy)}, n_boot=50)
