import math

import numpy as np
import pytest
from scipy import stats

from labelfuse import (
    LabelSet,
    audit_annotators,
    audit_items,
    average_confusion,
    dawid_skene,
    expected_entropy_reduction,
    majority_vote,
    plan_relabelling,
    posterior_entropy,
    simulate,
    spammer_score,
    symmetric_confusion,
    wilson_interval,
)


def test_spammer_score_closed_forms():
    assert spammer_score(np.eye(2)) == pytest.approx(1.0)
    assert spammer_score(np.eye(5)) == pytest.approx(1.0)
    assert spammer_score([[0.5, 0.25, 0.25]] * 3) == pytest.approx(0.0)
    assert spammer_score([[1, 0], [1, 0]]) == pytest.approx(0.0)
    # two classes: (sensitivity + specificity - 1)^2, also for an annotator who inverts
    assert spammer_score([[0.9, 0.1], [0.3, 0.7]]) == pytest.approx((0.9 + 0.7 - 1) ** 2)
    assert spammer_score([[0, 1], [1, 0]]) == pytest.approx(1.0)
    assert spammer_score(symmetric_confusion(0.6, 2)) == pytest.approx(0.04)


def test_spammer_score_equals_the_sum_over_pairs_of_rows():
    rng = np.random.default_rng(0)
    matrix = rng.dirichlet(np.ones(4), size=4)
    pairs = sum(((matrix[c] - matrix[d]) ** 2).sum() for c in range(4) for d in range(c + 1, 4))
    assert spammer_score(matrix) == pytest.approx(pairs / (4 * 3))
    with pytest.raises(ValueError, match="square"):
        spammer_score(np.ones((2, 3)) / 3)
    with pytest.raises(ValueError, match="square"):
        spammer_score([[1.0]])


@pytest.mark.parametrize(("k", "n"), [(8, 10), (0, 7), (50, 50), (123, 400)])
def test_wilson_interval_matches_scipy(k, n):
    expected = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    assert wilson_interval(k, n) == pytest.approx((expected.low, expected.high), abs=1e-9)
    assert wilson_interval(0, 0) == (0.0, 1.0)


@pytest.fixture(scope="module")
def crowd() -> LabelSet:
    """Binary labels from a good, a fair, a random, a constant and an inverting annotator."""
    confusion = np.array(
        [
            symmetric_confusion(0.9, 2),
            symmetric_confusion(0.8, 2),
            symmetric_confusion(0.8, 2),
            [[0.5, 0.5], [0.5, 0.5]],
            [[1.0, 0.0], [1.0, 0.0]],
            symmetric_confusion(0.1, 2),
        ]
    )
    return simulate(600, confusion, seed=7)


def test_annotator_audit_flags_random_and_constant_annotators(crowd):
    result = dawid_skene(crowd)
    reports = audit_annotators(crowd, result, seed=1)
    assert [report.annotator for report in reports] == list(crowd.annotators)
    assert [report.flag for report in reports] == [None, None, None, "random", "constant", None]
    good, _, _, random, constant, inverter = reports

    assert good.n_labels == 600
    assert good.accuracy == pytest.approx(0.9, abs=0.04)
    assert good.accuracy_interval[0] < good.accuracy < good.accuracy_interval[1]
    assert good.spammer_score == pytest.approx(0.64, abs=0.1)
    assert good.spammer_interval[0] < good.spammer_score < good.spammer_interval[1]
    np.testing.assert_allclose(good.confusion.sum(axis=1), 1.0)
    assert random.spammer_score < 0.02
    assert random.accuracy == pytest.approx(0.5, abs=0.06)
    assert constant.top_label == "c0"
    assert constant.top_label_share == 1.0
    assert constant.bias_class == "c0"
    assert constant.bias == pytest.approx(0.5, abs=0.06)
    # the inverter is wrong nine times in ten and still carries information
    assert inverter.accuracy == pytest.approx(0.1, abs=0.04)
    assert inverter.spammer_score > 0.5

    gold = np.array(crowd.gold)
    for j, report in enumerate(reports):
        mine = crowd.annotator_index == j
        given = np.array(crowd.classes)[crowd.label_index[mine]]
        truth = gold[crowd.item_index[mine]]
        assert report.n_gold == 600
        assert report.gold_accuracy == pytest.approx(np.mean(given == truth))
        assert report.gold_interval == wilson_interval(int(np.sum(given == truth)), 600)
        aggregate = np.array(result.labels)[crowd.item_index[mine]]
        assert report.agreement == pytest.approx(np.mean(given == aggregate))


def test_annotator_audit_thresholds_and_missing_gold(crowd):
    plain = LabelSet.from_records(list(crowd.records()))
    result = majority_vote(plain)
    reports = audit_annotators(plain, result, min_labels=1000, n_boot=20)
    assert all(report.flag is None for report in reports)
    assert all(report.gold_accuracy is None and report.n_gold == 0 for report in reports)
    strict = audit_annotators(plain, result, spam_threshold=2.0, constant_share=0.0, n_boot=20)
    assert {report.flag for report in strict} == {"constant"}
    with pytest.raises(ValueError, match="another label set"):
        audit_annotators(crowd, majority_vote(simulate(5, np.stack([np.eye(2)] * 2))))


def test_posterior_entropy_in_bits():
    rows = np.array([[0.5, 0.5, 0, 0], [1, 0, 0, 0], [0.25, 0.25, 0.25, 0.25]])
    np.testing.assert_allclose(posterior_entropy(rows), [1.0, 0.0, 2.0])


def binary_entropy(p: float) -> float:
    return 0.0 if p in (0, 1) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


@pytest.mark.parametrize(("q", "a"), [(0.5, 0.8), (0.7, 0.9), (0.9, 0.6), (0.3, 0.2)])
def test_expected_entropy_reduction_for_a_symmetric_annotator(q, a):
    """Binary symmetric channel: I(class; label) = H(q a + (1 - q)(1 - a)) - H(a)."""
    gain = expected_entropy_reduction(np.array([[q, 1 - q]]), symmetric_confusion(a, 2))
    expected = binary_entropy(q * a + (1 - q) * (1 - a)) - binary_entropy(a)
    assert gain[0] == pytest.approx(expected, abs=1e-12)


def test_expected_entropy_reduction_limits():
    posterior = np.array([[0.6, 0.4], [1.0, 0.0], [0.5, 0.5]])
    np.testing.assert_allclose(
        expected_entropy_reduction(posterior, [[0.5, 0.5]] * 2), 0, atol=1e-12
    )
    perfect = expected_entropy_reduction(posterior, np.eye(2))
    np.testing.assert_allclose(perfect, posterior_entropy(posterior), atol=1e-12)
    assert expected_entropy_reduction(posterior, symmetric_confusion(0.8, 2))[1] == 0.0


def test_item_audit_and_plan(crowd):
    result = dawid_skene(crowd)
    reports = audit_items(crowd, result, certain_at=0.9)
    assert len(reports) == 600
    first = reports[0]
    assert first.item == crowd.items[0]
    assert first.label == result.labels[0]
    assert first.probability == pytest.approx(result.posterior[0].max())
    assert first.margin == pytest.approx(abs(result.posterior[0, 0] - result.posterior[0, 1]))
    assert first.entropy == pytest.approx(binary_entropy(result.posterior[0, 0]))
    assert first.n_labels == 6
    assert sum(first.votes.values()) == 6
    uncertain = [report for report in reports if report.uncertain]
    assert 0 < len(uncertain) < 600
    assert all(report.probability < 0.9 for report in uncertain)

    pooled = average_confusion(crowd, result)
    np.testing.assert_allclose(pooled.sum(axis=1), 1.0)
    gain = expected_entropy_reduction(result.posterior, pooled)
    plan = plan_relabelling(crowd, result, 25)
    assert len(plan) == 25
    assert [row.expected_reduction for row in plan] == sorted(gain, reverse=True)[:25]
    assert plan[0].expected_reduction == pytest.approx(gain.max())
    assert plan[0].entropy > np.median([report.entropy for report in reports])
    assert len({row.item for row in plan}) == 25
    assert len(plan_relabelling(crowd, result, 10_000)) == 600
    assert plan_relabelling(crowd, result, 0) == []
    with pytest.raises(ValueError, match="budget"):
        plan_relabelling(crowd, result, -1)


def test_drop_annotators_keeps_items_and_gold(crowd):
    smaller = crowd.drop_annotators(["a4", "a5"])
    assert smaller.annotators == ("a1", "a2", "a3", "a6")
    assert smaller.items == crowd.items
    assert smaller.gold == crowd.gold
    assert smaller.n_labels == 2400
    kept = [record for record in crowd.records() if record[1] not in ("a4", "a5")]
    assert list(smaller.records()) == kept
    assert crowd.drop_annotators([]).n_labels == crowd.n_labels
    with pytest.raises(ValueError, match="unknown annotators"):
        crowd.drop_annotators(["nobody"])
    with pytest.raises(ValueError, match="no annotator would be left"):
        crowd.drop_annotators(crowd.annotators)


def test_item_without_labels_falls_back_to_the_prior():
    labels = LabelSet.from_records(
        [("i1", "a", "x"), ("i1", "b", "x"), ("i2", "a", "y"), ("i2", "b", "y"), ("i3", "c", "x")]
    )
    smaller = labels.drop_annotators(["c"])
    np.testing.assert_array_equal(smaller.labels_per_item(), [2, 2, 0])
    votes = majority_vote(smaller)
    np.testing.assert_allclose(votes.posterior[2], [0.5, 0.5])
    assert votes.labels[2] is None
    model = dawid_skene(smaller)
    np.testing.assert_allclose(model.posterior[2], model.class_prior)


def test_audits_reject_results_from_another_label_set(crowd):
    other = majority_vote(simulate(5, np.stack([np.eye(2)] * 2)))
    with pytest.raises(ValueError, match="another label set"):
        audit_items(crowd, other)
    with pytest.raises(ValueError, match="another label set"):
        plan_relabelling(crowd, other, 3)
    single = LabelSet.from_records([("i", "a", "x"), ("i", "b", "x")])
    with pytest.raises(ValueError, match="at least two classes"):
        audit_annotators(single, majority_vote(single))
