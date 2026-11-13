import numpy as np
import pytest

from labelfuse import LabelSet, majority_vote, resolve_labels, vote_shares

RECORDS = [
    ("i1", "a", "x"),
    ("i1", "b", "x"),
    ("i1", "c", "y"),
    ("i2", "a", "y"),
    ("i2", "b", "z"),
    ("i3", "a", "z"),
    ("i4", "a", "x"),
    ("i4", "b", "y"),
    ("i4", "c", "z"),
]


def test_posterior_is_the_vote_share():
    result = majority_vote(LabelSet.from_records(RECORDS))
    assert result.method == "majority"
    assert result.items == ("i1", "i2", "i3", "i4")
    assert result.classes == ("x", "y", "z")
    expected = [[2 / 3, 1 / 3, 0], [0, 1 / 2, 1 / 2], [0, 0, 1], [1 / 3, 1 / 3, 1 / 3]]
    np.testing.assert_allclose(result.posterior, expected)
    np.testing.assert_allclose(result.confidence, [2 / 3, 1 / 2, 1, 1 / 3])


def test_ties_abstain_by_default():
    result = majority_vote(LabelSet.from_records(RECORDS))
    assert result.labels == ("x", None, "z", None)
    np.testing.assert_array_equal(result.tied, [False, True, False, True])


def test_tie_first_takes_the_first_tied_class_in_class_order():
    result = majority_vote(LabelSet.from_records(RECORDS), tie="first")
    assert result.labels == ("x", "y", "z", "x")
    np.testing.assert_array_equal(result.tied, [False, True, False, True])


def test_tie_random_is_seeded_and_stays_within_the_tied_classes():
    labels = LabelSet.from_records(RECORDS)
    seen_i2, seen_i4 = set(), set()
    for seed in range(40):
        result = majority_vote(labels, tie="random", seed=seed)
        assert result.labels == majority_vote(labels, tie="random", seed=seed).labels
        assert result.labels[0] == "x"
        assert result.labels[2] == "z"
        seen_i2.add(result.labels[1])
        seen_i4.add(result.labels[3])
    assert seen_i2 == {"y", "z"}
    assert seen_i4 == {"x", "y", "z"}


def test_tie_random_is_roughly_uniform():
    posterior = np.tile([0.4, 0.2, 0.4], (3000, 1))
    labels, tied = resolve_labels(posterior, ("x", "y", "z"), tie="random", seed=1)
    assert tied.all()
    share = labels.count("x") / len(labels)
    assert labels.count("y") == 0
    # 3000 fair draws: 0.5 +- 4 standard errors of 0.0091
    assert abs(share - 0.5) < 0.037


def test_near_equal_posteriors_count_as_tied():
    posterior = np.array([[0.5 + 1e-12, 0.5 - 1e-12], [0.5 + 1e-6, 0.5 - 1e-6]])
    labels, tied = resolve_labels(posterior, ("x", "y"))
    assert labels == (None, "x")
    np.testing.assert_array_equal(tied, [True, False])


def test_unknown_tie_policy_is_an_error():
    with pytest.raises(ValueError, match="tie must be one of"):
        majority_vote(LabelSet.from_records(RECORDS), tie="coin")


def test_odd_number_of_binary_votes_never_ties():
    rng = np.random.default_rng(3)
    records = [(i, j, int(rng.integers(2))) for i in range(50) for j in range(5)]
    result = majority_vote(LabelSet.from_records(records))
    assert not result.tied.any()
    assert None not in result.labels


def test_vote_shares_of_an_item_without_votes_are_equal():
    np.testing.assert_allclose(vote_shares(np.array([[0, 0], [3, 1]])), [[0.5, 0.5], [0.75, 0.25]])


def test_to_records_lists_label_probability_and_posterior():
    records = majority_vote(LabelSet.from_records(RECORDS)).to_records()
    assert records[0] == {
        "item": "i1",
        "label": "x",
        "probability": pytest.approx(2 / 3),
        "tied": False,
        "posterior": {"x": pytest.approx(2 / 3), "y": pytest.approx(1 / 3), "z": 0.0},
    }
    assert records[1]["label"] is None
    assert records[1]["tied"] is True
