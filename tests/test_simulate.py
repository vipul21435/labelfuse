import numpy as np
import pytest

from labelfuse import simulate, symmetric_confusion

PLANTED = np.array(
    [
        [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]],
        [[0.6, 0.3, 0.1], [0.05, 0.9, 0.05], [0.3, 0.1, 0.6]],
        [[1 / 3, 1 / 3, 1 / 3]] * 3,
    ]
)


def test_symmetric_confusion():
    np.testing.assert_allclose(
        symmetric_confusion(0.7, 3), [[0.7, 0.15, 0.15], [0.15, 0.7, 0.15], [0.15, 0.15, 0.7]]
    )
    np.testing.assert_allclose(symmetric_confusion(0.5, 2), [[0.5, 0.5], [0.5, 0.5]])


@pytest.mark.parametrize(
    ("accuracy", "n_classes", "message"),
    [
        (1.2, 2, "accuracy must be between 0 and 1"),
        (-0.1, 3, "accuracy must be between 0 and 1"),
        (0.5, 1, "n_classes must be at least 2"),
    ],
)
def test_symmetric_confusion_rejects_bad_arguments(accuracy, n_classes, message):
    with pytest.raises(ValueError, match=message):
        symmetric_confusion(accuracy, n_classes)


def test_every_annotator_labels_every_item_by_default():
    labels = simulate(40, PLANTED, seed=2)
    assert (labels.n_items, labels.n_annotators, labels.n_classes) == (40, 3, 3)
    assert labels.n_labels == 120
    assert labels.items[0] == "i01"
    assert labels.items[-1] == "i40"
    assert labels.annotators == ("a1", "a2", "a3")
    assert labels.classes == ("c0", "c1", "c2")
    assert labels.n_gold == 40
    assert set(labels.gold) <= set(labels.classes)


def test_labels_per_item_draws_distinct_annotators():
    confusion = np.stack([symmetric_confusion(0.8, 2)] * 6)
    labels = simulate(500, confusion, labels_per_item=2, seed=4)
    np.testing.assert_array_equal(labels.labels_per_item(), 2)
    pairs = set(zip(labels.item_index.tolist(), labels.annotator_index.tolist(), strict=True))
    assert len(pairs) == 1000
    # each annotator is drawn for an item with probability 1/3: 167 +- 5 standard errors of 10.5
    assert np.abs(labels.labels_per_annotator() - 1000 / 6).max() < 53


def test_same_seed_same_labels_and_other_seed_other_labels():
    first = simulate(60, PLANTED, seed=9)
    again = simulate(60, PLANTED, seed=9)
    other = simulate(60, PLANTED, seed=10)
    np.testing.assert_array_equal(first.label_index, again.label_index)
    assert first.gold == again.gold
    assert not np.array_equal(first.label_index, other.label_index)


def test_label_frequencies_match_the_planted_parameters():
    prior = np.array([0.5, 0.3, 0.2])
    n_items = 20000
    labels = simulate(n_items, PLANTED, class_prior=prior, seed=1)
    truth = np.array([labels.classes.index(name) for name in labels.gold])
    class_share = np.bincount(truth, minlength=3) / n_items
    # standard error of a share is at most sqrt(0.25 / 20000) = 0.0035
    assert np.abs(class_share - prior).max() < 0.014
    observed = np.zeros_like(PLANTED)
    np.add.at(observed, (labels.annotator_index, truth[labels.item_index], labels.label_index), 1)
    observed /= observed.sum(axis=2, keepdims=True)
    # the smallest class has about 4000 items: standard error at most sqrt(0.25 / 4000) = 0.008
    assert np.abs(observed - PLANTED).max() < 0.032


def test_named_classes():
    labels = simulate(10, PLANTED, classes=["cat", "dog", "owl"], seed=0)
    assert labels.classes == ("cat", "dog", "owl")
    assert set(labels.gold) <= {"cat", "dog", "owl"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_items": 0, "confusion": PLANTED}, "n_items must be at least 1"),
        ({"n_items": 5, "confusion": PLANTED[0]}, "confusion must have shape"),
        ({"n_items": 5, "confusion": PLANTED * 2}, "must be a probability vector"),
        (
            {"n_items": 5, "confusion": PLANTED, "class_prior": [0.5, 0.5]},
            "class_prior must be a probability vector of length 3",
        ),
        (
            {"n_items": 5, "confusion": PLANTED, "class_prior": [0.5, 0.6, -0.1]},
            "class_prior must be a probability vector of length 3",
        ),
        (
            {"n_items": 5, "confusion": PLANTED, "labels_per_item": 4},
            "labels_per_item must be between 1 and 3",
        ),
        (
            {"n_items": 5, "confusion": PLANTED, "labels_per_item": 0},
            "labels_per_item must be between 1 and 3",
        ),
        (
            {"n_items": 5, "confusion": PLANTED, "classes": ["x", "x", "y"]},
            "classes must hold 3 distinct names",
        ),
    ],
)
def test_bad_arguments_are_errors(kwargs, message):
    with pytest.raises(ValueError, match=message):
        simulate(**kwargs)
