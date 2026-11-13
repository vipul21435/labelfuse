"""Simulated annotators with known confusion matrices, for testing recovery."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from labelfuse.data import LabelSet


def symmetric_confusion(accuracy: float, n_classes: int) -> np.ndarray:
    """Confusion matrix of an annotator who is right with probability ``accuracy``.

    The probability is the same for every true class, and a wrong answer falls on each of the
    other classes with equal probability. ``accuracy = 1 / n_classes`` is a random annotator.
    """
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    if not 0 <= accuracy <= 1:
        raise ValueError("accuracy must be between 0 and 1")
    matrix = np.full((n_classes, n_classes), (1 - accuracy) / (n_classes - 1))
    np.fill_diagonal(matrix, accuracy)
    return matrix


def _names(prefix: str, count: int, *, start: int = 1) -> list[str]:
    width = len(str(count - 1 + start))
    return [f"{prefix}{index:0{width}d}" for index in range(start, count + start)]


def simulate(
    n_items: int,
    confusion: np.ndarray | Sequence[Sequence[Sequence[float]]],
    *,
    class_prior: Sequence[float] | None = None,
    labels_per_item: int | None = None,
    seed: int = 0,
    classes: Sequence[str] | None = None,
) -> LabelSet:
    """Draw labels from the Dawid-Skene model with the given parameters.

    ``confusion[j, k, l]`` is the probability that annotator j answers class l on an item of true
    class k. The true class of each item is drawn from ``class_prior`` (uniform by default). Each
    item is labelled by every annotator, or by ``labels_per_item`` annotators chosen uniformly
    without replacement. The returned label set carries the true classes as gold; items are named
    ``i1, i2, ...``, annotators ``a1, a2, ...`` (in the order of ``confusion``) and classes
    ``c0, c1, ...`` unless ``classes`` names them.
    """
    matrices = np.asarray(confusion, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("confusion must have shape (n_annotators, n_classes, n_classes)")
    n_annotators, n_classes, _ = matrices.shape
    if (matrices < 0).any() or not np.allclose(matrices.sum(axis=2), 1.0):
        raise ValueError("every row of every confusion matrix must be a probability vector")
    if n_items < 1:
        raise ValueError("n_items must be at least 1")
    if class_prior is None:
        prior = np.full(n_classes, 1.0 / n_classes)
    else:
        prior = np.asarray(class_prior, dtype=float)
        if prior.shape != (n_classes,) or (prior < 0).any() or not np.isclose(prior.sum(), 1.0):
            raise ValueError(f"class_prior must be a probability vector of length {n_classes}")
    per_item = n_annotators if labels_per_item is None else labels_per_item
    if not 1 <= per_item <= n_annotators:
        raise ValueError(f"labels_per_item must be between 1 and {n_annotators}")
    if classes is None:
        class_names = _names("c", n_classes, start=0)
    else:
        class_names = [str(name) for name in classes]
        if len(class_names) != n_classes or len(set(class_names)) != n_classes:
            raise ValueError(f"classes must hold {n_classes} distinct names")

    rng = np.random.default_rng(seed)
    truth = rng.choice(n_classes, size=n_items, p=prior)
    if per_item == n_annotators:
        chosen = np.tile(np.arange(n_annotators), (n_items, 1))
    else:
        chosen = np.sort(rng.random((n_items, n_annotators)).argsort(axis=1)[:, :per_item], axis=1)
    item_index = np.repeat(np.arange(n_items), per_item)
    annotator_index = chosen.ravel()
    cumulative = matrices.cumsum(axis=2)[annotator_index, truth[item_index]]
    draws = rng.random(item_index.size)
    label_index = np.minimum((draws[:, None] > cumulative).sum(axis=1), n_classes - 1)

    item_names = _names("i", n_items)
    annotator_names = _names("a", n_annotators)
    return LabelSet(
        items=tuple(item_names),
        annotators=tuple(annotator_names),
        classes=tuple(class_names),
        item_index=item_index,
        annotator_index=annotator_index,
        label_index=label_index,
        gold=tuple(class_names[k] for k in truth),
    )
