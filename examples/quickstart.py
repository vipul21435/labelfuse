"""Aggregate the toy labels two ways and compare both with the planted classes.

Run from the repository root: python examples/quickstart.py
"""

from pathlib import Path

import numpy as np

from labelfuse import dawid_skene, majority_vote, read_labels

labels = read_labels(Path(__file__).with_name("toy.jsonl"), gold_col="gold")
print(f"{labels.n_items} items, {labels.n_annotators} annotators, {labels.n_labels} labels")


def correct(result) -> int:
    return sum(got == want for got, want in zip(result.labels, labels.gold, strict=True))


votes = majority_vote(labels)
# A tie broken at random is right with probability 1 / (number of tied classes) when the
# planted class is among them.
planted = np.array([labels.classes.index(name) for name in labels.gold])
top = votes.posterior == votes.posterior.max(axis=1, keepdims=True)
on_average = (top[np.arange(labels.n_items), planted] / top.sum(axis=1)).sum()
print(
    f"majority vote: {correct(votes)} of {labels.n_items} correct, "
    f"{int(votes.tied.sum())} tied and left without a label "
    f"({on_average:.1f} correct on average if ties are broken at random)"
)

model = dawid_skene(labels)
print(f"Dawid-Skene: {correct(model)} of {labels.n_items} correct, {int(model.tied.sum())} tied")
print("estimated accuracy per annotator (mean diagonal of the confusion matrix):")
for name, matrix in zip(model.annotators, model.confusion, strict=True):
    print(f"  {name}  {np.diag(matrix).mean():.2f}")
