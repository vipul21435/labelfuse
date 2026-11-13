# labelfuse

Turn several annotators' labels for the same items into one label with a probability, and report
which annotators and which items the result should not be trusted on.

**Status: early.** What works today is listed under [What works so far](#what-works-so-far); the
annotator and item audits, the evaluation against gold labels and the runs on public data are
[not written yet](#not-implemented-yet).

## Why this is hard

Majority vote treats a careful annotator and a random one alike. Weighing annotators by their
quality would fix that, but quality is measured against the true labels, and those are unknown:
the quality of each annotator has to be estimated from the same labels it is then used to weigh.

## Install

```
pip install git+https://github.com/vipul21435/labelfuse
```

Python 3.10 or later. The only dependency is numpy.

## What works so far

- Reading labels in long format (one row per label: item, annotator, label) from JSONL, CSV or
  TSV, with an optional gold column. Row order does not change a result. An annotator labelling
  the same item twice is an error unless a policy is chosen (`--duplicates keep|first|last`; the
  last two keep one label by row order).
- Majority vote with an explicit tie policy: `abstain` (default, the item gets no label), `first`
  (first tied class in class order) or `random` (seeded).
- Dawid-Skene fitted by EM: one confusion matrix per annotator, class prior, smoothing, several
  starts, log-likelihood trace. Returns the posterior of every item, not only the label.
- A simulator that draws labels from annotators with given confusion matrices.
- Two commands, `labelfuse aggregate` and `labelfuse simulate`, with Markdown and JSON output.

## Quickstart

`examples/toy.jsonl` holds simulated labels for 200 items in 3 classes from five annotators whose
planted accuracies are 0.9, 0.8, 0.7, 0.6 and 0.34 (with three classes the last one is close to
guessing). Each row is one label; the `gold` column is the planted class.

```
$ labelfuse simulate --items 200 --classes 3 --accuracy 0.9,0.8,0.7,0.6,0.34 --seed 7 -o examples/toy.jsonl
$ head -2 examples/toy.jsonl
{"item": "i001", "annotator": "a1", "label": "c1", "gold": "c1"}
{"item": "i001", "annotator": "a2", "label": "c1", "gold": "c1"}
```

Aggregate with Dawid-Skene (the default method):

```
$ labelfuse aggregate examples/toy.jsonl --top 5
# Aggregated labels

- items: 200
- annotators: 5
- labels: 1000 (per item: min 5, median 5, max 5)
- classes: c0, c1, c2
- method: dawid-skene, smoothing 0.01
- EM: 153 iterations, converged, log-likelihood -931.8092
- tied items: 0

## Items per class

| class | items | share |
|---|---:|---:|
| c0 | 69 | 0.345 |
| c1 | 69 | 0.345 |
| c2 | 62 | 0.310 |

## Least certain items (5 of 200)

| item | label | probability | votes |
|---|---|---:|---|
| i163 | c0 | 0.508 | c0 2, c1 2, c2 1 |
| i073 | c2 | 0.508 | c0 3, c2 2 |
| i048 | c1 | 0.518 | c0 2, c1 3 |
| i178 | c0 | 0.521 | c0 2, c1 1, c2 2 |
| i135 | c2 | 0.541 | c0 2, c1 1, c2 2 |
```

`--method majority` gives the vote shares instead, and `--format json` writes the full posterior
of every item together with the confusion matrix of every annotator, the class prior and the
log-likelihood trace.

The same from Python, compared with the planted classes:

```
$ python examples/quickstart.py
200 items, 5 annotators, 1000 labels
majority vote: 168 of 200 correct, 24 tied and left without a label (178.0 correct on average if ties are broken at random)
Dawid-Skene: 191 of 200 correct, 0 tied
estimated accuracy per annotator (mean diagonal of the confusion matrix):
  a1  0.93
  a2  0.78
  a3  0.69
  a4  0.67
  a5  0.41
```

This is one simulated file whose labels were drawn from the very model that Dawid-Skene fits, so
it shows that the code recovers what was planted. It says nothing yet about real annotators.

## How it works

**Majority vote.** The posterior of an item is its share of votes per class. That describes the
votes; it is not a calibrated probability. The label is the class with the most votes, and the
tie policy decides what happens when there is no single such class.

**Dawid-Skene.** Each item `i` has a hidden true class drawn from a class prior `p`. Annotator
`j`, shown an item of class `k`, answers `l` with probability `pi[j, k, l]`, so each annotator has
a confusion matrix of their own. Given the true class, labels are independent. EM starts from the
vote shares as the first posterior `T` and alternates:

```
M-step   pi[j, k, l] = (sum_i T[i, k] * n[i, j, l] + s) / (sum_i T[i, k] * n[i, j, .] + K * s)
         p[k]        = (sum_i T[i, k] + s) / (I + K * s)
E-step   T[i, k]  proportional to  p[k] * product over the labels (j, l) of item i of pi[j, k, l]
```

`n[i, j, l]` counts how often annotator `j` gave label `l` to item `i`, `K` is the number of
classes, `I` the number of items and `s` the smoothing pseudo-count (`--smoothing`, default 0.01).
With `s = 0` this is maximum likelihood. With `s > 0` it is the posterior mode under a
Dirichlet(1 + s) prior on every row of every confusion matrix and on the class prior. Iteration
stops once every posterior entry moves by less than `--tol` (default 1e-8) in one iteration.
`--restarts N` adds `N - 1` starts whose first posterior mixes the vote shares with random noise,
and returns the start with the highest objective.

The model and its EM fit are from A. P. Dawid and A. M. Skene, "Maximum Likelihood Estimation of
Observer Error-Rates Using the EM Algorithm", Applied Statistics 28(1), 1979,
[doi:10.2307/2346806](https://doi.org/10.2307/2346806).

The tests check one EM step against a calculation done by hand in exact fractions, three steps
against a plain-loop implementation in rational arithmetic, a symmetric data set whose maximum
likelihood solution has a closed form, recovery of planted confusion matrices from simulated
labels, and that renaming or reordering annotators and classes does not change the result.

## Not implemented yet

- Accuracy-weighted vote, the one-coin model, GLAD and MACE.
- The annotator audit (accuracy with an interval, spammer score, bias, flags for random and
  constant annotators) and the item audit (posterior entropy, relabelling planner).
- Evaluation against gold labels (bootstrap intervals, McNemar, calibration). The gold column is
  read but no command uses it yet.
- The `annotators`, `items`, `evaluate` and `plan` commands.
- Results on public crowdsourcing data and a comparison with another implementation.

## Limitations of what exists

- The Dawid-Skene likelihood has several local maxima. The default single start depends on the
  vote shares, and with `--restarts` the highest likelihood is not guaranteed to be the most
  accurate labelling. Classes are not re-aligned after a random start.
- The model assumes that annotators err independently given the true class and that an
  annotator's error rates do not depend on the item.
- A pseudo-count that is large next to the number of labels an annotator gave pulls that
  annotator's confusion matrix toward uniform, which removes their say in the result.

## Licence

MIT, see [LICENSE](LICENSE).

Built by Vipul Raj Jha · https://vipul21435.github.io
