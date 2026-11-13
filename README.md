# labelfuse

Turn several annotators' labels for the same items into one label with a probability, and report
which annotators and which items the result should not be trusted on.

## Why this is hard

Majority vote treats a careful annotator and a random one alike. Weighing annotators by their
quality would fix that, but quality is measured against the true labels, and those are unknown:
the quality of each annotator has to be estimated from the same labels it is then used to weigh.
The models that do this have several solutions, can be confidently wrong, and behave differently
on different data, so the result needs an audit and, where gold labels exist, an evaluation.

## Install

```
pip install git+https://github.com/vipul21435/labelfuse
```

Python 3.10 or later. Dependencies: numpy, and scipy for the optimiser inside GLAD.

## Quickstart

Input is one label per row (item, annotator, label, and optionally the gold label of the item) in
JSONL, CSV or TSV. `examples/toy.jsonl` holds simulated labels for 200 items in 3 classes from
five annotators whose planted accuracies are 0.9, 0.8, 0.7, 0.6 and 0.34 (with three classes the
last one is close to guessing); the `gold` column is the planted class.

```
$ labelfuse simulate --items 200 --classes 3 --accuracy 0.9,0.8,0.7,0.6,0.34 --seed 7 -o examples/toy.jsonl
$ head -2 examples/toy.jsonl
{"item": "i001", "annotator": "a1", "label": "c1", "gold": "c1"}
{"item": "i001", "annotator": "a2", "label": "c1", "gold": "c1"}
```

Aggregate with Dawid-Skene (the default method; `--method` also takes `majority`, `weighted`,
`one-coin` and `glad`). `--format json` writes the full posterior of every item and the fitted
parameters.

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

Compare every aggregator with the gold column:

```
$ labelfuse evaluate examples/toy.jsonl
# Evaluation against gold

- items with gold: 200 of 200
- annotators: 5, labels: 1000, classes: c0, c1, c2
- intervals: 95 % percentile bootstrap over items, 2000 resamples, seed 0
- accuracy gives 1/t credit when the gold class is one of t tied top classes; the McNemar counts break ties at random with seed 0

| method | accuracy | 95 % interval | ties | vs majority | 95 % interval | only method / only baseline right | McNemar p | ECE | Brier |
|---|---:|---|---:|---:|---|---|---:|---:|---:|
| majority | 0.890 | 0.853 to 0.927 | 24 | - | - | - | - | 0.178 | 0.227 |
| weighted | 0.930 | 0.895 to 0.965 | 0 | +0.040 | +0.020 to +0.062 | 5 / 2 | 0.453 | 0.191 | 0.192 |
| one-coin | 0.940 | 0.905 to 0.970 | 0 | +0.050 | +0.020 to +0.080 | 9 / 4 | 0.267 | 0.034 | 0.074 |
| dawid-skene | 0.955 | 0.925 to 0.985 | 0 | +0.065 | +0.033 to +0.100 | 12 / 4 | 0.077 | 0.023 | 0.080 |
```

Audit the annotators, and ask which items deserve another label:

```
$ labelfuse annotators examples/toy.jsonl
# Annotator audit

- annotators: 5, labels: 1000, classes: c0, c1, c2
- method: dawid-skene, smoothing 0.01
- EM: 153 iterations, converged, log-likelihood -931.8092
- flagged: 0 of 5 annotators (0.0%; random 0, constant 0), who gave 0 of 1000 labels
- rule: at least 10 labels and the upper end of the 95 % bootstrap interval of the spammer score below 0.05

## Least informative annotators (5 of 5)

| annotator | labels | accuracy | 95 % interval | spammer score | 95 % interval | bias | top label | agreement | flag |
|---|---:|---:|---|---:|---|---|---|---:|---|
| a5 | 200 | 0.416 | 0.350 to 0.485 | 0.017 | 0.004 to 0.051 | c0 +0.056 | c0 0.41 | 0.415 |  |
| a4 | 200 | 0.669 | 0.601 to 0.730 | 0.256 | 0.188 to 0.344 | c2 +0.033 | c1 0.36 | 0.680 |  |
| a3 | 200 | 0.689 | 0.622 to 0.749 | 0.283 | 0.203 to 0.389 | c2 +0.033 | c0 0.34 | 0.685 |  |
| a2 | 200 | 0.778 | 0.716 to 0.830 | 0.435 | 0.338 to 0.525 | c2 +0.008 | c1 0.35 | 0.785 |  |
| a1 | 200 | 0.932 | 0.888 to 0.959 | 0.786 | 0.726 to 0.837 | c2 +0.008 | c0 0.34 | 0.945 |  |
```

```
$ labelfuse plan examples/toy.jsonl --budget 5
# Relabelling plan

- method: dawid-skene, smoothing 0.01
- EM: 153 iterations, converged, log-likelihood -931.8092
- budget: 5 labels, planned: 5 (one more label for each listed item)
- expected drop in posterior entropy: 1.309 bits in total, from one more label by an average annotator on each of them

## Items to relabel (5 of 5)

| item | label | probability | entropy (bits) | expected drop (bits) | labels |
|---|---|---:|---:|---:|---:|
| i048 | c1 | 0.518 | 1.062 | 0.274 | 5 |
| i038 | c0 | 0.697 | 1.149 | 0.264 | 5 |
| i163 | c0 | 0.508 | 1.069 | 0.261 | 5 |
| i008 | c2 | 0.572 | 1.077 | 0.259 | 5 |
| i178 | c0 | 0.521 | 1.004 | 0.252 | 5 |
```

`labelfuse items` lists the items by posterior entropy. The same from Python:

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

The toy labels were drawn from the very model that Dawid-Skene fits, so this shows that the code
recovers what was planted and nothing more. Real data is under
[Results on public data](#results-on-public-data).

## How it works

Every aggregator returns a posterior over the classes for each item. Ids and labels are compared
as text, items, annotators and classes are put in a canonical order, and the rows are sorted, so
row order does not change a result. A whole number is written like an integer, so the JSON values
1 and 1.0 are the same label; NaN and infinities count as missing. An annotator labelling the
same item twice is an error unless `--duplicates keep|first|last` says what to do.

**Majority vote** (`majority`). The posterior is the share of votes per class: a description of
the votes, not a calibrated probability. A tie has an explicit policy, `--tie abstain` (default,
no label), `first` (first tied class in class order) or `random` (seeded).

**Accuracy-weighted vote** (`weighted`). Each label counts with the weight of its annotator. The
weight is the annotator's leave-one-out agreement: for each of their labels, the other labels of
the item are tallied, and the label scores 1 if it is their single most frequent class, 1/t if it
is one of t tied classes and 0 otherwise; the weight is `(score + 1) / (labels + K)` for `K`
classes. `weighted_vote(labels, weights=...)` takes known weights instead.

**Dawid-Skene** (`dawid-skene`). Each item `i` has a hidden true class drawn from a class prior
`p`. Annotator `j`, shown an item of class `k`, answers `l` with probability `pi[j, k, l]`, so
each annotator has a confusion matrix of their own. Given the true class, labels are independent.
EM starts from the vote shares as the first posterior `T` and alternates:

```
M-step   pi[j, k, l] = (sum_i T[i, k] * n[i, j, l] + s) / (sum_i T[i, k] * n[i, j, .] + K * s)
         p[k]        = (sum_i T[i, k] + s) / (I + K * s)
E-step   T[i, k]  proportional to  p[k] * product over the labels (j, l) of item i of pi[j, k, l]
```

`n[i, j, l]` counts how often annotator `j` gave label `l` to item `i`, `I` is the number of items
and `s` the smoothing pseudo-count (`--smoothing`, default 0.01). With `s = 0` this is maximum
likelihood; with `s > 0` it is the posterior mode under a Dirichlet(1 + s) prior on every row of
every confusion matrix and on the class prior. Iteration stops once every posterior entry moves
by less than `--tol` (default 1e-8) in one iteration. `--restarts N` adds `N - 1` starts whose
first posterior mixes the vote shares with random noise, and returns the start with the highest
objective. The model and its EM fit are from A. P. Dawid and A. M. Skene, "Maximum Likelihood
Estimation of Observer Error-Rates Using the EM Algorithm", Applied Statistics 28(1), 1979,
[doi:10.2307/2346806](https://doi.org/10.2307/2346806).

**One-coin model** (`one-coin`). Dawid-Skene with one parameter per annotator: annotator `j` is
right with probability `a[j]` whatever the true class, and a wrong answer falls evenly on the
other classes. The M-step is `a[j] = (expected correct labels + s) / (labels + K * s)`. The class
prior is fixed to uniform (see [Limitations](#limitations)).

**GLAD** (`glad`, two classes only). Annotator `j` labels item `i` correctly with probability
`sigmoid(alpha[j] * beta[i])`: `alpha` is the ability of the annotator (0 is random, negative
tends to give the opposite label) and `beta > 0` the easiness of the item. EM again; the M-step
has no closed form and is done with L-BFGS-B from scipy. The Gaussian prior `alpha ~ N(1, 1)` is
the one used in the paper (its section 3.1). For `log beta` the paper uses `N(1, 1)` too, while
this implementation centres it at 0, `log beta ~ N(0, 1)`. The optimiser keeps `alpha` within
[-8, 8] and `log beta` within [-4, 4], and the class prior is uniform. The model is from
J. Whitehill, T. Wu, J. Bergsma, J. R. Movellan and P. L. Ruvolo, "Whose Vote Should Count More:
Optimal Integration of Labels from Labelers of Unknown Expertise", NIPS 2009
([paper page](https://papers.nips.cc/paper_files/paper/2009/hash/f899139df5e1059396431415e770c6dd-Abstract.html)).

**Annotator audit** (`labelfuse annotators`). Computed from the labels and the posterior of any
aggregator. Each annotator gets a confusion matrix against the posterior (row `k` is the
distribution of their labels weighted by the posterior of class `k`, plus one pseudo-label spread
like their overall label distribution); the expected share of their labels that are right, with a
Wilson interval; accuracy on gold items where there are any; the class they give more often than
the posterior expects on the same items (bias); their agreement with the aggregate label; and the
spammer score of V. C. Raykar and S. Yu, "Eliminating Spammers and Ranking Annotators for
Crowdsourced Labeling Tasks", JMLR 13:491-518, 2012
([paper page](https://www.jmlr.org/papers/v13/raykar12a.html)):

```
S = 1 / (K (K - 1)) * sum over pairs c < c' and classes k of (A[c, k] - A[c', k])^2
```

`A` is the confusion matrix. An annotator whose answer does not depend on the true class has
equal rows and `S = 0`; a perfect one has `S = 1`; for two classes
`S = (sensitivity + specificity - 1)^2`. The score measures information, not accuracy: an
annotator who reliably gives the wrong class scores high too. It comes with a percentile
bootstrap interval from resampling the annotator's labels. An annotator with at least
`--min-labels` (10) labels is flagged when the upper end of that interval is below
`--spam-threshold` (0.05): `constant` if one class makes up at least `--constant-share` (0.95) of
their labels, `random` otherwise.

**Item audit and relabelling plan** (`labelfuse items`, `labelfuse plan`). Items are listed by
posterior entropy in bits; an item is uncertain when its top probability is below `--certain-at`
(0.9). The planner asks what one more label from an average annotator would buy. With `q` the
posterior of an item and `A` the confusion matrix of all labels pooled against the posterior, the
new label is `l` with probability `P(l) = sum_k q[k] A[k, l]` and turns the posterior into
`q[k] A[k, l] / P(l)`. The expected drop in entropy, `H(q) - sum_l P(l) H(q | l)`, is the mutual
information between the true class and the new label; the plan is the `--budget` items where it
is largest.

**Evaluation** (`labelfuse evaluate`). On the items with a gold label: accuracy, where an item
whose gold class is one of `t` classes tied at the top counts 1/t (the expected score of breaking
ties at random); a 95 % percentile bootstrap interval over items; the difference to majority vote
with a paired bootstrap interval; an exact McNemar test on the items only one of the two gets
right (ties broken at random with `--seed` for these counts); and for the posteriors a
reliability table (`--reliability`), the expected calibration error over ten bins of the top
probability, and the Brier score. `--drop-flagged` repeats the evaluation without the annotators
that the audit flags. A gold label that no annotator used can never be predicted: such items are
scored as wrong, and the report and a warning say how many there are.

**Tests.** One EM step is checked against a calculation by hand in exact fractions, three steps
against a plain-loop implementation in rational arithmetic, and a symmetric data set against its
closed-form maximum likelihood solution, where Dawid-Skene must equal majority vote. Planted
confusion matrices, accuracies and abilities are recovered from simulated labels. Renaming or
reordering annotators and classes must not change a result. McNemar and the Wilson interval are
compared with scipy, the bootstrap with the binomial distribution, the planner with the mutual
information of a binary symmetric channel. The `compare` extra pins crowd-kit 1.4.2
(`pip install -e ".[dev,compare]"`), the only version tried: with it, Dawid-Skene with
`smoothing=0` matches crowd-kit's posterior, prior and confusion matrices to 1e-6 on a simulated
set, and the vote shares match to 1e-12. Without crowd-kit those two tests are skipped, as they
are in CI.

## Results on public data

Public crowdsourcing data sets with gold labels. None of them is committed here: no source page
states a licence for the data itself, so this repository holds only the script that fetches them
and the numbers below.

| data set | source | licence stated at the source | items (with gold) | annotators | labels | classes |
|---|---|---|---:|---:|---:|---:|
| bluebirds | `demo/bluebirds` in [welinder/cubam](https://github.com/welinder/cubam), the code for Welinder, Branson, Belongie and Perona, "The Multidimensional Wisdom of Crowds", NIPS 2010 | the repository is under the Clear BSD licence (`LICENSE.TXT`); its README says nothing separate about the data | 108 (108) | 39 | 4212 | 2 |
| nist-trec-relevance | archive that [crowd-kit](https://github.com/Toloka/crowd-kit) downloads (`crowdkit/datasets/_loaders.py`): crowd judgements of web pages from the NIST TREC 2010 Relevance Feedback Track | none stated for the data; the crowd-kit code is Apache-2.0 | 20232 (4460) | 766 | 98453 | 4 |
| relevance-2 | same loader: binary relevance labels for query-document pairs from Yandex's 2016 "Relevance 2 Gradations" project | none stated for the data | 99319 (10079) | 7138 | 475536 | 2 |


```
python scripts/fetch_data.py --out data
labelfuse evaluate data/bluebirds.jsonl --drop-flagged
labelfuse evaluate data/nist-trec-relevance.jsonl --duplicates keep --drop-flagged
labelfuse evaluate data/relevance-2.jsonl --methods majority,weighted,one-coin,dawid-skene --tol 1e-6 --drop-flagged
```

The script checks the two crowd-kit archives against the md5 files published next to them. The
numbers are from runs on 2026-11-13 with Python 3.12.13, numpy 2.5.4 and scipy 1.18.1. Accuracy
gives 1/t credit for a t-way tie that contains the gold class; intervals are 95 % percentile
bootstrap intervals over items (2000 resamples, seed 0); "vs majority" is the paired difference;
ECE and Brier describe the posteriors.

### bluebirds

| method | accuracy | 95 % interval | ties | vs majority | 95 % interval | only method / only baseline right | McNemar p | ECE | Brier |
|---|---:|---|---:|---:|---|---|---:|---:|---:|
| majority | 0.759 | 0.676 to 0.843 | 0 | - | - | - | - | 0.129 | 0.328 |
| weighted | 0.759 | 0.676 to 0.843 | 0 | +0.000 | +0.000 to +0.000 | 0 / 0 | 1.000 | 0.062 | 0.312 |
| one-coin | 0.722 | 0.639 to 0.806 | 0 | -0.037 | -0.102 to +0.028 | 4 / 8 | 0.388 | 0.279 | 0.555 |
| dawid-skene | 0.898 | 0.833 to 0.954 | 0 | +0.139 | +0.074 to +0.204 | 15 / 0 | 6.1e-05 | 0.099 | 0.199 |
| glad | 0.722 | 0.639 to 0.806 | 0 | -0.037 | -0.102 to +0.028 | 4 / 8 | 0.388 | 0.253 | 0.487 |

Flagged by the dawid-skene audit: 6 of 39 annotators (15.4%; random 6, constant 0), who gave 648 of 4212 labels (15.4%). Accuracy with and without them:

| method | all annotators | without flagged | change |
|---|---:|---:|---:|
| majority | 0.759 | 0.759 | +0.000 |
| weighted | 0.759 | 0.769 | +0.009 |
| one-coin | 0.722 | 0.796 | +0.074 |
| dawid-skene | 0.898 | 0.898 | +0.000 |
| glad | 0.722 | 0.769 | +0.046 |

Dawid-Skene is 0.139 above majority vote: it gets 15 items right that majority vote gets wrong
and none the other way. The accuracy-weighted vote changes nothing. The one-coin model and GLAD
are 0.037 below majority vote, a difference whose interval (-0.102 to +0.028) includes zero.
Removing the six flagged annotators changes neither majority vote nor Dawid-Skene and lifts the
one-coin model by 0.074 and GLAD by 0.046.

### nist-trec-relevance

| method | accuracy | 95 % interval | ties | vs majority | 95 % interval | only method / only baseline right | McNemar p | ECE | Brier |
|---|---:|---|---:|---:|---|---|---:|---:|---:|
| majority | 0.541 | 0.529 to 0.554 | 916 | - | - | - | - | 0.062 | 0.606 |
| weighted | 0.559 | 0.544 to 0.572 | 1 | +0.017 | +0.011 to +0.023 | 214 / 139 | 7.7e-05 | 0.083 | 0.595 |
| one-coin | 0.440 | 0.426 to 0.454 | 75 | -0.101 | -0.115 to -0.087 | 386 / 836 | 1.3e-38 | 0.398 | 0.933 |
| dawid-skene | 0.610 | 0.596 to 0.623 | 0 | +0.069 | +0.055 to +0.082 | 713 / 409 | 8.9e-20 | 0.256 | 0.634 |

Flagged by the dawid-skene audit: 23 of 766 annotators (3.0%; random 9, constant 14), who gave 30394 of 98453 labels (30.9%). Accuracy with and without them:

| method | all annotators | without flagged | change |
|---|---:|---:|---:|
| majority | 0.541 | 0.590 | +0.049 |
| weighted | 0.559 | 0.615 | +0.057 |
| one-coin | 0.440 | 0.615 | +0.175 |
| dawid-skene | 0.610 | 0.613 | +0.004 |

Dawid-Skene is 0.069 above majority vote and the one-coin model 0.101 below it. The audit flags
3.0 % of the annotators, and those 23 gave 30.9 % of all labels. Without them the three simpler
methods rise by 0.049 to 0.175 and Dawid-Skene by 0.004, so that all four end between 0.590 and
0.615 and the lead of Dawid-Skene over majority vote shrinks from 0.069 to 0.023. Dawid-Skene's
posteriors are overconfident on this data: its expected calibration error is 0.256, against
0.062 for the plain vote shares. The file has 1570 repeated (item, annotator) pairs, which
`--duplicates keep` counts as separate judgements. GLAD does not apply to four classes.

### relevance-2

| method | accuracy | 95 % interval | ties | vs majority | 95 % interval | only method / only baseline right | McNemar p | ECE | Brier |
|---|---:|---|---:|---:|---|---|---:|---:|---:|
| majority | 0.772 | 0.764 to 0.780 | 214 | - | - | - | - | 0.042 | 0.330 |
| weighted | 0.774 | 0.766 to 0.782 | 5 | +0.002 | +0.000 to +0.003 | 60 / 46 | 0.206 | 0.077 | 0.322 |
| one-coin | 0.793 | 0.785 to 0.800 | 5 | +0.021 | +0.015 to +0.026 | 514 / 311 | 1.6e-12 | 0.134 | 0.343 |
| dawid-skene | 0.812 | 0.804 to 0.819 | 0 | +0.040 | +0.034 to +0.046 | 705 / 308 | 1.9e-36 | 0.118 | 0.310 |

Flagged by the dawid-skene audit: 471 of 7138 annotators (6.6%; random 207, constant 264), who gave 25446 of 475536 labels (5.4%). Accuracy with and without them:

| method | all annotators | without flagged | change |
|---|---:|---:|---:|
| majority | 0.772 | 0.779 | +0.007 |
| weighted | 0.774 | 0.787 | +0.013 |
| one-coin | 0.793 | 0.799 | +0.006 |
| dawid-skene | 0.812 | 0.811 | -0.000 |

Dawid-Skene is 0.040 above majority vote and the one-coin model 0.021: here the one-coin model
helps instead of hurting. The weighted vote is within 0.002 of majority vote. The audit flags
6.6 % of the annotators, who gave 5.4 % of the labels, and removing them moves every method by
0.013 or less. EM ran with `--tol 1e-6` instead of the default 1e-8 to shorten the run, and GLAD
is left out (see [Limitations](#limitations)).

### What this shows

Dawid-Skene was above majority vote on all three data sets, by 0.139, 0.069 and 0.040, and all
three intervals are clear of zero. Its posteriors were not as good as its labels: on
nist-trec-relevance they are overconfident. The one-coin model, which gives each annotator a
single accuracy, was below majority vote on bluebirds (by 0.037, interval includes zero) and on
nist-trec-relevance (by 0.101), and above it on relevance-2 (by 0.021). GLAD was run on bluebirds
only and was 0.037 below majority vote there. The accuracy-weighted vote was never more than
0.017 away from majority vote.

The audit flagged 15.4 %, 3.0 % and 6.6 % of the annotators. Without them Dawid-Skene changed by
0.004 or less on every data set. Majority vote changed by 0.000 on bluebirds, 0.049 on
nist-trec-relevance and 0.007 on relevance-2, the weighted vote by 0.009, 0.057 and 0.013, and
the one-coin model by 0.074, 0.175 and 0.006. The largest changes are on nist-trec-relevance,
where the 23 flagged annotators (9 random, 14 constant) had given 30.9 % of the labels: without
them the weighted vote and the one-coin model reach 0.615, above the 0.613 of Dawid-Skene, and
majority vote reaches 0.590. These runs do not test why Dawid-Skene moved so little.

Three data sets are too few to say when Dawid-Skene fails to beat majority vote. Here it never
was below it. On nist-trec-relevance its lead over majority vote shrank from 0.069 to 0.023 once
the flagged annotators were removed, and two simpler methods then scored above it.

## Limitations

- MACE is not implemented.
- GLAD handles two classes only. Its M-step re-optimises one parameter per annotator and per
  item in every EM iteration, which is slow on large data: on relevance-2 (the third data set
  that `scripts/fetch_data.py` fetches: 99319 items, 7138 annotators, 475536 labels) an
  evaluation that included GLAD was stopped unfinished after about eight minutes. Its time was
  not measured separately.
- The one-coin model and GLAD fix the class prior to uniform. With the prior estimated
  (`estimate_prior=True`) both drifted to "nearly every image is `false`" on bluebirds and
  scored 0.583 and 0.593 instead of 0.722:

  ```
  python -c "
  from labelfuse import evaluate, glad, majority_vote, one_coin, read_labels
  labels = read_labels('data/bluebirds.jsonl', gold_col='gold')
  results = {'majority': majority_vote(labels),
             'one-coin, free prior': one_coin(labels, estimate_prior=True),
             'glad, free prior': glad(labels, estimate_prior=True)}
  for row in evaluate(labels, results): print(row.method, round(row.accuracy, 3))"
  ```
- The Dawid-Skene likelihood has several local maxima. The default single start depends on the
  vote shares, and with `--restarts` the highest likelihood is not guaranteed to be the most
  accurate labelling. Classes are not re-aligned after a random start.
- EM can be slow. The stopping rule bounds the last move of the posterior, not the distance to
  the fixed point. On relevance-2 with the default `--tol 1e-8`, Dawid-Skene took 743 iterations
  and the one-coin model had not converged after the maximum of 1000
  (`labelfuse aggregate data/relevance-2.jsonl --method one-coin --top 0` prints the count).
- A smoothing pseudo-count that is large next to the number of labels an annotator gave pulls
  that annotator's confusion matrix toward uniform, which removes their say in the result.
- The models assume that annotators err independently given the true class. Dawid-Skene and the
  one-coin model also assume that an annotator's error rates do not depend on the item.
- The annotator audit rests on the posterior of the aggregate: where the aggregate is wrong, so
  is the audit. Its accuracy interval treats the posterior as known, and the flags are thresholds
  on an estimate, to be read as "look at this annotator", not as a verdict.
- The planner gives each item at most one new label, assumes the label comes from an average
  annotator, and knows nothing about item difficulty.
- The bootstrap intervals resample items only. They do not cover what a different set of
  annotators would have answered.
- Vote shares (majority, weighted) are not probabilities, and model posteriors can be
  overconfident, as the calibration columns above show.
- Agreement coefficients (kappa, alpha) are out of scope; see rubricops below.
- The public data sets are not redistributed here because their sources state no licence for
  the data; `scripts/fetch_data.py` downloads them.

## Related tools

- [crowd-kit](https://github.com/Toloka/crowd-kit) covers more aggregation methods: its README
  lists Majority Vote, One-coin Dawid-Skene, Dawid-Skene, Gold Majority Vote, M-MSR, Wawa,
  Zero-Based Skill, GLAD, KOS and MACE. labelfuse is smaller and centred on the audit, the
  evaluation with intervals and the command line.
- MACE: D. Hovy, T. Berg-Kirkpatrick, A. Vaswani and E. Hovy, "Learning Whom to Trust with MACE",
  NAACL 2013 ([paper page](https://aclanthology.org/N13-1132/)).
- S. Paun, B. Carpenter, J. Chamberlain, D. Hovy, U. Kruschwitz and M. Poesio, "Comparing
  Bayesian Models of Annotation", TACL 6:571-585, 2018
  ([paper page](https://aclanthology.org/Q18-1040/)), compares six models of annotation on the
  same data sets.
- R. Snow, B. O'Connor, D. Jurafsky and A. Ng, "Cheap and Fast - But is it Good? Evaluating
  Non-Expert Annotations for Natural Language Tasks", EMNLP 2008
  ([paper page](https://aclanthology.org/D08-1027/)).
- [rubricops](https://github.com/vipul21435/rubricops): rubrics and inter-rater agreement
  (kappa, alpha). Agreement says how much annotators agree; labelfuse says what the label
  probably is and whom to trust.
- [evalstats](https://github.com/vipul21435/evalstats): error bars and paired comparisons for
  evaluations.
- [judgebias](https://github.com/vipul21435/judgebias): audit of LLM judges.
- [prefpairs](https://github.com/vipul21435/prefpairs): pairwise preference data and
  Bradley-Terry, for judgements that compare two items instead of labelling one.

## Licence

MIT, see [LICENSE](LICENSE).

Built by Vipul Raj Jha · https://vipul21435.github.io
