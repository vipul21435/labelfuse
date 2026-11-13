# Changelog

## 0.1.0

First version.

- Long-format input (item, annotator, label, optional gold) from JSONL, CSV or TSV, in a
  canonical order so that row order does not change a result; explicit policy for an annotator
  labelling an item twice. Whole numbers are written like integers (1 and 1.0 are one label) and
  NaN counts as missing.
- Aggregators, each returning a posterior per item: majority vote with an explicit tie policy,
  accuracy-weighted vote, the one-coin model, Dawid-Skene EM (confusion matrices, class prior,
  smoothing, several starts, log-likelihood trace) and GLAD for binary labels.
- Annotator audit: confusion matrix against the aggregate, accuracy with an interval, spammer
  score of Raykar and Yu with a bootstrap interval, bias toward a class, agreement with the
  aggregate, flags for random and constant annotators.
- Item audit: posterior entropy, top probability, margin; relabelling planner by expected
  entropy reduction from one more label.
- Evaluation against gold: accuracy with a bootstrap interval, paired difference and exact
  McNemar test against majority vote, reliability table, expected calibration error, Brier score;
  the same without the flagged annotators. Gold labels that no annotator used are scored as wrong
  and reported with a warning.
- Simulation of annotators with given confusion matrices.
- Commands: `aggregate`, `annotators`, `items`, `evaluate`, `plan`, `simulate`, with Markdown and
  JSON output.
- Not included: MACE.
