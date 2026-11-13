"""Comparison with the Dawid-Skene and majority vote of crowd-kit, when it is installed.

Install it with ``pip install -e ".[dev,compare]"``; without it these tests are skipped.
"""

import numpy as np
import pytest

from labelfuse import dawid_skene, majority_vote, simulate, symmetric_confusion

pd = pytest.importorskip("pandas")
aggregation = pytest.importorskip("crowdkit.aggregation")

# crowd-kit raises deprecation warnings of its own with recent pandas
pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def labels():
    confusion = np.stack([symmetric_confusion(a, 3) for a in (0.9, 0.8, 0.7, 0.6, 0.4)])
    return simulate(120, confusion, labels_per_item=4, seed=5)


@pytest.fixture(scope="module")
def frame(labels):
    return pd.DataFrame(list(labels.records()), columns=["task", "worker", "label"])


def test_dawid_skene_matches_crowd_kit(labels, frame):
    theirs = aggregation.DawidSkene(n_iter=500, tol=0).fit(frame)
    ours = dawid_skene(labels, smoothing=0, tol=1e-12, max_iter=5000)
    assert ours.converged

    posterior = theirs.probas_.loc[list(labels.items), list(labels.classes)].to_numpy()
    np.testing.assert_allclose(ours.posterior, posterior, atol=1e-6)
    prior = theirs.priors_.loc[list(labels.classes)].to_numpy()
    np.testing.assert_allclose(ours.class_prior, prior, atol=1e-6)
    for j, annotator in enumerate(labels.annotators):
        for given, name in enumerate(labels.classes):
            row = theirs.errors_.loc[(annotator, name), list(labels.classes)].to_numpy()
            np.testing.assert_allclose(ours.confusion[j, :, given], row, atol=1e-6)
    agreed = [
        ours.labels[i] == theirs.labels_[item]
        for i, item in enumerate(labels.items)
        if not ours.tied[i]
    ]
    assert all(agreed)


def test_vote_shares_match_crowd_kit(labels, frame):
    theirs = aggregation.MajorityVote().fit_predict_proba(frame)
    shares = theirs.loc[list(labels.items), list(labels.classes)].to_numpy()
    np.testing.assert_allclose(majority_vote(labels).posterior, shares, atol=1e-12)
