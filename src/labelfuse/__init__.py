"""labelfuse: aggregate noisy labels from several annotators and audit the annotators."""

from labelfuse.data import LabelSet, dump_labels, read_labels, write_labels
from labelfuse.dawid_skene import DawidSkeneResult, dawid_skene, dawid_skene_posterior
from labelfuse.evaluate import (
    MethodEvaluation,
    ReliabilityBin,
    bootstrap_mean_interval,
    brier_score,
    evaluate,
    expected_calibration_error,
    mcnemar_exact,
    reliability_table,
    tie_credit,
)
from labelfuse.majority import majority_vote, vote_shares
from labelfuse.result import AggregationResult, resolve_labels
from labelfuse.simulate import simulate, symmetric_confusion

__version__ = "0.1.0"

__all__ = [
    "AggregationResult",
    "DawidSkeneResult",
    "LabelSet",
    "MethodEvaluation",
    "ReliabilityBin",
    "__version__",
    "bootstrap_mean_interval",
    "brier_score",
    "dawid_skene",
    "dawid_skene_posterior",
    "dump_labels",
    "evaluate",
    "expected_calibration_error",
    "majority_vote",
    "mcnemar_exact",
    "read_labels",
    "reliability_table",
    "resolve_labels",
    "simulate",
    "symmetric_confusion",
    "tie_credit",
    "vote_shares",
    "write_labels",
]
