"""labelfuse: aggregate noisy labels from several annotators and audit the annotators."""

from labelfuse.aggregators import METHODS, aggregate, applicable_methods
from labelfuse.audit import (
    AnnotatorReport,
    ItemReport,
    PlanRow,
    audit_annotators,
    audit_items,
    average_confusion,
    expected_entropy_reduction,
    plan_relabelling,
    posterior_entropy,
    spammer_score,
    wilson_interval,
)
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
from labelfuse.glad import GladResult, glad
from labelfuse.majority import majority_vote, vote_shares
from labelfuse.one_coin import OneCoinResult, one_coin
from labelfuse.result import AggregationResult, resolve_labels
from labelfuse.simulate import simulate, symmetric_confusion
from labelfuse.weighted import WeightedVoteResult, leave_one_out_agreement, weighted_vote

__version__ = "0.1.0"

__all__ = [
    "METHODS",
    "AggregationResult",
    "AnnotatorReport",
    "DawidSkeneResult",
    "GladResult",
    "ItemReport",
    "LabelSet",
    "MethodEvaluation",
    "OneCoinResult",
    "PlanRow",
    "ReliabilityBin",
    "WeightedVoteResult",
    "__version__",
    "aggregate",
    "applicable_methods",
    "audit_annotators",
    "audit_items",
    "average_confusion",
    "bootstrap_mean_interval",
    "brier_score",
    "dawid_skene",
    "dawid_skene_posterior",
    "dump_labels",
    "evaluate",
    "expected_calibration_error",
    "expected_entropy_reduction",
    "glad",
    "leave_one_out_agreement",
    "majority_vote",
    "mcnemar_exact",
    "one_coin",
    "plan_relabelling",
    "posterior_entropy",
    "read_labels",
    "reliability_table",
    "resolve_labels",
    "simulate",
    "spammer_score",
    "symmetric_confusion",
    "tie_credit",
    "vote_shares",
    "weighted_vote",
    "wilson_interval",
    "write_labels",
]
