"""labelfuse: aggregate noisy labels from several annotators and audit the annotators."""

from labelfuse.data import LabelSet, dump_labels, read_labels, write_labels
from labelfuse.majority import majority_vote, vote_shares
from labelfuse.result import AggregationResult, resolve_labels

__version__ = "0.1.0"

__all__ = [
    "AggregationResult",
    "LabelSet",
    "__version__",
    "dump_labels",
    "majority_vote",
    "read_labels",
    "resolve_labels",
    "vote_shares",
    "write_labels",
]
