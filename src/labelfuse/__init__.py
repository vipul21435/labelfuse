"""labelfuse: aggregate noisy labels from several annotators and audit the annotators."""

from labelfuse.data import LabelSet, dump_labels, read_labels, write_labels

__version__ = "0.1.0"

__all__ = ["LabelSet", "__version__", "dump_labels", "read_labels", "write_labels"]
