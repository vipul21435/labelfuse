"""What an aggregator returns: a posterior per item and the label read from it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

TiePolicy = Literal["abstain", "first", "random"]

_TIE_POLICIES = ("abstain", "first", "random")

#: Two classes of an item count as tied when their posteriors differ by no more than this.
TIE_TOLERANCE = 1e-9


def resolve_labels(
    posterior: np.ndarray,
    classes: tuple[str, ...],
    *,
    tie: TiePolicy = "abstain",
    seed: int = 0,
) -> tuple[tuple[str | None, ...], np.ndarray]:
    """Read one label per item from a posterior, with an explicit policy for ties.

    A tie is two or more classes sharing the largest posterior of an item (within
    ``TIE_TOLERANCE``). ``"abstain"`` gives such an item no label (``None``), ``"first"`` gives it
    the tied class that comes first in ``classes``, and ``"random"`` draws one of the tied classes
    uniformly with ``seed``. Returns the labels and a boolean array marking the tied items.
    """
    if tie not in _TIE_POLICIES:
        raise ValueError(f"tie must be one of {_TIE_POLICIES}, got {tie!r}")
    posterior = np.asarray(posterior, dtype=float)
    top = posterior.max(axis=1, keepdims=True)
    at_top = posterior >= top - TIE_TOLERANCE
    n_top = at_top.sum(axis=1)
    tied = n_top > 1
    choice = at_top.argmax(axis=1)
    if tie == "random" and tied.any():
        rng = np.random.default_rng(seed)
        pick = np.floor(rng.random(posterior.shape[0]) * n_top).astype(np.int64)
        rank = np.cumsum(at_top, axis=1) - 1
        choice = (at_top & (rank == pick[:, None])).argmax(axis=1)
    labels: list[str | None] = [classes[index] for index in choice]
    if tie == "abstain":
        for index in np.flatnonzero(tied):
            labels[index] = None
    return tuple(labels), tied


@dataclass(frozen=True, eq=False)
class AggregationResult:
    """One posterior row per item, in the order of ``items``; columns follow ``classes``."""

    method: str
    items: tuple[str, ...]
    classes: tuple[str, ...]
    posterior: np.ndarray
    labels: tuple[str | None, ...]
    tied: np.ndarray

    @property
    def confidence(self) -> np.ndarray:
        """Largest posterior of each item."""
        return self.posterior.max(axis=1)

    def to_records(self) -> list[dict[str, Any]]:
        """One plain dict per item: the label, its probability and the full posterior."""
        records = []
        for index, item in enumerate(self.items):
            row = self.posterior[index]
            records.append(
                {
                    "item": item,
                    "label": self.labels[index],
                    "probability": float(row.max()),
                    "tied": bool(self.tied[index]),
                    "posterior": {
                        name: float(value) for name, value in zip(self.classes, row, strict=True)
                    },
                }
            )
        return records
