"""Label sets: annotations in long format, indexed for the aggregators.

A label set is a list of (item, annotator, label) triples. Ids and labels are compared as
strings. Items, annotators and classes are put in a canonical order (numeric when every value is
an integer, lexicographic otherwise) and the triples are sorted, so the row order of the input
never changes a result.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TextIO

import numpy as np

DuplicatePolicy = Literal["error", "keep", "first", "last"]
FileFormat = Literal["jsonl", "csv", "tsv"]

_DUPLICATE_POLICIES = ("error", "keep", "first", "last")
_INTEGER = re.compile(r"-?\d+")
_SUFFIX_FORMATS = {
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".csv": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
}


def _ordered(values: Iterable[str]) -> list[str]:
    """Unique values in canonical order: by integer value when all are integers, else as text."""
    unique = set(values)
    if unique and all(_INTEGER.fullmatch(value) for value in unique):
        return sorted(unique, key=lambda value: (int(value), value))
    return sorted(unique)


def _text(value: Any) -> str | None:
    """Normalise a raw cell to text; ``None`` means the value is missing."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        raise ValueError(f"expected a scalar, got {type(value).__name__}")
    text = str(value).strip()
    return text or None


@dataclass(frozen=True, eq=False)
class LabelSet:
    """Annotations in long format with items, annotators and classes indexed.

    ``item_index``, ``annotator_index`` and ``label_index`` are parallel integer arrays with one
    entry per label; they index into ``items``, ``annotators`` and ``classes``. ``gold`` holds the
    known true label of each item (``None`` where unknown) or is ``None`` when no gold was given.
    Gold labels are kept as text and never change the set of classes.
    """

    items: tuple[str, ...]
    annotators: tuple[str, ...]
    classes: tuple[str, ...]
    item_index: np.ndarray
    annotator_index: np.ndarray
    label_index: np.ndarray
    gold: tuple[str | None, ...] | None = None
    n_duplicates: int = 0

    def __post_init__(self) -> None:
        arrays = {}
        for name, size in (
            ("item_index", len(self.items)),
            ("annotator_index", len(self.annotators)),
            ("label_index", len(self.classes)),
        ):
            array = np.array(getattr(self, name), dtype=np.int64)
            if array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if array.size and (array.min() < 0 or array.max() >= size):
                raise ValueError(f"{name} has values outside 0..{size - 1}")
            array.setflags(write=False)
            arrays[name] = array
            object.__setattr__(self, name, array)
        if len({array.size for array in arrays.values()}) != 1:
            raise ValueError("item_index, annotator_index and label_index differ in length")
        for name in ("items", "annotators", "classes"):
            values = tuple(getattr(self, name))
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains repeated values")
            object.__setattr__(self, name, values)
        if not self.classes:
            raise ValueError("a label set needs at least one class")
        if self.gold is not None:
            gold = tuple(self.gold)
            if len(gold) != len(self.items):
                raise ValueError("gold must have one entry per item")
            object.__setattr__(self, "gold", gold)

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def n_annotators(self) -> int:
        return len(self.annotators)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def n_labels(self) -> int:
        return int(self.item_index.size)

    @property
    def n_gold(self) -> int:
        """Number of items with a known gold label."""
        if self.gold is None:
            return 0
        return sum(value is not None for value in self.gold)

    def vote_counts(self) -> np.ndarray:
        """Number of labels per item and class, shape ``(n_items, n_classes)``."""
        flat = self.item_index * self.n_classes + self.label_index
        counts = np.bincount(flat, minlength=self.n_items * self.n_classes)
        return counts.reshape(self.n_items, self.n_classes)

    def labels_per_item(self) -> np.ndarray:
        return np.bincount(self.item_index, minlength=self.n_items)

    def labels_per_annotator(self) -> np.ndarray:
        return np.bincount(self.annotator_index, minlength=self.n_annotators)

    def records(self) -> Iterator[tuple[str, str, str]]:
        """The labels as (item, annotator, label) triples of text, in canonical order."""
        for i, j, k in zip(self.item_index, self.annotator_index, self.label_index, strict=True):
            yield self.items[i], self.annotators[j], self.classes[k]

    @classmethod
    def from_records(
        cls,
        records: Iterable[Sequence[Any]],
        *,
        gold: Mapping[Any, Any] | None = None,
        classes: Sequence[Any] | None = None,
        duplicates: DuplicatePolicy = "error",
    ) -> LabelSet:
        """Build a label set from (item, annotator, label) triples.

        ``classes`` fixes the set and order of classes; by default it is the observed labels in
        canonical order. ``duplicates`` says what to do when an annotator labels an item more than
        once: ``"error"`` refuses, ``"keep"`` counts every label as a separate judgement,
        ``"first"`` and ``"last"`` keep one of them by input order.
        """
        if duplicates not in _DUPLICATE_POLICIES:
            raise ValueError(f"duplicates must be one of {_DUPLICATE_POLICIES}, got {duplicates!r}")

        kept: list[tuple[str, str, str]] = []
        position: dict[tuple[str, str], int] = {}
        repeated: list[tuple[str, str]] = []
        for number, record in enumerate(records, start=1):
            if isinstance(record, (str, bytes)) or len(record) != 3:
                raise ValueError(f"record {number}: expected (item, annotator, label)")
            try:
                cells = [_text(cell) for cell in record]
            except ValueError as error:
                raise ValueError(f"record {number}: {error}") from None
            if any(cell is None for cell in cells):
                raise ValueError(f"record {number}: item, annotator and label must not be empty")
            item, annotator, label = (str(cell) for cell in cells)
            key = (item, annotator)
            if key not in position:
                position[key] = len(kept)
                kept.append((item, annotator, label))
                continue
            repeated.append(key)
            if duplicates == "keep":
                kept.append((item, annotator, label))
            elif duplicates == "last":
                kept[position[key]] = (item, annotator, label)
        if not kept:
            raise ValueError("no labels")
        if repeated and duplicates == "error":
            shown = ", ".join(f"({item!r}, {annotator!r})" for item, annotator in repeated[:3])
            raise ValueError(
                f"{len(repeated)} repeated (item, annotator) pairs, for example {shown}; "
                "choose duplicates='keep', 'first' or 'last' to accept them"
            )

        items = _ordered(item for item, _, _ in kept)
        annotators = _ordered(annotator for _, annotator, _ in kept)
        if classes is None:
            class_list = _ordered(label for _, _, label in kept)
        else:
            class_list = []
            for value in classes:
                text = _text(value)
                if text is None:
                    raise ValueError("classes must not contain empty values")
                class_list.append(text)
        item_of = {value: index for index, value in enumerate(items)}
        annotator_of = {value: index for index, value in enumerate(annotators)}
        class_of = {value: index for index, value in enumerate(class_list)}
        unknown = sorted({label for _, _, label in kept if label not in class_of})
        if unknown:
            raise ValueError(f"labels outside the given classes: {unknown[:5]}")

        item_index = np.array([item_of[item] for item, _, _ in kept], dtype=np.int64)
        annotator_index = np.array([annotator_of[a] for _, a, _ in kept], dtype=np.int64)
        label_index = np.array([class_of[label] for _, _, label in kept], dtype=np.int64)
        order = np.lexsort((label_index, annotator_index, item_index))

        gold_column: tuple[str | None, ...] | None = None
        if gold is not None:
            known: dict[str, str] = {}
            for raw_item, raw_label in gold.items():
                item_text, label_text = _text(raw_item), _text(raw_label)
                if item_text is not None and label_text is not None:
                    known[item_text] = label_text
            if known and not any(item in item_of for item in known):
                raise ValueError("no gold item id matches a labelled item")
            gold_column = tuple(known.get(item) for item in items)

        return cls(
            items=tuple(items),
            annotators=tuple(annotators),
            classes=tuple(class_list),
            item_index=item_index[order],
            annotator_index=annotator_index[order],
            label_index=label_index[order],
            gold=gold_column,
            n_duplicates=len(repeated),
        )


def _infer_format(path: Path) -> str:
    try:
        return _SUFFIX_FORMATS[path.suffix.lower()]
    except KeyError:
        raise ValueError(
            f"cannot tell the format of {path.name!r} from its suffix; pass fmt='jsonl', 'csv' "
            "or 'tsv'"
        ) from None


def _jsonl_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{number}: not valid JSON ({error.msg})") from None
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{number}: expected a JSON object")
            yield number, row


def _delimited_rows(
    path: Path, delimiter: str, required: Sequence[str]
) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        columns = list(reader.fieldnames or [])
        for column in required:
            if column not in columns:
                raise ValueError(f"{path.name}: no column {column!r}; columns are {columns}")
        for row in reader:
            yield reader.line_num, row


def read_labels(
    path: str | Path,
    *,
    item_col: str = "item",
    annotator_col: str = "annotator",
    label_col: str = "label",
    gold_col: str | None = None,
    fmt: FileFormat | None = None,
    duplicates: DuplicatePolicy = "error",
    classes: Sequence[Any] | None = None,
) -> LabelSet:
    """Read long-format labels from a JSONL, CSV or TSV file.

    Each row holds one label: an item id, an annotator id and the label. With ``gold_col`` the
    rows may also carry the true label of the item; it can be empty, and it only has to appear on
    one row of an item, but two different values for the same item are an error.
    """
    path = Path(path)
    kind = fmt or _infer_format(path)
    needed = [item_col, annotator_col, label_col] + ([gold_col] if gold_col else [])
    if kind == "jsonl":
        rows = _jsonl_rows(path)
    elif kind in ("csv", "tsv"):
        rows = _delimited_rows(path, "," if kind == "csv" else "\t", needed)
    else:
        raise ValueError(f"unknown format {kind!r}; use 'jsonl', 'csv' or 'tsv'")

    records: list[tuple[str, str, str]] = []
    gold: dict[str, str] = {}
    for number, row in rows:
        where = f"{path.name}:{number}"
        cells: list[str] = []
        for column in (item_col, annotator_col, label_col):
            try:
                cell = _text(row.get(column))
            except ValueError as error:
                raise ValueError(f"{where}: column {column!r}: {error}") from None
            if cell is None:
                raise ValueError(f"{where}: missing value for {column!r}")
            cells.append(cell)
        item, annotator, label = cells
        records.append((item, annotator, label))
        if gold_col:
            try:
                truth = _text(row.get(gold_col))
            except ValueError as error:
                raise ValueError(f"{where}: column {gold_col!r}: {error}") from None
            if truth is not None and gold.setdefault(item, truth) != truth:
                raise ValueError(
                    f"{where}: item {item!r} has two gold labels, {gold[item]!r} and {truth!r}"
                )
    if not records:
        raise ValueError(f"{path.name}: no labels")
    return LabelSet.from_records(
        records,
        gold=gold if gold_col else None,
        classes=classes,
        duplicates=duplicates,
    )


def dump_labels(
    labels: LabelSet,
    handle: TextIO,
    *,
    fmt: FileFormat = "jsonl",
    item_col: str = "item",
    annotator_col: str = "annotator",
    label_col: str = "label",
    gold_col: str = "gold",
) -> None:
    """Write a label set in long format to an open text file.

    The gold column is added when the set has gold labels.
    """
    if fmt not in ("jsonl", "csv", "tsv"):
        raise ValueError(f"unknown format {fmt!r}; use 'jsonl', 'csv' or 'tsv'")
    truth = dict(zip(labels.items, labels.gold, strict=True)) if labels.gold else None
    columns = [item_col, annotator_col, label_col] + ([gold_col] if truth is not None else [])
    writer = None
    if fmt != "jsonl":
        writer = csv.writer(handle, delimiter="," if fmt == "csv" else "\t", lineterminator="\n")
        writer.writerow(columns)
    for item, annotator, label in labels.records():
        cells: list[str | None] = [item, annotator, label]
        if truth is not None:
            cells.append(truth[item])
        if writer is None:
            row = dict(zip(columns, cells, strict=True))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            writer.writerow(["" if cell is None else cell for cell in cells])


def write_labels(
    labels: LabelSet,
    path: str | Path,
    *,
    fmt: FileFormat | None = None,
    item_col: str = "item",
    annotator_col: str = "annotator",
    label_col: str = "label",
    gold_col: str = "gold",
) -> None:
    """Write a label set in long format to ``path`` (format from the suffix unless given)."""
    path = Path(path)
    kind = fmt or _infer_format(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        dump_labels(
            labels,
            handle,
            fmt=kind,  # type: ignore[arg-type]
            item_col=item_col,
            annotator_col=annotator_col,
            label_col=label_col,
            gold_col=gold_col,
        )
