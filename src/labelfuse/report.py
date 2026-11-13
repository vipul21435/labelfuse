"""Markdown and JSON reports."""

from __future__ import annotations

from typing import Any

import numpy as np

from labelfuse.data import LabelSet
from labelfuse.dawid_skene import DawidSkeneResult
from labelfuse.result import AggregationResult


def _cell(text: object) -> str:
    """Text made safe for a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _table(header: list[str], rows: list[list[str]], align: str) -> list[str]:
    rule = {"l": "---", "r": "---:"}
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(rule[a] for a in align) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _fit_summary(result: AggregationResult) -> dict[str, Any] | None:
    if not isinstance(result, DawidSkeneResult):
        return None
    return {
        "smoothing": result.smoothing,
        "n_iter": result.n_iter,
        "converged": result.converged,
        "log_likelihood": result.log_likelihood,
        "log_likelihood_trace": list(result.log_likelihood_trace),
        "objective_trace": list(result.objective_trace),
        "restart_objectives": list(result.restart_objectives),
        "best_restart": result.best_restart,
    }


def aggregate_document(labels: LabelSet, result: AggregationResult) -> dict[str, Any]:
    """Everything an aggregation produced, as plain data ready for ``json.dumps``."""
    document: dict[str, Any] = {
        "method": result.method,
        "n_items": labels.n_items,
        "n_annotators": labels.n_annotators,
        "n_labels": labels.n_labels,
        "n_tied": int(result.tied.sum()),
        "classes": list(result.classes),
    }
    fit = _fit_summary(result)
    if fit is not None:
        document["fit"] = fit
    if isinstance(result, DawidSkeneResult):
        document["class_prior"] = {
            name: float(value)
            for name, value in zip(result.classes, result.class_prior, strict=True)
        }
        document["annotators"] = [
            {"annotator": name, "confusion": result.confusion[index].tolist()}
            for index, name in enumerate(result.annotators)
        ]
    document["items"] = result.to_records()
    return document


def aggregate_markdown(labels: LabelSet, result: AggregationResult, *, top: int = 10) -> str:
    """A short Markdown report: the data, the fit, labels per class, least certain items."""
    per_item = labels.labels_per_item()
    lines = [
        "# Aggregated labels",
        "",
        f"- items: {labels.n_items}",
        f"- annotators: {labels.n_annotators}",
        f"- labels: {labels.n_labels} (per item: min {per_item.min()}, "
        f"median {np.median(per_item):g}, max {per_item.max()})",
        f"- classes: {', '.join(labels.classes)}",
    ]
    if isinstance(result, DawidSkeneResult):
        state = "converged" if result.converged else "not converged"
        lines.append(f"- method: {result.method}, smoothing {result.smoothing:g}")
        lines.append(
            f"- EM: {result.n_iter} iterations, {state}, log-likelihood {result.log_likelihood:.4f}"
        )
        if len(result.restart_objectives) > 1:
            lines.append(
                f"- starts: {len(result.restart_objectives)}, objective from "
                f"{min(result.restart_objectives):.4f} to {max(result.restart_objectives):.4f}, "
                f"returned start {result.best_restart}"
            )
    else:
        lines.append(f"- method: {result.method}")
    lines.append(f"- tied items: {int(result.tied.sum())}")

    assigned = [label for label in result.labels if label is not None]
    rows = []
    for name in result.classes:
        count = assigned.count(name)
        rows.append([_cell(name), str(count), f"{count / labels.n_items:.3f}"])
    unlabelled = labels.n_items - len(assigned)
    if unlabelled:
        rows.append(["(no label)", str(unlabelled), f"{unlabelled / labels.n_items:.3f}"])
    lines += ["", "## Items per class", ""]
    lines += _table(["class", "items", "share"], rows, "lrr")

    shown = min(max(top, 0), labels.n_items)
    if shown:
        votes = labels.vote_counts()
        order = np.argsort(result.confidence, kind="stable")[:shown]
        rows = []
        for index in order:
            tally = ", ".join(
                f"{_cell(name)} {count}"
                for name, count in zip(labels.classes, votes[index], strict=True)
                if count
            )
            label = result.labels[index]
            rows.append(
                [
                    _cell(result.items[index]),
                    "(tie)" if label is None else _cell(label),
                    f"{result.confidence[index]:.3f}",
                    tally,
                ]
            )
        lines += ["", f"## Least certain items ({shown} of {labels.n_items})", ""]
        certainty = "vote share" if result.method == "majority" else "probability"
        lines += _table(["item", "label", certainty, "votes"], rows, "llrl")
    return "\n".join(lines) + "\n"
