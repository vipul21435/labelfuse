"""Markdown and JSON reports."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from labelfuse.audit import AnnotatorReport, ItemReport, PlanRow
from labelfuse.data import LabelSet
from labelfuse.dawid_skene import DawidSkeneResult
from labelfuse.evaluate import MethodEvaluation, gold_outside_classes
from labelfuse.glad import GladResult
from labelfuse.result import AggregationResult
from labelfuse.weighted import WeightedVoteResult


def _cell(text: object) -> str:
    """Text made safe for a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _table(header: list[str], rows: list[list[str]], align: str) -> list[str]:
    rule = {"l": "---", "r": "---:"}
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(rule[a] for a in align) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _span(interval: tuple[float, float], *, signed: bool = False) -> str:
    form = "{:+.3f} to {:+.3f}" if signed else "{:.3f} to {:.3f}"
    return form.format(*interval)


def _p_value(value: float) -> str:
    return f"{value:.3f}" if value >= 0.001 else f"{value:.1e}"


def _tally(classes: tuple[str, ...], counts: np.ndarray) -> str:
    return ", ".join(
        f"{_cell(name)} {count}" for name, count in zip(classes, counts, strict=True) if count
    )


def _data_lines(labels: LabelSet) -> list[str]:
    per_item = labels.labels_per_item()
    return [
        f"- items: {labels.n_items}",
        f"- annotators: {labels.n_annotators}",
        f"- labels: {labels.n_labels} (per item: min {per_item.min()}, "
        f"median {np.median(per_item):g}, max {per_item.max()})",
        f"- classes: {', '.join(labels.classes)}",
    ]


def _fit_lines(result: AggregationResult) -> list[str]:
    if isinstance(result, DawidSkeneResult):
        state = "converged" if result.converged else "not converged"
        lines = [
            f"- method: {result.method}, smoothing {result.smoothing:g}",
            f"- EM: {result.n_iter} iterations, {state}, "
            f"log-likelihood {result.log_likelihood:.4f}",
        ]
        if len(result.restart_objectives) > 1:
            lines.append(
                f"- starts: {len(result.restart_objectives)}, objective from "
                f"{min(result.restart_objectives):.4f} to {max(result.restart_objectives):.4f}, "
                f"returned start {result.best_restart}"
            )
        return lines
    if isinstance(result, GladResult):
        state = "converged" if result.converged else "not converged"
        return [
            f"- method: {result.method}",
            f"- EM: {result.n_iter} iterations, {state}, "
            f"log-likelihood {result.log_likelihood:.4f}",
        ]
    return [f"- method: {result.method}"]


def _fit_document(result: AggregationResult) -> dict[str, Any]:
    document: dict[str, Any] = {}
    if isinstance(result, DawidSkeneResult):
        document["fit"] = {
            "smoothing": result.smoothing,
            "n_iter": result.n_iter,
            "converged": result.converged,
            "log_likelihood": result.log_likelihood,
            "log_likelihood_trace": list(result.log_likelihood_trace),
            "objective_trace": list(result.objective_trace),
            "restart_objectives": list(result.restart_objectives),
            "best_restart": result.best_restart,
        }
        document["class_prior"] = {
            name: float(value)
            for name, value in zip(result.classes, result.class_prior, strict=True)
        }
        document["annotators"] = [
            {"annotator": name, "confusion": result.confusion[index].tolist()}
            for index, name in enumerate(result.annotators)
        ]
    elif isinstance(result, GladResult):
        document["fit"] = {
            "n_iter": result.n_iter,
            "converged": result.converged,
            "log_likelihood": result.log_likelihood,
            "objective_trace": list(result.objective_trace),
        }
        document["class_prior"] = {
            name: float(value)
            for name, value in zip(result.classes, result.class_prior, strict=True)
        }
        document["annotators"] = [
            {"annotator": name, "ability": float(value)}
            for name, value in zip(result.annotators, result.ability, strict=True)
        ]
        document["difficulty"] = {
            name: float(value) for name, value in zip(result.items, result.difficulty, strict=True)
        }
    elif isinstance(result, WeightedVoteResult):
        document["annotators"] = [
            {"annotator": name, "weight": float(value)}
            for name, value in zip(result.annotators, result.weights, strict=True)
        ]
    return document


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
    document.update(_fit_document(result))
    document["items"] = result.to_records()
    return document


def aggregate_markdown(labels: LabelSet, result: AggregationResult, *, top: int = 10) -> str:
    """A short Markdown report: the data, the fit, labels per class, least certain items."""
    lines = ["# Aggregated labels", "", *_data_lines(labels), *_fit_lines(result)]
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
            label = result.labels[index]
            rows.append(
                [
                    _cell(result.items[index]),
                    "(tie)" if label is None else _cell(label),
                    f"{result.confidence[index]:.3f}",
                    _tally(labels.classes, votes[index]),
                ]
            )
        lines += ["", f"## Least certain items ({shown} of {labels.n_items})", ""]
        certainty = "vote share" if result.method in ("majority", "weighted") else "probability"
        lines += _table(["item", "label", certainty, "votes"], rows, "llrl")
    return "\n".join(lines) + "\n"


def _evaluation_table(rows: list[MethodEvaluation], baseline: str) -> list[str]:
    body = []
    for row in rows:
        versus = ["-", "-", "-", "-"]
        if row.method != baseline:
            versus = [
                f"{row.difference:+.3f}",
                _span(row.difference_interval, signed=True),
                f"{row.only_method} / {row.only_baseline}",
                _p_value(row.mcnemar_p),
            ]
        body.append(
            [
                _cell(row.method),
                f"{row.accuracy:.3f}",
                _span(row.interval),
                str(row.n_tied),
                *versus,
                f"{row.ece:.3f}",
                f"{row.brier:.3f}",
            ]
        )
    header = [
        "method",
        "accuracy",
        "95 % interval",
        "ties",
        f"vs {baseline}",
        "95 % interval",
        "only method / only baseline right",
        "McNemar p",
        "ECE",
        "Brier",
    ]
    return _table(header, body, "lrlrrllrrr")


def evaluation_document(
    labels: LabelSet,
    rows: list[MethodEvaluation],
    *,
    baseline: str,
    n_boot: int,
    seed: int,
    without_flagged: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The evaluation as plain data ready for ``json.dumps``."""
    document: dict[str, Any] = {
        "n_items": labels.n_items,
        "n_gold": labels.n_gold,
        "n_gold_outside_classes": gold_outside_classes(labels)[0],
        "n_annotators": labels.n_annotators,
        "n_labels": labels.n_labels,
        "baseline": baseline,
        "bootstrap": n_boot,
        "seed": seed,
        "methods": [asdict(row) for row in rows],
    }
    if without_flagged is not None:
        rest = {key: value for key, value in without_flagged.items() if key != "rows"}
        rest["methods"] = [asdict(row) for row in without_flagged["rows"]]
        document["without_flagged"] = rest
    return document


def evaluation_markdown(
    labels: LabelSet,
    rows: list[MethodEvaluation],
    *,
    baseline: str,
    n_boot: int,
    seed: int,
    reliability: bool = False,
    without_flagged: dict[str, Any] | None = None,
) -> str:
    """Accuracy of every method against gold, the comparison with the baseline, calibration."""
    lines = [
        "# Evaluation against gold",
        "",
        f"- items with gold: {rows[0].n_items} of {labels.n_items}",
        f"- annotators: {labels.n_annotators}, labels: {labels.n_labels}, "
        f"classes: {', '.join(labels.classes)}",
        f"- intervals: 95 % percentile bootstrap over items, {n_boot} resamples, seed {seed}",
        "- accuracy gives 1/t credit when the gold class is one of t tied top classes; the "
        f"McNemar counts break ties at random with seed {seed}",
    ]
    n_outside, strays = gold_outside_classes(labels)
    if n_outside:
        shown = ", ".join(_cell(value) for value in strays[:5]) + (
            ", ..." if len(strays) > 5 else ""
        )
        lines.append(
            f"- gold labels that no annotator used: {shown} on {n_outside} of {rows[0].n_items} "
            "items with gold, scored as wrong"
        )
    lines += ["", *_evaluation_table(rows, baseline)]
    if without_flagged is not None:
        flagged = without_flagged["flagged"]
        dropped = without_flagged["n_labels_dropped"]
        lines += [
            "",
            "## Without flagged annotators",
            "",
            f"- flagged by the {without_flagged['audit_method']} audit: {len(flagged)} of "
            f"{labels.n_annotators} annotators ({len(flagged) / labels.n_annotators:.1%}; "
            f"random {without_flagged['n_random']}, constant {without_flagged['n_constant']}), "
            f"who gave {dropped} of {labels.n_labels} labels ({dropped / labels.n_labels:.1%})",
        ]
        if flagged:
            before = {row.method: row.accuracy for row in rows}
            lines += ["", *_evaluation_table(without_flagged["rows"], baseline), ""]
            change = [
                [
                    _cell(row.method),
                    f"{before[row.method]:.3f}",
                    f"{row.accuracy:.3f}",
                    f"{row.accuracy - before[row.method]:+.3f}",
                ]
                for row in without_flagged["rows"]
            ]
            header = ["method", "all annotators", "without flagged", "change"]
            lines += _table(header, change, "lrrr")
    if reliability:
        for row in rows:
            lines += ["", f"## Reliability of {row.method}", ""]
            body = [
                [
                    f"{entry.lower:.1f} to {entry.upper:.1f}",
                    str(entry.count),
                    f"{entry.mean_confidence:.3f}",
                    f"{entry.accuracy:.3f}",
                    f"{entry.accuracy - entry.mean_confidence:+.3f}",
                ]
                for entry in row.reliability
            ]
            header = ["top probability", "items", "mean probability", "accuracy", "gap"]
            lines += _table(header, body, "lrrrr")
    return "\n".join(lines) + "\n"


def annotators_document(
    labels: LabelSet, result: AggregationResult, reports: list[AnnotatorReport]
) -> dict[str, Any]:
    """The annotator audit as plain data ready for ``json.dumps``."""
    records = []
    for report in reports:
        record = asdict(report)
        record["confusion"] = report.confusion.tolist()
        records.append(record)
    flagged = [report for report in reports if report.flag]
    return {
        "method": result.method,
        "n_annotators": labels.n_annotators,
        "n_labels": labels.n_labels,
        "classes": list(labels.classes),
        "n_random": sum(report.flag == "random" for report in reports),
        "n_constant": sum(report.flag == "constant" for report in reports),
        "n_labels_flagged": sum(report.n_labels for report in flagged),
        "annotators": records,
    }


def annotators_markdown(
    labels: LabelSet,
    result: AggregationResult,
    reports: list[AnnotatorReport],
    *,
    spam_threshold: float,
    min_labels: int,
    top: int = 20,
) -> str:
    """The annotator audit, least informative annotators first."""
    flagged = [report for report in reports if report.flag]
    n_random = sum(report.flag == "random" for report in reports)
    lines = [
        "# Annotator audit",
        "",
        f"- annotators: {labels.n_annotators}, labels: {labels.n_labels}, "
        f"classes: {', '.join(labels.classes)}",
        *_fit_lines(result),
        f"- flagged: {len(flagged)} of {labels.n_annotators} annotators "
        f"({len(flagged) / labels.n_annotators:.1%}; random {n_random}, "
        f"constant {len(flagged) - n_random}), who gave "
        f"{sum(report.n_labels for report in flagged)} of {labels.n_labels} labels",
        f"- rule: at least {min_labels} labels and the upper end of the 95 % bootstrap interval "
        f"of the spammer score below {spam_threshold:g}",
    ]
    ordered = sorted(reports, key=lambda report: (report.spammer_interval[1], report.annotator))
    shown = ordered if top <= 0 else ordered[:top]
    with_gold = any(report.gold_accuracy is not None for report in reports)
    body = []
    for report in shown:
        row = [
            _cell(report.annotator),
            str(report.n_labels),
            f"{report.accuracy:.3f}",
            _span(report.accuracy_interval),
        ]
        if with_gold:
            if report.gold_accuracy is None or report.gold_interval is None:
                row += ["-", "-"]
            else:
                row += [
                    f"{report.gold_accuracy:.3f} ({report.n_gold})",
                    _span(report.gold_interval),
                ]
        row += [
            f"{report.spammer_score:.3f}",
            _span(report.spammer_interval),
            f"{_cell(report.bias_class)} {report.bias:+.3f}",
            f"{_cell(report.top_label)} {report.top_label_share:.2f}",
            f"{report.agreement:.3f}",
            report.flag or "",
        ]
        body.append(row)
    header = ["annotator", "labels", "accuracy", "95 % interval"]
    align = "lrrl"
    if with_gold:
        header += ["gold accuracy (items)", "95 % interval"]
        align += "rl"
    header += ["spammer score", "95 % interval", "bias", "top label", "agreement", "flag"]
    align += "rlllrl"
    lines += ["", f"## Least informative annotators ({len(shown)} of {len(reports)})", ""]
    lines += _table(header, body, align)
    return "\n".join(lines) + "\n"


def items_document(
    labels: LabelSet, result: AggregationResult, reports: list[ItemReport], *, certain_at: float
) -> dict[str, Any]:
    """The item audit as plain data ready for ``json.dumps``."""
    return {
        "method": result.method,
        "n_items": labels.n_items,
        "certain_at": certain_at,
        "n_uncertain": sum(report.uncertain for report in reports),
        "mean_entropy": float(np.mean([report.entropy for report in reports])),
        "items": [asdict(report) for report in reports],
    }


def items_markdown(
    labels: LabelSet,
    result: AggregationResult,
    reports: list[ItemReport],
    *,
    certain_at: float,
    top: int = 20,
) -> str:
    """The item audit, most uncertain items first."""
    n_uncertain = sum(report.uncertain for report in reports)
    lines = [
        "# Item audit",
        "",
        *_data_lines(labels),
        *_fit_lines(result),
        f"- uncertain items (top probability below {certain_at:g}): {n_uncertain} of "
        f"{labels.n_items} ({n_uncertain / labels.n_items:.1%})",
        f"- mean posterior entropy: {np.mean([report.entropy for report in reports]):.3f} bits",
    ]
    ordered = sorted(reports, key=lambda report: -report.entropy)
    shown = ordered if top <= 0 else ordered[:top]
    body = [
        [
            _cell(report.item),
            "(tie)" if report.label is None else _cell(report.label),
            f"{report.probability:.3f}",
            f"{report.entropy:.3f}",
            f"{report.margin:.3f}",
            str(report.n_labels),
            ", ".join(f"{_cell(name)} {count}" for name, count in report.votes.items()),
        ]
        for report in shown
    ]
    lines += ["", f"## Most uncertain items ({len(shown)} of {len(reports)})", ""]
    header = ["item", "label", "probability", "entropy (bits)", "margin", "labels", "votes"]
    lines += _table(header, body, "llrrrrl")
    return "\n".join(lines) + "\n"


def plan_document(result: AggregationResult, rows: list[PlanRow], *, budget: int) -> dict[str, Any]:
    """The relabelling plan as plain data ready for ``json.dumps``."""
    return {
        "method": result.method,
        "budget": budget,
        "n_planned": len(rows),
        "expected_reduction": float(sum(row.expected_reduction for row in rows)),
        "items": [asdict(row) for row in rows],
    }


def plan_markdown(
    result: AggregationResult, rows: list[PlanRow], *, budget: int, top: int = 20
) -> str:
    """Which items to send out for one more label each."""
    total = sum(row.expected_reduction for row in rows)
    lines = [
        "# Relabelling plan",
        "",
        *_fit_lines(result),
        f"- budget: {budget} labels, planned: {len(rows)} (one more label for each listed item)",
        f"- expected drop in posterior entropy: {total:.3f} bits in total, from one more label "
        "by an average annotator on each of them",
    ]
    shown = rows if top <= 0 else rows[:top]
    body = [
        [
            _cell(row.item),
            "(tie)" if row.label is None else _cell(row.label),
            f"{row.probability:.3f}",
            f"{row.entropy:.3f}",
            f"{row.expected_reduction:.3f}",
            str(row.n_labels),
        ]
        for row in shown
    ]
    lines += ["", f"## Items to relabel ({len(shown)} of {len(rows)})", ""]
    header = ["item", "label", "probability", "entropy (bits)", "expected drop (bits)", "labels"]
    lines += _table(header, body, "llrrrr")
    return "\n".join(lines) + "\n"
