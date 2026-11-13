"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from labelfuse import __version__
from labelfuse.aggregators import METHODS, aggregate, applicable_methods
from labelfuse.audit import audit_annotators, audit_items, plan_relabelling
from labelfuse.data import LabelSet, dump_labels, read_labels, write_labels
from labelfuse.dawid_skene import DEFAULT_MAX_ITER, DEFAULT_SMOOTHING, DEFAULT_TOL
from labelfuse.evaluate import GoldOutsideClassesWarning, evaluate
from labelfuse.report import (
    aggregate_document,
    aggregate_markdown,
    annotators_document,
    annotators_markdown,
    evaluation_document,
    evaluation_markdown,
    items_document,
    items_markdown,
    plan_document,
    plan_markdown,
)
from labelfuse.result import AggregationResult
from labelfuse.simulate import simulate, symmetric_confusion


def _add_input_options(parser: argparse.ArgumentParser, *, gold: str | None) -> None:
    parser.add_argument("labels", help="long-format labels: a .jsonl, .csv or .tsv file")
    group = parser.add_argument_group("input")
    group.add_argument("--item-col", default="item", help="item id column (default: item)")
    group.add_argument(
        "--annotator-col", default="annotator", help="annotator id column (default: annotator)"
    )
    group.add_argument("--label-col", default="label", help="label column (default: label)")
    group.add_argument(
        "--gold-col",
        default=gold,
        help="column with the true label of an item, where known"
        + (f" (default: {gold})" if gold else " (default: none)"),
    )
    group.add_argument(
        "--input-format",
        choices=("jsonl", "csv", "tsv"),
        help="file format (default: from the file suffix)",
    )
    group.add_argument(
        "--duplicates",
        choices=("error", "keep", "first", "last"),
        default="error",
        help="what to do when an annotator labels an item more than once (default: error)",
    )


def _add_model_options(parser: argparse.ArgumentParser, *, method: bool = True) -> None:
    group = parser.add_argument_group("aggregation")
    if method:
        group.add_argument(
            "--method", choices=METHODS, default="dawid-skene", help="default: dawid-skene"
        )
    group.add_argument(
        "--tie",
        choices=("abstain", "first", "random"),
        default="abstain",
        help="label of an item whose top classes are tied: none, the first class in class "
        "order, or a seeded random one (default: abstain)",
    )
    group.add_argument("--seed", type=int, default=0, help="seed for every random draw")
    group.add_argument(
        "--smoothing",
        type=float,
        default=DEFAULT_SMOOTHING,
        help="dawid-skene and one-coin: pseudo-count added to every confusion cell and prior "
        f"class (default: {DEFAULT_SMOOTHING:g})",
    )
    group.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help="dawid-skene and one-coin: most EM iterations per start "
        f"(default: {DEFAULT_MAX_ITER})",
    )
    group.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help="dawid-skene and one-coin: stop once every posterior entry moves by less than "
        f"this (default: {DEFAULT_TOL:g})",
    )
    group.add_argument(
        "--restarts",
        type=int,
        default=1,
        help="dawid-skene and one-coin: number of EM starts; the first is the vote shares, the "
        "others add noise (default: 1)",
    )


def _add_output_options(
    parser: argparse.ArgumentParser, *, top: int | None, what: str = ""
) -> None:
    group = parser.add_argument_group("output")
    group.add_argument("--format", choices=("markdown", "json"), default="markdown")
    if top is not None:
        group.add_argument(
            "--top", type=int, default=top, help=f"{what} to list, 0 for all (default: {top})"
        )
    group.add_argument("--output", "-o", help="write the report to this file instead of stdout")


def _add_audit_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("annotator flags")
    group.add_argument(
        "--spam-threshold",
        type=float,
        default=0.05,
        help="flag an annotator when the upper end of the interval of the spammer score is "
        "below this (default: 0.05)",
    )
    group.add_argument(
        "--constant-share",
        type=float,
        default=0.95,
        help="a flagged annotator is 'constant' when one class is at least this share of "
        "their labels, 'random' otherwise (default: 0.95)",
    )
    group.add_argument(
        "--min-labels", type=int, default=10, help="never flag below this many labels (default: 10)"
    )
    group.add_argument(
        "--audit-bootstrap",
        type=int,
        default=200,
        help="bootstrap resamples for the spammer score interval (default: 200)",
    )


def _read(args: argparse.Namespace) -> LabelSet:
    return read_labels(
        args.labels,
        item_col=args.item_col,
        annotator_col=args.annotator_col,
        label_col=args.label_col,
        gold_col=args.gold_col,
        fmt=args.input_format,
        duplicates=args.duplicates,
    )


def _aggregate(labels: LabelSet, args: argparse.Namespace, method: str) -> AggregationResult:
    return aggregate(
        labels,
        method,
        tie=args.tie,
        seed=args.seed,
        smoothing=args.smoothing,
        max_iter=args.max_iter,
        tol=args.tol,
        n_restarts=args.restarts,
    )


def _audit(labels: LabelSet, result: AggregationResult, args: argparse.Namespace) -> list[Any]:
    return audit_annotators(
        labels,
        result,
        spam_threshold=args.spam_threshold,
        constant_share=args.constant_share,
        min_labels=args.min_labels,
        n_boot=args.audit_bootstrap,
        seed=args.seed,
    )


def _emit(args: argparse.Namespace, markdown: str, document: dict[str, Any]) -> int:
    if args.format == "json":
        text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    else:
        text = markdown
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def _run_aggregate(args: argparse.Namespace) -> int:
    labels = _read(args)
    result = _aggregate(labels, args, args.method)
    return _emit(
        args, aggregate_markdown(labels, result, top=args.top), aggregate_document(labels, result)
    )


def _run_evaluate(args: argparse.Namespace) -> int:
    labels = _read(args)
    if labels.n_gold == 0:
        raise ValueError(f"no gold labels found in column {args.gold_col!r}")
    methods = args.methods or list(applicable_methods(labels))
    unknown = [name for name in methods if name not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; choose from {', '.join(METHODS)}")
    if args.baseline not in methods:
        methods = [args.baseline, *methods]

    def score(subset: LabelSet) -> list[Any]:
        results = {name: _aggregate(subset, args, name) for name in methods}
        return evaluate(
            subset,
            results,
            baseline=args.baseline,
            n_boot=args.bootstrap,
            seed=args.seed,
            n_bins=args.bins,
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", GoldOutsideClassesWarning)
        rows = score(labels)
        without = _without_flagged(labels, args, score, rows)
    for message in dict.fromkeys(str(entry.message) for entry in caught):
        print(f"labelfuse: warning: {message}", file=sys.stderr)
    options = {"baseline": args.baseline, "n_boot": args.bootstrap, "seed": args.seed}
    return _emit(
        args,
        evaluation_markdown(
            labels, rows, reliability=args.reliability, without_flagged=without, **options
        ),
        evaluation_document(labels, rows, without_flagged=without, **options),
    )


def _without_flagged(
    labels: LabelSet, args: argparse.Namespace, score: Any, rows: list[Any]
) -> dict[str, Any] | None:
    """The evaluation repeated without the annotators that the audit flags."""
    without = None
    if args.drop_flagged:
        reports = _audit(labels, _aggregate(labels, args, args.audit_method), args)
        flagged = [report for report in reports if report.flag]
        names = [report.annotator for report in flagged]
        without = {
            "audit_method": args.audit_method,
            "flagged": names,
            "n_random": sum(report.flag == "random" for report in flagged),
            "n_constant": sum(report.flag == "constant" for report in flagged),
            "n_labels_dropped": sum(report.n_labels for report in flagged),
            "rows": score(labels.drop_annotators(names)) if names else rows,
        }
    return without


def _run_annotators(args: argparse.Namespace) -> int:
    labels = _read(args)
    result = _aggregate(labels, args, args.method)
    reports = _audit(labels, result, args)
    markdown = annotators_markdown(
        labels,
        result,
        reports,
        spam_threshold=args.spam_threshold,
        min_labels=args.min_labels,
        top=args.top,
    )
    return _emit(args, markdown, annotators_document(labels, result, reports))


def _run_items(args: argparse.Namespace) -> int:
    labels = _read(args)
    result = _aggregate(labels, args, args.method)
    reports = audit_items(labels, result, certain_at=args.certain_at)
    markdown = items_markdown(labels, result, reports, certain_at=args.certain_at, top=args.top)
    return _emit(
        args, markdown, items_document(labels, result, reports, certain_at=args.certain_at)
    )


def _run_plan(args: argparse.Namespace) -> int:
    labels = _read(args)
    result = _aggregate(labels, args, args.method)
    rows = plan_relabelling(labels, result, args.budget)
    return _emit(
        args,
        plan_markdown(result, rows, budget=args.budget, top=args.top),
        plan_document(result, rows, budget=args.budget),
    )


def _floats(text: str) -> list[float]:
    try:
        return [float(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated numbers, got {text!r}"
        ) from None


def _names(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _run_simulate(args: argparse.Namespace) -> int:
    if args.confusion:
        confusion = np.array(json.loads(Path(args.confusion).read_text(encoding="utf-8")))
    elif args.accuracy:
        confusion = np.stack([symmetric_confusion(value, args.classes) for value in args.accuracy])
    else:
        raise ValueError("--accuracy needs at least one value")
    labels = simulate(
        args.items,
        confusion,
        class_prior=args.prior,
        labels_per_item=args.labels_per_item,
        seed=args.seed,
    )
    if args.output:
        write_labels(labels, args.output)
    else:
        dump_labels(labels, sys.stdout, fmt="jsonl")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="labelfuse",
        description="Aggregate noisy labels from several annotators and audit the annotators.",
    )
    parser.add_argument("--version", action="version", version=f"labelfuse {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser(
        "aggregate",
        help="one label and a posterior per item",
        description="Aggregate the labels of each item into a posterior over classes.",
    )
    _add_input_options(command, gold=None)
    _add_model_options(command)
    _add_output_options(command, top=10, what="least certain items")
    command.set_defaults(run=_run_aggregate)

    command = commands.add_parser(
        "evaluate",
        help="accuracy of every aggregator against gold labels",
        description="Score the aggregators on the items that have a gold label: accuracy with a "
        "bootstrap interval, an exact McNemar test against the baseline, and calibration.",
    )
    _add_input_options(command, gold="gold")
    _add_model_options(command, method=False)
    group = command.add_argument_group("evaluation")
    group.add_argument(
        "--methods", type=_names, help="comma-separated methods (default: all that apply)"
    )
    group.add_argument("--baseline", default="majority", help="default: majority")
    group.add_argument(
        "--bootstrap", type=int, default=2000, help="bootstrap resamples (default: 2000)"
    )
    group.add_argument("--bins", type=int, default=10, help="reliability bins (default: 10)")
    group.add_argument(
        "--reliability", action="store_true", help="add a reliability table for every method"
    )
    group.add_argument(
        "--drop-flagged",
        action="store_true",
        help="also evaluate without the annotators that the audit flags",
    )
    group.add_argument(
        "--audit-method",
        choices=METHODS,
        default="dawid-skene",
        help="aggregation behind the audit used by --drop-flagged (default: dawid-skene)",
    )
    _add_audit_options(command)
    _add_output_options(command, top=None)
    command.set_defaults(run=_run_evaluate)

    command = commands.add_parser(
        "annotators",
        help="audit the annotators",
        description="Per annotator: estimated accuracy with an interval, spammer score, bias "
        "toward a class, agreement with the aggregate, and flags for random and constant ones.",
    )
    _add_input_options(command, gold=None)
    _add_model_options(command)
    _add_audit_options(command)
    _add_output_options(command, top=20, what="annotators")
    command.set_defaults(run=_run_annotators)

    command = commands.add_parser(
        "items",
        help="audit the items",
        description="Per item: posterior entropy, top probability, margin and votes.",
    )
    _add_input_options(command, gold=None)
    _add_model_options(command)
    command.add_argument(
        "--certain-at",
        type=float,
        default=0.9,
        help="an item is uncertain when its top probability is below this (default: 0.9)",
    )
    _add_output_options(command, top=20, what="items")
    command.set_defaults(run=_run_items)

    command = commands.add_parser(
        "plan",
        help="choose the items that most deserve one more label",
        description="Rank items by the expected drop in posterior entropy from one more label "
        "by an average annotator and list the best ones within the budget.",
    )
    _add_input_options(command, gold=None)
    _add_model_options(command)
    command.add_argument("--budget", type=int, required=True, help="number of further labels")
    _add_output_options(command, top=20, what="planned items")
    command.set_defaults(run=_run_plan)

    command = commands.add_parser(
        "simulate",
        help="draw labels from annotators with known confusion matrices",
        description="Simulate annotators and write their labels in long format, with the true "
        "class of every item in the gold column.",
    )
    command.add_argument("--items", type=int, default=200, help="number of items (default: 200)")
    command.add_argument("--classes", type=int, default=2, help="number of classes (default: 2)")
    command.add_argument(
        "--accuracy",
        type=_floats,
        default=[0.9, 0.8, 0.7, 0.6, 0.5],
        help="one accuracy per annotator, comma-separated; a wrong answer is spread evenly over "
        "the other classes (default: 0.9,0.8,0.7,0.6,0.5)",
    )
    command.add_argument(
        "--confusion",
        help="JSON file with one confusion matrix per annotator (a list of square matrices, "
        "rows are true classes); replaces --accuracy and --classes",
    )
    command.add_argument("--prior", type=_floats, help="class probabilities (default: uniform)")
    command.add_argument(
        "--labels-per-item",
        type=int,
        help="annotators drawn at random for each item (default: all of them)",
    )
    command.add_argument("--seed", type=int, default=0)
    command.add_argument(
        "--output", "-o", help="output file, .jsonl, .csv or .tsv (default: stdout)"
    )
    command.set_defaults(run=_run_simulate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.run(args))
    except (ValueError, OSError, FloatingPointError) as error:
        print(f"labelfuse: error: {error}", file=sys.stderr)
        return 2
