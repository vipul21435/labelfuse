"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from labelfuse import __version__
from labelfuse.data import LabelSet, dump_labels, read_labels, write_labels
from labelfuse.dawid_skene import DEFAULT_MAX_ITER, DEFAULT_SMOOTHING, DEFAULT_TOL, dawid_skene
from labelfuse.majority import majority_vote
from labelfuse.report import aggregate_document, aggregate_markdown
from labelfuse.result import AggregationResult
from labelfuse.simulate import simulate, symmetric_confusion

METHODS = ("dawid-skene", "majority")


def _add_input_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("labels", help="long-format labels: a .jsonl, .csv or .tsv file")
    group = parser.add_argument_group("input")
    group.add_argument("--item-col", default="item", help="item id column (default: item)")
    group.add_argument(
        "--annotator-col", default="annotator", help="annotator id column (default: annotator)"
    )
    group.add_argument("--label-col", default="label", help="label column (default: label)")
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


def _read(args: argparse.Namespace) -> LabelSet:
    return read_labels(
        args.labels,
        item_col=args.item_col,
        annotator_col=args.annotator_col,
        label_col=args.label_col,
        fmt=args.input_format,
        duplicates=args.duplicates,
    )


def _emit(text: str, output: str | None) -> None:
    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def _aggregate(labels: LabelSet, args: argparse.Namespace) -> AggregationResult:
    if args.method == "majority":
        return majority_vote(labels, tie=args.tie, seed=args.seed)
    return dawid_skene(
        labels,
        smoothing=args.smoothing,
        max_iter=args.max_iter,
        tol=args.tol,
        n_restarts=args.restarts,
        seed=args.seed,
        tie=args.tie,
    )


def _run_aggregate(args: argparse.Namespace) -> int:
    labels = _read(args)
    result = _aggregate(labels, args)
    if args.format == "json":
        text = json.dumps(aggregate_document(labels, result), indent=2, ensure_ascii=False) + "\n"
    else:
        text = aggregate_markdown(labels, result, top=args.top)
    _emit(text, args.output)
    return 0


def _floats(text: str) -> list[float]:
    try:
        return [float(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated numbers, got {text!r}"
        ) from None


def _run_simulate(args: argparse.Namespace) -> int:
    if not args.accuracy:
        raise ValueError("--accuracy needs at least one value")
    confusion = np.stack([symmetric_confusion(value, args.classes) for value in args.accuracy])
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

    aggregate = commands.add_parser(
        "aggregate",
        help="one label and a posterior per item",
        description="Aggregate the labels of each item into a posterior over classes.",
    )
    _add_input_options(aggregate)
    aggregate.add_argument(
        "--method", choices=METHODS, default="dawid-skene", help="default: dawid-skene"
    )
    aggregate.add_argument(
        "--tie",
        choices=("abstain", "first", "random"),
        default="abstain",
        help="label of an item whose top classes are tied: none, the first class in class "
        "order, or a seeded random one (default: abstain)",
    )
    aggregate.add_argument("--seed", type=int, default=0, help="seed for restarts and random ties")
    model = aggregate.add_argument_group("dawid-skene")
    model.add_argument(
        "--smoothing",
        type=float,
        default=DEFAULT_SMOOTHING,
        help="pseudo-count added to every confusion cell and prior class "
        f"(default: {DEFAULT_SMOOTHING:g})",
    )
    model.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help=f"most EM iterations per start (default: {DEFAULT_MAX_ITER})",
    )
    model.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help=f"stop once every posterior entry moves by less than this (default: {DEFAULT_TOL:g})",
    )
    model.add_argument(
        "--restarts",
        type=int,
        default=1,
        help="number of EM starts; the first is the vote shares, the others add noise (default: 1)",
    )
    out = aggregate.add_argument_group("output")
    out.add_argument("--format", choices=("markdown", "json"), default="markdown")
    out.add_argument(
        "--top", type=int, default=10, help="least certain items to list (default: 10)"
    )
    out.add_argument("--output", "-o", help="write the report to this file instead of stdout")
    aggregate.set_defaults(run=_run_aggregate)

    sim = commands.add_parser(
        "simulate",
        help="draw labels from annotators with known accuracy",
        description="Simulate annotators and write their labels in long format, with the true "
        "class of every item in the gold column.",
    )
    sim.add_argument("--items", type=int, default=200, help="number of items (default: 200)")
    sim.add_argument("--classes", type=int, default=2, help="number of classes (default: 2)")
    sim.add_argument(
        "--accuracy",
        type=_floats,
        default=[0.9, 0.8, 0.7, 0.6, 0.5],
        help="one accuracy per annotator, comma-separated; a wrong answer is spread evenly over "
        "the other classes (default: 0.9,0.8,0.7,0.6,0.5)",
    )
    sim.add_argument("--prior", type=_floats, help="class probabilities (default: uniform)")
    sim.add_argument(
        "--labels-per-item",
        type=int,
        help="annotators drawn at random for each item (default: all of them)",
    )
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--output", "-o", help="output file, .jsonl, .csv or .tsv (default: stdout)")
    sim.set_defaults(run=_run_simulate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.run(args))
    except (ValueError, OSError, FloatingPointError) as error:
        print(f"labelfuse: error: {error}", file=sys.stderr)
        return 2
