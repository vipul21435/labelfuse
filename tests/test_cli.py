import json
import re
import subprocess
import sys

import numpy as np
import pytest

from labelfuse import __version__, read_labels
from labelfuse.cli import main


@pytest.fixture
def toy(tmp_path):
    path = tmp_path / "toy.jsonl"
    code = main(
        [
            "simulate",
            "--items",
            "60",
            "--classes",
            "3",
            "--accuracy",
            "0.9,0.8,0.7,0.34",
            "--seed",
            "3",
            "-o",
            str(path),
        ]
    )
    assert code == 0
    return path


def test_simulate_writes_long_format_with_gold(toy):
    labels = read_labels(toy, gold_col="gold")
    assert (labels.n_items, labels.n_annotators, labels.n_labels) == (60, 4, 240)
    assert labels.n_gold == 60
    first = json.loads(toy.read_text(encoding="utf-8").splitlines()[0])
    assert set(first) == {"item", "annotator", "label", "gold"}


def test_simulate_prints_jsonl_to_stdout_by_default(capsys):
    assert main(["simulate", "--items", "4", "--accuracy", "0.9,0.6", "--seed", "1"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 8
    assert rows[0]["item"] == "i1"
    assert rows[0]["annotator"] == "a1"


def test_simulate_csv_with_prior_and_labels_per_item(tmp_path):
    path = tmp_path / "sim.csv"
    arguments = ["simulate", "--items", "30", "--accuracy", "0.9,0.8,0.7", "--prior", "0.8,0.2"]
    assert main([*arguments, "--labels-per-item", "2", "-o", str(path)]) == 0
    labels = read_labels(path, gold_col="gold")
    assert labels.n_labels == 60
    assert path.read_text(encoding="utf-8").splitlines()[0] == "item,annotator,label,gold"


def test_aggregate_markdown_report(toy, capsys):
    assert main(["aggregate", str(toy), "--top", "3"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Aggregated labels\n")
    assert "- items: 60" in out
    assert "- annotators: 4" in out
    assert "- labels: 240 (per item: min 4, median 4, max 4)" in out
    assert "- method: dawid-skene, smoothing 0.01" in out
    assert "converged, log-likelihood -" in out
    assert "| class | items | share |" in out
    assert "## Least certain items (3 of 60)" in out
    assert len(re.findall(r"^\| i\d+ \| c\d \| 0\.\d{3} \| ", out, flags=re.MULTILINE)) == 3


def test_aggregate_markdown_lists_starts_and_can_hide_the_item_table(toy, capsys):
    assert main(["aggregate", str(toy), "--restarts", "3", "--top", "0"]) == 0
    out = capsys.readouterr().out
    assert "- starts: 3, objective from -" in out
    assert "Least certain items" not in out


def test_aggregate_majority_reports_ties(toy, capsys):
    assert main(["aggregate", str(toy), "--method", "majority"]) == 0
    out = capsys.readouterr().out
    assert "- method: majority" in out
    tied = int(out.split("- tied items: ")[1].splitlines()[0])
    assert tied > 0
    assert f"| (no label) | {tied} |" in out
    assert "| (tie) |" in out

    assert main(["aggregate", str(toy), "--method", "majority", "--tie", "first"]) == 0
    out = capsys.readouterr().out
    assert f"- tied items: {tied}" in out
    assert "(no label)" not in out


def test_aggregate_json_document(toy, capsys):
    assert main(["aggregate", str(toy), "--format", "json", "--restarts", "2"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["method"] == "dawid-skene"
    assert document["n_items"] == 60
    assert document["classes"] == ["c0", "c1", "c2"]
    assert len(document["items"]) == 60
    item = document["items"][0]
    assert item["item"] == "i01"
    assert set(item["posterior"]) == {"c0", "c1", "c2"}
    assert sum(item["posterior"].values()) == pytest.approx(1.0)
    assert item["probability"] == pytest.approx(max(item["posterior"].values()))
    assert len(document["annotators"]) == 4
    assert len(document["annotators"][0]["confusion"]) == 3
    assert sum(document["class_prior"].values()) == pytest.approx(1.0)
    fit = document["fit"]
    assert fit["converged"] is True
    assert fit["n_iter"] == len(fit["log_likelihood_trace"])
    assert len(fit["restart_objectives"]) == 2


def test_aggregate_json_for_majority_has_no_fit(toy, capsys):
    assert main(["aggregate", str(toy), "--method", "majority", "--format", "json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert "fit" not in document
    assert "annotators" not in document
    assert document["n_tied"] == sum(item["tied"] for item in document["items"])
    assert all(item["label"] is None for item in document["items"] if item["tied"])


def test_aggregate_writes_to_a_file(toy, tmp_path, capsys):
    report = tmp_path / "report.md"
    assert main(["aggregate", str(toy), "-o", str(report)]) == 0
    assert capsys.readouterr().out == ""
    assert report.read_text(encoding="utf-8").startswith("# Aggregated labels")


def test_aggregate_reads_csv_with_named_columns(tmp_path, capsys):
    path = tmp_path / "answers.csv"
    path.write_text(
        "question,worker,answer\nq1,w1,cat\nq1,w2,cat\nq1,w3,dog\nq2,w1,dog\nq2,w2,dog\n",
        encoding="utf-8",
    )
    columns = ["--item-col", "question", "--annotator-col", "worker", "--label-col", "answer"]
    assert main(["aggregate", str(path), "--method", "majority", "--format", "json", *columns]) == 0
    document = json.loads(capsys.readouterr().out)
    assert [item["label"] for item in document["items"]] == ["cat", "dog"]


def test_markdown_escapes_pipes_in_ids(tmp_path, capsys):
    path = tmp_path / "pipes.csv"
    path.write_text("item,annotator,label\na|b,w1,x|y\na|b,w2,x|y\n", encoding="utf-8")
    assert main(["aggregate", str(path), "--method", "majority"]) == 0
    out = capsys.readouterr().out
    assert "| a\\|b | x\\|y | 1.000 | x\\|y 2 |" in out


def test_duplicates_are_refused_unless_a_policy_is_chosen(tmp_path, capsys):
    path = tmp_path / "twice.csv"
    path.write_text("item,annotator,label\ni1,w1,x\ni1,w1,y\ni1,w2,y\n", encoding="utf-8")
    assert main(["aggregate", str(path), "--method", "majority"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "labelfuse: error: 1 repeated (item, annotator) pairs" in captured.err
    assert main(["aggregate", str(path), "--method", "majority", "--duplicates", "last"]) == 0
    assert "- labels: 2 " in capsys.readouterr().out


def test_errors_go_to_stderr_with_exit_code_2(tmp_path, capsys):
    assert main(["aggregate", str(tmp_path / "missing.jsonl")]) == 2
    assert "labelfuse: error:" in capsys.readouterr().err
    assert main(["simulate", "--accuracy", "1.5"]) == 2
    assert "accuracy must be between 0 and 1" in capsys.readouterr().err
    assert main(["simulate", "--accuracy", ","]) == 2
    assert "at least one value" in capsys.readouterr().err


def test_bad_number_list_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["simulate", "--accuracy", "0.9,high"])
    assert stop.value.code == 2
    assert "expected comma-separated numbers" in capsys.readouterr().err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.strip() == f"labelfuse {__version__}"


def test_module_entry_point(toy):
    done = subprocess.run(
        [sys.executable, "-m", "labelfuse", "aggregate", str(toy), "--method", "majority"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0
    assert done.stdout.startswith("# Aggregated labels")


@pytest.fixture
def binary(tmp_path):
    path = tmp_path / "binary.jsonl"
    arguments = ["simulate", "--items", "150", "--accuracy", "0.9,0.85,0.8,0.5,0.5", "--seed", "2"]
    assert main([*arguments, "-o", str(path)]) == 0
    return path


def test_aggregate_runs_every_method(binary, capsys):
    for method in ("majority", "weighted", "one-coin", "dawid-skene", "glad"):
        assert main(["aggregate", str(binary), "--method", method, "--format", "json"]) == 0
        document = json.loads(capsys.readouterr().out)
        assert document["method"] == method
        assert len(document["items"]) == 150
        if method in ("weighted", "glad", "one-coin", "dawid-skene"):
            assert len(document["annotators"]) == 5
    assert main(["aggregate", str(binary), "--method", "glad", "--top", "2"]) == 0
    assert "- method: glad\n- EM: " in capsys.readouterr().out


def test_evaluate_markdown(binary, capsys):
    assert main(["evaluate", str(binary), "--bootstrap", "200", "--reliability"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Evaluation against gold\n")
    assert "- items with gold: 150 of 150" in out
    assert "200 resamples, seed 0" in out
    header = "| method | accuracy | 95 % interval | ties | vs majority |"
    assert header in out
    rows = [line for line in out.splitlines() if re.match(r"\| [a-z-]+ \| 0\.\d{3} \| ", line)]
    assert [row.split(" | ")[0].strip("| ") for row in rows] == [
        "majority",
        "weighted",
        "one-coin",
        "dawid-skene",
        "glad",
    ]
    assert "| majority | 0." in out
    assert " | - | - | - | - | " in out
    assert "## Reliability of dawid-skene" in out
    assert "| top probability | items | mean probability | accuracy | gap |" in out


def test_evaluate_json_methods_baseline_and_dropping_flagged(binary, capsys):
    arguments = ["evaluate", str(binary), "--bootstrap", "100", "--format", "json"]
    assert main([*arguments, "--methods", "dawid-skene", "--drop-flagged"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert [row["method"] for row in document["methods"]] == ["majority", "dawid-skene"]
    assert document["n_gold"] == 150
    model = document["methods"][1]
    assert 0.8 < model["accuracy"] <= 1.0
    assert model["interval"][0] <= model["accuracy"] <= model["interval"][1]
    assert sum(row["count"] for row in model["reliability"]) == 150
    without = document["without_flagged"]
    assert without["audit_method"] == "dawid-skene"
    assert set(without["flagged"]) <= {"a4", "a5"}
    assert without["n_random"] + without["n_constant"] == len(without["flagged"])
    assert without["n_labels_dropped"] == 150 * len(without["flagged"])
    assert [row["method"] for row in without["methods"]] == ["majority", "dawid-skene"]


def test_evaluate_drop_flagged_markdown(binary, capsys):
    arguments = ["evaluate", str(binary), "--bootstrap", "100", "--drop-flagged"]
    assert main([*arguments, "--spam-threshold", "0.2"]) == 0
    out = capsys.readouterr().out
    assert "## Without flagged annotators" in out
    assert (
        "- flagged by the dawid-skene audit: 2 of 5 annotators (40.0%; random 2, constant 0)" in out
    )
    assert "who gave 300 of 750 labels (40.0%)" in out
    assert "| method | all annotators | without flagged | change |" in out
    assert main([*arguments, "--spam-threshold", "0"]) == 0
    out = capsys.readouterr().out
    assert "0 of 5 annotators" in out
    assert "all annotators | without flagged" not in out


def test_evaluate_errors(binary, tmp_path, capsys):
    assert main(["evaluate", str(binary), "--methods", "median"]) == 2
    assert "unknown methods" in capsys.readouterr().err
    assert main(["evaluate", str(binary), "--gold-col", "truth"]) == 2
    assert "no gold labels found in column 'truth'" in capsys.readouterr().err
    path = tmp_path / "nogold.csv"
    path.write_text("item,annotator,label,gold\ni1,a,x,\ni1,b,y,\n", encoding="utf-8")
    assert main(["evaluate", str(path)]) == 2
    assert "no gold labels found in column 'gold'" in capsys.readouterr().err


def test_annotators_report(binary, capsys):
    arguments = ["annotators", str(binary), "--spam-threshold", "0.2", "--audit-bootstrap", "100"]
    assert main([*arguments, "--gold-col", "gold", "--top", "2"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Annotator audit\n")
    assert "- flagged: 2 of 5 annotators (40.0%; random 2, constant 0), who gave 300 of 750" in out
    assert "## Least informative annotators (2 of 5)" in out
    assert "| gold accuracy (items) |" in out
    assert len(re.findall(r"^\| a[45] \| 150 \| .* \| random \|$", out, flags=re.MULTILINE)) == 2

    assert main([*arguments, "--format", "json", "--method", "majority"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["method"] == "majority"
    assert len(document["annotators"]) == 5
    first = document["annotators"][0]
    assert first["annotator"] == "a1"
    assert first["gold_accuracy"] is None
    assert np.array(first["confusion"]).shape == (2, 2)
    assert main([*arguments, "--top", "0"]) == 0
    out = capsys.readouterr().out
    assert "(5 of 5)" in out
    assert "gold accuracy" not in out


def test_items_report(binary, capsys):
    assert main(["items", str(binary), "--top", "3", "--certain-at", "0.99"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Item audit\n")
    assert "- uncertain items (top probability below 0.99): " in out
    assert "## Most uncertain items (3 of 150)" in out
    assert "| item | label | probability | entropy (bits) | margin | labels | votes |" in out
    assert main(["items", str(binary), "--format", "json", "--method", "majority"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert len(document["items"]) == 150
    assert document["n_uncertain"] == sum(item["uncertain"] for item in document["items"])


def test_items_and_plan_show_ties(tmp_path, capsys):
    path = tmp_path / "tie.csv"
    path.write_text("item,annotator,label\ni1,a,x\ni1,b,y\ni2,a,x\ni2,b,x\n", encoding="utf-8")
    assert main(["items", str(path), "--method", "majority", "--top", "0"]) == 0
    assert "| i1 | (tie) | 0.500 | 1.000 | 0.000 | 2 | x 1, y 1 |" in capsys.readouterr().out
    assert main(["plan", str(path), "--method", "majority", "--budget", "1"]) == 0
    assert "| i1 | (tie) | 0.500 | 1.000 | " in capsys.readouterr().out


def test_plan_report(binary, capsys):
    assert main(["plan", str(binary), "--budget", "12", "--top", "5"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Relabelling plan\n")
    assert "- budget: 12 labels, planned: 12 " in out
    assert "## Items to relabel (5 of 12)" in out
    assert main(["plan", str(binary), "--budget", "12", "--format", "json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["n_planned"] == 12
    drops = [item["expected_reduction"] for item in document["items"]]
    assert drops == sorted(drops, reverse=True)
    assert document["expected_reduction"] == pytest.approx(sum(drops))
    assert main(["plan", str(binary), "--budget", "3", "--method", "majority", "--top", "0"]) == 0
    assert "## Items to relabel (3 of 3)" in capsys.readouterr().out
    assert main(["plan", str(binary), "--budget", "-1"]) == 2
    assert "budget must not be negative" in capsys.readouterr().err


def test_simulate_from_confusion_matrices(tmp_path):
    spec = tmp_path / "annotators.json"
    spec.write_text(json.dumps([[[1, 0], [0, 1]], [[1, 0], [1, 0]]]), encoding="utf-8")
    out = tmp_path / "sim.jsonl"
    assert main(["simulate", "--items", "50", "--confusion", str(spec), "-o", str(out)]) == 0
    labels = read_labels(out, gold_col="gold")
    by_annotator = {}
    for item, annotator, label in labels.records():
        by_annotator.setdefault(annotator, []).append(
            (label, labels.gold[labels.items.index(item)])
        )
    assert all(label == gold for label, gold in by_annotator["a1"])
    assert {label for label, _ in by_annotator["a2"]} == {"c0"}


def test_evaluate_warns_about_gold_labels_nobody_used(tmp_path, capsys):
    path = tmp_path / "stray.csv"
    rows = "i1,a,x,x\ni1,b,x,\ni2,a,y,maybe\ni2,b,y,\ni3,a,y,y\ni3,b,y,\n"
    path.write_text("item,annotator,label,gold\n" + rows, encoding="utf-8")
    assert main(["evaluate", str(path), "--bootstrap", "50", "--drop-flagged"]) == 0
    captured = capsys.readouterr()
    warning = "labelfuse: warning: gold labels that are not among the classes ('x', 'y'): 'maybe'"
    assert captured.err.count(warning) == 1
    assert "on 1 of 3 items with gold; those items are scored as wrong" in captured.err
    line = "- gold labels that no annotator used: maybe on 1 of 3 items with gold, scored as wrong"
    assert line in captured.out
    assert "| majority | 0.667 |" in captured.out
    assert main(["evaluate", str(path), "--bootstrap", "50", "--format", "json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["n_gold_outside_classes"] == 1
    assert captured.err.count("labelfuse: warning:") == 1


def test_evaluate_is_silent_when_every_gold_label_is_a_class(binary, capsys):
    assert main(["evaluate", str(binary), "--bootstrap", "50", "--format", "json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["n_gold_outside_classes"] == 0
