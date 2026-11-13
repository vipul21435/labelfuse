import json
import re
import subprocess
import sys

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
