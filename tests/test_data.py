import json

import numpy as np
import pytest

from labelfuse import LabelSet, read_labels, write_labels

RECORDS = [
    ("q2", "blue", "yes"),
    ("q1", "blue", "no"),
    ("q1", "amber", "yes"),
    ("q2", "amber", "yes"),
    ("q3", "cyan", "no"),
]


def test_from_records_orders_items_annotators_and_classes():
    labels = LabelSet.from_records(RECORDS)
    assert labels.items == ("q1", "q2", "q3")
    assert labels.annotators == ("amber", "blue", "cyan")
    assert labels.classes == ("no", "yes")
    assert (labels.n_items, labels.n_annotators, labels.n_classes, labels.n_labels) == (3, 3, 2, 5)
    assert list(labels.records()) == [
        ("q1", "amber", "yes"),
        ("q1", "blue", "no"),
        ("q2", "amber", "yes"),
        ("q2", "blue", "yes"),
        ("q3", "cyan", "no"),
    ]


def test_row_order_does_not_change_the_label_set():
    forward = LabelSet.from_records(RECORDS)
    backward = LabelSet.from_records(RECORDS[::-1])
    assert forward.items == backward.items
    assert forward.annotators == backward.annotators
    for name in ("item_index", "annotator_index", "label_index"):
        np.testing.assert_array_equal(getattr(forward, name), getattr(backward, name))


def test_integer_ids_are_ordered_by_value_not_as_text():
    labels = LabelSet.from_records([(10, "a", 2), (9, "a", 10), (100, "a", 2)])
    assert labels.items == ("9", "10", "100")
    assert labels.classes == ("2", "10")


def test_mixed_ids_are_ordered_as_text():
    labels = LabelSet.from_records([("10", "a", "x"), ("9", "a", "x"), ("b", "a", "x")])
    assert labels.items == ("10", "9", "b")


def test_vote_counts_and_label_tallies():
    labels = LabelSet.from_records(RECORDS)
    np.testing.assert_array_equal(labels.vote_counts(), [[1, 1], [0, 2], [1, 0]])
    np.testing.assert_array_equal(labels.labels_per_item(), [2, 2, 1])
    np.testing.assert_array_equal(labels.labels_per_annotator(), [2, 2, 1])


def test_values_are_compared_as_trimmed_text():
    labels = LabelSet.from_records([(1, " a ", True), ("1", "b", "true"), (2, "a", False)])
    assert labels.items == ("1", "2")
    assert labels.annotators == ("a", "b")
    assert labels.classes == ("false", "true")


def test_explicit_classes_keep_their_order_and_may_be_unobserved():
    labels = LabelSet.from_records(RECORDS, classes=["yes", "maybe", "no"])
    assert labels.classes == ("yes", "maybe", "no")
    np.testing.assert_array_equal(labels.vote_counts()[0], [1, 0, 1])


def test_label_outside_explicit_classes_is_an_error():
    with pytest.raises(ValueError, match="outside the given classes"):
        LabelSet.from_records(RECORDS, classes=["yes"])


REPEATED = [("q1", "amber", "yes"), ("q1", "amber", "no"), ("q1", "blue", "no")]


def test_repeated_pair_is_an_error_by_default():
    with pytest.raises(ValueError, match="1 repeated"):
        LabelSet.from_records(REPEATED)


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("keep", [("q1", "amber", "no"), ("q1", "amber", "yes"), ("q1", "blue", "no")]),
        ("first", [("q1", "amber", "yes"), ("q1", "blue", "no")]),
        ("last", [("q1", "amber", "no"), ("q1", "blue", "no")]),
    ],
)
def test_duplicate_policies(policy, expected):
    labels = LabelSet.from_records(REPEATED, duplicates=policy)
    assert list(labels.records()) == expected
    assert labels.n_duplicates == 1


def test_unknown_duplicate_policy_is_an_error():
    with pytest.raises(ValueError, match="duplicates must be one of"):
        LabelSet.from_records(RECORDS, duplicates="merge")


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([], "no labels"),
        ([("q1", "amber")], r"record 1: expected \(item, annotator, label\)"),
        ([("q1", "amber", "yes"), ("q1", "blue", "")], "record 2: .* must not be empty"),
        ([("q1", None, "yes")], "record 1: .* must not be empty"),
        ([("q1", "amber", ["yes"])], "record 1: expected a scalar, got list"),
        (["abc"], r"record 1: expected \(item, annotator, label\)"),
    ],
)
def test_malformed_records_are_errors(records, message):
    with pytest.raises(ValueError, match=message):
        LabelSet.from_records(records)


def test_gold_is_kept_as_text_and_does_not_add_classes():
    gold = {"q1": "yes", "q2": " ", "q3": "other", "q9": "no"}
    labels = LabelSet.from_records(RECORDS, gold=gold)
    assert labels.gold == ("yes", None, "other")
    assert labels.n_gold == 2
    assert labels.classes == ("no", "yes")


def test_gold_matching_no_item_is_an_error():
    with pytest.raises(ValueError, match="no gold item id matches"):
        LabelSet.from_records(RECORDS, gold={"x1": "yes"})


def test_constructor_validates_indices():
    with pytest.raises(ValueError, match="outside"):
        LabelSet(("i",), ("a",), ("x",), np.array([1]), np.array([0]), np.array([0]))
    with pytest.raises(ValueError, match="differ in length"):
        LabelSet(("i",), ("a",), ("x",), np.array([0, 0]), np.array([0]), np.array([0]))
    with pytest.raises(ValueError, match="repeated"):
        LabelSet(("i", "i"), ("a",), ("x",), np.array([0]), np.array([0]), np.array([0]))
    with pytest.raises(ValueError, match="one entry per item"):
        LabelSet(("i",), ("a",), ("x",), np.array([0]), np.array([0]), np.array([0]), gold=())


def test_constructor_rejects_other_malformed_input():
    with pytest.raises(ValueError, match="one-dimensional"):
        LabelSet(("i",), ("a",), ("x",), np.array([[0]]), np.array([0]), np.array([0]))
    with pytest.raises(ValueError, match="at least one class"):
        LabelSet(("i",), ("a",), (), np.array([], dtype=int), np.array([], dtype=int), [])
    with pytest.raises(ValueError, match="classes must not contain empty values"):
        LabelSet.from_records(RECORDS, classes=["yes", " ", "no"])


def test_label_set_without_gold_counts_no_gold():
    assert LabelSet.from_records(RECORDS).n_gold == 0


def test_index_arrays_are_read_only():
    labels = LabelSet.from_records(RECORDS)
    with pytest.raises(ValueError, match="read-only"):
        labels.item_index[0] = 2


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_read_jsonl(tmp_path):
    path = write(
        tmp_path / "labels.jsonl",
        '{"item": 1, "annotator": "amber", "label": "yes"}\n'
        "\n"
        '{"item": 1, "annotator": "blue", "label": "no"}\n'
        '{"item": 2, "annotator": "amber", "label": "no"}\n',
    )
    labels = read_labels(path)
    assert labels.items == ("1", "2")
    assert list(labels.records()) == [
        ("1", "amber", "yes"),
        ("1", "blue", "no"),
        ("2", "amber", "no"),
    ]
    assert labels.gold is None


def test_read_csv_with_other_column_names_and_gold(tmp_path):
    path = write(
        tmp_path / "labels.csv",
        "question,worker,answer,truth\nq1,w1,cat,cat\nq1,w2,dog,\nq2,w1,dog,\n",
    )
    labels = read_labels(
        path, item_col="question", annotator_col="worker", label_col="answer", gold_col="truth"
    )
    assert labels.items == ("q1", "q2")
    assert labels.classes == ("cat", "dog")
    assert labels.gold == ("cat", None)


def test_read_tsv_and_explicit_format(tmp_path):
    text = "item\tannotator\tlabel\ni1\ta\tx\ni1\tb\ty\n"
    by_suffix = read_labels(write(tmp_path / "labels.tsv", text))
    by_argument = read_labels(write(tmp_path / "labels.txt", text), fmt="tsv")
    assert (
        list(by_suffix.records())
        == list(by_argument.records())
        == [
            ("i1", "a", "x"),
            ("i1", "b", "y"),
        ]
    )


def test_read_accepts_a_byte_order_mark(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_bytes(b"\xef\xbb\xbfitem,annotator,label\ni1,a,x\n")
    assert list(read_labels(path).records()) == [("i1", "a", "x")]


@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        ("a.jsonl", '{"item": 1, "annotator": "a"}\n', r"a\.jsonl:1: missing value for 'label'"),
        ("b.jsonl", '{"item": 1,\n', r"b\.jsonl:1: not valid JSON"),
        ("c.jsonl", "[1, 2, 3]\n", r"c\.jsonl:1: expected a JSON object"),
        (
            "d.jsonl",
            '{"item": 1, "annotator": "a", "label": {"x": 1}}\n',
            r"d\.jsonl:1: column 'label': expected a scalar",
        ),
        ("e.csv", "item,annotator\ni1,a\n", "no column 'label'"),
        ("f.csv", "item,annotator,label\ni1,a,\n", r"f\.csv:2: missing value for 'label'"),
        ("g.csv", "item,annotator,label\n", "no labels"),
        ("h.txt", "item,annotator,label\ni1,a,x\n", "cannot tell the format"),
    ],
)
def test_read_errors_name_the_place(tmp_path, name, text, message):
    with pytest.raises(ValueError, match=message):
        read_labels(write(tmp_path / name, text))


def test_conflicting_gold_for_one_item_is_an_error(tmp_path):
    path = write(
        tmp_path / "labels.csv",
        "item,annotator,label,gold\ni1,a,x,x\ni1,b,y,y\n",
    )
    with pytest.raises(ValueError, match="two gold labels"):
        read_labels(path, gold_col="gold")


def test_gold_must_be_a_scalar(tmp_path):
    path = write(
        tmp_path / "labels.jsonl",
        '{"item": 1, "annotator": "a", "label": "x", "gold": ["x"]}\n',
    )
    with pytest.raises(ValueError, match=r"labels\.jsonl:1: column 'gold': expected a scalar"):
        read_labels(path, gold_col="gold")


def test_unknown_format_is_an_error(tmp_path):
    path = write(tmp_path / "labels.csv", "item,annotator,label\ni1,a,x\n")
    with pytest.raises(ValueError, match="unknown format 'xml'"):
        read_labels(path, fmt="xml")
    with pytest.raises(ValueError, match="unknown format 'xml'"):
        write_labels(read_labels(path), tmp_path / "out.csv", fmt="xml")


def test_missing_gold_column_is_an_error(tmp_path):
    path = write(tmp_path / "labels.csv", "item,annotator,label\ni1,a,x\n")
    with pytest.raises(ValueError, match="no column 'gold'"):
        read_labels(path, gold_col="gold")


@pytest.mark.parametrize("suffix", [".jsonl", ".csv", ".tsv"])
def test_write_then_read_round_trip(tmp_path, suffix):
    labels = LabelSet.from_records(RECORDS, gold={"q1": "yes", "q3": "no"})
    path = tmp_path / f"out{suffix}"
    write_labels(labels, path)
    back = read_labels(path, gold_col="gold")
    assert list(back.records()) == list(labels.records())
    assert back.gold == labels.gold


def test_write_without_gold_has_three_columns(tmp_path):
    path = tmp_path / "out.jsonl"
    write_labels(LabelSet.from_records(RECORDS), path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0] == {"item": "q1", "annotator": "amber", "label": "yes"}
    assert len(rows) == 5
