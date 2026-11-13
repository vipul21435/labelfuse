"""Download three public crowdsourcing data sets and write them in long format.

    python scripts/fetch_data.py --out data

Nothing is downloaded twice: files already in ``<out>/raw`` are reused. The converted files are
``bluebirds.jsonl``, ``relevance-2.jsonl`` and ``nist-trec-relevance.jsonl``, each with the
columns item, annotator, label and, where the source has one, gold.

Sources
- bluebirds: demo/bluebirds/labels.yaml and gt.yaml in https://github.com/welinder/cubam
  (39 annotators give a true/false label to each of 108 images).
- relevance-2 and nist-trec-relevance: the archives that the crowd-kit library downloads,
  https://github.com/Toloka/crowd-kit (crowdkit/datasets/_loaders.py), checked against the md5
  files published next to them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

CUBAM = "https://raw.githubusercontent.com/welinder/cubam/public/demo/bluebirds/"
CROWD_KIT = "https://tlk.s3.yandex.net/dataset/crowd-kit/"
ARCHIVES = {"relevance-2": "relevance-2", "nist-trec-relevance": "relevance"}


def fetch(url: str, path: Path) -> bytes:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    return path.read_bytes()


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"{path}: {len(rows)} labels")


def bluebirds(out: Path) -> None:
    pair = re.compile(r"(\d+): (true|false)")
    truth = dict(pair.findall(fetch(CUBAM + "gt.yaml", out / "raw" / "bluebirds-gt.yaml").decode()))
    text = fetch(CUBAM + "labels.yaml", out / "raw" / "bluebirds-labels.yaml").decode()
    rows = []
    for annotator, body in re.findall(r"(?m)^(\d+): \{(.*?)\}", text, flags=re.S):
        for image, answer in pair.findall(body):
            rows.append(
                {"item": image, "annotator": annotator, "label": answer, "gold": truth[image]}
            )
    write_rows(out / "bluebirds.jsonl", rows)


def crowd_kit(out: Path, name: str, archive: str) -> None:
    blob = fetch(f"{CROWD_KIT}{archive}.zip", out / "raw" / f"{archive}.zip")
    expected = (
        fetch(f"{CROWD_KIT}{archive}.md5", out / "raw" / f"{archive}.md5").decode().split()[0]
    )
    if hashlib.md5(blob).hexdigest() != expected:
        raise SystemExit(f"{archive}.zip does not match its published md5")
    with zipfile.ZipFile(io.BytesIO(blob)) as bundle:
        gold_text = bundle.read(f"{archive}/gt.csv").decode()
        label_text = bundle.read(f"{archive}/crowd_labels.csv").decode()
    truth = {row["task"]: row["label"] for row in csv.DictReader(io.StringIO(gold_text))}
    rows = []
    for row in csv.DictReader(io.StringIO(label_text)):
        record = {"item": row["task"], "annotator": row["performer"], "label": row["label"]}
        if row["task"] in truth:
            record["gold"] = truth[row["task"]]
        rows.append(record)
    write_rows(out / f"{name}.jsonl", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data", help="output directory (default: data)")
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    bluebirds(out)
    for name, archive in ARCHIVES.items():
        crowd_kit(out, name, archive)


if __name__ == "__main__":
    main()
