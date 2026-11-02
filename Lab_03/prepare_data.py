"""Clean the raw WDBC export into something CsvExampleGen can ingest.

Two problems with the file as published:
  * the header line ends in a comma, so it declares 33 columns while every
    data row has 32. ExampleGen builds its schema from the header and would
    carry a phantom empty-named feature through the whole pipeline.
  * the file is CRLF, which leaves a trailing \r glued to the last value on
    each row and can turn a float feature into a byte string.
"""

from __future__ import annotations

import csv
from pathlib import Path

RAW_CSV = Path(__file__).resolve().parent / "data" / "breast_cancer_raw.csv"
OUT_DIR = Path(__file__).resolve().parent / "data" / "breast_cancer"
OUT_CSV = OUT_DIR / "breast_cancer_data.csv"


def prepare() -> Path:
    with open(RAW_CSV, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    header, *data = rows

    # Drop trailing empty header cells so the header matches the data width.
    while header and header[-1].strip() == "":
        header.pop()

    width = len(header)
    cleaned = [[value.strip() for value in row[:width]] for row in data if row]

    bad = [index for index, row in enumerate(cleaned, start=2) if len(row) != width]
    if bad:
        raise ValueError(f"{len(bad)} row(s) do not have {width} fields, first at line {bad[0]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # newline="" plus the default \r\n would reintroduce CRLF; force LF.
    with open(OUT_CSV, "w", newline="\n", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(cleaned)

    print(f"Wrote {len(cleaned)} rows x {width} columns -> {OUT_CSV}")
    return OUT_CSV


if __name__ == "__main__":
    prepare()
