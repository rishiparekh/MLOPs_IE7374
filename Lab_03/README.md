# Lab 03 — TFX Feature Engineering Pipeline (Breast Cancer)

A port of the TFX feature engineering lab to the Wisconsin Diagnostic Breast
Cancer dataset. The upstream lab runs the UCI Census Income data through five
TFX components; this one runs the same five over a dataset that is entirely
numeric, which changes what the `preprocessing_fn` has to do.

## Objectives

Build the feature engineering half of a TFX pipeline:

1. **`CsvExampleGen`** — ingest the CSV and split it into train and eval
2. **`StatisticsGen`** — compute descriptive statistics over each split
3. **`SchemaGen`** — infer types, domains and value ranges from those statistics
4. **`ExampleValidator`** — check the eval split against the schema for anomalies
5. **`Transform`** — apply `preprocessing_fn` to produce model-ready features

There is deliberately no `Trainer`, `Evaluator` or `Pusher`. The lab stops at
the point where features are materialised.

## Dataset

569 samples, 32 columns: an `id`, a `diagnosis` label (`M`/`B`, 212/357), and
30 numeric features. The 30 are 10 nucleus measurements — radius, texture,
perimeter, area, smoothness, compactness, concavity, concave points, symmetry,
fractal dimension — each recorded three ways: `_mean`, `_se` (standard error)
and `_worst` (mean of the three largest values).

The published CSV needs cleaning before `CsvExampleGen` will accept it:

- the header line ends in a comma, declaring 33 columns against 32 data
  columns, which would create a phantom empty-named feature
- the file is CRLF, leaving a trailing `\r` on the last value of every row

`prepare_data.py` fixes both and writes `data/breast_cancer/`. Run it once:

```bash
python prepare_data.py
```

`CsvExampleGen` takes an `input_base` **directory**, not a file path, which is
why the cleaned CSV lives in a subdirectory of its own.

## Running it

TFX cannot be installed natively on Windows: `tensorflow-data-validation` and
`ml-metadata` are C++ extensions published only as manylinux and macOS wheels,
with no Windows build and no pure-Python fallback. The lab therefore runs in a
container.

The image is built and published by CI, so running the lab is a pull, not a
build:

```bash
docker compose pull
```

```bash
docker compose up
```

Then open <http://localhost:8888/lab?token=tfx> and run
`breast_cancer_pipeline.ipynb`.

A local build from source takes roughly 15 minutes -- the wheels are large and
pyfarmhash compiles from source -- which is why
`.github/workflows/lab03_image.yml` does it once and pushes the result to
`ghcr.io/rishiparekh/lab03-tfx`. Build it yourself only when changing the
Dockerfile or the lock:

```bash
docker compose build
```

**Nothing needs to be run to review the lab.** `breast_cancer_pipeline.ipynb`
is committed with its outputs, so every component's result -- statistics,
inferred schema, anomaly report and the transformed feature ranges -- is
readable straight from the notebook on GitHub.

The lab directory is bind-mounted at `/workspace`, so edits you make on the
host — to the notebook or to `breast_cancer_transform.py` — take effect without
rebuilding. Component outputs are written back to `./pipeline` on the host.

To stop:

```bash
docker compose down
```

### Version pinning

| Package | Pin | Why |
|---|---|---|
| Python | 3.10 | `tfx` 1.16.0 declares `requires_python >=3.9,<3.11` |
| `tfx` | 1.16.0 | last release whose native deps ship `manylinux_2_17` wheels |
| TensorFlow | 2.16.x | pinned transitively by `tfx` 1.16.0 |

From TFX 1.17 onward, `tensorflow-data-validation` and `ml-metadata` are built
against glibc 2.39, which would force a Debian trixie or Ubuntu 24.04 base
image. Staying on 1.16.0 keeps `python:3.10-slim-bookworm` (glibc 2.36) viable.

### Why the image installs with uv

`tfx` depends on `apache-beam[gcp]`, whose dependency surface sends pip's
backtracking resolver into a search that does not terminate in any practical
time -- a plain `pip install tfx==1.16.0`, with nothing else requested, was
still resolving after 24 hours. uv's PubGrub resolver closes the same graph in
about 13 seconds.

`requirements.lock.txt` holds that resolved set, all 263 packages pinned. The
image installs from the lock, so the build performs no resolution at all.
`requirements.txt` stays as the human-readable input. Regenerate the lock only
when it changes:

```bash
uv pip compile requirements.txt --python-version 3.10 -o requirements.lock.txt
```

## Files

| File | Purpose |
|---|---|
| `breast_cancer_pipeline.ipynb` | the lab itself |
| `requirements.lock.txt` | fully resolved dependency set, what the image installs |
| `breast_cancer_constants.py` | feature groups, label key, bucket counts |
| `breast_cancer_transform.py` | `preprocessing_fn` |
| `prepare_data.py` | header and line-ending fixes for the raw CSV |
| `data/breast_cancer_raw.csv` | the file as published |
| `data/breast_cancer/` | cleaned output, what `CsvExampleGen` reads |

Unlike the upstream notebook, which writes its two module files out with
`%%writefile`, the modules here are checked in and imported. `Transform`
receives a `module_file` path either way, but keeping them as real files means
they are diffable, reviewable and testable on their own.

## Notes on the transform

`preprocessing_fn` runs **once**, to build a TensorFlow graph. The `tft.*`
analyzers inside it compute their statistics over the training split, bake the
results in as graph constants, and that same graph is then replayed over every
split. This is why the eval split is scaled using training statistics rather
than its own — the component exists to make training and serving agree.

Two details specific to this dataset:

- `id` is in the CSV and therefore in the inferred schema, but it is a record
  identifier. `SchemaGen` cannot know that; `preprocessing_fn` excludes it
  explicitly so the model cannot memorise rows.
- `radius_mean` is both a numeric feature and the bucketized one, so the
  bucketized output uses a `_bucketized_xf` suffix and both versions survive.
  (The census lab keeps its numeric and bucket key lists disjoint instead.)
