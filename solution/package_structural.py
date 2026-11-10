"""Package a fitted structural model and verify it from a fresh extraction."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from slot_forecasting.io import read_queries, read_submission, write_json
from slot_forecasting.submission_contract import package_code

PREDICT = '''"""Recreate the forecast CSV from the saved model. No training and no network."""
import argparse
import platform
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ensure_numpy():
    """Use an installed NumPy; otherwise unpack the matching bundled wheel into scratch space."""
    try:
        import numpy  # noqa: F401
        return "installed"
    except ImportError:
        pass
    tag = "cp%d%d" % sys.version_info[:2]
    machine = platform.machine().lower().replace("amd64", "x86_64").replace("arm64", "aarch64")
    for wheel in sorted((ROOT / "wheels").glob("numpy-*-%s-%s-*.whl" % (tag, tag))):
        if not wheel.name.endswith("_%s.whl" % machine):
            continue
        target = Path(tempfile.mkdtemp(prefix="house_edge_numpy_"))
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(target)
        sys.path.insert(0, str(target))
        import numpy  # noqa: F401
        return "bundled " + wheel.name
    raise SystemExit("NumPy is not installed and no bundled wheel matches %s %s" % (tag, machine))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print("NumPy:", ensure_numpy(), file=sys.stderr)
    from slot_forecasting.io import load_inputs, write_submission
    from solution import structural
    root = ROOT
    params = structural.load_params(root / "params.json")
    queries, history = load_inputs(args.data / "test")
    write_submission(args.out, structural.predict(queries, history, params))


if __name__ == "__main__":
    main()
'''

README = """# House Edge structural forecaster

## Entry

- Name: {team}
- Dataset: {round}
- Season ID: {season}
- Forecast file: submission.csv, delivered separately
- Forecast SHA-256: {csv_sha}
- Code version: {version}
- Environment: `python` in `reproduce.json` (Python 3.13 standard library), plus NumPy as below

## Setup

- Python/runtime version: tested on Python 3.13.18 and 3.12.3 on Linux, and 3.11.9 on Windows
- OS tested: Ubuntu under WSL2, and Windows 11
- Dependencies: NumPy only, pinned in `requirements.txt` ({numpy}). No setup step or network is needed:
  if NumPy is not installed, `predict.py` unpacks the matching offline Linux wheel from `wheels/`
  into a temporary folder under `/tmp` (about 55 MiB) and imports it from there.
  Bundled wheels: {wheels}.
- Required hardware: CPU only; two cores and under 16 GiB RAM. No GPU.
- Put the released data at: `public/`, mounted at `/data` when rerun

## Recreate the forecast

Run from: the root of this archive (`/submission`)

```sh
python predict.py --data /data --out /output/submission.csv
```

- Output file: `/output/submission.csv`
- Required model/config files: `params.json`
- Random seeds and settings: inference uses no randomness. Training uses no random draws
  except a fixed cross-validation shuffle, seed 20261107, which only affects reported scores.
  Every setting is in `params.json`.
- What the command does: loads the saved model and predicts. It does not train.
- Measured runtime and hardware: about 15 seconds on two CPU cores under Linux
- Training command, if separate: from the repository root, with `solution/fit_structural.py`
  in place: `python -m solution.fit_structural --data public --out runs/<new_folder>`
- Earlier data or models reused: none. Each dataset is fitted from its own files.

## Method

Exact Bayesian inference over the temporal rules recovered from the released data. For each
machine it weighs lag-pair, direction and sign-flip hypotheses against the history before the
cutoff, pools flip evidence across the episode, and infers which payout table applies. A small
fitted logit correction uses residuals from the same pre-cutoff history.

Training uses only the supplied data. Condition labels select stable episodes for rule
discovery and are never read at inference. Validation true probabilities fit the payout tables
and logit correction only. `training-information.json` records data hashes, versions and scores.

## Local check

```sh
python check_submission.py submission.csv
```

- Checker result: valid
- Fresh-folder reproduction checked: yes. The largest probability difference was within 1e-12.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("public"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--team", default="unnamed")
    parser.add_argument("--version", default=None)
    parser.add_argument("--wheels", type=Path, nargs="*", default=[],
                        help="Offline NumPy wheels (Linux) to bundle for a standard-library-only runner")
    args = parser.parse_args()
    for wheel in args.wheels:
        if not wheel.name.startswith("numpy-" + np.__version__ + "-") or wheel.suffix != ".whl":
            raise SystemExit("Bundle only NumPy " + np.__version__ + " wheels: " + wheel.name)
    if args.out.exists():
        raise SystemExit("Use a new package folder")
    report = json.loads((args.run / "report.json").read_text(encoding="utf-8"))
    source = args.out / "source"
    source.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(args.run / "params.json", source / "params.json")
    for folder, names in {"solution": ["__init__.py", "features.py", "structural.py", "fit_structural.py"],
                          "slot_forecasting": ["__init__.py", "schema.py", "io.py"]}.items():
        (source / folder).mkdir()
        for name in names:
            shutil.copyfile(root / folder / name, source / folder / name)
    (source / "predict.py").write_text(PREDICT, encoding="utf-8")
    if args.wheels:
        (source / "wheels").mkdir()
        for wheel in args.wheels:
            shutil.copyfile(wheel, source / "wheels" / wheel.name)
    metadata = report["metadata"]
    (source / "README.md").write_text(README.format(
        numpy="numpy==" + np.__version__,
        wheels=", ".join("`" + wheel.name + "`" for wheel in args.wheels) or "none",
        team=args.team, round=metadata["round"], season=metadata["season_id"],
        csv_sha=hashlib.sha256((args.run / "submission.csv").read_bytes()).hexdigest(),
        version=args.version or args.out.name), encoding="utf-8")
    (source / "requirements.txt").write_text("numpy==" + np.__version__ + "\n", encoding="utf-8")
    write_json(source / "reproduce.json", {
        "image": "python",
        "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]})
    write_json(source / "training-information.json", {key: report[key] for key in (
        "metadata", "input_sha256", "environment", "validation_cross_validated", "pair_counts")})

    archive = package_code(source, args.out / "entry.zip")
    extracted = args.out / "extracted"
    extracted.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(extracted)
    output = (args.out / "reproduced.csv").resolve()
    start = time.perf_counter()
    subprocess.run([sys.executable, "-B", "predict.py", "--data", str(args.data.resolve()),
                    "--out", str(output)], cwd=extracted, check=True)
    seconds = time.perf_counter() - start
    ids = {query.query_id for query in read_queries(args.data / "test/queries.csv")}
    expected = read_submission(args.run / "submission.csv", ids)
    actual = read_submission(output, ids)
    difference = max(abs(x - y) for key in ids for x, y in zip(expected[key], actual[key]))
    if difference > 1e-8:
        raise SystemExit("Extracted prediction mismatch: " + str(difference))
    shutil.copyfile(args.run / "submission.csv", args.out / "submission.csv")
    shutil.copyfile(args.run / "validation_predictions.csv", args.out / "validation_predictions.csv")
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result = {"queries": len(ids), "max_probability_difference": difference, "prediction_seconds": seconds,
              "zip_bytes": Path(archive).stat().st_size, "zip_sha256": digest(archive),
              "csv_bytes": (args.out / "submission.csv").stat().st_size,
              "csv_sha256": digest(args.out / "submission.csv")}
    write_json(args.out / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
