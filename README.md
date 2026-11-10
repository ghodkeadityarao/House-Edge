<h1 align="center">House Edge</h1>

<p align="center">
  Forecasting a slot machine's next payout odds (0×, 1×, 2×, 5× or 20×) from its spin history.<br>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue">
  <img alt="NumPy" src="https://img.shields.io/badge/numpy-only-informational">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green">
</p>

For each query the model returns five probabilities that sum to 1, using only ticks before the cutoff. It does not fit a generic regressor. It performs exact Bayesian inference over the rules that generate the data: which lagged payouts matter, in which direction, and when a machine's behavior changes.

## Results

Error is measured against the true probabilities on the validation data and normalized by a pooled-frequency baseline (`R = 1` means the same error as the baseline; lower is better).

| Dataset | Normalized error `R` | Baseline | Note |
| --- | --- | --- | --- |
| `results/v1` | 0.018 | 1.00 | Cross-validated; rules found: lag pairs (1,3), (2,5), (4,7) |
| `results/v2` | 0.015 | 1.00 | Cross-validated; rules found: lag pairs (1,4), (2,6), (3,8) |

The second dataset uses different lag rules. The model fitted to the first scored worse than the baseline there, so the rules are refitted from each dataset's own training data. Scores are cross-validated: each episode's logit correction is fitted without that episode. Held-out checks and a backtest on unseen histories matched these figures within their confidence intervals.

## How it works

```mermaid
flowchart LR
    A[Spin history before cutoff] --> B[Rule discovery on stable machines]
    B --> C[Bayesian inference per machine]
    C --> D[Pool flip evidence across the episode]
    D --> E[Logit correction]
    E --> F[Five payout probabilities]
```

1. **Rule discovery.** Stable histories are used to learn which lag sets (1 to 3 lags) predict a win, along with each set's truth table, the payout-size lags and the shared-machine lag.
2. **Inference.** For each machine, the model weighs rule direction, sign-flip timing and payout-table hypotheses against the history before the cutoff. It pools flip evidence across the episode.
3. **Correction.** A small fitted logit correction handles the remaining residuals.
4. **Check.** `family_check` in the fit report flags whether the learned rule family still describes the data.

## Layout

| Path | Contents |
| --- | --- |
| `solution/structural.py` | The model: Bayesian inference over rules and change points |
| `solution/fit_structural.py` | Fits all parameters from a dataset and writes forecasts and a report |
| `solution/package_structural.py` | Builds a self-contained `entry.zip` and checks that it reproduces the CSV |
| `solution/features.py` | History indexing helpers |
| `slot_forecasting/` | Data loading, schema, scoring and baseline library |
| `check_submission.py` | Validates a forecast CSV and scores it on validation data |
| `tools/verification/` | Hold-out scoring, backtest and residual-probe scripts |
| `tests/` | Unit tests, plus simulated datasets with changed rules |
| `results/v1`, `results/v2` | Forecasts and packaged models for the two datasets |
| `vendor/numpy-wheels/` | Offline NumPy wheels bundled into packaged entries |
| `DATA.md` | Input file formats |

Each `results/` folder holds `submission.csv` and the matching `entry.zip`. `results/v2` also holds the cross-validated and in-sample validation forecasts, the fit report and the reproduction check.

## Usage

Python 3.11+ and NumPy are required.

```sh
pip install -r requirements-solution.txt
```

Place a dataset in `public/` (see `DATA.md`). Then fit, check and package:

```sh
python -m solution.fit_structural --data public --out runs/fit_01
python check_submission.py runs/fit_01/validation_cv.csv --split validation
python check_submission.py runs/fit_01/submission.csv
python -m solution.package_structural --run runs/fit_01 --data public --out results/new --team NAME \
    --wheels vendor/numpy-wheels/numpy-2.3.4-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

A fit takes about 3 to 11 minutes on a laptop CPU. It is deterministic. `validation_cv.csv` is the out-of-sample estimate; `validation_predictions.csv` is in-sample for the correction and scores slightly higher.

To reproduce a packaged entry, unzip it and run:

```sh
python predict.py --data /path/to/public --out /path/to/submission.csv
```

Inference uses no randomness and takes about 15 seconds on two CPU cores.

## Tests

```sh
python -m unittest discover -s tests -p "test_*.py"
```
