# Flip-Scope

Code for analyzing how model scores and alert decisions vary across model retraining runs.

## Project structure

- `experiments/` — experiment implementations and benchmark processing code.
- `src/inflight/` — reusable score, calibration, and public-data utilities.
- `configs/` — experiment protocol configurations.
- `tests/` — focused checks for research calculations and utility functions.

Raw benchmark records, row-level mappings and predictions, trained model weights, generated reports, and run caches are not included. Obtain any benchmark data separately and follow its source license and citation requirements. Do not place credentials or local data in this repository.

## Setup

Requires Python 3.10 or newer.

```bash
python -m pip install -e '.[test]'
PYTHONPATH=src:. python -m experiments.core.run_w1_w4 --help
```

Experiment entry points may require locally prepared benchmark data and saved outputs. See `experiments/README.md` for the code layout.
