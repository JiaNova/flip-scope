# Experiment code

This directory contains implementations for the project's retraining, calibration, stability, and public benchmark analyses.

| Module group | Code areas |
|---|---|
| `core/` | Core model, calibration, audit, variance, timing, and sensitivity experiments |
| `public_benchmarks/` | Public benchmark preprocessing and experiment entry points |
| `adult_support/` | Adult benchmark preprocessing and analysis |

The scripts preserve experiment parameters and expected input paths. Benchmark data, individual-level outputs, model weights, and caches must be prepared locally and are not stored in this repository. Run modules from the repository root with `PYTHONPATH=src:.`.
