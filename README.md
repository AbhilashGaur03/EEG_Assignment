# EEGNet Subject-Generalization Experiments

This repository is a real Python refactor of the original Colab notebook for EEGBCI motor-imagery subject-generalization experiments. The notebook is preserved in `notebooks/`, but the runnable code is split into modules instead of placeholder folders.

## Structure

```text
.
├── main.py                  # Runs the full experiment pipeline
├── config.py                # Fixed protocol, seeds, hyperparameters, sweep values
├── data/
│   └── dataset.py           # EEGBCI download/loading, preprocessing, DataLoader
├── models/
│   └── eegnet.py            # EEGNet architecture and max-norm projection
├── methods/
│   ├── ap_mrg.py            # AP-MRG band weighting and signal mixing
│   └── ccsa.py              # Class-conditional subject-alignment objective
├── training/
│   ├── loops.py             # Training/evaluation loops and baseline/CCSA runners
│   └── ap_mrg_runner.py     # AP-MRG experiment runner
├── utils/
│   ├── common.py            # Seeds, directory creation, JSON saving
│   └── diagnostics.py       # Probes, aggregation, plotting, best/worst reporting
├── notebooks/
│   └── test_EEG_4.ipynb     # Original Colab notebook
└── results/
    └── figures/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the pipeline

```bash
python main.py
```

`main.py` runs the complete protocol without manual notebook steps: EEGBCI download through MNE, preprocessing, EEGNet baseline, CCSA sweep, AP-MRG sweep, figures, CSV tables, `metrics.json`, and `log.txt`.

The first run requires internet access because MNE downloads EEGBCI files.

## Reproducibility and reporting

The fixed seeds are documented in `config.py`:

```python
SEEDS = [42, 7, 123]
```

The code reports mean, standard deviation, best run, and worst run. It also saves explicit best/worst single-run tables:

```text
results/part1_best_worst_runs.csv
results/part2_ccsa_best_worst_runs.csv
results/part3_apmrg_best_worst_runs.csv
```

This avoids cherry-picking one favourable run.

## Quick smoke test

For a short check of imports and execution flow, set this in `config.py`:

```python
FAST_DEBUG = True
```

Then run:

```bash
python main.py
```
# EEG_Assignment
