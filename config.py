"""Configuration for EEGNet subject-generalization experiments."""

import os
import torch

# FAST_DEBUG is kept as an explicit reproducibility switch: False reproduces the
# submitted protocol, while True is only for local smoke tests before a commit.
FAST_DEBUG = False

if FAST_DEBUG:
    SEEDS = [42]
    EPOCHS_BASELINE = 60
    EPOCHS_SWEEP = 60
else:
    # Seeds are fixed and reported so every aggregate includes the same runs;
    # this avoids reporting only a favourable random initialization.
    SEEDS = [42, 7, 123]
    EPOCHS_BASELINE = 60
    EPOCHS_SWEEP = 60

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

# Assignment fixed subset
TRAIN_SUBJECTS_FULL = [1, 2, 3, 4, 5, 6, 7, 8]
TRAIN_SUBJECTS_SMALL = [1, 2, 3]
TEST_SUBJECTS = [9, 10]
RUNS = [6, 10, 14]

# Hyperparameter sweeps are centralized here so `python main.py` runs the full
# protocol without editing notebooks or scripts.
CCSA_BETAS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 3.0, 5, 7, 9, 10.0]
APMRG_LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# EEG preprocessing
LOW_FREQ = 4.0
HIGH_FREQ = 38.0
RESAMPLE_RATE = 160
TMIN = 0.5
TMAX = 2.5
EVENT_ID = {"T1": 0, "T2": 1}

# Training setup
BATCH_SIZE = 32
LR = 1e-3
USE_MAX_NORM = True

# Representative settings for plots/write-up; these are predeclared rather than
# selected from test accuracy, so the visualizations do not become cherry-picked.
MAIN_CCSA_BETA = 1
MAIN_APMRG_LAMBDA = 0.8

# AP-MRG frequency bands
MRG_BANDS = [
    ("theta_4_8", 4, 8),
    ("mu_8_12", 8, 12),
    ("low_beta_12_16", 12, 16),
    ("mid_beta_16_24", 16, 24),
    ("high_beta_24_38", 24, 38),
]
