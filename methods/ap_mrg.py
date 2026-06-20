"""Adaptive passband multi-resolution reweighting (AP-MRG)."""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from config import MRG_BANDS
from data.dataset import per_trial_zscore


def bandpass_epochs(X, fs, low, high, order=4):
    nyq = fs / 2
    high = min(high, nyq - 1.0)
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, X, axis=-1)


def band_logpower(X_band):
    return np.log(np.mean(X_band ** 2, axis=-1) + 1e-8)


def class_discriminability(F, y):
    F0, F1 = F[y == 0], F[y == 1]
    mu0, mu1 = F0.mean(axis=0), F1.mean(axis=0)
    std0, std1 = F0.std(axis=0), F1.std(axis=0)
    d = np.abs(mu1 - mu0) / (0.5 * (std0 + std1) + 1e-6)
    return float(np.mean(d))


def subject_nuisance(F, subjects):
    subject_means = np.stack([F[subjects == s].mean(axis=0) for s in np.unique(subjects)], axis=0)
    between = np.var(subject_means, axis=0)
    total = np.var(F, axis=0) + 1e-6
    return float(np.mean(between / total))


def softmax_np(x, temperature=1.0):
    x = np.asarray(x, dtype=np.float64) / temperature
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def compute_mrg_weights(X_train, y_train, subjects_train, fs=160, alpha=1.0):
    rows, reliability = [], []
    for band_name, low, high in MRG_BANDS:
        Xb = bandpass_epochs(X_train, fs=fs, low=low, high=high)
        F = band_logpower(Xb)
        D = class_discriminability(F, y_train)
        S = subject_nuisance(F, subjects_train)
        # A reliable band should separate classes but not mostly identify subjects;
        # dividing by subject nuisance encodes that design goal directly.
        R = D / ((S + 1e-6) ** alpha)
        rows.append({"band": band_name, "low_hz": low, "high_hz": high, "class_discriminability": D, "subject_nuisance": S, "raw_reliability": R})
        reliability.append(R)
    reliability = np.asarray(reliability)
    normalized = (reliability - reliability.mean()) / (reliability.std() + 1e-6)
    # Softmax keeps all bands active instead of hard-selecting one band, which is
    # safer for EEG where discriminative information can be distributed.
    weights = softmax_np(normalized, temperature=1.0)
    for i in range(len(rows)):
        rows[i]["normalized_reliability"] = float(normalized[i])
        rows[i]["mrg_weight"] = float(weights[i])
    return weights, pd.DataFrame(rows)


def apply_mrg(X, weights, fs=160):
    X_out = np.zeros_like(X)
    for w, (_, low, high) in zip(weights, MRG_BANDS):
        X_out += w * bandpass_epochs(X, fs=fs, low=low, high=high)
    return per_trial_zscore(X_out)


def mix_original_mrg(X_original, X_mrg, lam):
    if lam == 0.0:
        return X_original.copy()
    if lam == 1.0:
        return X_mrg.copy()
    # Convex mixing treats AP-MRG as a controlled perturbation of the original
    # signal rather than replacing the input with a completely new preprocessing.
    return per_trial_zscore((1.0 - lam) * X_original + lam * X_mrg)
