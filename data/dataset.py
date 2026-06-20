"""EEGBCI loading, preprocessing, and PyTorch dataset helpers."""

from dataclasses import dataclass
from typing import List

import mne
import numpy as np
import torch
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

from config import EVENT_ID, HIGH_FREQ, LOW_FREQ, RESAMPLE_RATE, TMAX, TMIN


@dataclass
class EpochSet:
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    ch_names: List[str]


def load_subject_raw(subject: int, runs: List[int]):
    raw_fnames = eegbci.load_data(subject, runs)
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
    raw = concatenate_raws(raws)

    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    # Missing channels are ignored because EEGBCI recordings are already standardized;
    # rejecting a subject for one absent montage label would reduce the fixed protocol.
    raw.set_montage(montage, on_missing="ignore")
    return raw


def preprocess_raw(raw):
    raw.pick_types(eeg=True, meg=False, stim=False)
    raw.filter(LOW_FREQ, HIGH_FREQ, fir_design="firwin", verbose=False)
    raw.resample(RESAMPLE_RATE, verbose=False)
    return raw


def raw_to_epochs(raw):
    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    valid_event_id = {}
    for key in EVENT_ID:
        if key in event_dict:
            valid_event_id[key] = event_dict[key]

    epochs = mne.Epochs(
        raw,
        events,
        event_id=valid_event_id,
        tmin=TMIN,
        tmax=TMAX,
        # No baseline correction is used because the downstream per-trial z-score
        # makes scale/offset comparable without using a hand-selected baseline window.
        baseline=None,
        preload=True,
        verbose=False,
    )

    X = epochs.get_data()
    inv_event_dict = {v: k for k, v in event_dict.items()}
    y = np.asarray([EVENT_ID[inv_event_dict[event_code]] for event_code in epochs.events[:, -1]], dtype=np.int64)
    return X, y, epochs.ch_names


def per_trial_zscore(X: np.ndarray) -> np.ndarray:
    # The assignment targets cross-subject generalization, so each trial is
    # normalized independently to suppress subject/session scale offsets while
    # retaining within-trial temporal and spatial EEG structure.
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True) + 1e-6
    return (X - mean) / std


def load_dataset(subjects: List[int], runs: List[int]) -> EpochSet:
    all_X, all_y, all_subjects = [], [], []
    ch_names_ref = None

    for subj in subjects:
        print(f"Loading subject {subj}...")
        raw = preprocess_raw(load_subject_raw(subj, runs))
        X, y, ch_names = raw_to_epochs(raw)
        X = per_trial_zscore(X)

        all_X.append(X)
        all_y.append(y)
        all_subjects.append(np.full(len(y), subj))
        if ch_names_ref is None:
            ch_names_ref = ch_names

    return EpochSet(
        X=np.concatenate(all_X, axis=0),
        y=np.concatenate(all_y, axis=0),
        subjects=np.concatenate(all_subjects, axis=0),
        ch_names=ch_names_ref,
    )


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, subjects):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.subjects = torch.tensor(subjects, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx], self.subjects[idx]


def make_loader(X, y, subjects, batch_size=32, shuffle=True):
    ds = EEGDataset(X, y, subjects)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
