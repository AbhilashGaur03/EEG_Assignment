"""Representation diagnostics, aggregation, and plotting utilities."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import SEEDS


def representation_diagnostics(Z, y, subjects):
    class_sil = silhouette_score(Z, y) if len(np.unique(y)) > 1 else np.nan
    subject_sil = silhouette_score(Z, subjects) if len(np.unique(subjects)) > 1 else np.nan
    return {"class_silhouette": float(class_sil), "subject_silhouette": float(subject_sil)}


def linear_probe_scores(Z, y, subjects, seed=42):
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y)
    scaler_c = StandardScaler()
    Ztr = scaler_c.fit_transform(Z[idx_train])
    Zte = scaler_c.transform(Z[idx_test])
    class_probe = LogisticRegression(max_iter=5000)
    class_probe.fit(Ztr, y[idx_train])
    class_acc = accuracy_score(y[idx_test], class_probe.predict(Zte))

    idx_train_s, idx_test_s = train_test_split(
        np.arange(len(subjects)), test_size=0.3, random_state=seed, stratify=subjects
    )
    scaler_s = StandardScaler()
    Ztr_s = scaler_s.fit_transform(Z[idx_train_s])
    Zte_s = scaler_s.transform(Z[idx_test_s])
    subject_probe = LogisticRegression(max_iter=5000)
    subject_probe.fit(Ztr_s, subjects[idx_train_s])
    subject_acc = accuracy_score(subjects[idx_test_s], subject_probe.predict(Zte_s))
    subject_chance = 1.0 / len(np.unique(subjects))
    return {
        "class_probe_accuracy": float(class_acc),
        "subject_probe_accuracy": float(subject_acc),
        "subject_chance_level": float(subject_chance),
    }


def aggregate_results(df, group_cols):
    metrics = [
        "test_balanced_accuracy",
        "class_probe_accuracy",
        "subject_probe_accuracy",
        "class_silhouette",
        "subject_silhouette",
    ]
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        for m in metrics:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_std"] = float(g[m].std(ddof=0))
            row[f"{m}_best"] = float(g[m].max())
            row[f"{m}_worst"] = float(g[m].min())
        rows.append(row)
    return pd.DataFrame(rows)



def best_worst_runs(df, group_cols, metric="test_balanced_accuracy"):
    """Return transparent best/worst single runs for every experiment group.

    The project reports both extremes because subject-generalization results can
    vary noticeably with seed; showing only the best seed would overstate the
    reliability of the method.
    """
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = {col: key for col, key in zip(group_cols, keys)}
        best = g.loc[g[metric].idxmax()].to_dict()
        worst = g.loc[g[metric].idxmin()].to_dict()
        rows.append({**label, "run_type": "best", **best})
        rows.append({**label, "run_type": "worst", **worst})
    return pd.DataFrame(rows)

def plot_umap(Z, color_values, title, filename):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=SEEDS[0])
    U = reducer.fit_transform(Z)
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(U[:, 0], U[:, 1], c=color_values, cmap="tab10", s=18, alpha=0.85)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(sc)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    return U


def plot_sweep_errorbars(df, x_col, title, filename):
    plt.figure(figsize=(7.2, 4.8))
    for y_col, label in [
        ("test_balanced_accuracy_mean", "Test balanced accuracy"),
        ("class_probe_accuracy_mean", "Class probe"),
        ("subject_probe_accuracy_mean", "Subject probe"),
    ]:
        plt.errorbar(df[x_col], df[y_col], yerr=df[y_col.replace("_mean", "_std")], marker="o", linewidth=2, capsize=4, label=label)
    plt.xlabel(x_col)
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()


def plot_metric_all_seeds(histories, metric, title, filename):
    curves = [np.asarray(h[metric], dtype=float) for h in histories if metric in h]
    if len(curves) == 0:
        print(f"No curves found for metric: {metric}")
        return
    min_len = min(len(c) for c in curves)
    curves = np.stack([c[:min_len] for c in curves], axis=0)
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0)
    epochs = np.arange(1, min_len + 1)
    plt.figure(figsize=(7.2, 4.6))
    for i in range(curves.shape[0]):
        plt.plot(epochs, curves[i], alpha=0.35, linewidth=1.2, label=f"Seed {i+1}")
    plt.plot(epochs, mean_curve, linewidth=2.8, label="Mean")
    plt.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.18, label="±1 std")
    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " "))
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()
