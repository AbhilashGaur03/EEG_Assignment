"""Run all EEGNet subject-generalization experiments.

This script reproduces the original Colab workflow in a GitHub-friendly form.
"""

import os

import pandas as pd

from config import (
    APMRG_LAMBDAS,
    BATCH_SIZE,
    CCSA_BETAS,
    DEVICE,
    EPOCHS_BASELINE,
    EPOCHS_SWEEP,
    FIG_DIR,
    LR,
    MAIN_APMRG_LAMBDA,
    MAIN_CCSA_BETA,
    RESAMPLE_RATE,
    RESULTS_DIR,
    RUNS,
    SEEDS,
    TEST_SUBJECTS,
    TRAIN_SUBJECTS_FULL,
    TRAIN_SUBJECTS_SMALL,
    USE_MAX_NORM,
)
from training.ap_mrg_runner import run_ap_mrg
from utils.common import ensure_dirs, save_json
from data.dataset import load_dataset
from utils.diagnostics import aggregate_results, best_worst_runs, plot_sweep_errorbars, plot_umap
from methods.ap_mrg import apply_mrg, compute_mrg_weights
from training.loops import run_ccsa, run_standard_eegnet


def compact_tables(part1_summary, ccsa_summary, apmrg_summary):
    cols = [
        "test_balanced_accuracy_mean",
        "test_balanced_accuracy_std",
        "test_balanced_accuracy_best",
        "test_balanced_accuracy_worst",
        "class_probe_accuracy_mean",
        "subject_probe_accuracy_mean",
    ]
    part1_compact = part1_summary[["experiment_name", *cols]].copy()
    part2_compact = ccsa_summary[["beta", *cols]].copy()
    part3_compact = apmrg_summary[["lambda", *cols]].copy()
    return part1_compact, part2_compact, part3_compact


def save_best_worst(raw_df, group_cols, filename):
    table = best_worst_runs(raw_df, group_cols)
    table.to_csv(os.path.join(RESULTS_DIR, filename), index=False)
    return table


def main():
    ensure_dirs(RESULTS_DIR, FIG_DIR)
    print("Device:", DEVICE)
    print("Seeds:", SEEDS)

    print("Loading EEGBCI data...")
    full_train_set = load_dataset(TRAIN_SUBJECTS_FULL, RUNS)
    small_train_set = load_dataset(TRAIN_SUBJECTS_SMALL, RUNS)
    test_set = load_dataset(TEST_SUBJECTS, RUNS)

    print("Full train:", full_train_set.X.shape)
    print("Small train:", small_train_set.X.shape)
    print("Test:", test_set.X.shape)
    print("Channels:", len(full_train_set.ch_names))
    print("Time samples:", full_train_set.X.shape[-1])

    # Part 1 intentionally trains with no validation split because the assignment
    # defines fixed train/test subjects; holding out validation subjects would change
    # the amount of assigned training data used by the method.
    part1_rows, part1_results = [], {}
    for seed in SEEDS:
        res_full = run_standard_eegnet(
            full_train_set,
            test_set,
            seed=seed,
            epochs=EPOCHS_BASELINE,
            experiment_name="Part1_EEGNet_full_train_1_to_8",
        )
        res_small = run_standard_eegnet(
            small_train_set,
            test_set,
            seed=seed,
            epochs=EPOCHS_BASELINE,
            experiment_name="Part1_EEGNet_reduced_train_1_to_3",
        )
        part1_rows.extend([res_full["row"], res_small["row"]])
        part1_results[("full", seed)] = res_full
        part1_results[("small", seed)] = res_small

    part1_df = pd.DataFrame(part1_rows)
    part1_df.to_csv(os.path.join(RESULTS_DIR, "part1_raw.csv"), index=False)
    part1_summary = aggregate_results(part1_df, ["experiment_name"])
    part1_summary.to_csv(os.path.join(RESULTS_DIR, "part1_summary.csv"), index=False)
    part1_best_worst = save_best_worst(part1_df, ["experiment_name"], "part1_best_worst_runs.csv")

    rep_seed = SEEDS[0]
    plot_umap(part1_results[("full", rep_seed)]["Z_train"], part1_results[("full", rep_seed)]["Y_train"], "Part 1 EEGNet full-train embeddings by class", os.path.join(FIG_DIR, "part1_full_by_class.png"))
    plot_umap(part1_results[("full", rep_seed)]["Z_train"], part1_results[("full", rep_seed)]["S_train"], "Part 1 EEGNet full-train embeddings by subject", os.path.join(FIG_DIR, "part1_full_by_subject.png"))
    plot_umap(part1_results[("small", rep_seed)]["Z_train"], part1_results[("small", rep_seed)]["Y_train"], "Part 1 EEGNet reduced-train embeddings by class", os.path.join(FIG_DIR, "part1_reduced_by_class.png"))
    plot_umap(part1_results[("small", rep_seed)]["Z_train"], part1_results[("small", rep_seed)]["S_train"], "Part 1 EEGNet reduced-train embeddings by subject", os.path.join(FIG_DIR, "part1_reduced_by_subject.png"))

    # Part 2: CCSA
    ccsa_betas = CCSA_BETAS
    ccsa_rows, ccsa_results = [], {}
    for seed in SEEDS:
        for beta in ccsa_betas:
            res = run_ccsa(full_train_set, test_set, beta=beta, seed=seed, epochs=EPOCHS_SWEEP)
            ccsa_rows.append(res["row"])
            ccsa_results[(beta, seed)] = res
    ccsa_df = pd.DataFrame(ccsa_rows)
    ccsa_df.to_csv(os.path.join(RESULTS_DIR, "part2_ccsa_raw.csv"), index=False)
    ccsa_summary = aggregate_results(ccsa_df, ["beta"])
    ccsa_summary.to_csv(os.path.join(RESULTS_DIR, "part2_ccsa_summary.csv"), index=False)
    ccsa_best_worst = save_best_worst(ccsa_df, ["beta"], "part2_ccsa_best_worst_runs.csv")
    plot_sweep_errorbars(ccsa_summary, "beta", "CCSA sweep across random seeds", os.path.join(FIG_DIR, "ccsa_sweep_all_seeds_errorbars.png"))
    if MAIN_CCSA_BETA in ccsa_betas:
        rep_ccsa = ccsa_results[(MAIN_CCSA_BETA, SEEDS[0])]
        plot_umap(rep_ccsa["Z_train"], rep_ccsa["Y_train"], f"Part 2 CCSA beta={MAIN_CCSA_BETA} embeddings by class", os.path.join(FIG_DIR, "part2_ccsa_by_class.png"))
        plot_umap(rep_ccsa["Z_train"], rep_ccsa["S_train"], f"Part 2 CCSA beta={MAIN_CCSA_BETA} embeddings by subject", os.path.join(FIG_DIR, "part2_ccsa_by_subject.png"))

    # Part 3: AP-MRG
    # Weights are learned only from the training subjects, keeping the test
    # subjects completely unseen until final evaluation.
    mrg_weights, mrg_band_table = compute_mrg_weights(full_train_set.X, full_train_set.y, full_train_set.subjects, fs=RESAMPLE_RATE, alpha=1.0)
    mrg_band_table.to_csv(os.path.join(RESULTS_DIR, "part3_mrg_band_table.csv"), index=False)
    X_train_mrg = apply_mrg(full_train_set.X, mrg_weights, fs=RESAMPLE_RATE)
    X_test_mrg = apply_mrg(test_set.X, mrg_weights, fs=RESAMPLE_RATE)

    apmrg_lambdas = APMRG_LAMBDAS
    apmrg_rows, apmrg_results = [], {}
    for seed in SEEDS:
        for lam in apmrg_lambdas:
            res = run_ap_mrg(full_train_set, test_set, X_train_mrg, X_test_mrg, lam=lam, seed=seed, epochs=EPOCHS_SWEEP)
            apmrg_rows.append(res["row"])
            apmrg_results[(lam, seed)] = res
    apmrg_df = pd.DataFrame(apmrg_rows)
    apmrg_df.to_csv(os.path.join(RESULTS_DIR, "part3_apmrg_raw.csv"), index=False)
    apmrg_summary = aggregate_results(apmrg_df, ["lambda"])
    apmrg_summary.to_csv(os.path.join(RESULTS_DIR, "part3_apmrg_summary.csv"), index=False)
    apmrg_best_worst = save_best_worst(apmrg_df, ["lambda"], "part3_apmrg_best_worst_runs.csv")
    plot_sweep_errorbars(apmrg_summary, "lambda", "AP-MRG sweep across random seeds", os.path.join(FIG_DIR, "apmrg_sweep_all_seeds_errorbars.png"))
    if MAIN_APMRG_LAMBDA in apmrg_lambdas:
        rep_apmrg = apmrg_results[(MAIN_APMRG_LAMBDA, SEEDS[0])]
        plot_umap(rep_apmrg["Z_train"], rep_apmrg["Y_train"], f"Part 3 AP-MRG lambda={MAIN_APMRG_LAMBDA} embeddings by class", os.path.join(FIG_DIR, "part3_apmrg_by_class.png"))
        plot_umap(rep_apmrg["Z_train"], rep_apmrg["S_train"], f"Part 3 AP-MRG lambda={MAIN_APMRG_LAMBDA} embeddings by subject", os.path.join(FIG_DIR, "part3_apmrg_by_subject.png"))

    part1_compact, part2_compact, part3_compact = compact_tables(part1_summary, ccsa_summary, apmrg_summary)
    part1_compact.to_csv(os.path.join(RESULTS_DIR, "part1_compact.csv"), index=False)
    part2_compact.to_csv(os.path.join(RESULTS_DIR, "part2_ccsa_compact.csv"), index=False)
    part3_compact.to_csv(os.path.join(RESULTS_DIR, "part3_apmrg_compact.csv"), index=False)

    final_metrics = {
        "configuration": {
            "train_subjects_full": TRAIN_SUBJECTS_FULL,
            "train_subjects_small": TRAIN_SUBJECTS_SMALL,
            "test_subjects": TEST_SUBJECTS,
            "runs": RUNS,
            "seeds": SEEDS,
            "seed_policy": "All reported summaries use every fixed seed; best and worst single runs are saved separately to avoid cherry-picking.",
            "training_protocol": "fixed epochs, no validation split, train on assigned training subjects",
            "epochs_baseline": EPOCHS_BASELINE,
            "epochs_sweep": EPOCHS_SWEEP,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "use_max_norm_eegnet": USE_MAX_NORM,
        },
        "part1_raw": part1_df.to_dict(orient="records"),
        "part1_summary": part1_summary.to_dict(orient="records"),
        "part1_best_worst_runs": part1_best_worst.to_dict(orient="records"),
        "part2_ccsa_raw": ccsa_df.to_dict(orient="records"),
        "part2_ccsa_summary": ccsa_summary.to_dict(orient="records"),
        "part2_ccsa_best_worst_runs": ccsa_best_worst.to_dict(orient="records"),
        "part3_mrg_band_table": mrg_band_table.to_dict(orient="records"),
        "part3_mrg_weights": mrg_weights.tolist(),
        "part3_apmrg_raw": apmrg_df.to_dict(orient="records"),
        "part3_apmrg_summary": apmrg_summary.to_dict(orient="records"),
        "part3_apmrg_best_worst_runs": apmrg_best_worst.to_dict(orient="records"),
    }
    save_json(final_metrics, os.path.join(RESULTS_DIR, "metrics.json"))

    lines = [
        "RESULTS SUMMARY",
        "Protocol: fixed epochs, no validation split; train on assigned training subjects.",
        f"Fixed random seeds: {SEEDS}",
        "",
        "PART 1",
        part1_compact.to_string(index=False),
        "",
        "PART 1 BEST/WORST SINGLE RUNS",
        part1_best_worst.to_string(index=False),
        "",
        "PART 2 CCSA",
        part2_compact.to_string(index=False),
        "",
        "PART 2 CCSA BEST/WORST SINGLE RUNS",
        ccsa_best_worst.to_string(index=False),
        "",
        "PART 3 AP-MRG",
        part3_compact.to_string(index=False),
        "",
        "PART 3 AP-MRG BEST/WORST SINGLE RUNS",
        apmrg_best_worst.to_string(index=False),
        "",
        "AP-MRG band reliability table",
        mrg_band_table.to_string(index=False),
    ]
    with open(os.path.join(RESULTS_DIR, "log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("\nSaved results under:", RESULTS_DIR)


if __name__ == "__main__":
    main()
