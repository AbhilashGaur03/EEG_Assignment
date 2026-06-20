"""AP-MRG experiment runner."""

import torch
import torch.nn as nn

from config import BATCH_SIZE, DEVICE, LR
from models.eegnet import EEGNet
from utils.common import set_seed
from data.dataset import EpochSet, make_loader
from utils.diagnostics import linear_probe_scores, representation_diagnostics
from methods.ap_mrg import mix_original_mrg
from training.loops import evaluate, extract_embeddings, train_one_epoch


def run_ap_mrg(train_set: EpochSet, test_set: EpochSet, X_train_mrg, X_test_mrg, lam: float, seed: int, epochs: int):
    set_seed(seed)
    Xtr = mix_original_mrg(train_set.X, X_train_mrg, lam)
    Xte = mix_original_mrg(test_set.X, X_test_mrg, lam)

    train_loader = make_loader(Xtr, train_set.y, train_set.subjects, BATCH_SIZE, True)
    test_loader = make_loader(Xte, test_set.y, test_set.subjects, BATCH_SIZE, False)
    dev_loader = make_loader(Xtr, train_set.y, train_set.subjects, BATCH_SIZE, False)

    model = EEGNet(n_channels=Xtr.shape[1], n_times=Xtr.shape[2]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = {"train_bal_acc": [], "train_loss": []}

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion)
        history["train_bal_acc"].append(float(train_stats["balanced_accuracy"]))
        history["train_loss"].append(float(train_stats["loss"]))
        if epoch == 1 or epoch % 10 == 0:
            print(f"AP-MRG lambda={lam} | seed={seed} | epoch={epoch:03d} | train_bal={train_stats['balanced_accuracy']:.3f} | train_loss={train_stats['loss']:.4f}")

    test_stats = evaluate(model, test_loader, criterion)
    Z, Y, S = extract_embeddings(model, dev_loader)
    row = {"method": "AP-MRG", "lambda": float(lam), "seed": int(seed), "test_accuracy": float(test_stats["accuracy"]), "test_balanced_accuracy": float(test_stats["balanced_accuracy"]), **representation_diagnostics(Z, Y, S), **linear_probe_scores(Z, Y, S, seed=seed)}
    return {"row": row, "model": model, "history": history, "Z_train": Z, "Y_train": Y, "S_train": S}
