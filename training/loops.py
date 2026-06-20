"""Training, evaluation, CCSA loss, and experiment runners."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from config import BATCH_SIZE, DEVICE, LR
from models.eegnet import EEGNet, project_eegnet_max_norm
from data.dataset import EpochSet, make_loader
from utils.diagnostics import linear_probe_scores, representation_diagnostics
from utils.common import set_seed
from methods.ccsa import ccsa_loss


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for X_batch, y_batch, _ in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        project_eegnet_max_norm(model)
        total_loss += loss.item() * len(y_batch)
        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
        all_labels.extend(y_batch.detach().cpu().numpy())
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
    }


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for X_batch, y_batch, _ in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
    }


@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()
    Z, Y, S = [], [], []
    for X_batch, y_batch, s_batch in loader:
        X_batch = X_batch.to(DEVICE)
        _, z = model(X_batch, return_features=True)
        Z.append(z.cpu().numpy())
        Y.append(y_batch.numpy())
        S.append(s_batch.numpy())
    return np.concatenate(Z), np.concatenate(Y), np.concatenate(S)




def train_one_epoch_ccsa(model, loader, optimizer, criterion, beta):
    model.train()
    total_loss, total_ce, total_align = 0.0, 0.0, 0.0
    all_preds, all_labels = [], []
    for X_batch, y_batch, s_batch in loader:
        X_batch, y_batch, s_batch = X_batch.to(DEVICE), y_batch.to(DEVICE), s_batch.to(DEVICE)
        optimizer.zero_grad()
        logits, z = model(X_batch, return_features=True)
        ce = criterion(logits, y_batch)
        align = ccsa_loss(z, y_batch, s_batch)
        loss = ce + beta * align
        loss.backward()
        optimizer.step()
        project_eegnet_max_norm(model)
        total_loss += loss.item() * len(y_batch)
        total_ce += ce.item() * len(y_batch)
        total_align += align.item() * len(y_batch)
        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
        all_labels.extend(y_batch.detach().cpu().numpy())
    return {
        "loss": total_loss / len(loader.dataset),
        "ce_loss": total_ce / len(loader.dataset),
        "align_loss": total_align / len(loader.dataset),
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
    }


def _make_standard_components(train_set: EpochSet, test_set: EpochSet):
    train_loader = make_loader(train_set.X, train_set.y, train_set.subjects, BATCH_SIZE, True)
    test_loader = make_loader(test_set.X, test_set.y, test_set.subjects, BATCH_SIZE, False)
    dev_loader = make_loader(train_set.X, train_set.y, train_set.subjects, BATCH_SIZE, False)
    model = EEGNet(n_channels=train_set.X.shape[1], n_times=train_set.X.shape[2]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    return train_loader, test_loader, dev_loader, model, optimizer, criterion


def run_standard_eegnet(train_set: EpochSet, test_set: EpochSet, seed: int, epochs: int, experiment_name: str):
    set_seed(seed)
    train_loader, test_loader, dev_loader, model, optimizer, criterion = _make_standard_components(train_set, test_set)
    history = {"train_bal_acc": [], "train_loss": []}
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion)
        history["train_bal_acc"].append(float(train_stats["balanced_accuracy"]))
        history["train_loss"].append(float(train_stats["loss"]))
        if epoch == 1 or epoch % 10 == 0:
            print(f"{experiment_name} | seed={seed} | epoch={epoch:03d} | train_bal={train_stats['balanced_accuracy']:.3f} | train_loss={train_stats['loss']:.4f}")
    test_stats = evaluate(model, test_loader, criterion)
    Z, Y, S = extract_embeddings(model, dev_loader)
    row = {"experiment_name": experiment_name, "seed": int(seed), "test_accuracy": float(test_stats["accuracy"]), "test_balanced_accuracy": float(test_stats["balanced_accuracy"]), **representation_diagnostics(Z, Y, S), **linear_probe_scores(Z, Y, S, seed=seed)}
    return {"row": row, "model": model, "history": history, "Z_train": Z, "Y_train": Y, "S_train": S}


def run_ccsa(train_set: EpochSet, test_set: EpochSet, beta: float, seed: int, epochs: int):
    set_seed(seed)
    train_loader, test_loader, dev_loader, model, optimizer, criterion = _make_standard_components(train_set, test_set)
    history = {"train_bal_acc": [], "train_loss": [], "ce_loss": [], "align_loss": []}
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch_ccsa(model, train_loader, optimizer, criterion, beta)
        history["train_bal_acc"].append(float(train_stats["balanced_accuracy"]))
        history["train_loss"].append(float(train_stats["loss"]))
        history["ce_loss"].append(float(train_stats["ce_loss"]))
        history["align_loss"].append(float(train_stats["align_loss"]))
        if epoch == 1 or epoch % 10 == 0:
            print(f"CCSA beta={beta} | seed={seed} | epoch={epoch:03d} | train_bal={train_stats['balanced_accuracy']:.3f} | CE={train_stats['ce_loss']:.4f} | Align={train_stats['align_loss']:.4f}")
    test_stats = evaluate(model, test_loader, criterion)
    Z, Y, S = extract_embeddings(model, dev_loader)
    row = {"method": "CCSA", "beta": float(beta), "seed": int(seed), "test_accuracy": float(test_stats["accuracy"]), "test_balanced_accuracy": float(test_stats["balanced_accuracy"]), **representation_diagnostics(Z, Y, S), **linear_probe_scores(Z, Y, S, seed=seed)}
    return {"row": row, "model": model, "history": history, "Z_train": Z, "Y_train": Y, "S_train": S}
