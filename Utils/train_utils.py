import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimal_thresh(fpr, tpr, thresholds):
    idx = np.argmin(fpr - tpr)
    return thresholds[idx]


def multi_label_roc(labels, predictions, num_classes):
    aucs = []
    for c in range(num_classes):
        binary = (labels == c).astype(int)
        if binary.sum() == 0:
            continue
        aucs.append(roc_auc_score(binary, predictions[:, c]))
    return aucs


@torch.no_grad()
def evaluate(model, loader, loss_fn, n_classes, bag_weight):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    total_loss, total = 0.0, 0

    for x5, x10, x20, label in loader:
        x5, x10, x20 = x5.to(device), x10.to(device), x20.to(device)
        label = torch.tensor(label).view(-1).to(device)

        logits, inst_loss = model(x5, x10, x20, label)
        bag_loss = loss_fn(logits, label)
        loss = bag_weight * bag_loss + (1.0 - bag_weight) * inst_loss

        total_loss += loss.item()
        total += label.size(0)

        probs = logits.softmax(dim=1)
        preds = logits.argmax(dim=1)

        y_true.extend(label.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_scores.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    auc = np.mean(multi_label_roc(y_true, y_scores, n_classes))

    return {
        "loss": total_loss / max(total, 1),
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "auc": auc
    }


def train_one_epoch(model, loader, optimizer, loss_fn, bag_weight):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x5, x10, x20, label in loader:
        x5, x10, x20 = x5.to(device), x10.to(device), x20.to(device)
        label = torch.tensor(label).view(-1).to(device)

        optimizer.zero_grad()
        logits, inst_loss = model(x5, x10, x20, label)
        bag_loss = loss_fn(logits, label)
        loss = bag_weight * bag_loss + (1.0 - bag_weight) * inst_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == label).sum().item()
        total += label.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def train_fold(model, train_loader, val_loader, test_loader, args, fold):
    model.to(device)
    os.makedirs(args.results_dir, exist_ok=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_auc = -np.inf
    ckpt = os.path.join(args.results_dir, f"fold_{fold}.pt")

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, args.bag_weight
        )

        val_stats = evaluate(
            model, val_loader, loss_fn, args.n_classes, args.bag_weight
        )

        print(
            f"[Fold {fold} | Epoch {epoch+1}] "
            f"Train Acc: {train_acc:.4f} | "
            f"Val AUC: {val_stats['auc']:.4f}"
        )

        if val_stats["auc"] > best_auc:
            best_auc = val_stats["auc"]
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    return evaluate(model, test_loader, loss_fn, args.n_classes, args.bag_weight)
