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
    roc_auc_score
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Early Stopping
# -------------------------
class EarlyStopping:
    def __init__(self, patience=20, stop_epoch=10):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, epoch, score, model, ckpt_path):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            self.counter += 1
            if self.counter >= self.patience and epoch >= self.stop_epoch:
                self.early_stop = True


# -------------------------
# Loss functions
# -------------------------
def get_loss_fns(args):
    if args.bag_loss == "svm":
        from topk.svm import SmoothTop1SVM
        bag_loss_fn = SmoothTop1SVM(n_classes=args.n_classes).to(device)
    else:
        bag_loss_fn = nn.CrossEntropyLoss()

    if args.inst_loss == "svm":
        from topk.svm import SmoothTop1SVM
        inst_loss_fn = SmoothTop1SVM(n_classes=2).to(device)
    else:
        inst_loss_fn = nn.CrossEntropyLoss()

    return bag_loss_fn, inst_loss_fn


# -------------------------
# Metrics
# -------------------------
def multi_label_auc(labels, scores, n_classes):
    aucs = []
    for c in range(n_classes):
        binary = (labels == c).astype(int)
        if binary.sum() == 0:
            continue
        aucs.append(roc_auc_score(binary, scores[:, c]))
    return np.mean(aucs)


# -------------------------
# Train / Eval loops
# -------------------------
def train_one_epoch(model, loader, optimizer, bag_loss_fn, bag_weight):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x5, x10, x20, label in loader:
        x5, x10, x20 = x5.to(device), x10.to(device), x20.to(device)
        label = torch.tensor(label).view(-1).to(device)

        optimizer.zero_grad()

        logits, instance_dict = model(x5, x10, x20, label)
        inst_loss = instance_dict["instance_loss"]

        bag_loss = bag_loss_fn(logits, label)
        loss = bag_weight * bag_loss + (1 - bag_weight) * inst_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == label).sum().item()
        total += label.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, bag_loss_fn, n_classes, bag_weight):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    total_loss, total = 0.0, 0

    for x5, x10, x20, label in loader:
        x5, x10, x20 = x5.to(device), x10.to(device), x20.to(device)
        label = torch.tensor(label).view(-1).to(device)

        logits, instance_dict = model(x5, x10, x20, label)
        inst_loss = instance_dict["instance_loss"]
        bag_loss = bag_loss_fn(logits, label)

        loss = bag_weight * bag_loss + (1 - bag_weight) * inst_loss
        total_loss += loss.item()
        total += label.size(0)

        probs = logits.softmax(dim=1)
        preds = logits.argmax(dim=1)

        y_true.extend(label.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_scores.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    auc = multi_label_auc(y_true, y_scores, n_classes)

    return {
        "loss": total_loss / total,
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "auc": auc
    }


# -------------------------
# Fold Training
# -------------------------
def train_fold(model, train_loader, val_loader, test_loader, args, fold):
    model.to(device)

    bag_loss_fn, _ = get_loss_fns(args)
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    ckpt = os.path.join(args.results_dir, f"fold_{fold}.pt")
    early_stopping = EarlyStopping() if args.early_stopping else None

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, bag_loss_fn, args.bag_weight
        )

        val_stats = evaluate(
            model, val_loader, bag_loss_fn, args.n_classes, args.bag_weight
        )

        print(
            f"[Fold {fold} | Epoch {epoch+1}] "
            f"Train Acc: {train_acc:.4f} | "
            f"Val AUC: {val_stats['auc']:.4f}"
        )

        if early_stopping:
            early_stopping(epoch, val_stats["auc"], model, ckpt)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        else:
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    return evaluate(model, test_loader, bag_loss_fn, args.n_classes, args.bag_weight)
