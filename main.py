import os
import argparse
import random
import numpy as np
import torch

from Model.cma-mil import CMA_MIL
from Utils.data_utils import data_generator
from Utils.train_eval import train_fold


def set_seed(seed):
    seed=0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser("CMA-MIL Training")

    # Data
    parser.add_argument("--data_root_5x", type=str, required=True)
    parser.add_argument("--data_root_10x", type=str, required=True)
    parser.add_argument("--data_root_20x", type=str, required=True)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--bag_weight", type=float, default=0.7)
    parser.add_argument("--n_classes", type=int, default=2)

    # Loss
    parser.add_argument("--bag_loss", type=str, default="svm", choices=["svm", "ce"])
    parser.add_argument("--inst_loss", type=str, default="svm", choices=["svm", "ce"])

    # CV / misc
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--early_stopping", action="store_true")

    return parser.parse_args()

def main():
    args = get_args()
    set_seed(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)

    print(" CMA-MIL Training")
    print(vars(args))

    all_fold_auc = []

    for fold in range(1, args.folds + 1):
        print(f"\n========== Fold {fold}/{args.folds} ==========\n")

        train_loader = list(
            data_generator(
                os.path.join(args.data_root_5x, str(fold), "train"),
                os.path.join(args.data_root_10x, str(fold), "train"),
                os.path.join(args.data_root_20x, str(fold), "train"),
            )
        )

        val_loader = list(
            data_generator(
                os.path.join(args.data_root_5x, str(fold), "val"),
                os.path.join(args.data_root_10x, str(fold), "val"),
                os.path.join(args.data_root_20x, str(fold), "val"),
            )
        )

        test_loader = list(
            data_generator(
                os.path.join(args.data_root_5x, str(fold), "test"),
                os.path.join(args.data_root_10x, str(fold), "test"),
                os.path.join(args.data_root_20x, str(fold), "test"),
            )
        )

        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()

        model = CMA_MIL(n_classes=args.num_classes, instance_loss_fn=instance_loss_fn)
        stats = train_fold(model, train_loader, val_loader, test_loader, args, fold)

        print(
            f"[Fold {fold}] "
            f"Test AUC: {stats['auc']:.4f} | "
            f"Acc: {stats['acc']:.4f} | "
            f"F1: {stats['f1']:.4f}"
        )

        all_fold_auc.append(stats["auc"])

    print("\n====================================")
    print(" Cross-validation Summary")
    print("====================================")
    print(f"Mean AUC : {np.mean(all_fold_auc):.4f}")
    print(f"Std  AUC : {np.std(all_fold_auc):.4f}")
    print("====================================")


if __name__ == "__main__":
    main()

