import os
from argparse import Namespace

from models.cma_mil import new
from data.data_utils import data_generator
from utils.train_eval import train_fold


args = Namespace(
    data_root_5x="path_to_5x",
    data_root_10x="path_to_10x",
    data_root_20x="path_to_20x",
    results_dir="results",
    n_classes=3,
    epochs=20,
    lr=1e-4,
    bag_weight=0.7,
)

num_folds = 5

for fold in range(1, num_folds + 1):
    print(f"\nStarting Fold {fold}\n")

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

    model = new()
    stats = train_fold(model, train_loader, val_loader, test_loader, args, fold)

    print(f"Fold {fold} Test AUC: {stats['auc']:.4f}")
