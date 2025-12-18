import os
import glob
import argparse
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

class BagDataset(Dataset):
    def __init__(self, patch_paths, transform=None):
        self.patch_paths = patch_paths
        self.transform = transform

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        img = Image.open(self.patch_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def build_shufflenet():
    pretrained_model = models.shufflenet_v2_x1_0(pretrained=True)

    feature_extractor = nn.Sequential(
        *list(pretrained_model.children())[:-1]  # remove classifier
    )

    projector = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True)
    )

    model = nn.Sequential(feature_extractor, projector)
    return model

@torch.no_grad()
def compute_features_for_bags(
    bags_list,
    model,
    save_path,
    batch_size,
    num_workers,
    device
):
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    for i, bag_path in enumerate(bags_list):
        patch_paths = (
            glob.glob(os.path.join(bag_path, "*.jpg")) +
            glob.glob(os.path.join(bag_path, "*.jpeg"))
        )

        if len(patch_paths) == 0:
            print(f"[Warning] No patches found in {bag_path}")
            continue

        dataset = BagDataset(patch_paths, transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        feats = []

        for patches in loader:
            patches = patches.to(device)
            outputs = model(patches)
            feats.append(outputs.cpu().numpy())

        feats = np.concatenate(feats, axis=0)

        class_name = os.path.basename(os.path.dirname(bag_path))
        slide_name = os.path.basename(bag_path)

        out_dir = os.path.join(save_path, class_name)
        os.makedirs(out_dir, exist_ok=True)

        out_csv = os.path.join(out_dir, f"{slide_name}.csv")
        pd.DataFrame(feats).to_csv(out_csv, index=False, float_format="%.4f")

        sys.stdout.write(
            f"\rComputed [{i+1}/{len(bags_list)}] : {slide_name}"
        )

    print("\nFeature extraction completed.")

def main():
    parser = argparse.ArgumentParser("Compute ShuffleNet Features for CMA-MIL")
    parser.add_argument("--patch_root", type=str, required=True,
                        help="Root folder (e.g., patches_5x / patches_10x / patches_20x)")
    parser.add_argument("--out_dir", type=str, default="features")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = build_shufflenet().to(device)

    bags_list = glob.glob(os.path.join(args.patch_root, "*", "*"))
    save_path = os.path.join(args.out_dir, os.path.basename(args.patch_root))
    os.makedirs(save_path, exist_ok=True)

    compute_features_for_bags(
        bags_list,
        model,
        save_path,
        args.batch_size,
        args.num_workers,
        device
    )

    # --------------------------------------------------
    # Create dataset index (paths + labels)
    # --------------------------------------------------
    all_rows = []
    class_dirs = sorted(glob.glob(os.path.join(save_path, "*")))

    for label, class_dir in enumerate(class_dirs):
        npy_files = glob.glob(os.path.join(class_dir, "*.npy"))
        for f in npy_files:
            all_rows.append([f, label])

    all_rows = shuffle(all_rows)
    np.save(os.path.join(save_path, "dataset_index.npy"), all_rows)

    print(f"\nDataset index saved to {save_path}/dataset_index.npy")

if __name__ == "__main__":
    main()

