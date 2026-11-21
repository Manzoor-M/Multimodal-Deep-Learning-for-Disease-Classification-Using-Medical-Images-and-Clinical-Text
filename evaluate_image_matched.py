import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "IU_Xray/indication_test.csv"

CHECKPOINT = "efficientnet_image_baseline.pth"

IMAGE_ROOT = "IU_Xray/NLMCXR_png"

TARGETS = [
    "cardiomegaly",
    "atelectasis",
    "pleural_effusion",
    "emphysema",
    "pulmonary_congestion",
    "opacity",
    "granulomatous_disease"
]

BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("MATCHED IMAGE-ONLY EVALUATION")
print("=" * 60)
print("Device:", DEVICE)


# ============================================================
# DATASET
# ============================================================

class ImageDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

        # Same preprocessing used for validation/test
        # in the original image-only baseline.
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # CSV contains the filename in the "image" column.
        image_filename = str(row["image"])

        image_path = os.path.join(
            IMAGE_ROOT,
            image_filename
        )

        if not os.path.exists(image_path):

            raise FileNotFoundError(
                f"\nImage not found:\n{image_path}\n"
                f"CSV row: {idx}\n"
                f"Image value: {image_filename}"
            )

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        labels = torch.tensor(
            [
                float(row[target])
                for target in TARGETS
            ],
            dtype=torch.float32
        )

        return image, labels


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

df = pd.read_csv(CSV_PATH)

print("Rows loaded from CSV:", len(df))

# ------------------------------------------------------------
# Keep only rows with valid indication text.
#
# This makes the image-only evaluation use EXACTLY the same
# test population used by the multimodal models.
# ------------------------------------------------------------

df = df[df["indication_text"].notna()]

df = df[
    df["indication_text"]
    .astype(str)
    .str.strip()
    != ""
]

df = df.reset_index(drop=True)

print(
    "Rows after removing missing indication:",
    len(df)
)

# Safety check

if len(df) != 1100:

    print(
        "\nWARNING:"
        f" Expected 1100 rows but found {len(df)}."
    )

    print(
        "Check the filtering/split before continuing."
    )


# ============================================================
# CHECK FOR DUPLICATE IMAGES
# ============================================================

duplicate_images = df["image"].duplicated().sum()

print(
    "Duplicate image filenames:",
    duplicate_images
)


# ============================================================
# CREATE DATASET
# ============================================================

dataset = ImageDataset(df)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    "Batches:",
    len(loader)
)


# ============================================================
# LOAD EFFICIENTNET-B0
# ============================================================

print("\nLoading EfficientNet-B0...")

model = models.efficientnet_b0(
    weights=None
)

num_features = (
    model.classifier[1].in_features
)

model.classifier[1] = nn.Linear(
    num_features,
    len(TARGETS)
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading checkpoint:")
print(CHECKPOINT)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

# Support both:
#
# 1. state_dict directly
# 2. checkpoint dictionary containing
#    "model_state_dict"

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# ============================================================
# RUN INFERENCE
# ============================================================

print("\nRunning inference...")

all_probabilities = []

all_labels = []

with torch.no_grad():

    for batch_idx, (images, labels) in enumerate(loader):

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.sigmoid(
            outputs
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        all_labels.append(
            labels.numpy()
        )

        if (batch_idx + 1) % 10 == 0:

            print(
                f"Processed batch "
                f"{batch_idx + 1}/{len(loader)}"
            )


y_prob = np.concatenate(
    all_probabilities,
    axis=0
)

y_true = np.concatenate(
    all_labels,
    axis=0
)

print("\nPrediction shape:", y_prob.shape)

print("Label shape:", y_true.shape)


# ============================================================
# SANITY CHECK
# ============================================================

assert y_prob.shape[0] == len(df)

assert y_true.shape[0] == len(df)

assert y_prob.shape[1] == len(TARGETS)

assert y_true.shape[1] == len(TARGETS)


# ============================================================
# CALCULATE METRICS
# ============================================================

results = []

for i, target in enumerate(TARGETS):

    true = y_true[:, i]

    probabilities = y_prob[:, i]

    # Same threshold used in the original evaluation
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # ROC-AUC
    try:

        auc = roc_auc_score(
            true,
            probabilities
        )

    except ValueError:

        auc = float("nan")


    # Precision
    precision = precision_score(
        true,
        predictions,
        zero_division=0
    )


    # Recall
    recall = recall_score(
        true,
        predictions,
        zero_division=0
    )


    # F1
    f1 = f1_score(
        true,
        predictions,
        zero_division=0
    )


    results.append({

        "Target": target,

        "ROC-AUC": auc,

        "Precision": precision,

        "Recall": recall,

        "F1": f1

    })


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# MACRO METRICS
# ============================================================

macro_f1 = results_df["F1"].mean()

macro_auc = results_df["ROC-AUC"].mean()


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("MATCHED IMAGE-ONLY TEST RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 80)

print(
    f"Macro F1:      {macro_f1:.4f}"
)

print(
    f"Macro ROC-AUC: {macro_auc:.4f}"
)

print("=" * 80)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = (
    "image_matched_results.csv"
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nResults saved to: {OUTPUT_FILE}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(
    f"Test images evaluated : {len(df)}"
)

print(
    f"Targets               : {len(TARGETS)}"
)

print(
    f"Macro ROC-AUC         : {macro_auc:.4f}"
)

print(
    f"Macro F1              : {macro_f1:.4f}"
)

print("=" * 60)