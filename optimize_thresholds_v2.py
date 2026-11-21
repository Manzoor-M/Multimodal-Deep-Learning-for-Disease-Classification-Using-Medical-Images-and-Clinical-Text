import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "IU_Xray"

VAL_CSV = os.path.join(
    DATA_DIR,
    "indication_val.csv"
)

TEST_CSV = os.path.join(
    DATA_DIR,
    "indication_test.csv"
)

IMAGE_DIR = os.path.join(
    DATA_DIR,
    "NLMCXR_png"
)

CHECKPOINT_PATH = "multimodal_v2_best.pth"

TEXT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"

IMAGE_SIZE = 224
MAX_TEXT_LENGTH = 128
BATCH_SIZE = 8
NUM_WORKERS = 0

TARGET_COLUMNS = [
    "cardiomegaly",
    "atelectasis",
    "pleural_effusion",
    "emphysema",
    "pulmonary_congestion",
    "opacity",
    "granulomatous_disease"
]

# Threshold search range.
# Fine enough to find a useful threshold without
# unnecessarily evaluating thousands of possibilities.
THRESHOLDS = np.arange(
    0.05,
    0.96,
    0.01
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("MULTIMODAL V2 - VALIDATION THRESHOLD OPTIMIZATION")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()


# ============================================================
# TRANSFORM
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TEXT_MODEL
)

print()


# ============================================================
# DATASET
# ============================================================

class MultimodalDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        tokenizer,
        transform
    ):

        self.df = dataframe.reset_index(drop=True)

        self.image_dir = image_dir

        self.tokenizer = tokenizer

        self.transform = transform


    def __len__(self):

        return len(self.df)


    def __getitem__(self, idx):

        row = self.df.iloc[idx]


        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image_path = os.path.join(
            self.image_dir,
            str(row["image"])
        )

        if not os.path.exists(image_path):

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)


        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        text = str(
            row["indication_text"]
        )

        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
            return_tensors="pt"
        )

        input_ids = (
            encoding["input_ids"]
            .squeeze(0)
        )

        attention_mask = (
            encoding["attention_mask"]
            .squeeze(0)
        )


        # ----------------------------------------------------
        # LABELS
        # ----------------------------------------------------

        labels = torch.tensor(
            row[TARGET_COLUMNS]
            .values
            .astype(np.float32)
        )


        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# ============================================================
# LOAD AND CLEAN DATA
# ============================================================

def load_dataset_csv(csv_path, name):

    df = pd.read_csv(csv_path)

    print(
        f"{name} original rows: {len(df)}"
    )

    # Same filtering used by the original V2
    # evaluation script.

    df = df[
        df["indication_text"].notna()
        &
        (
            df["indication_text"]
            .astype(str)
            .str.strip()
            != ""
        )
    ].reset_index(drop=True)

    print(
        f"{name} rows used: {len(df)}"
    )

    if "report_id" in df.columns:

        print(
            f"{name} reports: "
            f"{df['report_id'].nunique()}"
        )

    print()

    return df


val_df = load_dataset_csv(
    VAL_CSV,
    "Validation"
)

test_df = load_dataset_csv(
    TEST_CSV,
    "Test"
)


# ============================================================
# DATA LOADERS
# ============================================================

val_dataset = MultimodalDataset(
    val_df,
    IMAGE_DIR,
    tokenizer,
    test_transform
)

test_dataset = MultimodalDataset(
    test_df,
    IMAGE_DIR,
    tokenizer,
    test_transform
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


print(
    f"Validation batches: {len(val_loader)}"
)

print(
    f"Test batches:       {len(test_loader)}"
)

print()


# ============================================================
# V2 MODEL
# ============================================================

class MultimodalV2(nn.Module):

    def __init__(self, num_classes=7):

        super().__init__()


        # ----------------------------------------------------
        # IMAGE ENCODER
        # ----------------------------------------------------

        self.image_encoder = models.efficientnet_b0(
            weights=None
        )

        image_features = (
            self.image_encoder
            .classifier[1]
            .in_features
        )

        self.image_encoder.classifier = (
            nn.Identity()
        )


        self.image_projection = nn.Sequential(

            nn.Linear(
                image_features,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3)
        )


        # ----------------------------------------------------
        # TEXT ENCODER
        # ----------------------------------------------------

        self.text_encoder = AutoModel.from_pretrained(
            TEXT_MODEL
        )

        text_features = (
            self.text_encoder
            .config
            .hidden_size
        )


        self.text_projection = nn.Sequential(

            nn.Linear(
                text_features,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3)
        )


        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Linear(
                512,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2)
        )


        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            128,
            num_classes
        )


    def forward(
        self,
        image,
        input_ids,
        attention_mask
    ):

        # IMAGE

        image_features = (
            self.image_encoder(image)
        )

        image_features = (
            self.image_projection(
                image_features
            )
        )


        # TEXT

        text_output = (
            self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        )

        text_features = (
            text_output
            .last_hidden_state[:, 0, :]
        )

        text_features = (
            self.text_projection(
                text_features
            )
        )


        # FUSION

        fused_features = torch.cat(
            [
                image_features,
                text_features
            ],
            dim=1
        )

        fused_features = self.fusion(
            fused_features
        )


        # CLASSIFIER

        logits = self.classifier(
            fused_features
        )

        return logits


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading trained V2 model...")

model = MultimodalV2(
    num_classes=len(TARGET_COLUMNS)
)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


print(
    f"Loaded checkpoint from epoch "
    f"{checkpoint.get('epoch', 'unknown')}"
)

print(
    f"Validation loss stored in checkpoint: "
    f"{checkpoint.get('val_loss', 'unknown')}"
)

print()


# ============================================================
# INFERENCE FUNCTION
# ============================================================

def get_predictions(loader, dataset_name):

    print(
        f"Running {dataset_name} inference..."
    )

    all_labels = []

    all_probabilities = []


    with torch.no_grad():

        for batch_idx, batch in enumerate(
            loader
        ):

            images = batch["image"].to(
                device,
                non_blocking=True
            )

            input_ids = batch["input_ids"].to(
                device,
                non_blocking=True
            )

            attention_mask = (
                batch["attention_mask"]
                .to(
                    device,
                    non_blocking=True
                )
            )

            labels = batch["labels"]


            logits = model(
                images,
                input_ids,
                attention_mask
            )


            probabilities = torch.sigmoid(
                logits
            )


            all_labels.append(
                labels.numpy()
            )

            all_probabilities.append(
                probabilities
                .cpu()
                .numpy()
            )


            if (
                (batch_idx + 1) % 20 == 0
                or
                (batch_idx + 1) == len(loader)
            ):

                print(
                    f"  Processed "
                    f"{batch_idx + 1}/"
                    f"{len(loader)} batches"
                )


    y_true = np.concatenate(
        all_labels,
        axis=0
    )

    y_prob = np.concatenate(
        all_probabilities,
        axis=0
    )


    print(
        f"{dataset_name} prediction shape: "
        f"{y_prob.shape}"
    )

    print()

    return y_true, y_prob


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

val_true, val_prob = get_predictions(
    val_loader,
    "validation"
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

test_true, test_prob = get_predictions(
    test_loader,
    "test"
)


# ============================================================
# FIND BEST THRESHOLD ON VALIDATION SET
# ============================================================

print("=" * 70)
print("OPTIMIZING THRESHOLDS ON VALIDATION SET")
print("=" * 70)

threshold_results = []

best_thresholds = []


for i, target in enumerate(
    TARGET_COLUMNS
):

    true = val_true[:, i]

    probabilities = val_prob[:, i]


    best_threshold = 0.5

    best_f1 = -1.0


    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)


        f1 = f1_score(
            true,
            predictions,
            zero_division=0
        )


        if f1 > best_f1:

            best_f1 = f1

            best_threshold = threshold


    best_thresholds.append(
        best_threshold
    )


    threshold_results.append({

        "target": target,

        "threshold": best_threshold,

        "validation_f1": best_f1

    })


    print(
        f"{target:25s} "
        f"threshold={best_threshold:.2f} "
        f"validation_F1={best_f1:.4f}"
    )


threshold_df = pd.DataFrame(
    threshold_results
)


print()


# ============================================================
# SAVE THRESHOLDS
# ============================================================

threshold_file = (
    "multimodal_v2_thresholds.csv"
)

threshold_df.to_csv(
    threshold_file,
    index=False
)

print(
    f"Thresholds saved to: "
    f"{threshold_file}"
)

print()


# ============================================================
# EVALUATE TEST SET USING FROZEN THRESHOLDS
# ============================================================

print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_results = []


for i, target in enumerate(
    TARGET_COLUMNS
):

    true = test_true[:, i]

    probabilities = test_prob[:, i]

    threshold = best_thresholds[i]


    predictions = (
        probabilities >= threshold
    ).astype(int)


    # ROC-AUC does not depend on threshold.

    try:

        auc = roc_auc_score(
            true,
            probabilities
        )

    except ValueError:

        auc = float("nan")


    precision = precision_score(
        true,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        true,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        true,
        predictions,
        zero_division=0
    )


    test_results.append({

        "target": target,

        "threshold": threshold,

        "roc_auc": auc,

        "precision": precision,

        "recall": recall,

        "f1": f1

    })


    print()

    print(target)

    print(
        f"  Threshold: {threshold:.2f}"
    )

    print(
        f"  ROC-AUC:   {auc:.4f}"
    )

    print(
        f"  Precision: {precision:.4f}"
    )

    print(
        f"  Recall:    {recall:.4f}"
    )

    print(
        f"  F1:        {f1:.4f}"
    )


# ============================================================
# MACRO METRICS
# ============================================================

test_results_df = pd.DataFrame(
    test_results
)

macro_f1 = (
    test_results_df["f1"]
    .mean()
)

macro_auc = (
    test_results_df["roc_auc"]
    .mean()
)


print()
print("=" * 70)
print("FINAL MACRO RESULTS")
print("=" * 70)

print(
    f"Macro F1:       {macro_f1:.4f}"
)

print(
    f"Macro ROC-AUC: {macro_auc:.4f}"
)

print("=" * 70)


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

results_file = (
    "multimodal_v2_threshold_optimized_results.csv"
)

test_results_df.to_csv(
    results_file,
    index=False
)


print()
print(
    f"Final results saved to: "
    f"{results_file}"
)

print()
print("=" * 70)
print("THRESHOLD OPTIMIZATION COMPLETE")
print("=" * 70)