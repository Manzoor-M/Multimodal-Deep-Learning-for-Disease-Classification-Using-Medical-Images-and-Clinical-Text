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

TEST_CSV = os.path.join(
    DATA_DIR, "indication_test.csv"
)

IMAGE_DIR = os.path.join(
    DATA_DIR, "NLMCXR_png"
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


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("MULTIMODAL V2 EVALUATION")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()


# ============================================================
# TEST DATA
# ============================================================

test_df = pd.read_csv(TEST_CSV)

print(
    f"Original test rows: {len(test_df)}"
)

test_df = test_df[
    test_df["indication_text"].notna()
    & (
        test_df["indication_text"]
        .astype(str)
        .str.strip() != ""
    )
].reset_index(drop=True)

print(
    f"Test rows used: {len(test_df)}"
)

if "report_id" in test_df.columns:
    print(
        f"Test reports: "
        f"{test_df['report_id'].nunique()}"
    )

print()


# ============================================================
# TRANSFORMS
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

class MultimodalTestDataset(Dataset):

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

        # IMAGE
        image_path = os.path.join(
            self.image_dir,
            str(row["image"])
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        # TEXT
        text = str(
            row["indication_text"]
        )

        encoding = self.tokenizer(
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

        # LABELS
        labels = torch.tensor(
            row[TARGET_COLUMNS]
            .values.astype(np.float32)
        )

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


test_dataset = MultimodalTestDataset(
    test_df,
    IMAGE_DIR,
    tokenizer,
    test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print(
    f"Test batches: {len(test_loader)}"
)

print()


# ============================================================
# V2 MODEL
# ============================================================

class MultimodalV2(nn.Module):

    def __init__(self, num_classes=7):

        super().__init__()

        # IMAGE ENCODER
        self.image_encoder = (
            models.efficientnet_b0(
                weights=None
            )
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

        # TEXT ENCODER
        self.text_encoder = (
            AutoModel.from_pretrained(
                TEXT_MODEL
            )
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

        # FUSION
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

        # CLASSIFIER
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

        image_features = (
            self.image_encoder(image)
        )

        image_features = (
            self.image_projection(
                image_features
            )
        )

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
    f"Validation loss at checkpoint: "
    f"{checkpoint.get('val_loss', 'unknown')}"
)

print()


# ============================================================
# INFERENCE
# ============================================================

print("Running inference...")

all_labels = []
all_probabilities = []

with torch.no_grad():

    for batch in test_loader:

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
            probabilities.cpu().numpy()
        )


y_true = np.concatenate(
    all_labels,
    axis=0
)

y_prob = np.concatenate(
    all_probabilities,
    axis=0
)

y_pred = (
    y_prob >= 0.5
).astype(int)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("PER-CLASS RESULTS")
print("=" * 60)

results = []

for i, target in enumerate(
    TARGET_COLUMNS
):

    true = y_true[:, i]
    prob = y_prob[:, i]
    pred = y_pred[:, i]

    if len(np.unique(true)) == 2:
        auc = roc_auc_score(
            true,
            prob
        )
    else:
        auc = float("nan")

    precision = precision_score(
        true,
        pred,
        zero_division=0
    )

    recall = recall_score(
        true,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        true,
        pred,
        zero_division=0
    )

    results.append({
        "target": target,
        "roc_auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    print()
    print(target)

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

results_df = pd.DataFrame(results)

macro_f1 = results_df["f1"].mean()

macro_auc = results_df["roc_auc"].mean()

print()
print("=" * 60)
print("MACRO RESULTS")
print("=" * 60)

print(
    f"Macro F1:       {macro_f1:.4f}"
)

print(
    f"Macro ROC-AUC: {macro_auc:.4f}"
)


# ============================================================
# SAVE
# ============================================================

output_file = "multimodal_v2_results.csv"

results_df.to_csv(
    output_file,
    index=False
)

print()
print("Results saved to:")
print(output_file)

print()
print("=" * 60)
print("V2 EVALUATION COMPLETE")
print("=" * 60)