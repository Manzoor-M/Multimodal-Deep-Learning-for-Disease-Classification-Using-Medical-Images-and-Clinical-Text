import os
import pandas as pd
import numpy as np

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "IU_Xray"

TEST_CSV = os.path.join(
    BASE_DIR,
    "test.csv"
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "NLMCXR_png"
)

MODEL_PATH = "efficientnet_image_baseline.pth"


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
# DATASET
# ============================================================

class ChestXrayDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        transform=None
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = os.path.join(
            self.image_dir,
            row["image"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            row[TARGET_COLUMNS]
            .values
            .astype("float32")
        )

        return image, labels


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD TEST DATA
# ============================================================

test_df = pd.read_csv(
    TEST_CSV
)

print("\n==============================")
print("IMAGE BASELINE EVALUATION")
print("==============================")

print(
    "Test images:",
    len(test_df)
)

print(
    "Test reports:",
    test_df["report_id"].nunique()
)


# ============================================================
# DATA LOADER
# ============================================================

test_dataset = ChestXrayDataset(
    test_df,
    IMAGE_DIR,
    transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

model = efficientnet_b0(
    weights=None
)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    len(TARGET_COLUMNS)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# PREDICTIONS
# ============================================================

all_labels = []
all_probabilities = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        probabilities = torch.sigmoid(
            outputs
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


# ============================================================
# THRESHOLD
# ============================================================

y_pred = (
    y_prob >= 0.5
).astype(int)


# ============================================================
# PER-CLASS METRICS
# ============================================================

print("\n==============================")
print("PER-CLASS RESULTS")
print("==============================")

results = []


for i, target in enumerate(
    TARGET_COLUMNS
):

    true = y_true[:, i]

    prob = y_prob[:, i]

    pred = y_pred[:, i]


    # ROC-AUC

    try:

        auc = roc_auc_score(
            true,
            prob
        )

    except ValueError:

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
        "finding": target,
        "ROC-AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


    print(
        f"\n{target}"
    )

    print(
        f"  ROC-AUC:   {auc:.4f}"
    )

    print(
        f"  Precision:  {precision:.4f}"
    )

    print(
        f"  Recall:     {recall:.4f}"
    )

    print(
        f"  F1:         {f1:.4f}"
    )


# ============================================================
# MACRO METRICS
# ============================================================

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


macro_auc = np.nanmean([
    result["ROC-AUC"]
    for result in results
])


print("\n==============================")
print("MACRO RESULTS")
print("==============================")

print(
    f"Macro F1:       {macro_f1:.4f}"
)

print(
    f"Macro ROC-AUC:  {macro_auc:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "image_baseline_results.csv",
    index=False
)

print(
    "\nResults saved to:"
    "\nimage_baseline_results.csv"
)