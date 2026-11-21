import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "IU_Xray"

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
VAL_CSV = os.path.join(BASE_DIR, "val.csv")

IMAGE_DIR = os.path.join(BASE_DIR, "NLMCXR_png")

MODEL_PATH = "efficientnet_image_baseline.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 7

BATCH_SIZE = 16
NUM_EPOCHS = 10

LEARNING_RATE = 1e-4

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
# PRINT ENVIRONMENT
# ============================================================

print("\n==============================")
print("IMAGE-ONLY BASELINE")
print("==============================")

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )


# ============================================================
# DATASET
# ============================================================

class ChestXrayDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform=None):

        self.df = dataframe.reset_index(drop=True)
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
            row[TARGET_COLUMNS].values.astype("float32")
        )

        return image, labels


# ============================================================
# TRANSFORMS
# ============================================================

weights = EfficientNet_B0_Weights.DEFAULT

train_transform = transforms.Compose([

    transforms.Resize(256),

    transforms.RandomResizedCrop(
        224,
        scale=(0.85, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD CSV FILES
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = ChestXrayDataset(
    train_df,
    IMAGE_DIR,
    train_transform
)

val_dataset = ChestXrayDataset(
    val_df,
    IMAGE_DIR,
    val_transform
)


# ============================================================
# CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

model = efficientnet_b0(
    weights=weights
)

# Replace ImageNet classifier

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# CLASS WEIGHTS
# ============================================================

positive_counts = train_df[TARGET_COLUMNS].sum()

negative_counts = (
    len(train_df) - positive_counts
)

pos_weight = (
    negative_counts / positive_counts
)

pos_weight = torch.tensor(
    pos_weight.values,
    dtype=torch.float32
).to(DEVICE)

print("\nPositive class counts:")

for column, count in positive_counts.items():

    print(
        f"{column:25}: {int(count)}"
    )


print("\nPositive weights:")

for column, weight in zip(
    TARGET_COLUMNS,
    pos_weight
):

    print(
        f"{column:25}: {weight.item():.2f}"
    )


# ============================================================
# LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")


for epoch in range(NUM_EPOCHS):

    print(
        f"\n========== Epoch "
        f"{epoch + 1}/{NUM_EPOCHS} =========="
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc="Training"
    )

    for images, labels in progress_bar:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()


        train_loss += (
            loss.item()
            * images.size(0)
        )


        progress_bar.set_postfix(
            loss=loss.item()
        )


    train_loss /= len(train_dataset)


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, labels in tqdm(
            val_loader,
            desc="Validation"
        ):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += (
                loss.item()
                * images.size(0)
            )


    val_loss /= len(val_dataset)


    print(
        f"\nTrain Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss:   {val_loss:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "✓ Best model saved"
        )


print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")

print(
    "Best validation loss:",
    round(best_val_loss, 4)
)

print(
    "Model saved to:",
    MODEL_PATH
)