import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "IU_Xray"

TRAIN_CSV = os.path.join(DATA_DIR, "indication_train.csv")
VAL_CSV = os.path.join(DATA_DIR, "indication_val.csv")

IMAGE_DIR = os.path.join(DATA_DIR, "NLMCXR_png")

TEXT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"

IMAGE_SIZE = 224
MAX_TEXT_LENGTH = 128

BATCH_SIZE = 8
NUM_EPOCHS = 10

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

CHECKPOINT_PATH = "multimodal_best.pth"

SEED = 42

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
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("MULTIMODAL MEDICAL IMAGE + CLINICAL TEXT TRAINING")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print()


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print(f"Training rows:   {len(train_df)}")
print(f"Validation rows: {len(val_df)}")

print()


# Remove rows with missing indication text
train_df = train_df[
    train_df["indication_text"].notna()
    & (train_df["indication_text"].astype(str).str.strip() != "")
].reset_index(drop=True)

val_df = val_df[
    val_df["indication_text"].notna()
    & (val_df["indication_text"].astype(str).str.strip() != "")
].reset_index(drop=True)

print(f"Training rows after text filtering:   {len(train_df)}")
print(f"Validation rows after text filtering: {len(val_df)}")

print()


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.85, 1.0)
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# ============================================================

class MultimodalDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        tokenizer,
        transform=None
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

        image_name = str(row["image"])

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        text = str(row["indication_text"])

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)

        attention_mask = encoding["attention_mask"].squeeze(0)

        # ----------------------------------------------------
        # LABELS
        # ----------------------------------------------------

        labels = torch.tensor(
            row[TARGET_COLUMNS].values.astype(np.float32)
        )

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# ============================================================
# TOKENIZER
# ============================================================

print("Loading BioClinicalBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TEXT_MODEL
)

print("Tokenizer loaded.")
print()


# ============================================================
# DATASETS
# ============================================================

train_dataset = MultimodalDataset(
    train_df,
    IMAGE_DIR,
    tokenizer,
    train_transform
)

val_dataset = MultimodalDataset(
    val_df,
    IMAGE_DIR,
    tokenizer,
    val_transform
)


# ============================================================
# DATALOADERS
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

print(f"Training batches:   {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print()


# ============================================================
# MODEL
# ============================================================

class MultimodalModel(nn.Module):

    def __init__(self, num_classes=7):

        super().__init__()

        # ----------------------------------------------------
        # IMAGE ENCODER
        # ----------------------------------------------------

        print("Loading EfficientNet-B0...")

        self.image_encoder = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        image_features = (
            self.image_encoder.classifier[1].in_features
        )

        # Remove original ImageNet classifier
        self.image_encoder.classifier = nn.Identity()

        # Project image features
        self.image_projection = nn.Sequential(
            nn.Linear(image_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # ----------------------------------------------------
        # TEXT ENCODER
        # ----------------------------------------------------

        print("Loading BioClinicalBERT...")

        self.text_encoder = AutoModel.from_pretrained(
            TEXT_MODEL
        )

        text_features = self.text_encoder.config.hidden_size

        # Project text features
        self.text_projection = nn.Sequential(
            nn.Linear(text_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(256 + 256, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )


    def forward(
        self,
        image,
        input_ids,
        attention_mask
    ):

        # ----------------------------------------------------
        # IMAGE FEATURES
        # ----------------------------------------------------

        image_features = self.image_encoder(image)

        image_features = self.image_projection(
            image_features
        )

        # ----------------------------------------------------
        # TEXT FEATURES
        # ----------------------------------------------------

        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token representation
        text_features = text_output.last_hidden_state[:, 0, :]

        text_features = self.text_projection(
            text_features
        )

        # ----------------------------------------------------
        # FEATURE FUSION
        # ----------------------------------------------------

        fused_features = torch.cat(
            [
                image_features,
                text_features
            ],
            dim=1
        )

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        logits = self.classifier(
            fused_features
        )

        return logits


# ============================================================
# CREATE MODEL
# ============================================================

model = MultimodalModel(
    num_classes=len(TARGET_COLUMNS)
)

model = model.to(device)

print()
print("Multimodal model created.")
print()


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("Calculating class weights...")

positive_counts = train_df[TARGET_COLUMNS].sum()

negative_counts = (
    len(train_df) - positive_counts
)

pos_weight = (
    negative_counts / positive_counts
)

pos_weight = torch.tensor(
    pos_weight.values,
    dtype=torch.float32,
    device=device
)

print()

for name, weight in zip(
    TARGET_COLUMNS,
    pos_weight.tolist()
):
    print(
        f"{name:25s} weight = {weight:.2f}"
    )

print()


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
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):

        images = batch["image"].to(
            device,
            non_blocking=True
        )

        input_ids = batch["input_ids"].to(
            device,
            non_blocking=True
        )

        attention_mask = batch["attention_mask"].to(
            device,
            non_blocking=True
        )

        labels = batch["labels"].to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        logits = model(
            images,
            input_ids,
            attention_mask
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(
                device,
                non_blocking=True
            )

            input_ids = batch["input_ids"].to(
                device,
                non_blocking=True
            )

            attention_mask = batch["attention_mask"].to(
                device,
                non_blocking=True
            )

            labels = batch["labels"].to(
                device,
                non_blocking=True
            )

            logits = model(
                images,
                input_ids,
                attention_mask
            )

            loss = criterion(
                logits,
                labels
            )

            total_loss += loss.item()

    return total_loss / len(val_loader)


# ============================================================
# TRAINING LOOP
# ============================================================

print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

best_val_loss = float("inf")

for epoch in range(1, NUM_EPOCHS + 1):

    print()
    print(
        f"Epoch {epoch}/{NUM_EPOCHS}"
    )

    train_loss = train_one_epoch()

    val_loss = validate()

    print(
        f"Train Loss: {train_loss:.4f}"
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
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "target_columns": TARGET_COLUMNS
            },
            CHECKPOINT_PATH
        )

        print(
            f"✓ Best model saved "
            f"(val loss = {val_loss:.4f})"
        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Checkpoint saved to: "
    f"{CHECKPOINT_PATH}"
)

print()
print("Training finished successfully.")