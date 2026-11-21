import os
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "IU_Xray"

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

TEXT_COLUMN = "indication_text"

TARGETS = [
    "cardiomegaly",
    "atelectasis",
    "pleural_effusion",
    "emphysema",
    "pulmonary_congestion",
    "opacity",
    "granulomatous_disease"
]

MAX_LENGTH = 128

BATCH_SIZE = 8
EPOCHS = 5

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4

RANDOM_SEED = 42

MODEL_OUTPUT = "clinicalbert_indication_baseline.pth"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("INDICATION-ONLY TEXT BASELINE")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "indication_train.csv"
    )
)

val_df = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "indication_val.csv"
    )
)

print("\nOriginal dataset:")
print(
    f"Training rows:   {len(train_df)}"
)

print(
    f"Validation rows: {len(val_df)}"
)


# ============================================================
# REMOVE MISSING INDICATIONS
# ============================================================

train_df = train_df.dropna(
    subset=[TEXT_COLUMN]
).copy()

val_df = val_df.dropna(
    subset=[TEXT_COLUMN]
).copy()

train_df = train_df[
    train_df[TEXT_COLUMN]
    .astype(str)
    .str.strip() != ""
].copy()

val_df = val_df[
    val_df[TEXT_COLUMN]
    .astype(str)
    .str.strip() != ""
].copy()

print("\nAfter removing missing indications:")

print(
    f"Training rows:   {len(train_df)}"
)

print(
    f"Validation rows: {len(val_df)}"
)


# ============================================================
# DATASET
# ============================================================

class IndicationDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.tokenizer = tokenizer


    def __len__(self):

        return len(self.df)


    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        text = str(
            row[TEXT_COLUMN]
        )

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        labels = torch.tensor(
            row[TARGETS]
            .values
            .astype(np.float32)
        )

        return {
            "input_ids":
                encoding["input_ids"]
                .squeeze(0),

            "attention_mask":
                encoding["attention_mask"]
                .squeeze(0),

            "labels":
                labels
        }


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

train_dataset = IndicationDataset(
    train_df,
    tokenizer
)

val_dataset = IndicationDataset(
    val_df,
    tokenizer
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

class ClinicalBERTClassifier(
    torch.nn.Module
):

    def __init__(self):

        super().__init__()

        self.bert = AutoModel.from_pretrained(
            MODEL_NAME
        )

        hidden_size = (
            self.bert.config.hidden_size
        )

        self.dropout = torch.nn.Dropout(
            0.2
        )

        self.classifier = torch.nn.Linear(
            hidden_size,
            len(TARGETS)
        )


    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = (
            outputs.last_hidden_state[:, 0, :]
        )

        cls_embedding = self.dropout(
            cls_embedding
        )

        logits = self.classifier(
            cls_embedding
        )

        return logits


model = ClinicalBERTClassifier().to(device)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print(
    "\nCalculating positive class weights..."
)

positive_counts = (
    train_df[TARGETS]
    .sum()
    .values
)

negative_counts = (
    len(train_df) - positive_counts
)

pos_weights = (
    negative_counts /
    np.maximum(
        positive_counts,
        1
    )
)

pos_weights = torch.tensor(
    pos_weights,
    dtype=torch.float32,
    device=device
)

for target, positive, weight in zip(
    TARGETS,
    positive_counts,
    pos_weights.cpu().numpy()
):

    print(
        f"{target:25s} "
        f"positive={int(positive):4d} "
        f"pos_weight={weight:.2f}"
    )


# ============================================================
# LOSS
# ============================================================

criterion = BCEWithLogitsLoss(
    pos_weight=pos_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")

print("\n")
print("=" * 60)
print("TRAINING")
print("=" * 60)

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for batch in train_loader:

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids,
            attention_mask
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(
        train_loader
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for batch in val_loader:

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            logits = model(
                input_ids,
                attention_mask
            )

            loss = criterion(
                logits,
                labels
            )

            val_loss += loss.item()

    val_loss /= len(
        val_loader
    )


    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_OUTPUT
        )

        print(
            f"  ✓ Best model saved "
            f"(val loss = {val_loss:.4f})"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Model saved to: "
    f"{MODEL_OUTPUT}"
)