import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


DATA_DIR = "IU_Xray"

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MODEL_PATH = "clinicalbert_indication_baseline.pth"

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


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("INDICATION-ONLY TEXT BASELINE EVALUATION")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD TEST DATA
# ============================================================

test_df = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "indication_test.csv"
    )
)

print(
    f"\nOriginal test rows: {len(test_df)}"
)

test_df = test_df.dropna(
    subset=[TEXT_COLUMN]
).copy()

test_df = test_df[
    test_df[TEXT_COLUMN]
    .astype(str)
    .str.strip() != ""
].copy()

print(
    f"Test rows used: {len(test_df)}"
)

print(
    f"Test reports: "
    f"{test_df['report_id'].nunique()}"
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
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

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

test_dataset = IndicationDataset(
    test_df,
    tokenizer
)

test_loader = DataLoader(
    test_dataset,
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

        self.dropout = torch.nn.Dropout(0.2)

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

        return self.classifier(
            cls_embedding
        )


print("\nLoading trained model...")

model = ClinicalBERTClassifier().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# INFERENCE
# ============================================================

print("\nRunning inference...")

all_labels = []
all_probabilities = []

with torch.no_grad():

    for batch in test_loader:

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

        probabilities = torch.sigmoid(
            logits
        )

        all_labels.append(
            labels.cpu().numpy()
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

print("\n")
print("=" * 60)
print("PER-CLASS RESULTS")
print("=" * 60)

results = []

roc_auc_values = []
f1_values = []

for i, target in enumerate(TARGETS):

    true = y_true[:, i]
    prob = y_prob[:, i]
    pred = y_pred[:, i]

    try:
        roc_auc = roc_auc_score(
            true,
            prob
        )
    except ValueError:
        roc_auc = float("nan")

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

    if not np.isnan(roc_auc):
        roc_auc_values.append(
            roc_auc
        )

    f1_values.append(f1)

    print(f"\n{target}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

    results.append({
        "finding": target,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


# ============================================================
# MACRO RESULTS
# ============================================================

macro_f1 = np.mean(f1_values)
macro_roc_auc = np.mean(
    roc_auc_values
)

print("\n")
print("=" * 60)
print("MACRO RESULTS")
print("=" * 60)

print(
    f"Macro F1:       {macro_f1:.4f}"
)

print(
    f"Macro ROC-AUC:  {macro_roc_auc:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "indication_baseline_results.csv",
    index=False
)

print("\nResults saved to:")
print("indication_baseline_results.csv")