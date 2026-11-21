import pandas as pd


DATASET = "IU_Xray/multimodal_dataset.csv"

df = pd.read_csv(DATASET)

target_columns = [
    "cardiomegaly",
    "atelectasis",
    "pleural_effusion",
    "emphysema",
    "pulmonary_congestion",
    "opacity",
    "granulomatous_disease"
]


print("\n==============================")
print("MULTIMODAL DATASET INSPECTION")
print("==============================")


# Basic information

print(f"\nTotal rows: {len(df)}")
print(f"Unique reports: {df['report_id'].nunique()}")
print(f"Unique images: {df['image'].nunique()}")


# Missing values

print("\nMissing values:")

print(
    df[
        ["report_id", "image", "clinical_text"]
        + target_columns
    ].isnull().sum()
)


# Check binary targets

print("\nUnique values in target columns:")

for column in target_columns:

    print(
        f"{column:25} "
        f"{sorted(df[column].dropna().unique().tolist())}"
    )


# Positive counts

print("\nPositive samples:")

for column in target_columns:

    positives = int(df[column].sum())
    percentage = positives / len(df) * 100

    print(
        f"{column:25} "
        f"{positives:5} "
        f"({percentage:.2f}%)"
    )


# Number of positive labels per image

df["num_labels"] = df[target_columns].sum(axis=1)

print("\nLabels per image:")

print(
    df["num_labels"]
    .value_counts()
    .sort_index()
)


# Zero-label images

zero_labels = (df["num_labels"] == 0).sum()

print(
    f"\nImages with zero target findings: "
    f"{zero_labels}"
)


# Multi-label images

multi_label = (df["num_labels"] >= 2).sum()

print(
    f"Images with 2+ target findings: "
    f"{multi_label}"
)


# Clinical text

empty_text = (
    df["clinical_text"]
    .fillna("")
    .str.strip()
    .eq("")
    .sum()
)

print(
    f"\nImages with empty clinical text: "
    f"{empty_text}"
)


# Example records

print("\nExample records:")

print(
    df[
        ["image", "clinical_text"] + target_columns
    ].head(3).to_string(index=False)
)