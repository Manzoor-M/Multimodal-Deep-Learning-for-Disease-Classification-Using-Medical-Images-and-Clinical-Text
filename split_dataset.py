import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = "IU_Xray/multimodal_dataset.csv"

TRAIN_FILE = "IU_Xray/train.csv"
VAL_FILE = "IU_Xray/val.csv"
TEST_FILE = "IU_Xray/test.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\n==============================")
print("LOADING DATASET")
print("==============================")

print(f"Total images: {len(df)}")
print(f"Unique reports: {df['report_id'].nunique()}")


# ============================================================
# GET UNIQUE REPORT IDs
# ============================================================

report_ids = df["report_id"].unique()


# ============================================================
# 70% TRAIN / 30% TEMP
# ============================================================

train_reports, temp_reports = train_test_split(
    report_ids,
    test_size=0.30,
    random_state=42
)


# ============================================================
# 15% VALIDATION / 15% TEST
# ============================================================

val_reports, test_reports = train_test_split(
    temp_reports,
    test_size=0.50,
    random_state=42
)


# ============================================================
# CREATE DATASETS
# ============================================================

train_df = df[
    df["report_id"].isin(train_reports)
].copy()

val_df = df[
    df["report_id"].isin(val_reports)
].copy()

test_df = df[
    df["report_id"].isin(test_reports)
].copy()


# ============================================================
# SAVE
# ============================================================

train_df.to_csv(TRAIN_FILE, index=False)
val_df.to_csv(VAL_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)


# ============================================================
# REPORT
# ============================================================

print("\n==============================")
print("DATASET SPLIT COMPLETE")
print("==============================")

print(
    f"\nTrain:"
    f"\n  Reports: {train_df['report_id'].nunique()}"
    f"\n  Images:  {len(train_df)}"
)

print(
    f"\nValidation:"
    f"\n  Reports: {val_df['report_id'].nunique()}"
    f"\n  Images:  {len(val_df)}"
)

print(
    f"\nTest:"
    f"\n  Reports: {test_df['report_id'].nunique()}"
    f"\n  Images:  {len(test_df)}"
)


# ============================================================
# VERIFY NO REPORT LEAKAGE
# ============================================================

train_ids = set(train_df["report_id"])
val_ids = set(val_df["report_id"])
test_ids = set(test_df["report_id"])


print("\n==============================")
print("LEAKAGE CHECK")
print("==============================")

print(
    "Train ∩ Validation:",
    len(train_ids & val_ids)
)

print(
    "Train ∩ Test:",
    len(train_ids & test_ids)
)

print(
    "Validation ∩ Test:",
    len(val_ids & test_ids)
)