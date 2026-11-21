import os
import pandas as pd

DATA_DIR = "IU_Xray"

INDICATION_FILE = os.path.join(
    DATA_DIR,
    "indication_dataset.csv"
)

TRAIN_FILE = os.path.join(
    DATA_DIR,
    "train.csv"
)

VAL_FILE = os.path.join(
    DATA_DIR,
    "val.csv"
)

TEST_FILE = os.path.join(
    DATA_DIR,
    "test.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("CREATING INDICATION-ONLY SPLITS")
print("=" * 60)

indication_df = pd.read_csv(
    INDICATION_FILE
)

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"\nIndication dataset: {len(indication_df)} rows")
print(f"Existing train:     {len(train_df)} rows")
print(f"Existing val:       {len(val_df)} rows")
print(f"Existing test:      {len(test_df)} rows")


# ============================================================
# PRESERVE EXISTING REPORT-LEVEL SPLIT
# ============================================================

train_reports = set(
    train_df["report_id"].astype(str)
)

val_reports = set(
    val_df["report_id"].astype(str)
)

test_reports = set(
    test_df["report_id"].astype(str)
)

print("\nExisting report split:")
print(f"Train reports: {len(train_reports)}")
print(f"Val reports:   {len(val_reports)}")
print(f"Test reports:  {len(test_reports)}")


# ============================================================
# NORMALIZE REPORT IDs
# ============================================================

indication_df["report_id"] = (
    indication_df["report_id"]
    .astype(str)
)

# ============================================================
# CREATE SPLITS
# ============================================================

indication_train = indication_df[
    indication_df["report_id"].isin(train_reports)
].copy()

indication_val = indication_df[
    indication_df["report_id"].isin(val_reports)
].copy()

indication_test = indication_df[
    indication_df["report_id"].isin(test_reports)
].copy()


# ============================================================
# CHECK REPORT OVERLAP
# ============================================================

train_result_reports = set(
    indication_train["report_id"]
)

val_result_reports = set(
    indication_val["report_id"]
)

test_result_reports = set(
    indication_test["report_id"]
)

print("\nReport overlap checks:")

print(
    "Train ∩ Val:",
    len(train_result_reports & val_result_reports)
)

print(
    "Train ∩ Test:",
    len(train_result_reports & test_result_reports)
)

print(
    "Val ∩ Test:",
    len(val_result_reports & test_result_reports)
)


# ============================================================
# MISSING INDICATION CHECK
# ============================================================

print("\nMissing indication text:")

for name, df in [
    ("Train", indication_train),
    ("Validation", indication_val),
    ("Test", indication_test)
]:

    missing = (
        df["indication_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        f"{name}: {missing}"
    )


# ============================================================
# SAVE
# ============================================================

train_output = os.path.join(
    DATA_DIR,
    "indication_train.csv"
)

val_output = os.path.join(
    DATA_DIR,
    "indication_val.csv"
)

test_output = os.path.join(
    DATA_DIR,
    "indication_test.csv"
)

indication_train.to_csv(
    train_output,
    index=False
)

indication_val.to_csv(
    val_output,
    index=False
)

indication_test.to_csv(
    test_output,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("SPLITS CREATED")
print("=" * 60)

print(
    f"Train rows:      {len(indication_train)}"
)

print(
    f"Validation rows: {len(indication_val)}"
)

print(
    f"Test rows:       {len(indication_test)}"
)

print("\nFiles:")

print(
    f"  {train_output}"
)

print(
    f"  {val_output}"
)

print(
    f"  {test_output}"
)