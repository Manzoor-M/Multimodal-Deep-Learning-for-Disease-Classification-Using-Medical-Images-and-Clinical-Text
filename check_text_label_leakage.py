import pandas as pd
import re

DATA_DIR = "IU_Xray"

TARGETS = [
    "cardiomegaly",
    "atelectasis",
    "pleural_effusion",
    "emphysema",
    "pulmonary_congestion",
    "opacity",
    "granulomatous_disease"
]

# Terms that may appear in the report when the corresponding
# target is present.

KEYWORDS = {
    "cardiomegaly": [
        "cardiomegaly",
        "enlarged cardiac",
        "enlarged heart",
        "cardiac enlargement"
    ],

    "atelectasis": [
        "atelectasis"
    ],

    "pleural_effusion": [
        "pleural effusion",
        "pleural fluid"
    ],

    "emphysema": [
        "emphysema",
        "emphysematous"
    ],

    "pulmonary_congestion": [
        "pulmonary congestion",
        "vascular congestion",
        "pulmonary vascular congestion"
    ],

    "opacity": [
        "opacity",
        "opacities",
        "airspace opacity",
        "airspace opacities"
    ],

    "granulomatous_disease": [
        "granulomatous",
        "granuloma"
    ]
}


def contains_keyword(text, keywords):

    text = str(text).lower()

    return any(
        keyword.lower() in text
        for keyword in keywords
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    f"{DATA_DIR}/test.csv"
)

df = df.dropna(
    subset=["clinical_text"]
).copy()

df["clinical_text"] = (
    df["clinical_text"]
    .astype(str)
)


# ============================================================
# AUDIT
# ============================================================

print("=" * 70)
print("TEXT LABEL LEAKAGE AUDIT")
print("=" * 70)

print(f"Test rows: {len(df)}")

print("\n")


for target in TARGETS:

    keywords = KEYWORDS[target]

    target_positive = df[target] == 1

    keyword_present = df["clinical_text"].apply(
        lambda text: contains_keyword(
            text,
            keywords
        )
    )

    # How often the target-positive report explicitly
    # contains its target terminology.
    positive_with_keyword = (
        target_positive & keyword_present
    ).sum()

    positive_count = target_positive.sum()

    # How often target-negative reports contain the term.
    negative_with_keyword = (
        (~target_positive) & keyword_present
    ).sum()

    negative_count = (~target_positive).sum()

    positive_rate = (
        positive_with_keyword / positive_count
        if positive_count > 0
        else 0
    )

    negative_rate = (
        negative_with_keyword / negative_count
        if negative_count > 0
        else 0
    )

    print(target)
    print("-" * 40)

    print(
        f"Positive cases:              {positive_count}"
    )

    print(
        f"Positive cases containing term: "
        f"{positive_with_keyword}"
    )

    print(
        f"Positive keyword rate:        "
        f"{positive_rate:.2%}"
    )

    print(
        f"Negative cases containing term: "
        f"{negative_with_keyword}"
    )

    print(
        f"Negative keyword rate:        "
        f"{negative_rate:.2%}"
    )

    print(
        f"Keywords checked: {keywords}"
    )

    print()


# ============================================================
# SHOW EXAMPLES
# ============================================================

print("=" * 70)
print("EXAMPLES OF POSSIBLE LABEL LEAKAGE")
print("=" * 70)

for target in TARGETS:

    keywords = KEYWORDS[target]

    mask = (
        (df[target] == 1)
        &
        df["clinical_text"].apply(
            lambda text: contains_keyword(
                text,
                keywords
            )
        )
    )

    examples = df.loc[
        mask,
        ["clinical_text", target]
    ].head(2)

    print(f"\n### {target}")

    for _, row in examples.iterrows():

        print(
            "\nClinical text:"
        )

        print(
            row["clinical_text"][:1000]
        )

        print(
            f"\nLabel: {int(row[target])}"
        )