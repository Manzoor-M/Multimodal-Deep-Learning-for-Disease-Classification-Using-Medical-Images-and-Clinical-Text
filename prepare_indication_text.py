import os
import re
import pandas as pd
import xml.etree.ElementTree as ET


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "IU_Xray"
REPORT_DIR = os.path.join(
    DATA_DIR,
    "NLMCXR_reports",
    "ecgen-radiology"
)

MULTIMODAL_FILE = os.path.join(
    DATA_DIR,
    "multimodal_dataset.csv"
)


# ============================================================
# EXTRACT INDICATION FROM XML
# ============================================================

def extract_indication(xml_path):

    try:

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for abstract in root.iter("AbstractText"):

            label = abstract.attrib.get(
                "Label",
                ""
            ).upper()

            if label == "INDICATION":

                text = "".join(
                    abstract.itertext()
                ).strip()

                # Normalize whitespace
                text = re.sub(
                    r"\s+",
                    " ",
                    text
                )

                return text

    except Exception as e:

        print(
            f"Error reading {xml_path}: {e}"
        )

    return ""


# ============================================================
# LOAD ORIGINAL MULTIMODAL DATASET
# ============================================================

print("=" * 60)
print("BUILDING INDICATION-ONLY DATASET")
print("=" * 60)

df = pd.read_csv(
    MULTIMODAL_FILE
)

print(
    f"\nOriginal rows: {len(df)}"
)

print(
    f"Unique reports: "
    f"{df['report_id'].nunique()}"
)


# ============================================================
# BUILD REPORT → INDICATION MAPPING
# ============================================================

print("\nReading XML reports...")

indication_map = {}

xml_files = [
    f
    for f in os.listdir(REPORT_DIR)
    if f.lower().endswith(".xml")
]

print(
    f"XML reports found: {len(xml_files)}"
)


for filename in xml_files:

    xml_path = os.path.join(
        REPORT_DIR,
        filename
    )

    # XML filename is the report ID
    report_id = os.path.splitext(
        filename
    )[0]

    indication = extract_indication(
        xml_path
    )

    indication_map[report_id] = indication


# ============================================================
# MAP INDICATION TO DATASET
# ============================================================

df["indication_text"] = (
    df["report_id"]
    .astype(str)
    .map(indication_map)
    .fillna("")
)


# ============================================================
# CLEAN TEXT
# ============================================================

df["indication_text"] = (
    df["indication_text"]
    .astype(str)
    .str.replace(
        r"\s+",
        " ",
        regex=True
    )
    .str.strip()
)


# ============================================================
# STATISTICS
# ============================================================

missing = (
    df["indication_text"]
    .str.strip()
    .eq("")
    .sum()
)

non_empty = len(df) - missing

print("\nIndication statistics:")
print(
    f"Rows with indication: "
    f"{non_empty}"
)

print(
    f"Rows without indication: "
    f"{missing}"
)

print(
    f"Percentage with indication: "
    f"{100 * non_empty / len(df):.2f}%"
)


# ============================================================
# SHOW EXAMPLES
# ============================================================

print("\n")
print("=" * 60)
print("EXAMPLES")
print("=" * 60)

examples = (
    df[
        df["indication_text"].str.strip() != ""
    ]
    [["report_id", "indication_text"]]
    .drop_duplicates("report_id")
    .head(10)
)

for _, row in examples.iterrows():

    print(
        f"\nReport {row['report_id']}:"
    )

    print(
        row["indication_text"]
    )


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    DATA_DIR,
    "indication_dataset.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\n")
print("=" * 60)
print("COMPLETE")
print("=" * 60)

print(
    f"Saved to: {output_file}"
)